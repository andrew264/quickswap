import argparse
import os
import sys
import glob
import time
import subprocess
import threading
import queue
import urllib.request

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

# ==========================================
# 1. TEMPLATES & MATH
# ==========================================

TEMPLATES = {
  'arcface_112': np.array([[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]]),
  'arcface_128': np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]]),
  'ffhq_512': np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904], [0.39308822, 0.72541100], [0.61150205, 0.72490465]])
}


def warp_face(frame, landmarks, template_name, crop_size):
  template = TEMPLATES[template_name] * crop_size
  affine_matrix = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
  crop = cv2.warpAffine(frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
  return crop, affine_matrix


def paste_back(full_frame, crop, affine_matrix):
  inverse_matrix = cv2.invertAffineTransform(affine_matrix)
  h, w = crop.shape[:2]

  # Create Box Mask with blur (face_mask_blur = 0.3)
  mask = np.ones((h, w), dtype=np.float32)
  blur_amount = int(w * 0.5 * 0.3)
  if blur_amount > 0:
    mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
  mask = mask[..., np.newaxis]

  # Map back to full frame
  fh, fw = full_frame.shape[:2]
  inverse_mask = cv2.warpAffine(mask, inverse_matrix, (fw, fh)).clip(0, 1)[..., np.newaxis]
  inverse_crop = cv2.warpAffine(crop, inverse_matrix, (fw, fh), borderMode=cv2.BORDER_REPLICATE)

  return full_frame * (1 - inverse_mask) + inverse_crop * inverse_mask


def implode_pixel_boost(crop, boost_total=2, model_size=(256, 256)):
  boosted = crop.reshape(model_size[0], boost_total, model_size[1], boost_total, 3)
  return boosted.transpose(1, 3, 0, 2, 4).reshape(boost_total ** 2, model_size[0], model_size[1], 3)


def explode_pixel_boost(frames, boost_total=2, model_size=(256, 256), boost_size=(512, 512)):
  crop = np.stack(frames).reshape(boost_total, boost_total, model_size[0], model_size[1], 3)
  return crop.transpose(2, 0, 3, 1, 4).reshape(boost_size[0], boost_size[1], 3)


# ==========================================
# 2. MODELS & DOWNLOADER
# ==========================================

MODELS = {
  'yoloface': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx',
  'arcface': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx',
  'hyperswap': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx',
  'gfpgan': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx'
}


def download_models(model_dir=".models"):
  os.makedirs(model_dir, exist_ok=True)
  paths = {}
  for name, url in MODELS.items():
    path = os.path.join(model_dir, f"{name}.onnx")
    if not os.path.exists(path):
      print(f"Downloading {name}...")
      urllib.request.urlretrieve(url, path)
    paths[name] = path
  return paths


# ==========================================
# 3. ML RUNNERS
# ==========================================

class Pipeline:
  def __init__(self, model_paths):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    opt = onnxruntime.SessionOptions()
    opt.log_severity_level = 3

    self.yolo = onnxruntime.InferenceSession(model_paths['yoloface'], sess_options=opt, providers=providers)
    self.arcface = onnxruntime.InferenceSession(model_paths['arcface'], sess_options=opt, providers=providers)
    self.swapper = onnxruntime.InferenceSession(model_paths['hyperswap'], sess_options=opt, providers=providers)
    self.enhancer = onnxruntime.InferenceSession(model_paths['gfpgan'], sess_options=opt, providers=providers)

  def detect_faces(self, frame):
    h, w = frame.shape[:2]
    scale = min(640 / h, 640 / w)
    nw, nh = int(w * scale), int(h * scale)
    pad_x, pad_y = max(0, 640 - nw) // 2, max(0, 640 - nh) // 2

    resized = cv2.resize(frame, (nw, nh))
    padded = np.pad(resized, ((pad_y, 640 - nh - pad_y), (pad_x, 640 - nw - pad_x), (0, 0)))
    blob = (padded[..., ::-1] / 255.0).transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

    detection = self.yolo.run(None, {'input': blob})[0]
    detection = np.squeeze(detection).T
    bboxes_raw, scores_raw, landmarks_raw = np.split(detection, [4, 5], axis=1)

    keep = np.where(scores_raw > 0.5)[0]
    faces = []
    if len(keep) > 0:
      bboxes = bboxes_raw[keep]
      scores = scores_raw[keep].ravel()
      landmarks = landmarks_raw[keep].reshape(-1, 5, 3)[..., :2]

      bboxes = np.column_stack(
        [(bboxes[:, 0] - bboxes[:, 2] / 2 - pad_x) / scale, (bboxes[:, 1] - bboxes[:, 3] / 2 - pad_y) / scale, (bboxes[:, 0] + bboxes[:, 2] / 2 - pad_x) / scale,
          (bboxes[:, 1] + bboxes[:, 3] / 2 - pad_y) / scale])
      landmarks = (landmarks - [pad_x, pad_y]) / scale

      bboxes_nms = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in bboxes]
      indices = cv2.dnn.NMSBoxes(bboxes_nms, scores, 0.5, 0.4)
      for i in indices:
        faces.append({'bbox': bboxes[i], 'landmarks': landmarks[i], 'score': float(scores[i])})
    return faces

  def get_embedding(self, frame, landmarks):
    crop, _ = warp_face(frame, landmarks, 'arcface_112', (112, 112))
    blob = (crop / 127.5 - 1)[..., ::-1].transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
    embed = self.arcface.run(None, {'input': blob})[0].ravel()
    return embed / np.linalg.norm(embed)

  def process_frame(self, frame, pairs, frame_num=0, debug=False):
    faces = self.detect_faces(frame)
    out_frame = frame.copy()
    debug_data = []

    for face in faces:
      vid_emb = self.get_embedding(frame, face['landmarks'])

      best_dist = 1.0
      best_src_emb = None
      for ref_emb, src_emb in pairs:
        dist = 1 - np.dot(vid_emb, ref_emb)
        if dist < best_dist and dist < 0.4:
          best_dist = dist
          best_src_emb = src_emb

      debug_data.append({'face': face, 'dist': best_dist, 'swapped': best_src_emb is not None})
      if best_src_emb is None:
        continue

      # 1. Hyperswap with 512x512 Pixel Boost
      crop_swap, mat_swap = warp_face(out_frame, face['landmarks'], 'arcface_128', (512, 512))
      crop_swap_norm = (crop_swap[..., ::-1] / 255.0 - 0.5) / 0.5
      tiles = implode_pixel_boost(crop_swap_norm, 2, (256, 256))
      out_tiles = []

      for tile in tiles:
        blob = tile.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]
        out_tile = self.swapper.run(None, {'source': best_src_emb.reshape(1, -1), 'target': blob})[0][0]
        out_tiles.append(out_tile.transpose(1, 2, 0))

      swapped_512 = explode_pixel_boost(out_tiles, 2, (256, 256), (512, 512))
      swapped_512 = swapped_512 * 0.5 + 0.5
      swapped_512 = np.clip(swapped_512, 0, 1)
      swapped_512 = np.round(swapped_512 * 255.0).astype(np.uint8)[..., ::-1]
      out_frame = paste_back(out_frame, swapped_512, mat_swap)

      # 2. Enhance with GFPGAN (80% blend)
      crop_enh, mat_enh = warp_face(out_frame, face['landmarks'], 'ffhq_512', (512, 512))
      blob_enh = (crop_enh[..., ::-1] / 255.0 - 0.5) / 0.5
      blob_enh = blob_enh.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

      enhanced = self.enhancer.run(None, {'input': blob_enh})[0][0]
      enhanced = np.clip(enhanced, -1.0, 1.0)
      enhanced = (enhanced + 1.0) / 2.0
      enhanced = enhanced.transpose(1, 2, 0)
      enhanced = np.round(enhanced * 255.0).astype(np.uint8)[..., ::-1]

      crop_enh = crop_enh.astype(np.uint8)
      blended = cv2.addWeighted(crop_enh, 0.2, enhanced, 0.8, 0)
      out_frame = paste_back(out_frame, blended, mat_enh)
    if debug:
      cv2.putText(out_frame, f"Frame: {frame_num}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      for data in debug_data:
        face = data['face']
        bbox = face['bbox'].astype(int)
        score = face['score']
        dist = data['dist']
        swapped = data['swapped']

        color = (0, 255, 0) if swapped else (0, 0, 255)
        cv2.rectangle(out_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        sim_score = 1.0 - dist
        label = f"Conf:{score:.2f} Sim:{sim_score:.2f}"
        text_y = max(bbox[1] - 10, 20)
        cv2.putText(out_frame, label, (bbox[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out_frame.astype(np.uint8)


# ==========================================
# 4. FFMPEG QUEUE STREAMING
# ==========================================

def get_video_info(path):
  cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames', '-of', 'csv=p=0', path]
  out = subprocess.check_output(cmd).decode('utf-8').strip().split(',')
  w, h = int(out[0]), int(out[1])
  num, den = out[2].split('/')
  fps = float(num) / float(den)
  frames = int(out[3]) if len(out) > 3 and out[3].strip() else 0
  return w, h, fps, frames


def get_best_encoder():
  try:
    res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    if "av1_nvenc" in res.stdout: return "av1_nvenc"
    if "hevc_nvenc" in res.stdout: return "hevc_nvenc"
    if "h264_nvenc" in res.stdout: return "h264_nvenc"
  except: pass
  return "libx264"


# ==========================================
# 5. MAIN
# ==========================================

def main():
  parser = argparse.ArgumentParser(description="QuickSwap")
  parser.add_argument("p1_dir", help="Path to p1 directory")
  parser.add_argument("--debug", action="store_true", help="Draw debug information on frames")
  parser.add_argument("--range", type=str, help="Process specific frames (e.g. '300,500' or '300' or ',500')")
  args = parser.parse_args()

  p1_dir = args.p1_dir

  start_frame, end_frame = 1, float('inf')

  if args.range:
    parts = args.range.split(',')
    if len(parts) == 1:
      start_frame = int(parts[0]) if parts[0].strip() else 1
    else:
      start_frame = int(parts[0]) if parts[0].strip() else 1
      end_frame = int(parts[1]) if parts[1].strip() else float('inf')

  in_video = os.path.join(p1_dir, "input.mp4")
  out_video = os.path.join(p1_dir, "output.mp4")
  ref_dir = os.path.join(p1_dir, "references")
  src_dir = os.path.join(p1_dir, "sources")

  if not os.path.exists(in_video):
    print(f"Error: {in_video} not found.")
    sys.exit(1)

  for d in [ref_dir, src_dir]:
    if not os.path.exists(d):
      print(f"Error: Directory {d} not found.")
      sys.exit(1)

  model_paths = download_models()
  pipe = Pipeline(model_paths)

  # 1. Load Explicit Pairs (matches exact filenames)
  pairs = []
  ref_imgs = glob.glob(os.path.join(ref_dir, "*.*"))

  for ref_path in ref_imgs:
    base_name = os.path.basename(ref_path)
    src_path = os.path.join(src_dir, base_name)

    if not os.path.exists(src_path):
      print(f"Warning: Match for {base_name} not found in sources/. Skipping.")
      continue

    ref_img = cv2.imread(ref_path)
    src_img = cv2.imread(src_path)

    ref_faces = pipe.detect_faces(ref_img)
    src_faces = pipe.detect_faces(src_img)

    if not ref_faces or not src_faces:
      print(f"Warning: Could not detect faces in {base_name} pair. Skipping.")
      continue

    ref_emb = pipe.get_embedding(ref_img, ref_faces[0]['landmarks'])
    src_emb = pipe.get_embedding(src_img, src_faces[0]['landmarks'])
    pairs.append((ref_emb, src_emb))

  if not pairs:
    print("Error: No valid reference-source pairs found. Exiting.")
    sys.exit(1)

  print(f"Loaded {len(pairs)} reference-source pairs.")

  # 2. Setup FFmpeg Queues
  w, h, fps, total_frames = get_video_info(in_video)
  encoder = get_best_encoder()
  print(f"Video: {w}x{h} @ {fps:.2f}fps. Using encoder: {encoder}")

  frame_size = w * h * 3
  q_in = queue.Queue(maxsize=60)
  q_out = queue.Queue(maxsize=60)

  tmp_video = os.path.join(p1_dir, "temp_video_only.mp4")

  # Reader Thread
  def reader():
    cmd = ['ffmpeg', '-i', in_video, '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while True:
      raw = proc.stdout.read(frame_size)
      if not raw: break
      q_in.put(np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)))
    q_in.put(None)
    proc.communicate()

  # Writer Thread
  def writer():
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-framerate', str(fps), '-i', '-', '-c:v', encoder, '-cq', '20',
      '-preset', 'p4', '-pix_fmt', 'yuv420p', tmp_video]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while True:
      frame = q_out.get()
      if frame is None: break
      proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.communicate()

  threading.Thread(target=reader, daemon=True).start()
  threading.Thread(target=writer, daemon=True).start()

  # 3. Processing Loop (Main Thread)
  print("Processing video frames in VRAM...")

  expected_frames = total_frames - start_frame + 1 if total_frames > 0 else 0
  if end_frame != float('inf'):
    expected_frames = min(expected_frames, end_frame - start_frame + 1)

  pbar = tqdm(total=expected_frames if expected_frames > 0 else None, unit='frames')
  frame_idx = 1

  while True:
    frame = q_in.get()
    if frame is None:
      q_out.put(None)
      break

    if frame_idx < start_frame:
      frame_idx += 1
      continue

    if frame_idx > end_frame:
      q_out.put(None)
      break

    out_frame = pipe.process_frame(frame, pairs, frame_num=frame_idx, debug=args.debug)

    q_out.put(out_frame)
    pbar.update(1)
    frame_idx += 1
  pbar.close()

  # Wait for writer to finish emptying the queue
  while not q_out.empty():
    time.sleep(0.5)
  time.sleep(2)

  # 4. Restore Audio
  print("Restoring audio...")
  start_time = max(0.0, (start_frame - 1) / fps)
  cmd_audio = ['ffmpeg', '-y', '-i', tmp_video, '-ss', str(start_time), '-i', in_video, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0?', '-shortest', out_video]
  subprocess.run(cmd_audio, stderr=subprocess.DEVNULL)

  if os.path.exists(tmp_video):
    os.remove(tmp_video)

  print(f"Done! Saved to {out_video}")


if __name__ == "__main__":
  main()
