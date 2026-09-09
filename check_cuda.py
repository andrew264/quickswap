#!/usr/bin/env python3
"""Check the local CUDA, TensorRT, and ONNX Runtime GPU installation.

Library paths are prepared by check_cuda.sh before this file imports
onnxruntime.  A model can be supplied to additionally test session creation.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OK = "OK"
FAIL = "FAIL"
WARN = "WARN"


def report(status: str, message: str) -> None:
  print(f"[{status:4}] {message}")


def unique_paths(paths: Iterable[Path]) -> list[Path]:
  result: list[Path] = []
  seen: set[Path] = set()
  for path in paths:
    path = path.resolve()
    if path not in seen and path.is_dir():
      result.append(path)
      seen.add(path)
  return result


def package_dirs() -> list[Path]:
  candidates = [Path(sysconfig.get_paths()["purelib"])]
  candidates.extend(ROOT.glob(".venv/lib/python*/site-packages"))
  return unique_paths(candidates)


def find_library(directories: Iterable[Path], name: str) -> Path | None:
  for directory in directories:
    exact = directory / name
    if exact.is_file():
      return exact
    matches = sorted(directory.glob(f"{name}.*"))
    if matches:
      return matches[0]
  return None


def check_driver() -> bool:
  nvidia_smi = shutil.which("nvidia-smi")
  if not nvidia_smi:
    report(FAIL, "nvidia-smi was not found; install/load the NVIDIA driver")
    return False

  result = subprocess.run(
    [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    detail = (result.stderr or result.stdout).strip().splitlines()
    report(FAIL, f"NVIDIA driver is not usable: {detail[-1] if detail else 'nvidia-smi failed'}")
    return False

  gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
  report(OK, f"NVIDIA driver is usable ({'; '.join(gpus)})")
  return True


def check_libraries(site_dirs: list[Path]) -> tuple[bool, list[Path]]:
  cuda_dirs = [directory for site in site_dirs for directory in (site / "nvidia").glob("*/lib")]
  trt_dirs = [site / "tensorrt_libs" for site in site_dirs]
  ort_dirs = [site / "onnxruntime" / "capi" for site in site_dirs]

  checks = [
    ("CUDA runtime", cuda_dirs, "libcudart.so.12"),
    ("cuBLAS", cuda_dirs, "libcublas.so.12"),
    ("cuDNN", cuda_dirs, "libcudnn.so.9"),
    ("TensorRT", trt_dirs, "libnvinfer.so.10"),
    ("TensorRT plugin", trt_dirs, "libnvinfer_plugin.so.10"),
    ("TensorRT ONNX parser", trt_dirs, "libnvonnxparser.so.10"),
    ("ONNX Runtime core", ort_dirs, "libonnxruntime.so.1"),
    ("ONNX Runtime CUDA provider", ort_dirs, "libonnxruntime_providers_cuda.so"),
    ("ONNX Runtime TensorRT provider", ort_dirs, "libonnxruntime_providers_tensorrt.so"),
  ]

  all_ok = True
  found: list[Path] = []
  for label, directories, name in checks:
    path = find_library(directories, name)
    if path is None:
      report(FAIL, f"{label} library not found ({name})")
      all_ok = False
    else:
      report(OK, f"{label}: {path}")
      found.append(path)

  return all_ok, found


def load_libraries(paths: Iterable[Path]) -> bool:
  all_ok = True
  mode = getattr(ctypes, "RTLD_GLOBAL", 0)
  for path in paths:
    try:
      ctypes.CDLL(str(path), mode=mode)
    except OSError as error:
      report(FAIL, f"Could not load {path.name}: {error}")
      all_ok = False
  return all_ok


def import_onnxruntime():
  try:
    import onnxruntime as ort
  except Exception as error:  # noqa: BLE001 - this is a diagnostic tool
    report(FAIL, f"onnxruntime import failed: {error}")
    return None

  if not hasattr(ort, "get_available_providers"):
    report(FAIL, "onnxruntime package is incomplete (get_available_providers is missing)")
    report(WARN, "Repair the virtualenv with: uv sync")
    return None

  try:
    version = importlib.metadata.version("onnxruntime-gpu")
  except importlib.metadata.PackageNotFoundError:
    version = "unknown version"
  report(OK, f"onnxruntime-gpu {version} imported with Python {sys.version.split()[0]}")
  return ort


def check_providers(ort) -> bool:
  try:
    providers = ort.get_available_providers()
  except Exception as error:  # noqa: BLE001 - this is a diagnostic tool
    report(FAIL, f"Could not enumerate ONNX Runtime providers: {error}")
    return False

  report(OK, f"ONNX Runtime providers: {', '.join(providers)}")
  all_ok = True
  for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
    if provider in providers:
      report(OK, f"{provider} is registered")
    else:
      report(FAIL, f"{provider} is not registered")
      all_ok = False
  return all_ok


def model_candidates(explicit_model: str | None) -> list[Path]:
  if explicit_model:
    return [Path(explicit_model).expanduser().resolve()]

  candidates = sorted((ROOT / ".models").glob("*.onnx"))
  candidates.extend(sorted(ROOT.glob("**/onnxruntime/datasets/mul_1.onnx")))
  return candidates


def check_session(ort, model: Path | None) -> bool:
  if model is None:
    report(WARN, "No ONNX model supplied; provider registration was checked, session creation was not")
    report(WARN, "After downloading models, run: bash check_cuda.sh --model .models/yoloface.onnx")
    return True
  if not model.is_file():
    report(FAIL, f"Model does not exist: {model}")
    return False

  try:
    session = ort.InferenceSession(
      str(model),
      providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = session.get_providers()
  except Exception as error:  # noqa: BLE001 - this is a diagnostic tool
    report(FAIL, f"ONNX Runtime could not create a GPU session for {model.name}: {error}")
    return False

  report(OK, f"GPU session created for {model} ({', '.join(active)})")
  return "TensorrtExecutionProvider" in active


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", help="Optional ONNX model to use for a real session-creation test")
  args = parser.parse_args()

  print(f"Project: {ROOT}")
  print(f"Python:  {sys.executable}")
  print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")

  sites = package_dirs()
  if not sites:
    report(FAIL, "No Python site-packages directory found")
    return 1

  driver_ok = check_driver()
  libraries_ok, provider_libraries = check_libraries(sites)
  load_ok = load_libraries(provider_libraries)
  ort = import_onnxruntime()
  providers_ok = bool(ort) and check_providers(ort)

  model = next(iter(model_candidates(args.model)), None)
  session_ok = bool(ort) and check_session(ort, model)

  print()
  if driver_ok and libraries_ok and load_ok and providers_ok and session_ok:
    report(OK, "CUDA + TensorRT are ready")
    return 0

  report(FAIL, "CUDA + TensorRT are not ready")
  print("Run this diagnostic again after fixing the first FAIL line.")
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
