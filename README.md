Check the GPU setup at any time with:

```shell
bash check_cuda.sh
```

Once the models exist, pass one to also test ONNX Runtime session creation:

```shell
bash check_cuda.sh --model .models/yoloface.onnx
```

Run QuickSwap with the same automatically discovered CUDA/TensorRT library
paths:

```shell
bash run_quickswap.sh <p1_dir>
```

The checker is deliberately separate from model downloading, so it is safe to
run after the project has been idle for a while. It exits with status `0` only
when the NVIDIA driver, CUDA libraries, TensorRT libraries, and both ONNX
Runtime GPU providers are ready.
