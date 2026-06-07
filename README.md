```shell
LD_LIBRARY_PATH="$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':').venv/lib/python3.14/site-packages/tensorrt_libs:$LD_LIBRARY_PATH" \
  uv run quickswap.py
```