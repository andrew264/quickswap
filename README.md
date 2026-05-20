```shell
LD_LIBRARY_PATH="$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH" uv run quickswap.py
```