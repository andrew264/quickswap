#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Create the environment first with: uv sync" >&2
  exit 1
fi

shopt -s nullglob
library_dirs=()
for site_packages in "$ROOT_DIR"/.venv/lib/python*/site-packages; do
  library_dirs+=("$site_packages"/nvidia/*/lib)
  library_dirs+=("$site_packages/tensorrt_libs")
  library_dirs+=("$site_packages/onnxruntime/capi")
done

library_path=""
for directory in "${library_dirs[@]}"; do
  [[ -d "$directory" ]] || continue
  case ":$library_path:" in
    *":$directory:"*) ;;
    *) library_path+="${library_path:+:}$directory" ;;
  esac
done

export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/quickswap.py" "$@"
