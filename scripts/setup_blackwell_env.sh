#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-multi-db-blackwell}"
CUDA_BACKEND="${CUDA_BACKEND:-cu129}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 127
fi

conda env create -n "$ENV_NAME" -f environment-blackwell.yml || \
  conda env update -n "$ENV_NAME" -f environment-blackwell.yml --prune

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install -U pip uv

uv pip install vllm --torch-backend="$CUDA_BACKEND"
python -m pip install -e .

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
PY

echo
echo "Blackwell environment is ready: $ENV_NAME"
echo "Activate it with: conda activate $ENV_NAME"
