#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_DIR="${MODEL_DIR:-./models/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-qwen3-30b-a3b-instruct-2507}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
GPU_IDS="${GPU_IDS:-3,4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MOE_BACKEND="${MOE_BACKEND:-triton}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ "${USE_SYSTEM_CUDA:-0}" != "1" ]]; then
  unset CUDA_HOME
  unset CUDA_PATH
fi

NVIDIA_LIBRARY_PATHS="$(
  python - <<'PY'
import pathlib
import site

paths = []
for site_package in site.getsitepackages():
    nvidia_root = pathlib.Path(site_package) / "nvidia"
    if not nvidia_root.exists():
        continue
    for library_dir in sorted(nvidia_root.glob("*/lib")):
        if any(library_dir.glob("*.so*")):
            paths.append(str(library_dir))
print(":".join(dict.fromkeys(paths)))
PY
)"

if [[ -n "$NVIDIA_LIBRARY_PATHS" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "${USE_SYSTEM_CUDA:-0}" == "1" && -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  unset CUDA_HOME
fi

if [[ "${USE_SYSTEM_CUDA:-0}" == "1" && -n "${CUDA_PATH:-}" && ! -x "${CUDA_PATH}/bin/nvcc" ]]; then
  unset CUDA_PATH
fi

if [[ -z "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
  TENSOR_PARALLEL_SIZE="${#VISIBLE_GPU_ARRAY[@]}"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Create/update the conda environment first." >&2
  exit 127
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Local model directory does not exist: $MODEL_DIR" >&2
  echo "Run: ./scripts/download_qwen3_30b_a3b.sh" >&2
  exit 2
fi

echo "Serving local model: $MODEL_DIR"
echo "Served model name: $SERVED_MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "VLLM_USE_FLASHINFER_SAMPLER: $VLLM_USE_FLASHINFER_SAMPLER"
echo "MoE backend: $MOE_BACKEND"
echo "Endpoint: http://localhost:${PORT}/v1/chat/completions"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --moe-backend "$MOE_BACKEND"
