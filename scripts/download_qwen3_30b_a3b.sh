#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_DIR="${MODEL_DIR:-./models/Qwen3-30B-A3B-Instruct-2507}"

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(hf download "$MODEL_ID" --local-dir "$MODEL_DIR")
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_CMD=(huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR")
else
  echo "Neither hf nor huggingface-cli is installed. Run: python -m pip install -U huggingface_hub" >&2
  exit 127
fi

mkdir -p "$(dirname "$MODEL_DIR")"

echo "Downloading $MODEL_ID"
echo "Target: $MODEL_DIR"

"${HF_DOWNLOAD_CMD[@]}"

echo
echo "Done."
echo "Local model path: $MODEL_DIR"
