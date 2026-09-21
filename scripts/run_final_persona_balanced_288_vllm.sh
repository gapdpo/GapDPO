#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${MODE:-persona_balanced_288}"
RECORDS="${RECORDS:-288}"
OUTPUT_PATH="${OUTPUT_PATH:-data/generated/final/run_final_persona_balanced_288_vllm.jsonl}"
OUT_DIR="${OUT_DIR:-benchmark_final/output/persona_balanced_288}"

MODE="$MODE" RECORDS="$RECORDS" OUTPUT_PATH="$OUTPUT_PATH" OUT_DIR="$OUT_DIR" \
  bash scripts/run_final_vllm_pipeline.sh
