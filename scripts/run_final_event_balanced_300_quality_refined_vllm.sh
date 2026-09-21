#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${MODE:-event_balanced_300}"
RECORDS="${RECORDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
OUTPUT_PATH="${OUTPUT_PATH:-data/generated/final/run_final_event_balanced_300_vllm_quality_refined.jsonl}"
OUT_DIR="${OUT_DIR:-benchmark_final/output/event_balanced_300_quality_refined}"

MODE="$MODE" RECORDS="$RECORDS" MAX_RETRIES="$MAX_RETRIES" OUTPUT_PATH="$OUTPUT_PATH" OUT_DIR="$OUT_DIR" \
  bash scripts/run_final_vllm_pipeline.sh
