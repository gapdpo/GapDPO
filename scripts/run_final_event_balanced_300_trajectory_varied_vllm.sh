#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${MODE:-event_balanced_300_trajectory_varied}"
RECORDS="${RECORDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-4}"
SHORTCUT_GATE="${SHORTCUT_GATE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-data/generated/final/run_final_event_balanced_300_trajectory_varied_vllm.jsonl}"
OUT_DIR="${OUT_DIR:-benchmark_final/output/event_balanced_300_trajectory_varied}"

export MODE RECORDS MAX_RETRIES SHORTCUT_GATE OUTPUT_PATH OUT_DIR

bash scripts/run_final_vllm_pipeline.sh
