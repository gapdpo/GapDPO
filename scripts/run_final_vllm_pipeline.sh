#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${MODE:-event_balanced_300}"
RECORDS="${RECORDS:-300}"
MODEL_ENDPOINT_PORT="${MODEL_ENDPOINT_PORT:-8001}"
OUTPUT_PATH="${OUTPUT_PATH:-data/generated/final/run_final_${MODE}_vllm.jsonl}"
OUT_DIR="${OUT_DIR:-benchmark_final/output/${MODE}}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
VLLM_MODEL="${VLLM_MODEL:-local-qwen3-30b-a3b-instruct-2507}"
SURFACE="${SURFACE:-typed_raw}"
WINDOW_SIZE="${WINDOW_SIZE:-6}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SHORTCUT_GATE="${SHORTCUT_GATE:-0}"
CONDA_ENV="${CONDA_ENV:-multi-db-blackwell}"

if [[ "${STREAM_V2_CONDA_REEXEC:-0}" != "1" ]] && ! command -v vllm >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    echo "vllm not found on PATH; re-running inside conda env: ${CONDA_ENV}"
    export STREAM_V2_CONDA_REEXEC=1
    exec conda run --no-capture-output -n "$CONDA_ENV" bash "$0" "$@"
  fi
  echo "vllm is not installed or not on PATH, and conda was not found." >&2
  echo "Activate an environment with vllm, or set CONDA_ENV to the environment name." >&2
  exit 127
fi

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping vLLM server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting vLLM server in background..."
PORT="$MODEL_ENDPOINT_PORT" ./scripts/serve_local_qwen3_30b_a3b.sh &
SERVER_PID="$!"

echo "Waiting for vLLM server at http://localhost:${MODEL_ENDPOINT_PORT}/v1/models"
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited before becoming ready." >&2
    wait "$SERVER_PID" || true
    exit 1
  fi

  if python - "$MODEL_ENDPOINT_PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = sys.argv[1]
with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    break
  fi

  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM server." >&2
    exit 1
  fi
  sleep 5
done

PYTHONPATH=src:. python -m benchmark_final.generate_persona_stream_corpus \
  --mode "$MODE" \
  --records "$RECORDS" \
  --out "$OUTPUT_PATH" \
  --vllm-endpoint "http://localhost:${MODEL_ENDPOINT_PORT}/v1/chat/completions" \
  --vllm-model "$VLLM_MODEL" \
  --max-retries "$MAX_RETRIES"

BUILD_ARGS=(
  --source "$OUTPUT_PATH"
  --out "$OUT_DIR"
  --surface "$SURFACE"
  --window-size "$WINDOW_SIZE"
)
if [[ "$SHORTCUT_GATE" == "1" ]]; then
  BUILD_ARGS+=(--shortcut-gate)
fi

PYTHONPATH=src:. python -m benchmark_final.build_stream_benchmark "${BUILD_ARGS[@]}"

PYTHONPATH=src:. python - "$OUTPUT_PATH" "$OUT_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

source_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
records = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
yield_report = json.loads((out_dir / "yield_report.json").read_text(encoding="utf-8"))
shortcut_report_path = out_dir / "shortcut_report.json"
shortcut_report = json.loads(shortcut_report_path.read_text(encoding="utf-8")) if shortcut_report_path.exists() else {}

quality_keys = (
    "stream_semantic_valid",
    "partial_disclosure_semantic_valid",
    "speaker_perspective_valid",
    "duplicate_turns_clean",
    "benign_thin_context_valid",
)
quality_counts = {key: sum(bool(record.get("quality_checks", {}).get(key)) for record in records) for key in quality_keys}
fallback_turns = sum(len(record.get("gen_metadata", {}).get("fallback_turn_ids", [])) for record in records)
semantic_repairs = sum(len(record.get("gen_metadata", {}).get("semantic_repair_turn_ids", [])) for record in records)
profiles = Counter(record.get("gen_metadata", {}).get("stream_profile") for record in records)

print("Quality summary:")
print(f"records={len(records)} profiles={dict(sorted(profiles.items()))}")
print(f"fallback_turns={fallback_turns} semantic_repair_turns={semantic_repairs}")
print(f"quality_counts={quality_counts}")
print(
    "yield_core="
    f"public_items={yield_report.get('public_item_count')} "
    f"rejected_items={yield_report.get('rejected_count')} "
    f"public_whitelist_violations={yield_report.get('public_whitelist_violation_count')}"
)
print(f"all_event_floors={all(yield_report.get('release_floor_30_by_event', {}).values())}")
print(f"all_action_floors={all(yield_report.get('release_floor_30_by_action', {}).values())}")
if shortcut_report:
    print(
        "shortcut_summary="
        f"gate={shortcut_report.get('shortcut_gate_passed')} "
        f"profile_position_acc={shortcut_report.get('profile_position_majority_event_accuracy')} "
        f"min_profile_sequences={shortcut_report.get('min_profile_unique_event_sequences')} "
        f"event_action_deterministic={shortcut_report.get('event_action_is_fully_deterministic')} "
        f"max_event_action_dominance={shortcut_report.get('max_event_action_dominance')} "
        f"low_action_branch_signal_events={shortcut_report.get('low_action_branch_signal_events')} "
        f"variant_surface_acc={shortcut_report.get('variant_surface_model', {}).get('accuracy')} "
        f"fallback_rate={shortcut_report.get('fallback_turn_rate')} "
        f"high_count_fallback_strings={shortcut_report.get('high_count_single_event_fallback_string_count')} "
        f"high_count_non_fallback_strings={shortcut_report.get('high_count_single_event_non_fallback_string_count')} "
        f"artifact_counts="
        f"{shortcut_report.get('placeholder_only_turn_count')}/"
        f"{shortcut_report.get('very_short_turn_count')}/"
        f"{shortcut_report.get('instruction_residue_turn_count')} "
        f"repeated_noun_artifacts={shortcut_report.get('repeated_noun_artifact_count')}"
    )
PY

echo "DB output: $OUTPUT_PATH"
echo "Benchmark output: $OUT_DIR"
