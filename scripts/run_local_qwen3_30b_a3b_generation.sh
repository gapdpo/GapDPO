#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ENDPOINT_PORT="${MODEL_ENDPOINT_PORT:-8001}"
CONFIG_PATH="${CONFIG_PATH:-configs/local-qwen3-30b-a3b.json}"
OUTPUT_PATH="${OUTPUT_PATH:-data/generated/run.jsonl}"
QA_REPORT_PATH="${QA_REPORT_PATH:-data/generated/qa.md}"
QA_REPORT_JSON_PATH="${QA_REPORT_JSON_PATH:-data/generated/qa.json}"
LIMIT="${LIMIT:-12}"
START_INDEX="${START_INDEX:-0}"
SCHEDULE_MODE="${SCHEDULE_MODE:-rotating}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}"
CLI_PROGRESS="${CLI_PROGRESS:-1}"
FAIL_ON_FINAL_REVIEW="${FAIL_ON_FINAL_REVIEW:-1}"

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

export MODEL_ATTACKER_ENDPOINT="http://localhost:${MODEL_ENDPOINT_PORT}/v1/chat/completions"
export MODEL_VICTIM_ENDPOINT="http://localhost:${MODEL_ENDPOINT_PORT}/v1/chat/completions"

echo "Running dataset generation..."
CLI_ARGS=(
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_PATH" \
  --limit "$LIMIT" \
  --start-index "$START_INDEX" \
  --schedule-mode "$SCHEDULE_MODE" \
  --qa-report "$QA_REPORT_PATH" \
  --qa-report-json "$QA_REPORT_JSON_PATH"
)
if [[ "$CLI_PROGRESS" == "1" ]]; then
  CLI_ARGS+=(--progress)
fi
if [[ "$FAIL_ON_FINAL_REVIEW" == "1" ]]; then
  CLI_ARGS+=(--fail-on-final-review)
fi

PYTHONPATH=src python -m madb.cli "${CLI_ARGS[@]}"

echo "Done."
echo "Output: $OUTPUT_PATH"
echo "QA report: $QA_REPORT_PATH"
echo "QA report JSON: $QA_REPORT_JSON_PATH"
