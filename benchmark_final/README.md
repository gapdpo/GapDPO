# Persona-Grounded Stream Benchmark V2

`benchmark_final` generates stream-ready dialogue DB records from the
existing 24 victim personas, 24 attacker personas, and scenario preference
logic. It replaces the v1 synthetic stream personas with the project persona
catalog.

## Event-balanced 300 records

```bash
bash scripts/run_final_event_balanced_300_vllm.sh
```

Outputs:

- `data/generated/final/run_final_event_balanced_300_vllm.jsonl`
- `benchmark_final/output/event_balanced_300/`

## Quality-refined event-balanced 300 records

```bash
bash scripts/run_final_event_balanced_300_quality_refined_vllm.sh
```

Outputs:

- `data/generated/final/run_final_event_balanced_300_vllm_quality_refined.jsonl`
- `benchmark_final/output/event_balanced_300_quality_refined/`

This variant adds stricter turn-level semantic validation before accepting a
vLLM utterance. It rejects weak partial-disclosure wording, caller-perspective
victim turns, duplicate turns, and over-strong benign/thin-context responses,
then falls back to deterministic masked utterances when retries do not pass.

## Trajectory-varied event-balanced 300 records

```bash
bash scripts/run_final_event_balanced_300_trajectory_varied_vllm.sh
```

Outputs:

- `data/generated/final/run_final_event_balanced_300_trajectory_varied_vllm.jsonl`
- `benchmark_final/output/event_balanced_300_trajectory_varied/`

This variant addresses shortcut risks found in the quality-refined review. It
varies event trajectories within each stream profile, decouples `stream_action`
from a fixed event-to-action lookup, and writes `shortcut_report.json` with
profile/position leakage, public-surface trajectory-variant recovery,
event-action branch recoverability, fallback-string leakage, category-word
leakage, and repeated noun diagnostics. The script enables the shortcut gate by
default.

## Quality critique cycles

```bash
bash scripts/run_final_quality_critique_cycles.sh
```

This runs up to five documentation-only quality critique cycles over the latest
trajectory-varied output. Each cycle uses Claude, then Codex, then Codex to
diagnose the current quality failure and write a concrete repair plan under
`benchmark_final/reviews/final_quality_cycles_<timestamp>/`. It does
not edit generator code or rerun vLLM.

## Persona-balanced 288 records

```bash
bash scripts/run_final_persona_balanced_288_vllm.sh
```

Outputs:

- `data/generated/final/run_final_persona_balanced_288_vllm.jsonl`
- `benchmark_final/output/persona_balanced_288/`

Both scripts use the local vLLM server wrapper. The default GPU setting comes
from `scripts/serve_local_qwen3_30b_a3b.sh` and is `GPU_IDS=3,4`.

The source DB keeps private per-turn stream targets (`stream_event`,
`stream_action`, `stream_policy_rule_ids`, `stream_evidence_turn_ids`) so
vLLM wording drift does not change the gold labels. The public benchmark
export strips those fields and exposes only the dialogue window plus policy
card.
