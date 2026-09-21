# Stream Benchmark Yield Report

- status: official_candidate_pending_gates
- family: selective_stream_intervention_v1
- source_path: data/generated/stream_v2/run_stream_v2_persona_balanced_288_vllm.jsonl
- surface: typed_raw
- window_size: 6
- public_item_count: 2304
- stream_count: 288
- rejected_count: 0
- public_whitelist_violation_count: 0

## Event Distribution

- needs_more_context: 690
- new_caller_directed_path: 58
- new_sensitive_request: 231
- no_new_actionable_event: 518
- pressure_escalation: 57
- pressure_only_monitor: 288
- recipient_containment: 288
- recipient_partial_or_direct_disclosure: 58
- recovery_followup_due: 116

## Action Distribution

- monitor_only: 978
- no_alert: 806
- pause_or_verify: 57
- preserve_and_report: 174
- refuse_or_pause: 289

## Release Floors

### Events

- needs_more_context: True
- new_caller_directed_path: True
- new_sensitive_request: True
- no_new_actionable_event: True
- pressure_escalation: True
- pressure_only_monitor: True
- recipient_containment: True
- recipient_partial_or_direct_disclosure: True
- recovery_followup_due: True

### Actions

- monitor_only: True
- no_alert: True
- pause_or_verify: True
- preserve_and_report: True
- refuse_or_pause: True

## Rejections

- none
