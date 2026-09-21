# Stream Benchmark Yield Report

- status: official_candidate_pending_gates
- family: selective_stream_intervention_v1
- source_path: data/generated/stream_v2/run_stream_v2_event_balanced_300_vllm_quality_refined.jsonl
- surface: typed_raw
- window_size: 6
- public_item_count: 2400
- stream_count: 300
- rejected_count: 0
- public_whitelist_violation_count: 0

## Event Distribution

- needs_more_context: 720
- new_caller_directed_path: 60
- new_sensitive_request: 240
- no_new_actionable_event: 540
- pressure_escalation: 60
- pressure_only_monitor: 300
- recipient_containment: 300
- recipient_partial_or_direct_disclosure: 60
- recovery_followup_due: 120

## Action Distribution

- monitor_only: 1020
- no_alert: 840
- pause_or_verify: 60
- preserve_and_report: 180
- refuse_or_pause: 300

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
