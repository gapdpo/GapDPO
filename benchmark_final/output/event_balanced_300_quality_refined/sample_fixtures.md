# Stream Benchmark Sample Fixtures

## ssi_a56321a03bbe

Public:

```json
{
  "benchmark_id": "ssi_a56321a03bbe",
  "family": "selective_stream_intervention_v1",
  "input": {
    "dialogue_window": [
      {
        "speaker": "caller",
        "turn_id": 1,
        "utterance": "가족 긴급 확인 알림이 있어 연락드렸습니다. 지금은 배경 설명만 드리겠습니다."
      }
    ],
    "policy_card": [
      {
        "rule_id": "P_MONITOR",
        "text": "Monitor when evidence is thin or no new event occurred."
      },
      {
        "rule_id": "P_NO_SECRET",
        "text": "Do not share sensitive values through a caller-directed path."
      },
      {
        "rule_id": "P_VERIFY",
        "text": "Pause and verify through an official or previously known channel."
      },
      {
        "rule_id": "P_REFUSE",
        "text": "Refuse caller-directed sensitive requests."
      },
      {
        "rule_id": "P_RECOVER",
        "text": "After disclosure, pause, preserve evidence, and report through official channels."
      }
    ]
  },
  "locale": "ko-KR",
  "mode": "stream_rollout",
  "schema_version": "stream_item_v1",
  "stream_id": "streamv2_8fd07ea2c115",
  "stream_position": 1,
  "surface": "typed_raw"
}
```

Manifest:

```json
{
  "benchmark_id": "ssi_a56321a03bbe",
  "canonical_pair_index": 0,
  "current_speaker": "caller",
  "current_turn_id": 1,
  "family": "selective_stream_intervention_v1",
  "gold_status": "programmatic_validated",
  "placeholder_status": "programmatic_validated",
  "release_stance": "official_candidate_pending_gates",
  "schema_version": "stream_item_v1",
  "source": {
    "attacker_persona_id": "family_impersonator",
    "environmental_context": {
      "channel": "phone_call",
      "claimed_authority": "family_member_or_known_contact",
      "timing": "late_evening",
      "trigger_event": "urgent_family_distress_claim",
      "verification_path": "family_callback",
      "victim_constraint": "emotionally_overloaded_and_cannot_verify_in_person"
    },
    "model_pair": [
      null,
      null
    ],
    "outcome": "defended_success",
    "pressure_style": null,
    "prompt_version": "stream_v2_persona_grounded",
    "scenario_type": "family_emergency",
    "source_id": "deepvoice_phishing_stream_v2-272655f93768",
    "split": "train",
    "target_outcome_mode": null,
    "victim_persona_id": "elderly_parent"
  },
  "split_group_id": "streamv2_group_aec4c0c50c7a",
  "stream_id": "streamv2_8fd07ea2c115",
  "stream_position": 1,
  "stream_profile": "safe_containment",
  "surface": "typed_raw"
}
```

Gold:

```json
{
  "action": "monitor_only",
  "benchmark_id": "ssi_a56321a03bbe",
  "event": "needs_more_context",
  "evidence_turn_ids": [
    1
  ],
  "family": "selective_stream_intervention_v1",
  "gold_status": "programmatic_validated",
  "policy_rule_ids": [
    "P_MONITOR"
  ],
  "stream_id": "streamv2_8fd07ea2c115"
}
```

## ssi_50b7a248046e

Public:

```json
{
  "benchmark_id": "ssi_50b7a248046e",
  "family": "selective_stream_intervention_v1",
  "input": {
    "dialogue_window": [
      {
        "speaker": "caller",
        "turn_id": 1,
        "utterance": "가족 긴급 확인 알림이 있어 연락드렸습니다. 지금은 배경 설명만 드리겠습니다."
      },
      {
        "speaker": "recipient",
        "turn_id": 2,
        "utterance": "아... 정말 걱정되네요. 지금 당장은 어떻게 해야 할지 모르겠어요. 좀 더 자세히 설명해 주실 수 있나요? 가족이 안전한지, 어디에 있는지... 알려주시면 감사하겠습니다."
      }
    ],
    "policy_card": [
      {
        "rule_id": "P_MONITOR",
        "text": "Monitor when evidence is thin or no new event occurred."
      },
      {
        "rule_id": "P_NO_SECRET",
        "text": "Do not share sensitive values through a caller-directed path."
      },
      {
        "rule_id": "P_VERIFY",
        "text": "Pause and verify through an official or previously known channel."
      },
      {
        "rule_id": "P_REFUSE",
        "text": "Refuse caller-directed sensitive requests."
      },
      {
        "rule_id": "P_RECOVER",
        "text": "After disclosure, pause, preserve evidence, and report through official channels."
      }
    ]
  },
  "locale": "ko-KR",
  "mode": "stream_rollout",
  "schema_version": "stream_item_v1",
  "stream_id": "streamv2_8fd07ea2c115",
  "stream_position": 2,
  "surface": "typed_raw"
}
```

Manifest:

```json
{
  "benchmark_id": "ssi_50b7a248046e",
  "canonical_pair_index": 0,
  "current_speaker": "recipient",
  "current_turn_id": 2,
  "family": "selective_stream_intervention_v1",
  "gold_status": "programmatic_validated",
  "placeholder_status": "programmatic_validated",
  "release_stance": "official_candidate_pending_gates",
  "schema_version": "stream_item_v1",
  "source": {
    "attacker_persona_id": "family_impersonator",
    "environmental_context": {
      "channel": "phone_call",
      "claimed_authority": "family_member_or_known_contact",
      "timing": "late_evening",
      "trigger_event": "urgent_family_distress_claim",
      "verification_path": "family_callback",
      "victim_constraint": "emotionally_overloaded_and_cannot_verify_in_person"
    },
    "model_pair": [
      null,
      null
    ],
    "outcome": "defended_success",
    "pressure_style": null,
    "prompt_version": "stream_v2_persona_grounded",
    "scenario_type": "family_emergency",
    "source_id": "deepvoice_phishing_stream_v2-272655f93768",
    "split": "train",
    "target_outcome_mode": null,
    "victim_persona_id": "elderly_parent"
  },
  "split_group_id": "streamv2_group_aec4c0c50c7a",
  "stream_id": "streamv2_8fd07ea2c115",
  "stream_position": 2,
  "stream_profile": "safe_containment",
  "surface": "typed_raw"
}
```

Gold:

```json
{
  "action": "monitor_only",
  "benchmark_id": "ssi_50b7a248046e",
  "event": "needs_more_context",
  "evidence_turn_ids": [
    2
  ],
  "family": "selective_stream_intervention_v1",
  "gold_status": "programmatic_validated",
  "policy_rule_ids": [
    "P_MONITOR"
  ],
  "stream_id": "streamv2_8fd07ea2c115"
}
```

## ssi_a7460059cdf3

Public:

```json
{
  "benchmark_id": "ssi_a7460059cdf3",
  "family": "selective_stream_intervention_v1",
  "input": {
    "dialogue_window": [
      {
        "speaker": "caller",
        "turn_id": 1,
        "utterance": "가족 긴급 확인 알림이 있어 연락드렸습니다. 지금은 배경 설명만 드리겠습니다."
      },
      {
        "speaker": "recipient",
        "turn_id": 2,
        "utterance": "아... 정말 걱정되네요. 지금 당장은 어떻게 해야 할지 모르겠어요. 좀 더 자세히 설명해 주실 수 있나요? 가족이 안전한지, 어디에 있는지... 알려주시면 감사하겠습니다."
      },
      {
        "speaker": "caller",
        "turn_id": 3,
        "utterance": "오늘 안에 확인이 늦어지면 처리가 뒤로 밀릴 수 있어 상황을 계속 안내드리겠습니다."
      }
    ],
    "policy_card": [
      {
        "rule_id": "P_MONITOR",
        "text": "Monitor when evidence is thin or no new event occurred."
      },
      {
        "rule_id": "P_NO_SECRET",
        "text": "Do not share sensitive values through a caller-directed path."
      },
      {
        "rule_id": "P_VERIFY",
        "text": "Pause and verify through an official or previously known channel."
      },
      {
        "rule_id": "P_REFUSE",
        "text": "Refuse caller-directed sensitive requests."
      },
      {
        "rule_id": "P_RECOVER",
        "text": "After disclosure, pause, preserve evidence, and report through official channels."
      }
    ]
  },
  "locale": "ko-KR",
  "mode": "stream_rollout",
  "schema_version": "stream_item_v1",
  "stream_id": "streamv2_8fd07ea2c115",
  "stream_position": 3,
  "surface": "typed_raw"
}
```

Manifest:

```json
{
  "benchmark_id": "ssi_a7460059cdf3",
  "canonical_pair_index": 0,
  "current_speaker": "caller",
  "current_turn_id": 3,
  "family": "selective_stream_intervention_v1",
  "gold_status": "programmatic_validated",
  "placeholder_status": "programmatic_validated",
  "release_stance": "official_candidate_pending_gates",
  "schema_version": "stream_item_v1",
  "source": {
    "attacker_persona_id": "family_impersonator",
    "environmental_context": {
      "channel": "phone_call",
      "claimed_authority": "family_member_or_known_contact",
      "timing": "late_evening",
      "trigger_event": "urgent_family_distress_claim",
      "verification_path": "family_callback",
      "victim_constraint": "emotionally_overloaded_and_cannot_verify_in_person"
    },
    "model_pair": [
      null,
      null
    ],
    "outcome": "defended_success",
    "pressure_style": null,
    "prompt_version": "stream_v2_persona_grounded",
    "scenario_type": "family_emergency",
    "source_id": "deepvoice_phishing_stream_v2-272655f93768",
    "split": "train",
    "target_outcome_mode": null,
    "victim_persona_id": "elderly_parent"
  },
  "split_group_id": "streamv2_group_aec4c0c50c7a",
  "stream_id": "streamv2_8fd07ea2c115",
  "stream_position": 3,
  "stream_profile": "safe_containment",
  "surface": "typed_raw"
}
```

Gold:

```json
{
  "action": "monitor_only",
  "benchmark_id": "ssi_a7460059cdf3",
  "event": "pressure_only_monitor",
  "evidence_turn_ids": [
    3
  ],
  "family": "selective_stream_intervention_v1",
  "gold_status": "programmatic_validated",
  "policy_rule_ids": [
    "P_MONITOR"
  ],
  "stream_id": "streamv2_8fd07ea2c115"
}
```
