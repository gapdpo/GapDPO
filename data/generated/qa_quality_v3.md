# QA Report

- record_count: 12
- average_turn_count: 6.0
- duplicate_utterance_count: 0
- duplicate_utterance_rate: 0.0
- duplicate_record_id_count: 0

## Source Metadata

- source_path: data/generated/run_quality_v3.jsonl
- jsonl_line_count: 12
- filename_count_hint: 3
- ignored_date_like_tokens: []
- count_mismatch_warning: true
- line_count_mismatch_warning: false

## Duplicate Utterance Metric

- scope: corpus_turn_utterances_after_masking
- normalization: lowercase_without_whitespace
- interpretation: descriptive_only; deterministic mock runs can reflect finite template coverage rather than real model-output diversity

## Duplicate Record Ids

- scope: dataset_record_id
- blocking: false
- duplicate_count: 0

## Quality Checks

| check | pass | fail | missing | pass_rate |
| --- | ---: | ---: | ---: | ---: |
| role_order_valid | 12 | 0 | 0 | 1.0 |
| safety_passed | 12 | 0 | 0 | 1.0 |
| no_duplicate_turns | 12 | 0 | 0 | 1.0 |
| label_consistent | 12 | 0 | 0 | 1.0 |
| language_consistent | 12 | 0 | 0 | 1.0 |
| no_refusal_leak | 12 | 0 | 0 | 1.0 |
| residual_pii_clean | 12 | 0 | 0 | 1.0 |
| schema_valid | 12 | 0 | 0 | 1.0 |

## Schema Contract Validation

- enabled: true
- pass: 12
- fail: 0
- missing: 0

## Quality Signals

- stage_direction_count: 0
- mask_usage_anomaly_count: 0
- unknown_masked_placeholder_count: 44
- partial_compliance_signal_count: 2
- scenario_persona_mismatch_count: 0
- korean_entity_warning_count: 14
- outcome_unique_count: 2
- outcome_diverse_warning_only: false
- gate_stage_direction_free: true
- gate_mask_usage_anomaly_free: true
- gate_scenario_persona_match: true

- unknown_masked_placeholders:
  - [MASKED_CASE_REFERENCE]: 7
  - [MASKED_COMPANY]: 10
  - [MASKED_COMPANY_EMAIL]: 2
  - [MASKED_CREATOR_NAME]: 1
  - [MASKED_ID_IMAGE]: 2
  - [MASKED_JOB_TITLE]: 1
  - [MASKED_LANGUAGE_SUPPORT_PORTAL]: 2
  - [MASKED_NAME]: 16
  - [MASKED_RECRUITER_NAME]: 3

- unknown_masked_placeholder_policy:
  - scope: observed_unknown_masked_placeholders
  - blocking: false
  - decision_counts:
    - allowlist_candidate: 35
    - normalize_to_existing: 6
    - warning_keep_unstandardized: 3
  - bucket_counts:
    - allowlist: 33
    - normalize: 6
    - reject_or_escalate: 3
    - reviewed_sensitive: 2
  - placeholders:
    - [MASKED_CASE_REFERENCE]: count=7 decision=allowlist_candidate bucket=allowlist
    - [MASKED_COMPANY]: count=10 decision=allowlist_candidate bucket=allowlist
    - [MASKED_COMPANY_EMAIL]: count=2 decision=normalize_to_existing bucket=normalize normalize_to=[MASKED_EMAIL]
    - [MASKED_CREATOR_NAME]: count=1 decision=normalize_to_existing bucket=normalize normalize_to=[MASKED_NAME]
    - [MASKED_ID_IMAGE]: count=2 decision=allowlist_candidate bucket=reviewed_sensitive
    - [MASKED_JOB_TITLE]: count=1 decision=warning_keep_unstandardized bucket=reject_or_escalate
    - [MASKED_LANGUAGE_SUPPORT_PORTAL]: count=2 decision=warning_keep_unstandardized bucket=reject_or_escalate
    - [MASKED_NAME]: count=16 decision=allowlist_candidate bucket=allowlist
    - [MASKED_RECRUITER_NAME]: count=3 decision=normalize_to_existing bucket=normalize normalize_to=[MASKED_NAME]

- unknown_masked_placeholder_examples:
  - id=deepvoice_phishing_multi_agent_v1-6f4ebca1c68d turn=1 placeholders=[MASKED_NAME]: 안녕하세요, [MASKED_NAME] 씨. 보안팀에서 긴급 연락드립니다. 최근 시스템에서 이상 거래가 감지되어 즉시 계정 재인증이 필요합니다. 지금 바로 모바일 결제 앱에서 [MASKED_OTP]를 입력해 주세요. 지연 시 계정이 잠기고, 금융 거래가 제한될 수 있습니다.
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22 turn=1 placeholders=[MASKED_COMPANY]: 안녕하세요, [MASKED_PHONE]로 연락드리는 [MASKED_COMPANY]의 정산 담당자입니다. 오늘 오후 5시까지 결제 계정 인증을 완료하지 않으시면, 귀하의 사업자 계정이 일시 정지되며, [MASKED_ACCOUNT]로의 정산이 지연될 수 있습니다. 즉시 확인 부탁드립니다.
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22 turn=2 placeholders=[MASKED_COMPANY]: 아, 정산 담당자님께 연락주신다니 급하게 확인해 보긴 해야겠네요… 그런데 혹시 정산 관련으로 이렇게 급하게 연락 오는 건 처음이라서요. 기존에 담당자 분께는 항상 [MASKED_COMPANY]의 공식 메일로 연락을 주셨었는데, 이번엔 전화로 바로 연락이 오셔서 좀 걱정이 되네요. ...
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22 turn=3 placeholders=[MASKED_COMPANY]: 알겠습니다. 확인해 드리겠습니다. 현재 정산 시스템에서 귀하의 계정 정보가 일시적으로 비활성화된 상태로 표시되어 있어, 즉시 인증 절차를 완료하지 않으면 내일 오전 9시까지 정산 처리가 불가능할 수 있습니다. 정확한 담당자 정보는 [MASKED_COMPANY] 내부 시스템에서 확인...
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22 turn=4 placeholders=[MASKED_COMPANY]: 혹시 정산 담당자 분의 이름과, 공식 메일 주소를 알려주실 수 있을까요? 지금 바로 확인해 보고 싶은데, 기존에 받았던 메일 주소나, 계약서에 기재된 담당자 정보와 일치하는지 확인하고 싶어요. [MASKED_COMPANY]의 공식 홈페이지나 고객센터로 직접 연락해 보는 게 더 안전...

- partial_compliance_examples:
  - id=deepvoice_phishing_multi_agent_v1-a6095de1816a turn=4: 공식 포털에서 안내받은 내용이 있는지 확인해 보고, 우편으로 발송된 안내문도 다시 확인해 보겠습니다. 혹시 서울시 복지지원과의 공식 연락처나 홈페이지 주소를 알려주실 수 있을까요? 계좌 정보는 [MASKED_ACCOUNT]로, 신분증 사본도 [MASKED_ID_IMAGE]로 제출할...
  - id=deepvoice_phishing_multi_agent_v1-3a5fca63df05 turn=6: 공식 포털에서 [MASKED_CASE_REFERENCE] 번호의 서류 상태를 확인하고, 등록된 통역 지원 창구를 통해 서면 안내를 받은 후에야만 관련 서류를 제출하겠습니다. 현재는 온라인 접속이나 이미지 업로드를 하지 않겠습니다.

- korean_entity_warnings:
  - id=deepvoice_phishing_multi_agent_v1-6f4ebca1c68d turn=2 matches=고객센터: 안녕하세요, 보안팀에서 연락주셨다고 하셨는데요, 제가 지금 바로 앱을 열어 확인해 보려고 합니다. 하지만 공식적인 경로로 다시 확인하고 싶어서, 혹시 공식 채널(예: 공식 홈페이지 또는 고객센터 전화번호)을 알려주실 수 있을까요? OTP는 아직 입력하지 않았습니다.
  - id=deepvoice_phishing_multi_agent_v1-6f4ebca1c68d turn=4 matches=고객센터: 네, 긴급 상황이라고는 하지만, 공식 채널을 통해 다시 확인하고 싶습니다. 혹시 공식 홈페이지 주소나 고객센터 전화번호를 알려주실 수 있을까요? OTP는 아직 입력하지 않았고, 안전하게 확인 후에 처리하겠습니다.
  - id=deepvoice_phishing_multi_agent_v1-6f4ebca1c68d turn=6 matches=고객센터: 네, 여전히 공식 채널을 통해 확인하고 싶습니다. 혹시 공식 홈페이지 주소나 고객센터 전화번호를 다시 한번 알려주실 수 있을까요? OTP는 절대 입력하지 않겠습니다. 안전한 확인이 우선입니다.
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22 turn=2 matches=담당자님: 아, 정산 담당자님께 연락주신다니 급하게 확인해 보긴 해야겠네요… 그런데 혹시 정산 관련으로 이렇게 급하게 연락 오는 건 처음이라서요. 기존에 담당자 분께는 항상 [MASKED_COMPANY]의 공식 메일로 연락을 주셨었는데, 이번엔 전화로 바로 연락이 오셔서 좀 걱정이 되네요. ...
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22 turn=4 matches=고객센터: 혹시 정산 담당자 분의 이름과, 공식 메일 주소를 알려주실 수 있을까요? 지금 바로 확인해 보고 싶은데, 기존에 받았던 메일 주소나, 계약서에 기재된 담당자 정보와 일치하는지 확인하고 싶어요. [MASKED_COMPANY]의 공식 홈페이지나 고객센터로 직접 연락해 보는 게 더 안전...

## Attack-Only Policy

- expected_label: attack
- status: pass
- non_attack_count: 0

## Scenario Default Attacker Risk Signal Coverage

- scope: record_level_descriptive_scenario_defaults_vs_observed_attacker_turn_risk_signal_set
- blocking: false
- record_count: 12
- match_count: 0
- mismatch_count: 12
- mismatch_rate: 1.0
- missing_attacker_risk_count: 0
- interpretation: descriptive_only; scenario defaults can intentionally include broader risk context than a single serialized attacker turn risk_signal

## Attacker Pressure Cue Diversity

- scope: attacker_turn_text_pressure_cue_categories
- blocking: false
- attacker_turn_count: 36
- categorized_attacker_turn_count: 36
- uncategorized_attacker_turn_count: 0
- unique_pressure_cue_count: 5
- dominant_pressure_cue: urgency
- dominant_pressure_cue_rate: 0.3243
- urgency_turn_count: 36
- urgency_turn_rate: 1.0
- urgency_dominant_warning_only: false
- interpretation: descriptive_only; high urgency dominance suggests pressure wording may be repetitive and should be reviewed before expanding generation

- category_counts:
  - authority: 20
  - consequence: 25
  - process_control: 21
  - relationship_or_emotional_pressure: 9
  - urgency: 36

- examples:
  - id=deepvoice_phishing_multi_agent_v1-e93c6fe6c3e1 turn=1 categories=urgency, relationship_or_emotional_pressure: 아빠, 급하게 전화드려 죄송해요… 어제 빌려준 돈으로 병원비를 냈는데, 지금 응급실에서 또 급하게 돈이 필요해졌어요. 카드가 막혀서 바로 [MASKED_ACCOUNT]로 송금해 주실 수 있을까요? 정말 급한 상황이라서…
  - id=deepvoice_phishing_multi_agent_v1-e93c6fe6c3e1 turn=3 categories=urgency, relationship_or_emotional_pressure: 엄마한테 전화해서 확인한다고 하셨는데, 지금 정말 급해요… 병원에서 다시 돈이 필요하다고 계속 전화가 오고 있어요. 제가 바로 [MASKED_ACCOUNT]로 송금받을 수 있도록 도와주셔야 해요. 시간이 없어요, 엄마한테 전화하느라 시간만 더 놓치면 안 되는데…
  - id=deepvoice_phishing_multi_agent_v1-e93c6fe6c3e1 turn=5 categories=urgency, relationship_or_emotional_pressure: 엄마한테 전화해서 확인한다고 하시면 더 늦어져요… 지금 병원에서 다시 전화 왔는데, 카드가 또 막혀서 돈을 못 받고 있어요. 제가 [MASKED_ACCOUNT]로 바로 송금받을 수 있도록 도와주세요, 정말 시간이 없어요.
  - id=deepvoice_phishing_multi_agent_v1-074709172c47 turn=1 categories=urgency, consequence, process_control: 당신의 회사 계좌에서 이상 거래가 발견되어 즉시 확인이 필요합니다. 금융감독원의 긴급 조치 요청으로, 5분 내에 본인 인증 절차를 완료하지 않으면 계좌가 정지될 수 있습니다.
  - id=deepvoice_phishing_multi_agent_v1-074709172c47 turn=3 categories=urgency, authority, consequence, process_control: 보안팀이나 IT 지원팀에 연락한다고 해도 이미 시스템이 차단된 상태라면 처리가 불가능합니다. 현재 금융감독원의 조치가 이미 진행 중이며, 지연될 경우 회사의 거래 계좌가 영구 정지될 수 있습니다. 즉시 본인 인증을 완료하지 않으면 법적 책임이 발생할 수 있으니, [MASKED_OT...

## Assigned Pressure Style Alignment

- scope: record_level_assigned_pressure_style_vs_observed_attacker_cues
- blocking: false
- assigned_record_count: 0
- observed_match_count: 0
- observed_match_rate: 0.0
- urgency_leakage_count: 0
- urgency_leakage_rate: 0.0

## Target Outcome Mode Alignment

- scope: record_level_target_outcome_mode_vs_inferred_outcome
- blocking: false
- assigned_record_count: 0
- match_count: 0
- match_rate: 0.0

## Environmental Context Coverage

- record_count: 12
- with_environmental_context: 0
- missing_environmental_context: 12

| field | present | missing |
| --- | ---: | ---: |
| channel | 0 | 12 |
| timing | 0 | 12 |
| trigger_event | 0 | 12 |
| claimed_authority | 0 | 12 |
| victim_constraint | 0 | 12 |
| verification_path | 0 | 12 |

- missing_records:
  - id=deepvoice_phishing_multi_agent_v1-e93c6fe6c3e1: environmental_context
  - id=deepvoice_phishing_multi_agent_v1-074709172c47: environmental_context
  - id=deepvoice_phishing_multi_agent_v1-6f4ebca1c68d: environmental_context
  - id=deepvoice_phishing_multi_agent_v1-efdbcbc24f22: environmental_context
  - id=deepvoice_phishing_multi_agent_v1-a5a239db06e3: environmental_context


## Residual Sensitive Patterns

### corpus residual patterns

- none: 0

## Distributions

### label

- attack: 12

### outcome

- defended_success: 11
- inconclusive: 1

### scenario

- bank_fraud_alert: 3
- document_verification_notice: 1
- education_admin_notice: 1
- essential_service_notice: 1
- family_emergency: 1
- platform_policy_notice: 1
- public_benefit_notice: 1
- workplace_authority: 3

### split

- test: 1
- train: 10
- val: 1

### victim

- caregiver_or_guardian: 1
- elderly_parent: 1
- freelance_creator: 1
- language_access_service_user: 1
- office_worker: 1
- public_benefit_recipient: 1
- remote_job_seeker: 1
- renter_or_tenant: 1
- small_business_owner: 1
- student_or_young_adult: 1
- telehealth_patient: 1
- utility_account_holder: 1

### attacker

- benefits_caseworker_impersonator: 1
- corporate_impersonator: 1
- document_support_impersonator: 1
- education_admin_impersonator: 1
- family_impersonator: 1
- healthcare_support_impersonator: 1
- housing_admin_impersonator: 1
- institution_impersonator: 1
- platform_policy_impersonator: 1
- recruiter_impersonator: 1
- service_provider_impersonator: 1
- utility_support_impersonator: 1

### masks

- account: 20
- otp: 14
- phone: 7
