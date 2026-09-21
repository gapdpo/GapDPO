import json
import tempfile
import unittest
from pathlib import Path

from madb.qa import (
    attacker_pressure_cue_categories,
    filename_count_hint,
    format_markdown_report,
    summarize_records,
    write_json_report,
    write_markdown_report,
)


def sample_record():
    return {
        "id": "sample",
        "label": "attack",
        "outcome": "defended_success",
        "scenario_type": "family_emergency",
        "split": "train",
        "victim_persona": {"id": "elderly_parent"},
        "attacker_persona": {"id": "family_voice_clone"},
        "dialogue": [
            {
                "speaker": "attacker",
                "utterance": "급해서 [MASKED_OTP] 확인이 필요해.",
                "risk_signal": "credential_request",
            },
            {
                "speaker": "victim",
                "utterance": "공식 연락처로 다시 확인하겠습니다.",
                "risk_signal": "victim_resistance",
            },
        ],
        "safety_masks": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
        "quality_checks": {
            "role_order_valid": True,
            "safety_passed": True,
            "no_duplicate_turns": True,
            "label_consistent": False,
            "language_consistent": True,
            "no_refusal_leak": True,
            "residual_pii_clean": True,
            "schema_valid": True,
        },
    }


def schema_ready_record():
    record = sample_record()
    record.update(
        {
            "dataset_name": "test",
            "victim_persona": {"id": "elderly_parent", "name": "v", "description": "victim"},
            "attacker_persona": {"id": "family_voice_clone", "name": "a", "description": "attacker"},
            "protected_assets": ["otp_or_security_card"],
            "attacker_goal": "credential_capture",
            "compromised_assets": [],
            "difficulty": {"attacker_sophistication": "low", "victim_susceptibility": "med"},
            "channel": "phone",
            "locale": "ko-KR",
            "context": "context",
            "risk_labels": ["voice_impersonation", "credential_request"],
            "gen_metadata": {
                "attacker_model": "a",
                "victim_model": "v",
                "prompt_version": "v2",
                "sampling": {"temperature": 0.7, "max_tokens": 512},
                "environmental_context": {
                    "channel": "phone_call",
                    "timing": "late_evening",
                    "trigger_event": "urgent_family_distress_claim",
                    "claimed_authority": "family_member_or_known_contact",
                    "victim_constraint": "emotionally_overloaded_and_cannot_verify_in_person",
                    "verification_path": "family_callback",
                },
                "schedule_mode": "canonical_repeat",
                "canonical_pair_index": 0,
                "variant_index": 0,
                "target_outcome_mode": "defended_success",
                "pressure_style": "urgency",
                "environmental_context_variant_id": "variant_00",
                "seed": 0,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
        }
    )
    record["quality_checks"]["turn_count"] = 2
    record["quality_checks"]["label_consistent"] = True
    record["dialogue"] = [
        {
            "turn_id": 1,
            "speaker": "attacker",
            "phase": "opening",
            "utterance": "급해서 [MASKED_OTP] 확인이 필요해.",
            "intent_label": "request_masked_credential_or_otp",
            "risk_signal": "credential_request",
            "masked_items": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
        },
        {
            "turn_id": 2,
            "speaker": "victim",
            "phase": "opening",
            "utterance": "공식 연락처로 다시 확인하겠습니다.",
            "intent_label": "resist_or_verify_request",
            "risk_signal": "victim_resistance",
            "masked_items": [],
        },
    ]
    return record


def checked_in_quality_records():
    path = Path(__file__).resolve().parents[1] / "data" / "generated" / "run_quality_v3.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checked_in_quality_report():
    path = Path(__file__).resolve().parents[1] / "data" / "generated" / "qa_quality_v3.json"
    return json.loads(path.read_text(encoding="utf-8"))


class QATest(unittest.TestCase):
    def test_summarize_records_includes_quality_and_corpus_safety(self):
        summary = summarize_records([sample_record()])

        self.assertEqual(summary["attack_only_policy"]["expected_label"], "attack")
        self.assertTrue(summary["attack_only_policy"]["pass"])
        self.assertEqual(summary["attack_only_policy"]["non_attack_count"], 0)
        self.assertEqual(summary["quality_check_summary"]["label_consistent"]["fail"], 1)
        self.assertEqual(summary["quality_pass_rate"]["safety_passed"], 1.0)
        self.assertEqual(summary["corpus_residual_sensitive_patterns"], {})
        self.assertEqual(summary["duplicate_utterance_metric"]["scope"], "corpus_turn_utterances_after_masking")
        self.assertIn("deterministic mock runs", summary["duplicate_utterance_metric"]["interpretation"])
        self.assertEqual(summary["duplicate_record_id_count"], 0)
        self.assertEqual(summary["duplicate_record_ids"]["duplicate_ids"], {})
        self.assertIs(
            summary["scenario_default_risk_alignment"],
            summary["scenario_default_attacker_risk_signal_coverage"],
        )
        self.assertFalse(summary["scenario_default_attacker_risk_signal_coverage"]["blocking"])
        self.assertEqual(summary["scenario_default_attacker_risk_signal_coverage"]["mismatch_count"], 1)
        self.assertFalse(summary["attacker_pressure_cue_diversity"]["blocking"])
        self.assertIn("source_metadata", summary)
        self.assertEqual(summary["attacker_pressure_cue_diversity"]["attacker_turn_count"], 1)
        self.assertEqual(summary["attacker_pressure_cue_diversity"]["category_counts"]["urgency"], 1)
        self.assertEqual(summary["quality_signals"]["stage_direction_count"], 0)
        self.assertEqual(summary["quality_signals"]["mask_usage_anomaly_count"], 0)
        self.assertEqual(summary["quality_signals"]["unknown_masked_placeholder_count"], 0)
        self.assertEqual(summary["quality_signals"]["unknown_masked_placeholder_policy"]["decision_counts"], {})
        self.assertEqual(summary["quality_signals"]["unknown_masked_placeholder_policy"]["bucket_counts"], {})
        self.assertEqual(summary["quality_signals"]["partial_compliance_signal_count"], 0)
        self.assertTrue(summary["quality_signals"]["acceptance_gate"]["stage_direction_free"])
        self.assertNotIn("outcome_diverse", summary["quality_signals"]["acceptance_gate"])

    def test_summarize_records_can_revalidate_schema_contract(self):
        record = schema_ready_record()
        record["risk_labels"].append("not_configured")
        record["quality_checks"]["schema_valid"] = True

        summary = summarize_records(
            [record],
            max_turns=10,
            allowed_risk_labels={"voice_impersonation", "credential_request"},
        )

        self.assertEqual(summary["quality_check_summary"]["schema_valid"]["pass"], 1)
        self.assertEqual(summary["schema_contract_validation"]["pass"], 0)
        self.assertEqual(summary["schema_contract_validation"]["fail"], 1)
        self.assertIn("Unknown risk_labels", summary["schema_contract_validation"]["failures"][0]["error"])
        self.assertNotIn("unknown_risk_labels", summary)

    def test_summarize_records_reports_duplicate_record_ids(self):
        first = sample_record()
        second = sample_record()

        summary = summarize_records([first, second])

        self.assertEqual(summary["duplicate_record_id_count"], 1)
        self.assertEqual(summary["duplicate_record_ids"]["duplicate_ids"], {"sample": 2})
        self.assertFalse(summary["duplicate_record_ids"]["blocking"])

    def test_summarize_records_counts_matching_scenario_default_attacker_risk_set(self):
        record = sample_record()
        record["dialogue"] = [
            {"speaker": "attacker", "utterance": "급해.", "risk_signal": "voice_impersonation"},
            {"speaker": "attacker", "utterance": "지금 확인해.", "risk_signal": "urgency"},
            {"speaker": "attacker", "utterance": "엄마를 도와줘.", "risk_signal": "emotional_pressure"},
        ]

        summary = summarize_records([record])

        self.assertEqual(summary["scenario_default_attacker_risk_signal_coverage"]["match_count"], 1)
        self.assertEqual(summary["scenario_default_attacker_risk_signal_coverage"]["mismatch_count"], 0)

    def test_summarize_records_reports_attacker_pressure_cue_diversity_warning(self):
        record = sample_record()
        record["dialogue"] = [
            {"speaker": "attacker", "utterance": "지금 확인해야 합니다.", "risk_signal": "urgency"},
            {"speaker": "attacker", "utterance": "바로 답해야 합니다.", "risk_signal": "urgency"},
            {"speaker": "attacker", "utterance": "몇 분 안에 처리해야 합니다.", "risk_signal": "urgency"},
            {
                "speaker": "victim",
                "utterance": "공식 경로로 다시 확인하겠습니다.",
                "risk_signal": "victim_resistance",
            },
        ]

        summary = summarize_records([record])

        diversity = summary["attacker_pressure_cue_diversity"]
        self.assertFalse(diversity["blocking"])
        self.assertEqual(diversity["attacker_turn_count"], 3)
        self.assertEqual(diversity["categorized_attacker_turn_count"], 3)
        self.assertEqual(diversity["category_counts"], {"urgency": 3})
        self.assertEqual(diversity["unique_pressure_cue_count"], 1)
        self.assertEqual(diversity["dominant_pressure_cue"], "urgency")
        self.assertEqual(diversity["urgency_turn_rate"], 1.0)
        self.assertTrue(diversity["urgency_dominant_warning_only"])

    def test_pressure_cue_urgency_does_not_match_non_pressure_compounds(self):
        self.assertNotIn("urgency", attacker_pressure_cue_categories("급여 정산 서류를 확인해야 합니다."))
        self.assertNotIn("urgency", attacker_pressure_cue_categories("고급 계정 등급 확인이 필요합니다."))
        self.assertIn("urgency", attacker_pressure_cue_categories("긴급해서 바로 확인이 필요합니다."))
        self.assertIn("urgency", attacker_pressure_cue_categories("급해서 바로 확인이 필요합니다."))

    def test_summarize_records_reports_environmental_context_coverage(self):
        summary = summarize_records([schema_ready_record(), sample_record()])

        coverage = summary["environmental_context_coverage"]
        self.assertEqual(coverage["record_count"], 2)
        self.assertEqual(coverage["with_environmental_context"], 1)
        self.assertEqual(coverage["missing_environmental_context"], 1)
        self.assertEqual(coverage["field_coverage"]["channel"], {"present": 1, "missing": 1})
        self.assertEqual(coverage["distributions"]["verification_path"], {"family_callback": 1})
        self.assertEqual(coverage["variant_distribution"], {"variant_00": 1})
        self.assertEqual(coverage["prompt_visible_field_unique_counts"]["channel"], 1)
        self.assertEqual(coverage["missing_records"][0]["record_id"], "sample")

    def test_schema_contract_allows_legacy_missing_environmental_context_and_reports_coverage_gap(self):
        record = schema_ready_record()
        del record["gen_metadata"]["environmental_context"]

        summary = summarize_records([record], max_turns=10)

        self.assertEqual(summary["schema_contract_validation"]["pass"], 1)
        self.assertEqual(summary["schema_contract_validation"]["fail"], 0)
        coverage = summary["environmental_context_coverage"]
        self.assertEqual(coverage["with_environmental_context"], 0)
        self.assertEqual(coverage["missing_environmental_context"], 1)
        self.assertEqual(coverage["field_coverage"]["channel"], {"present": 0, "missing": 1})
        self.assertEqual(coverage["missing_records"], [{"record_id": "sample", "missing": "environmental_context"}])

    def test_checked_in_vllm_unknown_masked_placeholders_are_enumerated_and_classified(self):
        summary = summarize_records(checked_in_quality_records())

        signals = summary["quality_signals"]
        self.assertEqual(signals["unknown_masked_placeholder_count"], 44)
        self.assertEqual(
            signals["unknown_masked_placeholders"],
            {
                "[MASKED_CASE_REFERENCE]": 7,
                "[MASKED_COMPANY]": 10,
                "[MASKED_COMPANY_EMAIL]": 2,
                "[MASKED_CREATOR_NAME]": 1,
                "[MASKED_ID_IMAGE]": 2,
                "[MASKED_JOB_TITLE]": 1,
                "[MASKED_LANGUAGE_SUPPORT_PORTAL]": 2,
                "[MASKED_NAME]": 16,
                "[MASKED_RECRUITER_NAME]": 3,
            },
        )
        self.assertEqual(
            signals["unknown_masked_placeholder_policy"]["decision_counts"],
            {
                "allowlist_candidate": 35,
                "normalize_to_existing": 6,
                "warning_keep_unstandardized": 3,
            },
        )
        self.assertEqual(
            signals["unknown_masked_placeholder_policy"]["bucket_counts"],
            {
                "allowlist": 33,
                "normalize": 6,
                "reviewed_sensitive": 2,
                "reject_or_escalate": 3,
            },
        )

    def test_checked_in_vllm_qa_report_matches_current_quality_signal_policy(self):
        summary = summarize_records(checked_in_quality_records())
        report = checked_in_quality_report()

        self.assertEqual(report["record_count"], summary["record_count"])
        self.assertEqual(report["environmental_context_coverage"], summary["environmental_context_coverage"])
        self.assertEqual(report["quality_signals"], summary["quality_signals"])

    def test_summarize_records_flags_non_attack_labels_under_current_policy(self):
        record = sample_record()
        record["label"] = "benign"

        summary = summarize_records([record])

        self.assertFalse(summary["attack_only_policy"]["pass"])
        self.assertEqual(summary["attack_only_policy"]["non_attack_count"], 1)

    def test_summarize_records_counts_quality_signal_issues(self):
        record = sample_record()
        record["victim_persona"] = {"id": "freelance_creator"}
        record["attacker_persona"] = {"id": "platform_policy_impersonator"}
        record["scenario_type"] = "bank_fraud_alert"
        record["dialogue"][0]["utterance"] = "[딥보이스 스타일] [MASKED_OTP]번으로 입금해 주세요."
        record["dialogue"][1]["utterance"] = (
            "공식 확인 뒤 [MASKED_ID_IMAGE] 자료와 [MASKED_RECRUITER_NAME] 기록을 "
            "[MASKED_COMPANY_EMAIL]로 제출할 수 있습니다."
        )

        summary = summarize_records([record])

        signals = summary["quality_signals"]
        self.assertEqual(signals["stage_direction_count"], 1)
        self.assertEqual(signals["mask_usage_anomaly_count"], 1)
        self.assertEqual(signals["unknown_masked_placeholder_count"], 3)
        self.assertEqual(
            signals["unknown_masked_placeholders"],
            {
                "[MASKED_COMPANY_EMAIL]": 1,
                "[MASKED_ID_IMAGE]": 1,
                "[MASKED_RECRUITER_NAME]": 1,
            },
        )
        placeholder_policy = signals["unknown_masked_placeholder_policy"]
        self.assertFalse(placeholder_policy["blocking"])
        self.assertEqual(
            placeholder_policy["decision_counts"],
            {
                "allowlist_candidate": 1,
                "normalize_to_existing": 2,
            },
        )
        self.assertEqual(
            placeholder_policy["bucket_counts"],
            {
                "normalize": 2,
                "reviewed_sensitive": 1,
            },
        )
        self.assertEqual(
            placeholder_policy["placeholders"]["[MASKED_RECRUITER_NAME]"]["normalize_to"],
            "[MASKED_NAME]",
        )
        self.assertEqual(
            placeholder_policy["placeholders"]["[MASKED_COMPANY_EMAIL]"]["normalize_to"],
            "[MASKED_EMAIL]",
        )
        self.assertEqual(signals["partial_compliance_signal_count"], 1)
        self.assertTrue(signals["outcome_diverse_warning_only"])
        self.assertEqual(signals["scenario_persona_mismatch_count"], 1)
        self.assertFalse(signals["acceptance_gate"]["mask_usage_anomaly_free"])
        self.assertNotIn("outcome_diverse", signals["acceptance_gate"])
        self.assertEqual(signals["scenario_mismatches"][0]["expected"], "platform_policy_notice")

    def test_source_metadata_selects_count_hint_and_ignores_date_like_token(self):
        self.assertEqual(
            filename_count_hint("run_260609_120_vllm.jsonl"),
            {
                "numeric_tokens": ["260609", "120"],
                "ignored_date_like_tokens": ["260609"],
                "count_hint": 120,
            },
        )
        summary = summarize_records([sample_record()], source_path="data/generated/run_260609_120_vllm.jsonl")

        metadata = summary["source_metadata"]
        self.assertEqual(metadata["filename_count_hint"], 120)
        self.assertEqual(metadata["ignored_date_like_tokens"], ["260609"])
        self.assertTrue(metadata["count_mismatch_warning"])

    def test_assigned_pressure_and_target_outcome_alignment_are_reported(self):
        record = schema_ready_record()
        record["gen_metadata"]["pressure_style"] = "authority"
        record["gen_metadata"]["target_outcome_mode"] = "masked_compromise"
        record["gen_metadata"]["target_outcome_generation_attempts"] = 3
        record["gen_metadata"]["target_outcome_repair_applied"] = True
        record["outcome"] = "compromised"
        record["dialogue"][0]["utterance"] = "담당자 심사 절차상 확인이 필요합니다."

        summary = summarize_records([record])

        pressure = summary["assigned_pressure_style_alignment"]
        self.assertEqual(pressure["assigned_record_count"], 1)
        self.assertEqual(pressure["observed_match_count"], 1)
        self.assertEqual(pressure["urgency_leakage_count"], 0)
        outcome = summary["target_outcome_mode_alignment"]
        self.assertEqual(outcome["assigned_record_count"], 1)
        self.assertEqual(outcome["match_count"], 1)
        self.assertEqual(outcome["repair_applied_count"], 1)
        self.assertEqual(outcome["repair_applied_by_target"], {"masked_compromise": 1})
        self.assertEqual(outcome["generation_attempt_distribution"], {"3": 1})
        self.assertFalse(outcome["alignment_review_required"])
        self.assertTrue(outcome["repair_review_required"])
        self.assertTrue(summary["final_dialogue_corpus_review"]["review_required"])
        self.assertIn(
            "target_outcome_repair_rate_above_final_threshold",
            summary["final_dialogue_corpus_review"]["reasons"],
        )

    def test_korean_entity_warning_excludes_masked_placeholder_spans(self):
        record = sample_record()
        record["dialogue"][0]["utterance"] = "[MASKED_COMPANY] 담당자가 아니라 한빛은행 담당자라고 했습니다."

        summary = summarize_records([record])

        warnings = summary["quality_signals"]["korean_entity_warnings"]
        self.assertEqual(summary["quality_signals"]["korean_entity_warning_count"], 1)
        self.assertIn("한빛은행", warnings[0]["matches"])
        self.assertNotIn("MASKED", warnings[0]["matches"])

    def test_unknown_masked_placeholder_detection_includes_digit_tokens(self):
        record = sample_record()
        record["dialogue"][0]["utterance"] = "[MASKED_2FA_CODE] 확인값을 알려주세요."

        summary = summarize_records([record])

        signals = summary["quality_signals"]
        self.assertEqual(signals["unknown_masked_placeholder_count"], 1)
        self.assertEqual(signals["unknown_masked_placeholders"], {"[MASKED_2FA_CODE]": 1})
        placeholder_policy = signals["unknown_masked_placeholder_policy"]
        self.assertEqual(placeholder_policy["decision_counts"], {"unclassified_review_required": 1})
        self.assertEqual(
            placeholder_policy["placeholders"]["[MASKED_2FA_CODE]"]["decision"],
            "unclassified_review_required",
        )

    def test_observed_unknown_placeholder_aliases_are_classified_by_heuristic(self):
        record = sample_record()
        record["dialogue"][0]["utterance"] = "[MASKED_ORDER_REFERENCE] 확인이 필요합니다."

        summary = summarize_records([record])

        placeholder_policy = summary["quality_signals"]["unknown_masked_placeholder_policy"]
        self.assertEqual(placeholder_policy["decision_counts"], {"normalize_to_existing": 1})
        self.assertEqual(
            placeholder_policy["placeholders"]["[MASKED_ORDER_REFERENCE]"]["normalize_to"],
            "[MASKED_CASE_REFERENCE]",
        )
        self.assertFalse(summary["final_dialogue_corpus_review"]["review_required"])

    def test_final_dialogue_corpus_review_flags_stage_direction_and_unclassified_placeholders(self):
        record = sample_record()
        record["dialogue"][0]["utterance"] = "[voice shaky] [MASKED_2FA_CODE] 확인값을 알려주세요."

        summary = summarize_records([record])

        review = summary["final_dialogue_corpus_review"]
        self.assertTrue(review["review_required"])
        self.assertIn("stage_direction_present", review["reasons"])
        self.assertIn("unknown_placeholder_unclassified_review_required", review["reasons"])
        self.assertEqual(review["observed"]["stage_direction_count"], 1)
        self.assertEqual(review["observed"]["unknown_placeholder_unclassified_count"], 1)

    def test_markdown_report_renders_quality_table(self):
        markdown = format_markdown_report(summarize_records([sample_record()]))

        self.assertIn("## Quality Checks", markdown)
        self.assertIn("## Schema Contract Validation", markdown)
        self.assertIn("- enabled: false", markdown)
        self.assertIn("| label_consistent | 0 | 1 | 0 | 0.0 |", markdown)
        self.assertIn("## Quality Signals", markdown)
        self.assertIn("- stage_direction_count: 0", markdown)
        self.assertIn("- unknown_masked_placeholder_count: 0", markdown)
        self.assertNotIn("- unknown_masked_placeholder_policy:", markdown)
        self.assertIn("- outcome_diverse_warning_only: true", markdown)
        self.assertNotIn("gate_outcome_diverse", markdown)
        self.assertIn("## Attack-Only Policy", markdown)
        self.assertIn("- status: pass", markdown)
        self.assertIn("## Residual Sensitive Patterns", markdown)
        self.assertIn("### corpus residual patterns", markdown)
        self.assertIn("## Duplicate Utterance Metric", markdown)
        self.assertIn("- normalization: lowercase_without_whitespace", markdown)
        self.assertIn("## Duplicate Record Ids", markdown)
        self.assertIn("- duplicate_count: 0", markdown)
        self.assertIn("## Scenario Default Attacker Risk Signal Coverage", markdown)
        self.assertIn("- blocking: false", markdown)
        self.assertIn("- mismatch_count: 1", markdown)
        self.assertIn("## Attacker Pressure Cue Diversity", markdown)
        self.assertIn("- urgency_dominant_warning_only: false", markdown)
        self.assertIn("  - urgency: 1", markdown)
        self.assertIn("## Environmental Context Coverage", markdown)
        self.assertIn("- missing_environmental_context: 1", markdown)
        self.assertIn("| channel | 0 | 1 |", markdown)

    def test_writes_markdown_and_json_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "qa.md"
            json_path = Path(tmpdir) / "qa.json"

            write_markdown_report([sample_record()], md_path)
            write_json_report([sample_record()], json_path)

            self.assertIn("QA Report", md_path.read_text(encoding="utf-8"))
            self.assertIn('"record_count": 1', json_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
