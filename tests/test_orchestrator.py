import json
import copy
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from madb.agents import AttackerAgent, VictimAgent
from madb.config import GenerationConfig, load_json_list
from madb.llm_client import MockLLMClient
from madb.labeling import annotate_turn
from madb.orchestrator import (
    ENVIRONMENTAL_CONTEXTS,
    PAIR_SCENARIO_PREFERENCES,
    SCENARIO_CONTEXTS,
    SCENARIO_RISK_LABELS,
    DatasetOrchestrator,
    choose_scenario_type,
    default_assets_for_scenario,
    environmental_context_for_scenario,
    make_record_id,
    persona_pair_for_index,
    schedule_assignment_for_index,
    target_outcome_repair_utterance,
)
from madb.qa import summarize_records
from madb.safety import SafetyMasker
from madb.schema import SchemaError, validate_record


class FailingFirstOrchestrator(DatasetOrchestrator):
    def generate_one(self, victim, attacker, scenario_type, index, **kwargs):
        if index == 0:
            raise ValueError("synthetic failure")
        return super().generate_one(victim, attacker, scenario_type, index, **kwargs)


class RawSensitivePatternClient:
    def chat(self, messages, *, model, temperature, max_tokens):
        joined = "\n".join(message["content"] for message in messages)
        if "ROLE=attacker" in joined:
            return (
                "진료 확인을 위해 인증번호 123456, https://example.test, 010-1234-5678, "
                "900101-1234567, password:abcd1234, 123-456-789012 확인이 필요합니다."
            )
        return "공식 앱과 대표번호로 확인하기 전에는 통화 중 정보를 제공하지 않겠습니다."


class RawPinCredentialClient:
    def chat(self, messages, *, model, temperature, max_tokens):
        joined = "\n".join(message["content"] for message in messages)
        if "ROLE=attacker" in joined:
            return "본인 확인을 위해 pin:1234 값을 알려주세요."
        return "공식 앱과 대표번호로 확인하기 전에는 통화 중 정보를 제공하지 않겠습니다."


class DefensiveDriftClient:
    def chat(self, messages, *, model, temperature, max_tokens):
        joined = "\n".join(message["content"] for message in messages)
        phase = "opening"
        for line in joined.splitlines():
            if line.startswith("phase="):
                phase = line.split("=", 1)[1]
                break
        if "ROLE=attacker" in joined:
            return f"{phase} 단계의 보안 확인 절차상 [MASKED_OTP] 인증이 필요합니다."
        return f"{phase} 단계에서도 공식 앱에서 확인하기 전에는 [MASKED_OTP]를 알려주지 않겠습니다."


class RetryAwareTargetClient:
    def chat(self, messages, *, model, temperature, max_tokens):
        joined = "\n".join(message["content"] for message in messages)
        phase = "opening"
        target = "defended_success"
        retry_attempt = 0
        for line in joined.splitlines():
            if line.startswith("phase="):
                phase = line.split("=", 1)[1]
            elif line.startswith("target_outcome_mode="):
                target = line.split("=", 1)[1]
            elif "outcome_retry_attempt=1" in line:
                retry_attempt = 1
        if "ROLE=attacker" in joined:
            return f"{phase} 단계의 보안 확인 절차상 [MASKED_OTP] 인증이 필요합니다."
        if retry_attempt == 0:
            return f"{phase} 단계에서는 공식 앱에서 확인하기 전에는 [MASKED_OTP]를 알려주지 않겠습니다."
        if target == "inconclusive":
            utterances = {
                "opening": "갑작스러운 연락이라 지금 바로 판단하기 어렵습니다.",
                "establish_context": "말씀하신 상황은 이해했지만 아직 결정하기가 어렵습니다.",
                "make_request": "요청하신 내용은 알겠지만 지금 결정하기가 어렵습니다.",
                "handle_resistance": "계속 재촉받으니 더 망설여집니다.",
                "close": "조금 더 생각한 뒤 대답하겠습니다.",
            }
            return utterances.get(phase, utterances["close"])
        if target == "masked_compromise":
            if phase == "make_request":
                return "[MASKED_OTP]를 입력하겠습니다."
            return "말씀하신 절차를 더 듣고 진행해 보겠습니다."
        return "이 통화에서는 [MASKED_OTP]를 알려주지 않고 공식 경로로 확인하겠습니다."


class RewriteAwareTargetClient:
    def chat(self, messages, *, model, temperature, max_tokens):
        joined = "\n".join(message["content"] for message in messages)
        phase = "opening"
        target = "defended_success"
        rewrite_attempt = False
        for line in joined.splitlines():
            if line.startswith("phase="):
                phase = line.split("=", 1)[1]
            elif line.startswith("target_outcome_mode="):
                target = line.split("=", 1)[1]
            elif "victim_rewrite_attempt=" in line:
                rewrite_attempt = True
        if "ROLE=attacker" in joined:
            return f"{phase} 단계의 확인 절차상 [MASKED_OTP] 인증이 필요합니다."
        if rewrite_attempt and target == "inconclusive":
            utterances = {
                "opening": "갑작스러운 연락이라 지금은 마음이 좀 흔들립니다.",
                "establish_context": "말씀은 이해했지만 아직 어느 쪽인지 잘 모르겠습니다.",
                "make_request": "지금 정하라고 하시니 더 망설여져서 잠깐 생각하고 싶습니다.",
                "handle_resistance": "재촉을 들으니 더 혼란스러워서 바로 답하기 어렵습니다.",
                "close": "조금 시간을 두고 생각한 뒤에야 말할 수 있을 것 같습니다.",
            }
            return utterances.get(phase, utterances["close"])
        return f"{phase} 단계에서는 공식 앱에서 확인하기 전에는 [MASKED_OTP]를 알려주지 않겠습니다."


def record_for_pair(records, victim_id, attacker_id):
    matches = [
        record
        for record in records
        if record["victim_persona"]["id"] == victim_id
        and record["attacker_persona"]["id"] == attacker_id
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one record for {victim_id}/{attacker_id}, found {len(matches)}")
    return matches[0]


def persona_pair_ids_for_index(victims, attackers, index):
    victim, attacker = persona_pair_for_index(victims, attackers, index)
    return victim["id"], attacker["id"]


class OrchestratorTest(unittest.TestCase):
    def test_generates_mock_records(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=3,
        )

        self.assertEqual(len(records), 3)
        self.assertEqual({record["label"] for record in records}, {"attack"})
        self.assertEqual({record["victim_persona"]["id"] for record in records}, {
            "elderly_parent",
            "office_worker",
            "student_or_young_adult",
        })
        for record in records:
            validate_record(record, max_turns=config.max_turns)
            self.assertLessEqual(len(record["dialogue"]), 10)
            self.assertIn(record["outcome"], {"defended_success", "compromised", "aborted", "inconclusive"})
            self.assertTrue(record["protected_assets"])
            self.assertTrue(record["risk_labels"])
            self.assertEqual(
                set(record["gen_metadata"]["environmental_context"]),
                {
                    "channel",
                    "timing",
                    "trigger_event",
                    "claimed_authority",
                    "victim_constraint",
                    "verification_path",
                },
            )
            self.assertIn({"label": "account", "placeholder": "[MASKED_ACCOUNT]"}, record["safety_masks"])
            self.assertIn({"label": "otp", "placeholder": "[MASKED_OTP]"}, record["safety_masks"])
            self.assertTrue(record["quality_checks"]["no_duplicate_turns"])
            self.assertTrue(record["quality_checks"]["schema_valid"])

    def test_mock_default_corpus_keeps_duplicate_utterance_rate_bounded(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.default_limit,
        )
        summary = summarize_records(records)

        self.assertLessEqual(summary["duplicate_utterance_rate"], 0.2)
        self.assertIn("deterministic mock runs", summary["duplicate_utterance_metric"]["interpretation"])

    def test_generation_schema_uses_configured_risk_label_set(self):
        config = replace(GenerationConfig.from_file("configs/default.json"), risk_label_set=["urgency"])
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        with self.assertRaisesRegex(SchemaError, "Unknown risk_labels"):
            orchestrator.generate_one(victims[0], attackers[0], "family_emergency", 0)

    def test_default_schedule_keeps_expected_persona_pair_scenarios(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.default_limit,
        )

        expected_by_victim = {
            "elderly_parent": ("family_impersonator", "family_emergency"),
            "office_worker": ("institution_impersonator", "bank_fraud_alert"),
            "student_or_young_adult": ("corporate_impersonator", "workplace_authority"),
            "small_business_owner": ("service_provider_impersonator", "workplace_authority"),
            "remote_job_seeker": ("recruiter_impersonator", "workplace_authority"),
            "telehealth_patient": ("healthcare_support_impersonator", "bank_fraud_alert"),
            "renter_or_tenant": ("housing_admin_impersonator", "bank_fraud_alert"),
            "utility_account_holder": ("utility_support_impersonator", "essential_service_notice"),
            "public_benefit_recipient": ("benefits_caseworker_impersonator", "public_benefit_notice"),
            "freelance_creator": ("platform_policy_impersonator", "platform_policy_notice"),
            "caregiver_or_guardian": ("education_admin_impersonator", "education_admin_notice"),
            "language_access_service_user": ("document_support_impersonator", "document_verification_notice"),
        }
        actual_by_victim = {
            record["victim_persona"]["id"]: (
                record["attacker_persona"]["id"],
                record["scenario_type"],
            )
            for record in records
        }

        self.assertEqual(len(actual_by_victim), len(records))
        self.assertEqual(actual_by_victim, expected_by_victim)

    def test_default_mock_run_preserves_conditional_partial_compliance_as_inconclusive(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.default_limit,
        )
        outcomes = Counter(record["outcome"] for record in records)
        partial_records = [
            record
            for record in records
            if any(
                turn["speaker"] == "victim"
                and "[MASKED_OTP]를 입력해볼게요" in turn["utterance"]
                and "공식 경로에서만 처리" in turn["utterance"]
                for turn in record["dialogue"]
            )
        ]

        self.assertEqual(outcomes["inconclusive"], 3)
        self.assertEqual(outcomes["compromised"], 0)
        self.assertEqual(len(partial_records), 3)
        self.assertTrue(all(record["compromised_assets"] == [] for record in partial_records))
        self.assertTrue(all(record["quality_checks"]["schema_valid"] for record in partial_records))

    def test_persona_pair_for_index_pairs_first_pass_then_rotates_attackers(self):
        victims = [{"id": f"victim_{index}"} for index in range(3)]
        attackers = [{"id": f"attacker_{index}"} for index in range(3)]

        self.assertEqual(
            [persona_pair_ids_for_index(victims, attackers, index) for index in range(3)],
            [
                ("victim_0", "attacker_0"),
                ("victim_1", "attacker_1"),
                ("victim_2", "attacker_2"),
            ],
        )
        self.assertEqual(
            [persona_pair_ids_for_index(victims, attackers, index) for index in range(3, 6)],
            [
                ("victim_0", "attacker_1"),
                ("victim_1", "attacker_2"),
                ("victim_2", "attacker_0"),
            ],
        )

    def test_canonical_repeat_schedule_repeats_canonical_pairs_without_rotating_attackers(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)
        assignments = [
            schedule_assignment_for_index(
                victims,
                attackers,
                config.scenario_types,
                index,
                schedule_mode="canonical_repeat",
            )
            for index in range(240)
        ]

        self.assertEqual(Counter(assignment.victim["id"] for assignment in assignments), Counter(v["id"] for v in victims for _ in range(10)))
        self.assertEqual(Counter(assignment.attacker["id"] for assignment in assignments), Counter(a["id"] for a in attackers for _ in range(10)))
        self.assertEqual({assignment.variant_index for assignment in assignments}, set(range(10)))
        self.assertEqual({assignment.schedule_mode for assignment in assignments}, {"canonical_repeat"})
        self.assertNotIn("aborted", {assignment.target_outcome_mode for assignment in assignments})
        for assignment in assignments:
            pair_key = (assignment.victim["id"], assignment.attacker["id"])
            expected = PAIR_SCENARIO_PREFERENCES.get(pair_key)
            if expected is not None:
                self.assertEqual(assignment.scenario_type, expected)

    def test_canonical_repeat_scenario_distribution_matches_preference_map_targets(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)
        distribution = Counter(
            schedule_assignment_for_index(
                victims,
                attackers,
                config.scenario_types,
                index,
                schedule_mode="canonical_repeat",
            ).scenario_type
            for index in range(240)
        )

        self.assertEqual(distribution["bank_fraud_alert"], 30)
        self.assertEqual(distribution["workplace_authority"], 30)
        self.assertEqual(distribution["family_emergency"], 10)
        for scenario in set(PAIR_SCENARIO_PREFERENCES.values()):
            self.assertEqual(distribution[scenario], 10, scenario)

    def test_canonical_repeat_records_include_repair_metadata_and_context_variants(self):
        config = replace(GenerationConfig.from_file("configs/default.json"), schedule_mode="canonical_repeat")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=26,
        )

        first = records[0]["gen_metadata"]
        repeated = records[24]["gen_metadata"]
        for metadata in (first, repeated):
            self.assertEqual(metadata["schedule_mode"], "canonical_repeat")
            self.assertIn(metadata["target_outcome_mode"], {"defended_success", "inconclusive", "masked_compromise"})
            self.assertIn(metadata["pressure_style"], {"urgency", "authority", "consequence", "relationship_or_emotional_pressure", "process_control"})
            self.assertIn("environmental_context_variant_id", metadata)
        self.assertEqual(first["canonical_pair_index"], 0)
        self.assertEqual(first["variant_index"], 0)
        self.assertEqual(repeated["canonical_pair_index"], 0)
        self.assertEqual(repeated["variant_index"], 1)
        self.assertNotEqual(first["environmental_context"], repeated["environmental_context"])

    def test_target_outcome_repair_aligns_defensive_vllm_drift(self):
        config = replace(GenerationConfig.from_file("configs/default.json"), schedule_mode="canonical_repeat")
        client = DefensiveDriftClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=3,
        )
        summary = summarize_records(
            records,
            max_turns=config.max_turns,
            allowed_risk_labels=config.risk_label_set,
        )

        self.assertEqual(
            [record["gen_metadata"]["target_outcome_mode"] for record in records],
            ["defended_success", "inconclusive", "masked_compromise"],
        )
        self.assertEqual(
            [record["outcome"] for record in records],
            ["defended_success", "inconclusive", "compromised"],
        )
        self.assertEqual(
            [record["gen_metadata"]["target_outcome_repair_applied"] for record in records],
            [False, True, True],
        )
        self.assertEqual(
            [record["gen_metadata"]["target_outcome_generation_attempts"] for record in records],
            [1, 3, 3],
        )
        self.assertEqual(summary["target_outcome_mode_alignment"]["match_count"], 3)
        self.assertEqual(summary["target_outcome_mode_alignment"]["repair_applied_count"], 2)
        self.assertEqual(
            summary["target_outcome_mode_alignment"]["repair_applied_by_target"],
            {"inconclusive": 1, "masked_compromise": 1},
        )
        self.assertEqual(records[2]["compromised_assets"], ["masked_otp"])
        self.assertIn({"label": "otp", "placeholder": "[MASKED_OTP]"}, records[2]["safety_masks"])
        self.assertTrue(all(record["quality_checks"]["schema_valid"] for record in records))

    def test_inconclusive_repair_uses_record_index_for_variation(self):
        utterances = {
            target_outcome_repair_utterance(
                "inconclusive",
                ["case_reference_number"],
                phase="make_request",
                is_final=False,
                victim_id="public_benefit_recipient",
                scenario_type="public_benefit_notice",
                environmental_context=environmental_context_for_scenario("public_benefit_notice"),
                variant_index=0,
                record_index=index,
            )
            for index in range(20)
        }

        self.assertGreaterEqual(len(utterances), 5)

    def test_target_outcome_retry_can_align_without_repair(self):
        config = replace(GenerationConfig.from_file("configs/default.json"), schedule_mode="canonical_repeat")
        client = RetryAwareTargetClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=3,
        )
        summary = summarize_records(
            records,
            max_turns=config.max_turns,
            allowed_risk_labels=config.risk_label_set,
        )

        self.assertEqual(
            [record["outcome"] for record in records],
            ["defended_success", "inconclusive", "compromised"],
        )
        self.assertEqual(
            [record["gen_metadata"]["target_outcome_generation_attempts"] for record in records],
            [1, 2, 2],
        )
        self.assertEqual(
            [record["gen_metadata"]["target_outcome_repair_applied"] for record in records],
            [False, False, False],
        )
        self.assertEqual(summary["target_outcome_mode_alignment"]["retry_success_count"], 2)
        self.assertEqual(summary["target_outcome_mode_alignment"]["repair_applied_count"], 0)

    def test_victim_turn_rewrite_can_align_inconclusive_without_deterministic_repair(self):
        config = replace(GenerationConfig.from_file("configs/default.json"), schedule_mode="canonical_repeat")
        client = RewriteAwareTargetClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=2,
        )
        rewritten = records[1]
        summary = summarize_records(
            records,
            max_turns=config.max_turns,
            allowed_risk_labels=config.risk_label_set,
        )

        self.assertEqual(rewritten["gen_metadata"]["target_outcome_mode"], "inconclusive")
        self.assertEqual(rewritten["outcome"], "inconclusive")
        self.assertTrue(rewritten["gen_metadata"]["target_outcome_llm_rewrite_applied"])
        self.assertEqual(rewritten["gen_metadata"]["target_outcome_llm_rewrite_attempts"], 1)
        self.assertFalse(rewritten["gen_metadata"]["target_outcome_deterministic_repair_applied"])
        self.assertFalse(rewritten["gen_metadata"]["target_outcome_repair_applied"])
        outcome = summary["target_outcome_mode_alignment"]
        self.assertEqual(outcome["match_count"], 2)
        self.assertEqual(outcome["repair_applied_count"], 0)
        self.assertEqual(outcome["llm_rewrite_applied_count"], 1)
        self.assertEqual(outcome["llm_rewrite_applied_by_target"], {"inconclusive": 1})

    def test_default_smoke_schedule_covers_configured_prefix_once(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        # The default limit is the smoke-run size, not the catalog capacity.
        self.assertLessEqual(config.default_limit, len(victims))
        self.assertLessEqual(config.default_limit, len(attackers))

        records = orchestrator.generate(victims, attackers, limit=config.default_limit)

        self.assertEqual(
            Counter(record["victim_persona"]["id"] for record in records),
            Counter(victim["id"] for victim in victims[: config.default_limit]),
        )
        self.assertEqual(
            Counter(record["attacker_persona"]["id"] for record in records),
            Counter(attacker["id"] for attacker in attackers[: config.default_limit]),
        )

    def test_manual_limit_can_generate_beyond_default_limit_within_catalog_capacity(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        self.assertLess(config.default_limit, len(victims))
        self.assertLessEqual(len(victims), config.persona_catalog_limit)
        records = orchestrator.generate(
            victims,
            attackers,
            limit=config.default_limit + 1,
        )
        extra_record = record_for_pair(records, "seasonal_tax_filer", "tax_portal_impersonator")

        self.assertEqual(len(records), config.default_limit + 1)
        self.assertEqual(extra_record["scenario_type"], "tax_refund_notice")
        self.assertTrue(extra_record["quality_checks"]["schema_valid"])

    def test_extended_schedule_includes_creator_policy_personas(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=10,
        )
        creator_record = record_for_pair(records, "freelance_creator", "platform_policy_impersonator")

        self.assertEqual(creator_record["victim_persona"]["id"], "freelance_creator")
        self.assertEqual(creator_record["attacker_persona"]["id"], "platform_policy_impersonator")
        self.assertEqual(creator_record["scenario_type"], "platform_policy_notice")
        self.assertEqual(creator_record["attacker_goal"], "credential_capture")
        self.assertIn("콘텐츠 또는 후원 플랫폼 정책 담당자", creator_record["context"])
        self.assertIn("플랫폼 정책 심사 알림", creator_record["dialogue"][0]["utterance"])
        self.assertIn("플랫폼 공식 알림함과 등록된 정산 메뉴", creator_record["dialogue"][1]["utterance"])
        self.assertNotIn("money_transfer_request", creator_record["risk_labels"])
        self.assertTrue(creator_record["quality_checks"]["schema_valid"])

    def test_extended_schedule_includes_guardian_education_personas(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=11,
        )
        guardian_record = record_for_pair(records, "caregiver_or_guardian", "education_admin_impersonator")

        self.assertEqual(guardian_record["victim_persona"]["id"], "caregiver_or_guardian")
        self.assertEqual(guardian_record["attacker_persona"]["id"], "education_admin_impersonator")
        self.assertEqual(guardian_record["scenario_type"], "education_admin_notice")
        self.assertEqual(guardian_record["attacker_goal"], "identity_information")
        self.assertIn("학교나 돌봄 활동 운영 담당자", guardian_record["context"])
        self.assertIn("보호자 행정 확인 알림", guardian_record["dialogue"][0]["utterance"])
        self.assertIn("공식 알림장과 등록된 보호자 확인 경로", guardian_record["dialogue"][1]["utterance"])
        self.assertTrue(guardian_record["quality_checks"]["schema_valid"])

    def test_extended_schedule_includes_language_access_document_personas(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.default_limit,
        )
        document_record = record_for_pair(records, "language_access_service_user", "document_support_impersonator")

        self.assertEqual(document_record["victim_persona"]["id"], "language_access_service_user")
        self.assertEqual(document_record["attacker_persona"]["id"], "document_support_impersonator")
        self.assertEqual(document_record["scenario_type"], "document_verification_notice")
        self.assertEqual(document_record["attacker_goal"], "identity_information")
        self.assertIn("서류 또는 언어 지원 담당자", document_record["context"])
        self.assertIn("서류 지원 보완 알림", document_record["dialogue"][0]["utterance"])
        self.assertIn("공식 포털과 등록된 지원 창구", document_record["dialogue"][1]["utterance"])
        self.assertIn("identity_information_request", document_record["risk_labels"])
        self.assertTrue(document_record["quality_checks"]["schema_valid"])

    def test_twenty_four_record_mock_schedule_implements_repaired_batch2_pairs(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.persona_catalog_limit,
        )

        expected_suffix = [
            ("seasonal_tax_filer", "tax_portal_impersonator", "tax_refund_notice"),
            ("travel_booking_customer", "travel_rebooking_impersonator", "travel_booking_notice"),
            ("property_claim_policyholder", "property_claim_adjuster_impersonator", "property_claim_notice"),
            ("disaster_preparedness_resident", "emergency_drill_coordinator_impersonator", "emergency_drill_notice"),
            ("community_volunteer_coordinator", "nonprofit_admin_impersonator", "nonprofit_admin_notice"),
            ("pet_care_client", "veterinary_scheduler_impersonator", "pet_appointment_notice"),
            ("research_participant", "university_research_impersonator", "research_study_notice"),
            ("library_account_holder", "library_services_impersonator", "library_account_notice"),
            ("vehicle_owner", "vehicle_recall_impersonator", "vehicle_service_notice"),
            ("event_ticket_buyer", "ticketing_refund_impersonator", "ticketing_refund_notice"),
            ("museum_member", "membership_services_impersonator", "membership_access_notice"),
            ("infant_childcare_applicant", "childcare_waitlist_impersonator", "childcare_service_notice"),
        ]
        actual_suffix = [
            (
                record["victim_persona"]["id"],
                record["attacker_persona"]["id"],
                record["scenario_type"],
            )
            for record in records[12:24]
        ]

        self.assertEqual(len(records), 24)
        self.assertEqual(actual_suffix, expected_suffix)
        self.assertTrue(all(record["quality_checks"]["schema_valid"] for record in records))
        self.assertEqual(
            record_for_pair(records, "disaster_preparedness_resident", "emergency_drill_coordinator_impersonator")["protected_assets"],
            [
                "preparedness_notice_reference",
                "drill_callback_reference",
                "registered_contact_details",
                "callback_window_preference",
                "voice_sample",
            ],
        )
        self.assertEqual(
            record_for_pair(records, "pet_care_client", "veterinary_scheduler_impersonator")["protected_assets"],
            [
                "appointment_reference_number",
                "pet_profile_reference",
                "registered_contact_details",
                "clinic_callback_reference",
                "voice_sample",
            ],
        )
        self.assertEqual(
            record_for_pair(records, "museum_member", "membership_services_impersonator")["protected_assets"],
            [
                "membership_reference_number",
                "member_contact_details",
                "entry_reservation_status",
                "mailed_notice_reference",
                "voice_sample",
            ],
        )
        self.assertEqual(
            record_for_pair(records, "infant_childcare_applicant", "childcare_waitlist_impersonator")["protected_assets"],
            [
                "guardian_identity_reference",
                "waitlist_reference_number",
                "registered_contact_details",
                "intake_appointment_details",
                "voice_sample",
            ],
        )

        repaired_dialogue_text = "\n".join(
            "\n".join(turn["utterance"] for turn in record["dialogue"])
            for record in records
            if record["scenario_type"]
            in {"emergency_drill_notice", "pet_appointment_notice", "membership_access_notice", "childcare_service_notice"}
        )
        forbidden_terms = [
            "evacuation",
            "shelter",
            "access code",
            "payment",
            "credential",
            "diagnosis",
            "dosage",
            "insurance",
            "ticket transfer",
            "resale",
            "refund",
            "donation",
            "school pickup",
            "student records",
            "[MASKED_CHILD_RECORD]",
            "대피",
            "피난",
            "접근 코드",
            "결제",
            "진단",
            "투약",
            "보험",
            "환불",
            "기부",
            "학교 픽업",
            "학생 기록",
        ]
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, repaired_dialogue_text)

    def test_start_index_can_generate_batch2_only(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=12,
            start_index=12,
        )

        self.assertEqual(len(records), 12)
        self.assertEqual(records[0]["victim_persona"]["id"], "seasonal_tax_filer")
        self.assertEqual(records[0]["attacker_persona"]["id"], "tax_portal_impersonator")
        self.assertEqual(records[0]["scenario_type"], "tax_refund_notice")
        self.assertEqual(records[-1]["victim_persona"]["id"], "infant_childcare_applicant")
        self.assertEqual(records[-1]["attacker_persona"]["id"], "childcare_waitlist_impersonator")
        self.assertEqual(records[-1]["scenario_type"], "childcare_service_notice")
        self.assertTrue(all(record["quality_checks"]["schema_valid"] for record in records))

    def test_mock_creator_policy_make_request_is_deterministically_labeled_as_credential(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        utterance = client.chat(
            [
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "ROLE=attacker",
                            "phase=make_request",
                            "scenario_type=platform_policy_notice",
                            "attacker_goal=credential_capture",
                            "attacker_persona=platform_policy_impersonator: mock",
                        ]
                    ),
                }
            ],
            model=config.attacker_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        _intent, risk = annotate_turn("attacker", utterance, "credential_request")

        self.assertIn("플랫폼 정책 심사 알림", utterance)
        self.assertEqual(risk, "credential_request")

    def test_family_emergency_mock_money_transfer_label_is_backed_by_turn_signal(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        record = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=1,
        )[0]
        attacker_risks = {
            turn["risk_signal"]
            for turn in record["dialogue"]
            if turn["speaker"] == "attacker"
        }

        self.assertEqual(record["scenario_type"], "family_emergency")
        self.assertIn("money_transfer_request", record["risk_labels"])
        self.assertIn("money_transfer_request", attacker_risks)

    def test_default_schedule_exercises_utility_personas_with_service_scenario(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.default_limit,
        )
        utility_records = [
            record
            for record in records
            if record["victim_persona"]["id"] == "utility_account_holder"
            or record["attacker_persona"]["id"] == "utility_support_impersonator"
        ]

        self.assertEqual(len(utility_records), 1)
        utility_record = utility_records[0]
        self.assertEqual(utility_record["victim_persona"]["id"], "utility_account_holder")
        self.assertEqual(utility_record["attacker_persona"]["id"], "utility_support_impersonator")
        self.assertEqual(utility_record["scenario_type"], "essential_service_notice")
        self.assertIn("필수 생활 서비스", utility_record["context"])
        self.assertIn("필수 서비스 중단 알림", utility_record["dialogue"][0]["utterance"])

    def test_default_schedule_does_not_assign_service_scenario_to_non_utility_personas(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        records = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=config.default_limit,
        )
        unintended_service_records = [
            record
            for record in records
            if record["scenario_type"] == "essential_service_notice"
            and (
                record["victim_persona"]["id"] != "utility_account_holder"
                or record["attacker_persona"]["id"] != "utility_support_impersonator"
            )
        ]

        self.assertEqual(unintended_service_records, [])

    def test_service_scenario_is_pair_only_for_direct_scenario_selection(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)
        utility_victim = next(victim for victim in victims if victim["id"] == "utility_account_holder")
        utility_attacker = next(attacker for attacker in attackers if attacker["id"] == "utility_support_impersonator")
        document_victim = next(victim for victim in victims if victim["id"] == "language_access_service_user")
        document_attacker = next(attacker for attacker in attackers if attacker["id"] == "document_support_impersonator")
        benefit_victim = next(victim for victim in victims if victim["id"] == "public_benefit_recipient")
        benefit_attacker = next(attacker for attacker in attackers if attacker["id"] == "benefits_caseworker_impersonator")
        creator_victim = next(victim for victim in victims if victim["id"] == "freelance_creator")
        creator_attacker = next(attacker for attacker in attackers if attacker["id"] == "platform_policy_impersonator")
        guardian_victim = next(victim for victim in victims if victim["id"] == "caregiver_or_guardian")
        guardian_attacker = next(attacker for attacker in attackers if attacker["id"] == "education_admin_impersonator")

        self.assertEqual(
            choose_scenario_type(utility_victim, utility_attacker, config.scenario_types, 0),
            "essential_service_notice",
        )
        self.assertEqual(
            choose_scenario_type(document_victim, document_attacker, config.scenario_types, 0),
            "document_verification_notice",
        )
        self.assertEqual(
            choose_scenario_type(benefit_victim, benefit_attacker, config.scenario_types, 0),
            "public_benefit_notice",
        )
        self.assertEqual(
            choose_scenario_type(creator_victim, creator_attacker, config.scenario_types, 0),
            "platform_policy_notice",
        )
        self.assertEqual(
            choose_scenario_type(guardian_victim, guardian_attacker, config.scenario_types, 0),
            "education_admin_notice",
        )
        for victim in victims:
            if victim["id"] != "utility_account_holder":
                self.assertNotEqual(
                    choose_scenario_type(victim, utility_attacker, config.scenario_types, 0),
                    "essential_service_notice",
                )
        for attacker in attackers:
            if attacker["id"] != "utility_support_impersonator":
                self.assertNotEqual(
                    choose_scenario_type(utility_victim, attacker, config.scenario_types, 0),
                    "essential_service_notice",
                )

    def test_pair_only_scenario_can_still_be_used_when_it_is_the_only_configured_scenario(self):
        config = GenerationConfig.from_file("configs/default.json")
        victim = load_json_list(config.victims_path)[0]
        attacker = load_json_list(config.attackers_path)[0]

        self.assertEqual(
            choose_scenario_type(victim, attacker, ["essential_service_notice"], 0),
            "essential_service_notice",
        )

    def test_configured_scenarios_have_metadata_assets_and_mock_coverage(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        expected_attacker_cues = {
            "family_emergency": "가족 사고 확인",
            "bank_fraud_alert": "은행 보안 알림",
            "workplace_authority": "업무 결재 예외",
            "essential_service_notice": "필수 서비스 중단 알림",
            "document_verification_notice": "서류 보완 확인 알림",
            "public_benefit_notice": "공공 지원 자격 확인 알림",
            "platform_policy_notice": "플랫폼 정책 심사 알림",
            "education_admin_notice": "보호자 행정 확인 알림",
            "tax_refund_notice": "세무 환급 상태 확인 알림",
            "travel_booking_notice": "여행 재예약 일정 확인 알림",
            "property_claim_notice": "재산 보험 청구 보완 알림",
            "emergency_drill_notice": "사전 재난 대비 훈련 콜백 알림",
            "nonprofit_admin_notice": "봉사 행사 콜백 확인 알림",
            "pet_appointment_notice": "반려동물 예약 접수 확인 알림",
            "research_study_notice": "연구 참여 일정 확인 알림",
            "library_account_notice": "도서관 이용 상태 확인 알림",
            "vehicle_service_notice": "차량 리콜 서비스 일정 알림",
            "ticketing_refund_notice": "예매 환불 상태 확인 알림",
            "membership_access_notice": "문화시설 멤버십 접근 확인 알림",
            "childcare_service_notice": "영아 대기자 상담 일정 확인 알림",
        }
        expected_victim_verifiers = {
            "family_emergency": "가족에게 직접 다시 전화",
            "bank_fraud_alert": "은행 공식 앱과 대표번호",
            "workplace_authority": "사내 결재 라인과 보안팀",
            "essential_service_notice": "공식 앱과 등록된 고객센터",
            "document_verification_notice": "공식 포털과 서면 안내문",
            "public_benefit_notice": "공식 포털과 우편 안내문",
            "platform_policy_notice": "플랫폼 공식 알림함과 등록된 정산 메뉴",
            "education_admin_notice": "공식 알림장과 등록된 보호자 확인 경로",
            "tax_refund_notice": "세무 공식 포털과 우편 고지",
            "travel_booking_notice": "예약 앱과 저장된 고객센터 번호",
            "property_claim_notice": "보험사 앱과 등록된 보상 담당자 callback",
            "emergency_drill_notice": "우편 안내문과 지자체 알림함",
            "nonprofit_admin_notice": "저장된 단체 담당자 번호와 기존 행사 공지",
            "pet_appointment_notice": "저장된 병원 번호와 예약 앱",
            "research_study_notice": "연구 포털 알림함과 등록된 연구실 번호",
            "library_account_notice": "도서관 앱과 저장된 지점 번호",
            "vehicle_service_notice": "공식 리콜 조회와 저장된 서비스센터 번호",
            "ticketing_refund_notice": "예매 계정 알림함과 주문 내역",
            "membership_access_notice": "멤버십 포털 알림함과 우편 안내문",
            "childcare_service_notice": "센터 사무실 callback과 공식 영아 대기자 안내문",
        }

        for scenario_type in config.scenario_types:
            self.assertIn(scenario_type, SCENARIO_CONTEXTS)
            self.assertIn(scenario_type, SCENARIO_RISK_LABELS)
            self.assertIn(scenario_type, ENVIRONMENTAL_CONTEXTS)
            self.assertEqual(
                set(environmental_context_for_scenario(scenario_type)),
                {
                    "channel",
                    "timing",
                    "trigger_event",
                    "claimed_authority",
                    "victim_constraint",
                    "verification_path",
                },
            )
            self.assertNotEqual(default_assets_for_scenario(scenario_type), ["identity_information", "voice_sample"])
            self.assertIn(scenario_type, expected_attacker_cues)
            self.assertIn(scenario_type, expected_victim_verifiers)
            attacker_utterance = client.chat(
                [
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "ROLE=attacker",
                                "phase=opening",
                                f"scenario_type={scenario_type}",
                                "attacker_goal=credential_capture",
                                "attacker_persona=mock_attacker: mock",
                            ]
                        ),
                    }
                ],
                model=config.attacker_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            victim_utterance = client.chat(
                [
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "ROLE=victim",
                                "phase=opening",
                                f"scenario_type={scenario_type}",
                                "victim_persona=mock_victim: mock",
                            ]
                        ),
                    }
                ],
                model=config.victim_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            self.assertIn(expected_attacker_cues[scenario_type], attacker_utterance)
            self.assertIn(expected_victim_verifiers[scenario_type], victim_utterance)

    def test_generation_masks_raw_sensitive_patterns_before_export_and_qa(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = RawSensitivePatternClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        record = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=1,
        )[0]
        summary = summarize_records([record])

        exported_text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("123456", exported_text)
        self.assertNotIn("https://example.test", exported_text)
        self.assertNotIn("010-1234-5678", exported_text)
        self.assertNotIn("900101-1234567", exported_text)
        self.assertNotIn("password:abcd1234", exported_text)
        self.assertNotIn("123-456-789012", exported_text)
        self.assertEqual(summary["corpus_residual_sensitive_patterns"], {})
        self.assertTrue(record["quality_checks"]["residual_pii_clean"])
        self.assertTrue(record["quality_checks"]["safety_passed"])
        self.assertEqual(
            {mask["label"] for mask in record["safety_masks"]},
            {"account", "credential", "otp", "phone", "rrn", "url"},
        )

    def test_generation_labels_credentials_detected_by_safety_masking(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = RawPinCredentialClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        record = orchestrator.generate(
            load_json_list(config.victims_path),
            load_json_list(config.attackers_path),
            limit=1,
        )[0]
        first_turn = record["dialogue"][0]

        self.assertEqual(first_turn["utterance"], "본인 확인을 위해 [MASKED_CREDENTIAL] 값을 알려주세요.")
        self.assertEqual(first_turn["risk_signal"], "credential_request")
        self.assertIn("credential_request", record["risk_labels"])
        self.assertIn({"label": "credential", "placeholder": "[MASKED_CREDENTIAL]"}, first_turn["masked_items"])

    def test_iter_generate_isolates_record_errors(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = FailingFirstOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        events = list(
            orchestrator.iter_generate(
                load_json_list(config.victims_path),
                load_json_list(config.attackers_path),
                limit=2,
            )
        )

        errors = [event.error for event in events if event.error is not None]
        records = [event.record for event in events if event.record is not None]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "ValueError")
        self.assertEqual(len(records), 2)

    def test_iter_generate_skips_existing_record_ids_for_resume(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)
        existing = orchestrator.generate(victims, attackers, limit=1)[0]["id"]

        events = list(orchestrator.iter_generate(victims, attackers, limit=1, skip_record_ids={existing}))
        records = [event.record for event in events if event.record is not None]

        self.assertEqual(len(records), 1)
        self.assertNotEqual(records[0]["id"], existing)

    def test_iter_generate_raises_when_scan_limit_is_exhausted_by_existing_ids(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)
        victim, attacker = persona_pair_for_index(victims, attackers, 0)
        scenario_type = choose_scenario_type(victim, attacker, config.scenario_types, 0)
        existing = make_record_id(config.dataset_name, victim["id"], attacker["id"], scenario_type, 0)

        with self.assertRaisesRegex(RuntimeError, "Unable to generate requested records within scan limit"):
            list(orchestrator.iter_generate(victims, attackers, limit=1, skip_record_ids={existing}, max_attempts=1))

    def test_iter_generate_rejects_invalid_personas_before_record_loop(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        with self.assertRaisesRegex(ValueError, "victim personas must contain at least one persona"):
            list(orchestrator.iter_generate([], load_json_list(config.attackers_path), limit=1))

    def test_iter_generate_rejects_empty_attackers_before_record_loop(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        with self.assertRaisesRegex(ValueError, "attacker personas must contain at least one persona"):
            list(orchestrator.iter_generate(load_json_list(config.victims_path), [], limit=1))

    def test_generate_rejects_empty_personas_through_public_path(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        with self.assertRaisesRegex(ValueError, "victim personas must contain at least one persona"):
            orchestrator.generate([], attackers, limit=1)
        with self.assertRaisesRegex(ValueError, "attacker personas must contain at least one persona"):
            orchestrator.generate(victims, [], limit=1)

    def test_empty_scenario_types_rejected_before_generation_loop(self):
        config = GenerationConfig.from_file("configs/default.json")

        with self.assertRaisesRegex(ValueError, "scenario_types must contain at least one entry"):
            replace(config, scenario_types=[])

    def test_invalid_config_values_rejected_before_generation_loop(self):
        config = GenerationConfig.from_file("configs/default.json")

        with self.assertRaisesRegex(ValueError, "max_turns must be at least 2"):
            replace(config, max_turns=1)
        with self.assertRaisesRegex(ValueError, "max_turns must be at most 10"):
            replace(config, max_turns=11)
        with self.assertRaisesRegex(ValueError, "default_limit must be non-negative"):
            replace(config, default_limit=-1)
        with self.assertRaisesRegex(ValueError, "persona_catalog_limit must be at least 1"):
            replace(config, persona_catalog_limit=0)
        with self.assertRaisesRegex(ValueError, "default_limit must be less than or equal to persona_catalog_limit"):
            replace(config, default_limit=config.persona_catalog_limit + 1)
        with self.assertRaisesRegex(ValueError, "scenario_types entries must be non-empty strings"):
            replace(config, scenario_types=["family_emergency", ""])

    def test_from_file_rejects_max_turns_above_supported_bound(self):
        config_path = Path("configs/default.json")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data["max_turns"] = 11

        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_config = Path(tmpdir) / "default.json"
            invalid_config.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "max_turns must be at most 10"):
                GenerationConfig.from_file(invalid_config)

    def test_generate_rejects_negative_limit(self):
        config = GenerationConfig.from_file("configs/default.json")
        client = MockLLMClient()
        orchestrator = DatasetOrchestrator(
            config=config,
            attacker_agent=AttackerAgent(client, config.attacker_model, config.temperature, config.max_tokens),
            victim_agent=VictimAgent(client, config.victim_model, config.temperature, config.max_tokens),
            safety=SafetyMasker(),
        )

        with self.assertRaisesRegex(ValueError, "limit must be non-negative"):
            orchestrator.generate(
                load_json_list(config.victims_path),
                load_json_list(config.attackers_path),
                limit=-1,
            )


if __name__ == "__main__":
    unittest.main()
