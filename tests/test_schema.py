import unittest

from madb.schema import (
    MASKED_PLACEHOLDER_PATTERN,
    MASKED_SECRET_PATTERN,
    STANDARD_MASKED_PLACEHOLDERS,
    SchemaError,
    validate_record,
)


def valid_record():
    return {
        "id": "sample",
        "dataset_name": "test",
        "label": "attack",
        "scenario_type": "family_emergency",
        "victim_persona": {"id": "v", "name": "v", "description": ""},
        "attacker_persona": {"id": "a", "name": "a", "description": ""},
        "protected_assets": ["otp_or_security_card"],
        "attacker_goal": "credential_capture",
        "outcome": "defended_success",
        "compromised_assets": [],
        "difficulty": {"attacker_sophistication": "low", "victim_susceptibility": "med"},
        "channel": "phone",
        "locale": "ko-KR",
        "split": "train",
        "context": "context",
        "dialogue": [
            {
                "turn_id": 1,
                "speaker": "attacker",
                "phase": "opening",
                "utterance": "masked request",
                "intent_label": "request_masked_credential_or_otp",
                "risk_signal": "credential_request",
                "masked_items": [],
            },
            {
                "turn_id": 2,
                "speaker": "victim",
                "phase": "opening",
                "utterance": "verify first",
                "intent_label": "resist_or_verify_request",
                "risk_signal": "victim_resistance",
                "masked_items": [],
            },
        ],
        "risk_labels": ["voice_impersonation"],
        "safety_masks": [],
        "gen_metadata": {
            "attacker_model": "a",
            "victim_model": "v",
            "prompt_version": "v2",
            "sampling": {"temperature": 0.7, "max_tokens": 512},
            "seed": 0,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        "quality_checks": {
            "turn_count": 2,
            "role_order_valid": True,
            "safety_passed": True,
            "no_duplicate_turns": True,
            "label_consistent": True,
            "language_consistent": True,
            "no_refusal_leak": True,
            "residual_pii_clean": True,
            "schema_valid": True,
        },
    }


class SchemaTest(unittest.TestCase):
    def test_valid_record_passes(self):
        validate_record(valid_record(), max_turns=10)

    def test_rejects_wrong_speaker_order(self):
        record = valid_record()
        record["dialogue"][0]["speaker"] = "victim"

        with self.assertRaises(SchemaError):
            validate_record(record, max_turns=10)

    def test_rejects_too_many_turns(self):
        record = valid_record()
        record["dialogue"] = record["dialogue"] * 6
        for index, turn in enumerate(record["dialogue"], start=1):
            turn["turn_id"] = index
            turn["speaker"] = "attacker" if index % 2 == 1 else "victim"

        with self.assertRaises(SchemaError):
            validate_record(record, max_turns=10)

    def test_rejects_original_in_safety_masks(self):
        record = valid_record()
        record["safety_masks"] = [{"label": "otp", "placeholder": "[MASKED_OTP]", "original": "123456"}]

        with self.assertRaises(SchemaError):
            validate_record(record, max_turns=10)

    def test_rejects_original_in_turn_masked_items(self):
        record = valid_record()
        record["dialogue"][0]["masked_items"] = [
            {"label": "otp", "placeholder": "[MASKED_OTP]", "original": "123456"}
        ]

        with self.assertRaisesRegex(SchemaError, "masked_items must not expose original"):
            validate_record(record, max_turns=10)

    def test_rejects_malformed_mask_entries(self):
        record = valid_record()
        record["safety_masks"] = "not-a-list"

        with self.assertRaisesRegex(SchemaError, "safety_masks must be a list"):
            validate_record(record, max_turns=10)

        record = valid_record()
        record["dialogue"][0]["masked_items"] = [{"label": "otp"}]

        with self.assertRaisesRegex(SchemaError, "masked_items\\[0\\] is missing placeholder"):
            validate_record(record, max_turns=10)

    def test_rejects_quality_check_type_mismatch(self):
        record = valid_record()
        record["quality_checks"]["schema_valid"] = "true"

        with self.assertRaises(SchemaError):
            validate_record(record, max_turns=10)

    def test_rejects_non_object_gen_metadata(self):
        record = valid_record()
        record["gen_metadata"] = "not-an-object"

        with self.assertRaisesRegex(SchemaError, "gen_metadata must be an object"):
            validate_record(record, max_turns=10)

    def test_rejects_malformed_environmental_context_when_present(self):
        record = valid_record()
        record["gen_metadata"]["environmental_context"] = {
            "channel": "phone_call",
            "timing": "late_evening",
            "trigger_event": "urgent_family_distress_claim",
            "claimed_authority": "family_member_or_known_contact",
            "victim_constraint": "",
        }

        with self.assertRaisesRegex(SchemaError, "environmental_context is missing fields"):
            validate_record(record, max_turns=10)

        record = valid_record()
        record["gen_metadata"]["environmental_context"] = {
            "channel": "phone_call",
            "timing": "late_evening",
            "trigger_event": "urgent_family_distress_claim",
            "claimed_authority": "family_member_or_known_contact",
            "victim_constraint": "",
            "verification_path": "family_callback",
        }

        with self.assertRaisesRegex(SchemaError, "victim_constraint must be a non-empty string"):
            validate_record(record, max_turns=10)

    def test_rejects_turn_count_mismatch(self):
        record = valid_record()
        record["quality_checks"]["turn_count"] = 1

        with self.assertRaises(SchemaError):
            validate_record(record, max_turns=10)

    def test_rejects_non_string_risk_labels(self):
        record = valid_record()
        record["risk_labels"] = ["voice_impersonation", ""]

        with self.assertRaisesRegex(SchemaError, "risk_labels entries"):
            validate_record(record, max_turns=10)

    def test_rejects_risk_labels_outside_allowed_set(self):
        record = valid_record()
        record["risk_labels"] = ["voice_impersonation", "not_configured"]

        with self.assertRaisesRegex(SchemaError, "Unknown risk_labels"):
            validate_record(
                record,
                max_turns=10,
                allowed_risk_labels={"voice_impersonation", "urgency"},
            )

    def test_masked_secret_pattern_does_not_overlap_unknown_placeholders(self):
        unknown_placeholders = ("[MASKED_2FA_CODE]", "[MASKED_COMPANY_EMAIL]", "[MASKED_ID_IMAGE]")

        for placeholder in STANDARD_MASKED_PLACEHOLDERS:
            with self.subTest(standard=placeholder):
                self.assertRegex(placeholder, MASKED_PLACEHOLDER_PATTERN)
                self.assertRegex(placeholder, MASKED_SECRET_PATTERN)

        for placeholder in unknown_placeholders:
            with self.subTest(unknown=placeholder):
                self.assertRegex(placeholder, MASKED_PLACEHOLDER_PATTERN)
                self.assertNotRegex(placeholder, MASKED_SECRET_PATTERN)


if __name__ == "__main__":
    unittest.main()
