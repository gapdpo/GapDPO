import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from madb.config import (
    GenerationConfig,
    load_json_list,
    validate_persona_catalog_capacity,
    validate_personas,
)

EXPECTED_BATCH1_VICTIMS = [
    "elderly_parent",
    "office_worker",
    "student_or_young_adult",
    "small_business_owner",
    "remote_job_seeker",
    "telehealth_patient",
    "renter_or_tenant",
    "utility_account_holder",
    "public_benefit_recipient",
    "freelance_creator",
    "caregiver_or_guardian",
    "language_access_service_user",
]

EXPECTED_BATCH1_ATTACKERS = [
    "family_impersonator",
    "institution_impersonator",
    "corporate_impersonator",
    "service_provider_impersonator",
    "recruiter_impersonator",
    "healthcare_support_impersonator",
    "housing_admin_impersonator",
    "utility_support_impersonator",
    "benefits_caseworker_impersonator",
    "platform_policy_impersonator",
    "education_admin_impersonator",
    "document_support_impersonator",
]

EXPECTED_BATCH2_VICTIMS = [
    "seasonal_tax_filer",
    "travel_booking_customer",
    "property_claim_policyholder",
    "disaster_preparedness_resident",
    "community_volunteer_coordinator",
    "pet_care_client",
    "research_participant",
    "library_account_holder",
    "vehicle_owner",
    "event_ticket_buyer",
    "museum_member",
    "infant_childcare_applicant",
]

EXPECTED_BATCH2_ATTACKERS = [
    "tax_portal_impersonator",
    "travel_rebooking_impersonator",
    "property_claim_adjuster_impersonator",
    "emergency_drill_coordinator_impersonator",
    "nonprofit_admin_impersonator",
    "veterinary_scheduler_impersonator",
    "university_research_impersonator",
    "library_services_impersonator",
    "vehicle_recall_impersonator",
    "ticketing_refund_impersonator",
    "membership_services_impersonator",
    "childcare_waitlist_impersonator",
]


class PersonaValidationTest(unittest.TestCase):
    def test_default_persona_files_have_required_fields_and_unique_ids(self):
        config = GenerationConfig.from_file("configs/default.json")

        validate_personas(load_json_list(config.victims_path), kind="victim")
        validate_personas(load_json_list(config.attackers_path), kind="attacker")

    def test_persona_catalog_limit_covers_current_persona_catalogs(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        self.assertEqual(len(victims), 24)
        self.assertEqual(len(attackers), 24)
        self.assertEqual(config.default_limit, 12)
        self.assertLessEqual(config.default_limit, config.persona_catalog_limit)
        self.assertEqual(config.persona_catalog_limit, 24)
        validate_persona_catalog_capacity(victims, attackers, limit=config.persona_catalog_limit)

    def test_default_and_local_configs_share_persona_capacity_contract(self):
        for config_path in ("configs/default.json", "configs/local-qwen3-30b-a3b.json"):
            with self.subTest(config_path=config_path):
                config = GenerationConfig.from_file(config_path)
                victims = load_json_list(config.victims_path)
                attackers = load_json_list(config.attackers_path)

                self.assertEqual(config.default_limit, 12)
                self.assertEqual(config.persona_catalog_limit, 24)
                self.assertEqual(len(victims), 24)
                self.assertEqual(len(attackers), 24)
                validate_persona_catalog_capacity(victims, attackers, limit=config.persona_catalog_limit)

    def test_persona_catalog_order_preserves_batch1_and_appends_batch2(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        self.assertEqual([victim["id"] for victim in victims[:12]], EXPECTED_BATCH1_VICTIMS)
        self.assertEqual([attacker["id"] for attacker in attackers[:12]], EXPECTED_BATCH1_ATTACKERS)
        self.assertEqual([victim["id"] for victim in victims[12:24]], EXPECTED_BATCH2_VICTIMS)
        self.assertEqual([attacker["id"] for attacker in attackers[12:24]], EXPECTED_BATCH2_ATTACKERS)

    def test_config_without_persona_catalog_limit_falls_back_to_default_limit(self):
        config_path = Path("configs/default.json")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        del data["persona_catalog_limit"]

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_config = Path(tmpdir) / "default.json"
            legacy_config.write_text(json.dumps(data), encoding="utf-8")

            config = GenerationConfig.from_file(legacy_config)

        self.assertEqual(config.persona_catalog_limit, config.default_limit)
        self.assertEqual(config.schedule_mode, "rotating")

    def test_rejects_unknown_schedule_mode(self):
        config = GenerationConfig.from_file("configs/default.json")

        with self.assertRaisesRegex(ValueError, "schedule_mode must be one of"):
            replace(config, schedule_mode="unknown")

    def test_rejects_default_limit_above_persona_catalog_limit(self):
        config = GenerationConfig.from_file("configs/default.json")

        with self.assertRaisesRegex(ValueError, "default_limit must be less than or equal to persona_catalog_limit"):
            GenerationConfig(
                dataset_name=config.dataset_name,
                max_turns=config.max_turns,
                default_limit=config.persona_catalog_limit + 1,
                persona_catalog_limit=config.persona_catalog_limit,
                schedule_mode=config.schedule_mode,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                attacker_model=config.attacker_model,
                victim_model=config.victim_model,
                victims_path=config.victims_path,
                attackers_path=config.attackers_path,
                scenario_types=config.scenario_types,
                risk_label_set=config.risk_label_set,
            )

    def test_rejects_persona_catalogs_above_configured_capacity(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = load_json_list(config.victims_path)
        attackers = load_json_list(config.attackers_path)

        with self.assertRaisesRegex(ValueError, "victim persona catalog size"):
            validate_persona_catalog_capacity(victims + [copy.deepcopy(victims[0])], attackers, limit=len(victims))

        with self.assertRaisesRegex(ValueError, "attacker persona catalog size"):
            validate_persona_catalog_capacity(victims, attackers + [copy.deepcopy(attackers[0])], limit=len(attackers))

    def test_rejects_duplicate_persona_ids(self):
        personas = [
            {
                "id": "duplicate",
                "name": "First",
                "description": "A valid victim persona.",
                "vulnerabilities": ["urgency"],
                "resistance_style": "Verify through a trusted channel.",
                "protected_assets": ["account_password"],
            },
            {
                "id": "duplicate",
                "name": "Second",
                "description": "Another valid victim persona.",
                "vulnerabilities": ["authority"],
                "resistance_style": "Ask for official confirmation.",
                "protected_assets": ["otp_or_security_card"],
            },
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate victim persona id"):
            validate_personas(personas, kind="victim")

    def test_rejects_missing_required_persona_field(self):
        config = GenerationConfig.from_file("configs/default.json")
        attackers = copy.deepcopy(load_json_list(config.attackers_path))
        del attackers[0]["default_goals"]

        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_personas(attackers, kind="attacker")

    def test_rejects_empty_required_persona_list(self):
        config = GenerationConfig.from_file("configs/default.json")
        victims = copy.deepcopy(load_json_list(config.victims_path))
        victims[0]["protected_assets"] = []

        with self.assertRaisesRegex(ValueError, "non-empty list protected_assets"):
            validate_personas(victims, kind="victim")

    def test_rejects_empty_persona_collection(self):
        with self.assertRaisesRegex(ValueError, "at least one persona"):
            validate_personas([], kind="victim")


if __name__ == "__main__":
    unittest.main()
