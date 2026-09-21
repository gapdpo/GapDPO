import json
import unittest

from madb.prompts import attacker_messages, victim_messages


class PromptTest(unittest.TestCase):
    def test_environmental_context_is_rendered_as_stable_json(self):
        environmental_context = {
            "timing": "late_evening",
            "channel": "phone_call",
            "trigger_event": "urgent_family_distress_claim",
            "claimed_authority": "family_member_or_known_contact",
            "victim_constraint": "emotionally_overloaded_and_cannot_verify_in_person",
            "verification_path": "family_callback",
        }
        victim = {"id": "elderly_parent", "description": "victim", "resistance_style": "calls back"}
        attacker = {"id": "family_impersonator", "description": "attacker"}

        attacker_prompt = attacker_messages(
            victim,
            attacker,
            "family_emergency",
            "money_transfer",
            ["otp_or_security_card"],
            [],
            "opening",
            "open",
            environmental_context,
        )[1]["content"]
        victim_prompt = victim_messages(
            victim,
            "family_emergency",
            ["otp_or_security_card"],
            [],
            "opening",
            "verify",
            "defensive",
            environmental_context,
        )[1]["content"]

        rendered = json.dumps(environmental_context, ensure_ascii=False, sort_keys=True)
        self.assertIn(f"environmental_context={rendered}", attacker_prompt)
        self.assertIn(f"environmental_context={rendered}", victim_prompt)
        self.assertNotIn("environmental_context={'", attacker_prompt)
        self.assertNotIn("environmental_context={'", victim_prompt)

    def test_inconclusive_and_non_urgency_prompt_contracts_are_explicit(self):
        victim = {"id": "public_benefit_recipient", "description": "victim", "resistance_style": "hesitates"}
        attacker = {"id": "benefits_caseworker_impersonator", "description": "attacker"}

        attacker_prompt = attacker_messages(
            victim,
            attacker,
            "public_benefit_notice",
            "identity_information",
            ["case_reference_number"],
            [],
            "make_request",
            "request context",
            {},
            "inconclusive",
            "authority",
        )[1]["content"]
        victim_prompt = victim_messages(
            victim,
            "public_benefit_notice",
            ["case_reference_number"],
            [],
            "make_request",
            "stay uncertain",
            "uncertain",
            {},
            "inconclusive",
        )[1]["content"]

        self.assertIn("avoid forcing a binary comply/refuse decision", attacker_prompt)
        self.assertIn("Avoid urgent words", attacker_prompt)
        self.assertIn("do not use official/trusted verification", victim_prompt)
        self.assertIn("uncertainty, delay, confusion, or difficulty deciding", victim_prompt)


if __name__ == "__main__":
    unittest.main()
