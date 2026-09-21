import unittest

from madb.labeling import (
    ATTACK_KEYWORDS,
    BASE_ATTACK_KEYWORDS,
    MASKED_PLACEHOLDER_RISK_POLICY,
    annotate_turn,
    collect_risk_labels,
    infer_record_outcome,
)
from madb.safety import SafetyMasker


class LabelingTest(unittest.TestCase):
    def test_all_safety_placeholders_have_explicit_labeling_policy(self):
        placeholders = {placeholder for _label, placeholder, _pattern in SafetyMasker.PATTERNS}

        self.assertEqual(set(MASKED_PLACEHOLDER_RISK_POLICY), placeholders)
        for placeholder, expected_risk in MASKED_PLACEHOLDER_RISK_POLICY.items():
            normalized_placeholder = placeholder.lower()
            mapped_risks = {
                risk_signal
                for risk_signal, keywords in ATTACK_KEYWORDS.items()
                if normalized_placeholder in keywords
            }
            if expected_risk is None:
                self.assertEqual(mapped_risks, set(), placeholder)
            else:
                self.assertEqual(mapped_risks, {expected_risk}, placeholder)

    def test_masked_placeholder_attack_keywords_are_derived_from_policy(self):
        for placeholder, risk_signal in MASKED_PLACEHOLDER_RISK_POLICY.items():
            normalized_placeholder = placeholder.lower()
            for candidate_risk, base_keywords in BASE_ATTACK_KEYWORDS.items():
                self.assertNotIn(normalized_placeholder, base_keywords, candidate_risk)
            mapped_risks = {
                candidate_risk
                for candidate_risk, keywords in ATTACK_KEYWORDS.items()
                if normalized_placeholder in keywords
            }
            expected = set() if risk_signal is None else {risk_signal}
            self.assertEqual(mapped_risks, expected, placeholder)

    def test_money_transfer_requires_transfer_action_not_account_placeholder_only(self):
        _intent, risk = annotate_turn(
            "attacker",
            "[MASKED_ACCOUNT] 상태 확인 뒤 [MASKED_CREDENTIAL]와 [MASKED_OTP] 확인이 필요합니다.",
            "credential_request",
        )

        self.assertEqual(risk, "credential_request")

    def test_credential_request_takes_precedence_over_identity_context_when_otp_is_requested(self):
        _intent, risk = annotate_turn(
            "attacker",
            "신원 확인 자료와 [MASKED_ACCOUNT], [MASKED_OTP] 검증이 필요합니다.",
            "identity_information_request",
        )

        self.assertEqual(risk, "credential_request")

    def test_identity_request_applies_when_no_credential_material_is_requested(self):
        _intent, risk = annotate_turn(
            "attacker",
            "신원 확인 자료와 생년월일 확인이 필요합니다.",
            "identity_information_request",
        )

        self.assertEqual(risk, "identity_information_request")

    def test_masked_rrn_is_identity_information_request(self):
        _intent, risk = annotate_turn(
            "attacker",
            "본인 확인을 위해 [MASKED_RRN] 값을 확인해야 합니다.",
            "credential_request",
        )

        self.assertEqual(risk, "identity_information_request")

    def test_masked_phone_is_identity_information_request(self):
        _intent, risk = annotate_turn(
            "attacker",
            "신원 확인을 위해 [MASKED_PHONE] 연락처가 필요합니다.",
            "credential_request",
        )

        self.assertEqual(risk, "identity_information_request")

    def test_identity_phrase_does_not_override_credential_request(self):
        _intent, risk = annotate_turn(
            "attacker",
            "본인 확인을 위해 [MASKED_OTP] 인증이 필요합니다.",
            "identity_information_request",
        )

        self.assertEqual(risk, "credential_request")

    def test_english_password_marker_is_credential_request(self):
        _intent, risk = annotate_turn(
            "attacker",
            "본인 확인을 위해 password:abcd1234 값을 알려주세요.",
            "identity_information_request",
        )

        self.assertEqual(risk, "credential_request")

    def test_money_transfer_action_still_takes_precedence(self):
        _intent, risk = annotate_turn(
            "attacker",
            "[MASKED_ACCOUNT]로 결제 확인과 [MASKED_OTP] 검증이 필요합니다.",
            "credential_request",
        )

        self.assertEqual(risk, "money_transfer_request")

    def test_account_placeholder_with_transfer_action_is_money_transfer_request(self):
        _intent, risk = annotate_turn(
            "attacker",
            "[MASKED_ACCOUNT]로 지금 이체 확인이 필요합니다.",
            "credential_request",
        )

        self.assertEqual(risk, "money_transfer_request")

    def test_victim_password_reference_can_remain_benign_defense(self):
        intent, risk = annotate_turn(
            "victim",
            "password는 통화 중 제공하지 않고 공식 앱에서 직접 확인하겠습니다.",
            "credential_request",
        )

        self.assertEqual(intent, "resist_or_verify_request")
        self.assertEqual(risk, "victim_resistance")

    def test_victim_passwd_reference_can_remain_benign_defense(self):
        intent, risk = annotate_turn(
            "victim",
            "passwd 값은 알려주지 않고 은행에 직접 확인하겠습니다.",
            "credential_request",
        )

        self.assertEqual(intent, "resist_or_verify_request")
        self.assertEqual(risk, "victim_resistance")

    def test_collect_risk_labels_ignores_victim_state_signals(self):
        labels = collect_risk_labels(
            [
                {"speaker": "attacker", "risk_signal": "credential_request"},
                {"speaker": "victim", "risk_signal": "asset_disclosure"},
                {"speaker": "victim", "risk_signal": "victim_resistance"},
            ],
            ["voice_impersonation"],
        )

        self.assertEqual(labels, ["credential_request", "voice_impersonation"])

    def test_partial_victim_compliance_can_infer_compromised_masked_asset(self):
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_OTP] 인증이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_OTP]를 입력해볼게요.",
                    "masked_items": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
                },
            ]
        )

        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_otp"])

    def test_conditional_partial_credential_entry_is_uncertain_not_disclosure(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_OTP]를 입력해볼게요, 다만 실제 값은 공식 경로에서만 처리하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_OTP] 인증이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_OTP]를 입력해볼게요, 다만 실제 값은 공식 경로에서만 처리하겠습니다.",
                    "masked_items": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
                },
            ]
        )

        self.assertEqual(intent, "uncertain_response")
        self.assertEqual(risk, "victim_uncertainty")
        self.assertEqual(outcome, "inconclusive")
        self.assertEqual(assets, [])

    def test_v1_target_outcome_mode_fixture_patterns_are_labelable(self):
        fixtures = [
            (
                "defended_success",
                "공식 연락처로 확인하기 전에는 [MASKED_OTP]를 알려주지 않겠습니다.",
                "defended_success",
                [],
            ),
            (
                "inconclusive",
                "[MASKED_OTP]는 공식 확인 절차를 거친 후에만 입력하겠습니다.",
                "inconclusive",
                [],
            ),
            (
                "masked_compromise",
                "[MASKED_OTP]를 입력하겠습니다.",
                "compromised",
                ["masked_otp"],
            ),
        ]
        masker = SafetyMasker()
        for target_mode, utterance, expected_outcome, expected_assets in fixtures:
            with self.subTest(target_mode=target_mode):
                safe = masker.mask_text(utterance)
                _intent, risk = annotate_turn("victim", safe.text, "victim_resistance")
                outcome, assets = infer_record_outcome(
                    [
                        {
                            "speaker": "victim",
                            "utterance": safe.text,
                            "risk_signal": risk,
                            "masked_items": masker.masks_to_dicts(safe.masked_items),
                        }
                    ]
                )

                self.assertEqual(outcome, expected_outcome)
                self.assertEqual(assets, expected_assets)

    def test_formal_masked_credential_entry_is_disclosure(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_OTP]를 입력하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_OTP] 인증이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_OTP]를 입력하겠습니다.",
                    "masked_items": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_otp"])

    def test_spaced_masked_credential_entry_is_disclosure(self):
        intent, risk = annotate_turn(
            "victim",
            "우선 [MASKED_OTP]를 입력해 보겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_OTP] 인증이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "우선 [MASKED_OTP]를 입력해 보겠습니다.",
                    "masked_items": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_otp"])

    def test_generic_processing_and_progression_do_not_imply_compromise(self):
        cases = [
            (
                "[MASKED_LINK] 안 누르고 그냥 진행하겠습니다.",
                [{"label": "link", "placeholder": "[MASKED_LINK]"}],
            ),
            (
                "[MASKED_BANK] 앱에서 제가 직접 처리하겠습니다.",
                [{"label": "bank", "placeholder": "[MASKED_BANK]"}],
            ),
        ]
        for utterance, masked_items in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_resistance")
                outcome, assets = infer_record_outcome(
                    [
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": masked_items,
                        },
                    ]
                )

                self.assertNotEqual(intent, "disclose_or_comply")
                self.assertNotEqual(risk, "asset_disclosure")
                self.assertNotEqual(outcome, "compromised")
                self.assertEqual(assets, [])

    def test_informal_generic_processing_and_progression_do_not_imply_compromise(self):
        cases = [
            (
                "[MASKED_LINK] 안 누르고 그냥 진행할게.",
                [{"label": "link", "placeholder": "[MASKED_LINK]"}],
            ),
            (
                "[MASKED_BANK] 앱에서 제가 직접 처리할게.",
                [{"label": "bank", "placeholder": "[MASKED_BANK]"}],
            ),
            (
                "[MASKED_ACCOUNT]로 안 이체할게.",
                [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
            ),
        ]
        for utterance, masked_items in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_resistance")
                outcome, assets = infer_record_outcome(
                    [
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": masked_items,
                        },
                    ]
                )

                self.assertNotEqual(intent, "disclose_or_comply")
                self.assertNotEqual(risk, "asset_disclosure")
                self.assertNotEqual(outcome, "compromised")
                self.assertEqual(assets, [])

    def test_explicit_negated_asset_actions_are_victim_resistance(self):
        cases = [
            (
                "[MASKED_ACCOUNT]로 안 이체할게.",
                [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
            ),
            (
                "[MASKED_URL]은 안 누를게.",
                [{"label": "url", "placeholder": "[MASKED_URL]"}],
            ),
            (
                "[MASKED_OTP]는 안 알려줄게.",
                [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
            ),
        ]
        for utterance, masked_items in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_uncertainty")
                outcome, assets = infer_record_outcome(
                    [
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": masked_items,
                        },
                    ]
                )

                self.assertEqual(intent, "resist_or_verify_request")
                self.assertEqual(risk, "victim_resistance")
                self.assertEqual(outcome, "defended_success")
                self.assertEqual(assets, [])

    def test_polite_negated_asset_actions_are_victim_resistance(self):
        cases = [
            (
                "[MASKED_OTP]는 알려드리지 않겠습니다.",
                [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
            ),
            (
                "[MASKED_OTP]는 보내지 않겠습니다.",
                [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
            ),
            (
                "[MASKED_ID_IMAGE]는 보내드리지 않겠습니다.",
                [{"label": "id_image", "placeholder": "[MASKED_ID_IMAGE]"}],
            ),
        ]
        for utterance, masked_items in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_uncertainty")
                outcome, assets = infer_record_outcome(
                    [
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": masked_items,
                        },
                    ]
                )

                self.assertEqual(intent, "resist_or_verify_request")
                self.assertEqual(risk, "victim_resistance")
                self.assertEqual(outcome, "defended_success")
                self.assertEqual(assets, [])

    def test_formal_asset_action_keywords_infer_compromised_assets(self):
        cases = [
            ("[MASKED_ACCOUNT] 정보를 제출하겠습니다.", "account", "account"),
            ("[MASKED_ACCOUNT]로 이체하겠습니다.", "account", "account"),
            ("[MASKED_ACCOUNT]로 송금하겠습니다.", "account", "account"),
            ("[MASKED_ACCOUNT]로 결제하겠습니다.", "account", "account"),
        ]
        for utterance, label, expected_asset in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_resistance")
                outcome, assets = infer_record_outcome(
                    [
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": [{"label": label, "placeholder": "[MASKED_ACCOUNT]"}],
                        },
                    ]
                )

                self.assertEqual(intent, "disclose_or_comply")
                self.assertEqual(risk, "asset_disclosure")
                self.assertEqual(outcome, "compromised")
                self.assertEqual(assets, [f"masked_{expected_asset}"])

    def test_mixed_placeholder_turn_only_attributes_complied_asset(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_OTP]는 알려주지 않고 [MASKED_ACCOUNT]로 이체하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_OTP]는 알려주지 않고 [MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [
                        {"label": "otp", "placeholder": "[MASKED_OTP]"},
                        {"label": "account", "placeholder": "[MASKED_ACCOUNT]"},
                    ],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_mixed_refusal_and_partial_submission_keeps_only_complied_asset(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_ID_IMAGE]는 보내드리지 않겠습니다. 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_ID_IMAGE]는 보내드리지 않겠습니다. 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
                    "masked_items": [
                        {"label": "id_image", "placeholder": "[MASKED_ID_IMAGE]"},
                        {"label": "account", "placeholder": "[MASKED_ACCOUNT]"},
                    ],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_separate_verification_sentence_does_not_cancel_actual_compliance(self):
        intent, risk = annotate_turn(
            "victim",
            "공식 연락처는 별도로 확인하겠습니다. 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "공식 연락처는 별도로 확인하겠습니다. 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_separate_weak_condition_sentence_does_not_cancel_actual_compliance(self):
        intent, risk = annotate_turn(
            "victim",
            "담당자는 공식 포털에서 확인되면 다시 연락하겠습니다. 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "담당자는 공식 포털에서 확인되면 다시 연락하겠습니다. 계좌 정보는 [MASKED_ACCOUNT]로 제출하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_separate_verification_and_refusal_only_attributes_complied_asset(self):
        intent, risk = annotate_turn(
            "victim",
            "공식 연락처는 확인하겠습니다. [MASKED_OTP]는 알려주지 않고 [MASKED_ACCOUNT]로 이체하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "공식 연락처는 확인하겠습니다. [MASKED_OTP]는 알려주지 않고 [MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [
                        {"label": "otp", "placeholder": "[MASKED_OTP]"},
                        {"label": "account", "placeholder": "[MASKED_ACCOUNT]"},
                    ],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_attached_contrastive_refusal_does_not_cancel_later_compliance(self):
        intent, risk = annotate_turn(
            "victim",
            "인증 요청은 거부하지만 [MASKED_ACCOUNT]로 이체하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "인증 요청은 거부하지만 [MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_attached_contrastive_refusal_without_space_still_finds_later_compliance(self):
        intent, risk = annotate_turn(
            "victim",
            "인증 요청은 거부하지만[MASKED_ACCOUNT]로 이체하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "인증 요청은 거부하지만[MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_attached_contrastive_refusal_and_digit_placeholder_compliance_cross_case(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_2FA_CODE] 입력 요청은 거부하지만[MASKED_ACCOUNT]로 이체하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_2FA_CODE] 입력 요청은 거부하지만[MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [
                        {"label": "2fa_code", "placeholder": "[MASKED_2FA_CODE]"},
                        {"label": "account", "placeholder": "[MASKED_ACCOUNT]"},
                    ],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_attached_contrastive_refusal_and_quoted_digit_placeholder_stays_defended(self):
        intent, risk = annotate_turn(
            "victim",
            "이체는 거부하지만[MASKED_2FA_CODE]를 입력하겠습니다라고 말하지 않겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "이체는 거부하지만[MASKED_2FA_CODE]를 입력하겠습니다라고 말하지 않겠습니다.",
                    "masked_items": [{"label": "2fa_code", "placeholder": "[MASKED_2FA_CODE]"}],
                },
            ]
        )

        self.assertEqual(intent, "resist_or_verify_request")
        self.assertEqual(risk, "victim_resistance")
        self.assertEqual(outcome, "defended_success")
        self.assertEqual(assets, [])

    def test_attached_past_contrastive_refusal_without_space_finds_later_compliance(self):
        intent, risk = annotate_turn(
            "victim",
            "인증 요청은 거부했지만[MASKED_ACCOUNT]로 이체하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "인증 요청은 거부했지만[MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_attached_past_contrastive_quoted_digit_placeholder_stays_defended(self):
        intent, risk = annotate_turn(
            "victim",
            "이체는 거부했지만[MASKED_2FA_CODE]를 입력하겠습니다라고 말하지 않겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {
                    "speaker": "victim",
                    "utterance": "이체는 거부했지만[MASKED_2FA_CODE]를 입력하겠습니다라고 말하지 않겠습니다.",
                    "masked_items": [{"label": "2fa_code", "placeholder": "[MASKED_2FA_CODE]"}],
                },
            ]
        )

        self.assertEqual(intent, "resist_or_verify_request")
        self.assertEqual(risk, "victim_resistance")
        self.assertEqual(outcome, "defended_success")
        self.assertEqual(assets, [])

    def test_multi_turn_unconditional_masked_transfer_still_compromises_account(self):
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_ACCOUNT]로 지금 이체 확인이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "먼저 내용을 확인했습니다.",
                    "masked_items": [],
                },
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_ACCOUNT]로 이체하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_conditional_formal_masked_transfer_is_uncertain_not_disclosure(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_ACCOUNT]로 이체하겠습니다. 다만 공식 확인 절차를 거친 후에만 처리하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_ACCOUNT]로 지금 이체 확인이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_ACCOUNT]로 이체하겠습니다. 다만 공식 확인 절차를 거친 후에만 처리하겠습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(intent, "uncertain_response")
        self.assertEqual(risk, "victim_uncertainty")
        self.assertEqual(outcome, "inconclusive")
        self.assertEqual(assets, [])

    def test_strong_conditional_masked_submission_remains_inconclusive(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_ID_IMAGE]는 공식 확인 절차를 거친 후에만 제출하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_ID_IMAGE] 제출이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_ID_IMAGE]는 공식 확인 절차를 거친 후에만 제출하겠습니다.",
                    "masked_items": [{"label": "id_image", "placeholder": "[MASKED_ID_IMAGE]"}],
                },
            ]
        )

        self.assertEqual(intent, "uncertain_response")
        self.assertEqual(risk, "victim_uncertainty")
        self.assertEqual(outcome, "inconclusive")
        self.assertEqual(assets, [])

    def test_weak_conditional_masked_submission_remains_inconclusive(self):
        cases = [
            "[MASKED_ACCOUNT] 정보는 공식 포털에서 확인되면 제출하겠습니다.",
            "공식 포털에서 확인되는 대로 [MASKED_ACCOUNT] 정보를 제출하겠습니다.",
            "담당자 검증되면 [MASKED_ACCOUNT]로 이체하겠습니다.",
            "관리자 승인되면 [MASKED_ACCOUNT] 정보를 제출하겠습니다.",
            "정식 절차가 끝나면 [MASKED_ACCOUNT]로 이체하겠습니다.",
        ]
        for utterance in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_resistance")
                outcome, assets = infer_record_outcome(
                    [
                        {"speaker": "attacker", "utterance": "[MASKED_ACCOUNT] 계좌 정보를 제출해 주세요."},
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                        },
                    ]
                )

                self.assertEqual(intent, "uncertain_response")
                self.assertEqual(risk, "victim_uncertainty")
                self.assertEqual(outcome, "inconclusive")
                self.assertEqual(assets, [])

    def test_placeholder_based_partial_submission_can_infer_compromised_asset(self):
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_ACCOUNT] 계좌 정보를 제출해 주세요."},
                {
                    "speaker": "victim",
                    "utterance": "계좌 정보는 [MASKED_ACCOUNT]로 제출할 수 있습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_account"])

    def test_digit_masked_placeholder_partial_submission_can_infer_compromised_asset(self):
        intent, risk = annotate_turn(
            "victim",
            "[MASKED_2FA_CODE]를 입력하겠습니다.",
            "victim_resistance",
        )
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_2FA_CODE] 확인값을 요청합니다."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_2FA_CODE]를 입력하겠습니다.",
                    "masked_items": [{"label": "2fa_code", "placeholder": "[MASKED_2FA_CODE]"}],
                },
            ]
        )

        self.assertEqual(intent, "disclose_or_comply")
        self.assertEqual(risk, "asset_disclosure")
        self.assertEqual(outcome, "compromised")
        self.assertEqual(assets, ["masked_2fa_code"])

    def test_digit_masked_placeholder_refusal_references_do_not_infer_compromise(self):
        cases = [
            "[MASKED_2FA_CODE]를 입력하겠습니다라고 말하지 않겠습니다.",
            "[MASKED_2FA_CODE]를 제출하겠습니다라는 요구는 거절합니다.",
            "[MASKED_2FA_CODE] 입력 요청은 거부하겠습니다.",
            "[MASKED_2FA_CODE]를 보내라는 요청은 거부합니다.",
        ]
        for utterance in cases:
            with self.subTest(utterance=utterance):
                intent, risk = annotate_turn("victim", utterance, "victim_uncertainty")
                outcome, assets = infer_record_outcome(
                    [
                        {"speaker": "attacker", "utterance": "[MASKED_2FA_CODE] 확인값을 요청합니다."},
                        {
                            "speaker": "victim",
                            "utterance": utterance,
                            "masked_items": [{"label": "2fa_code", "placeholder": "[MASKED_2FA_CODE]"}],
                        },
                    ]
                )

                self.assertEqual(intent, "resist_or_verify_request")
                self.assertEqual(risk, "victim_resistance")
                self.assertEqual(outcome, "defended_success")
                self.assertEqual(assets, [])

    def test_conditional_partial_submission_inferrs_inconclusive(self):
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_ACCOUNT] 계좌 정보를 제출해 주세요."},
                {
                    "speaker": "victim",
                    "utterance": "[MASKED_ACCOUNT]는 공식 확인 절차를 거친 후에만 제출할 수 있습니다.",
                    "masked_items": [{"label": "account", "placeholder": "[MASKED_ACCOUNT]"}],
                },
            ]
        )

        self.assertEqual(outcome, "inconclusive")
        self.assertEqual(assets, [])

    def test_negated_disclosure_remains_defended_success(self):
        outcome, assets = infer_record_outcome(
            [
                {"speaker": "attacker", "utterance": "[MASKED_OTP] 인증이 필요합니다."},
                {
                    "speaker": "victim",
                    "utterance": "공식 채널 확인 전에는 [MASKED_OTP]를 제공할 수 없습니다.",
                    "masked_items": [{"label": "otp", "placeholder": "[MASKED_OTP]"}],
                },
            ]
        )

        self.assertEqual(outcome, "defended_success")
        self.assertEqual(assets, [])


if __name__ == "__main__":
    unittest.main()
