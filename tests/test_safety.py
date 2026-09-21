import unittest

from madb.safety import SafetyMasker


class SafetyMaskerTest(unittest.TestCase):
    def test_masks_sensitive_patterns(self):
        masker = SafetyMasker()
        text = (
            "인증번호 123456, https://example.test, 010-1234-5678, "
            "900101-1234567, user@example.test, password:abcd1234, 123-456-789012"
        )

        result = masker.mask_text(text)

        self.assertIn("[MASKED_OTP]", result.text)
        self.assertIn("[MASKED_URL]", result.text)
        self.assertIn("[MASKED_PHONE]", result.text)
        self.assertIn("[MASKED_RRN]", result.text)
        self.assertIn("[MASKED_EMAIL]", result.text)
        self.assertIn("[MASKED_CREDENTIAL]", result.text)
        self.assertIn("[MASKED_ACCOUNT]", result.text)
        self.assertEqual(len(result.masked_items), 7)

    def test_exported_masks_do_not_include_original(self):
        masker = SafetyMasker()
        result = masker.mask_text("인증번호 123456")

        exported = masker.masks_to_dicts(result.masked_items)

        self.assertEqual(exported, [{"label": "otp", "placeholder": "[MASKED_OTP]"}])

    def test_pre_masked_placeholders_are_exported(self):
        masker = SafetyMasker()
        result = masker.mask_text("[MASKED_ACCOUNT]와 [MASKED_OTP] 처리가 필요합니다.")

        exported = masker.masks_to_dicts(result.masked_items)

        self.assertEqual(
            exported,
            [
                {"label": "account", "placeholder": "[MASKED_ACCOUNT]"},
                {"label": "otp", "placeholder": "[MASKED_OTP]"},
            ],
        )

    def test_normalizes_nonstandard_bracket_name_placeholder(self):
        masker = SafetyMasker()
        result = masker.mask_text("[victim_name]님, 일정 확인이 필요합니다.")

        self.assertEqual(result.text, "[MASKED_NAME]님, 일정 확인이 필요합니다.")

    def test_normalizes_repeated_masked_placeholder_aliases(self):
        masker = SafetyMasker()
        result = masker.mask_text(
            "[MASKED_CONTACT]와 [MASKED_CASE_NUMBER], [MASKED_MANAGER_NAME], [MASKED_ID_DOC] 확인이 필요합니다."
        )

        self.assertIn("[MASKED_PHONE]", result.text)
        self.assertIn("[MASKED_CASE_REFERENCE]", result.text)
        self.assertIn("[MASKED_NAME]", result.text)
        self.assertIn("[MASKED_ID_IMAGE]", result.text)
        self.assertIn({"label": "phone", "placeholder": "[MASKED_PHONE]"}, masker.masks_to_dicts(result.masked_items))

    def test_normalizes_lowercase_bracket_office_placeholder(self):
        masker = SafetyMasker()
        result = masker.mask_text("[childcare_center_office]에서 상담 일정을 확인해야 합니다.")

        self.assertEqual(result.text, "[MASKED_COMPANY]에서 상담 일정을 확인해야 합니다.")

    def test_normalizes_observed_masked_placeholder_aliases_by_heuristic(self):
        masker = SafetyMasker()
        result = masker.mask_text(
            "[MASKED_ORDER_REFERENCE], [MASKED_APPOINTMENT_REFERENCE], [MASKED_DOCUMENT_IMAGE], "
            "[MASKED_PET_NAME], [MASKED_LANGUAGE_SUPPORT_PORTAL_CREDENTIALS] 확인이 필요합니다."
        )

        self.assertIn("[MASKED_CASE_REFERENCE]", result.text)
        self.assertIn("[MASKED_ID_IMAGE]", result.text)
        self.assertIn("[MASKED_NAME]", result.text)
        self.assertIn("[MASKED_CREDENTIAL]", result.text)

    def test_flags_forbidden_operational_terms(self):
        masker = SafetyMasker()

        self.assertTrue(masker.contains_forbidden_operational_detail("인증 우회 절차를 설명한다."))
        self.assertFalse(masker.contains_forbidden_operational_detail("공식 경로로 확인한다."))

    def test_residual_pattern_detection_ignores_dates(self):
        masker = SafetyMasker()

        self.assertEqual(masker.find_sensitive_patterns("timestamp 2026-06-02"), [])
        self.assertEqual(masker.find_sensitive_patterns("contact user@example.test"), ["email"])
        self.assertEqual(masker.find_sensitive_patterns("계좌 123-456-789012"), ["account"])


if __name__ == "__main__":
    unittest.main()
