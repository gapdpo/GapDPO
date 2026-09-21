from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import MASKED_SECRET_PATTERN, STANDARD_PLACEHOLDER_BY_LABEL


@dataclass(frozen=True)
class MaskedItem:
    label: str
    placeholder: str
    original: str


@dataclass(frozen=True)
class SafetyResult:
    text: str
    masked_items: list[MaskedItem]


class SafetyMasker:
    """Masks sensitive values while preserving enough signal for defensive labels."""

    PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
        ("url", STANDARD_PLACEHOLDER_BY_LABEL["url"], re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)),
        ("rrn", STANDARD_PLACEHOLDER_BY_LABEL["rrn"], re.compile(r"\b\d{6}-[1-4]\d{6}\b")),
        (
            "otp",
            STANDARD_PLACEHOLDER_BY_LABEL["otp"],
            re.compile(r"(?i)\b(?:otp|인증번호|보안코드|verification code)[:\s-]*\d{4,8}\b"),
        ),
        ("phone", STANDARD_PLACEHOLDER_BY_LABEL["phone"], re.compile(r"\b(?:010|011|016|017|018|019)-?\d{3,4}-?\d{4}\b")),
        (
            "email",
            STANDARD_PLACEHOLDER_BY_LABEL["email"],
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        ),
        (
            "account",
            STANDARD_PLACEHOLDER_BY_LABEL["account"],
            re.compile(r"\b(?!(?:19|20)\d{2}-\d{2}-\d{2}\b)\d{2,6}-\d{2,6}-\d{5,8}\b"),
        ),
        (
            "credential",
            STANDARD_PLACEHOLDER_BY_LABEL["credential"],
            re.compile(r"(?i)\b(?:password|passwd|비밀번호|pin)[:\s-]*[A-Za-z0-9!@#$%^&*]{4,}\b"),
        ),
    )
    PLACEHOLDER_LABELS = {placeholder: label for label, placeholder, _pattern in PATTERNS}
    PLACEHOLDER_PATTERN = MASKED_SECRET_PATTERN
    REVIEWED_NONSTANDARD_PLACEHOLDERS = {
        "[MASKED_CASE_REFERENCE]",
        "[MASKED_COMPANY]",
        "[MASKED_ID_IMAGE]",
        "[MASKED_NAME]",
        "[MASKED_VOICE_SAMPLE]",
    }
    PLACEHOLDER_ALIASES = {
        "[MASKED_ACCOUNT_NUMBER]": "[MASKED_ACCOUNT]",
        "[MASKED_BANK_ACCOUNT]": "[MASKED_ACCOUNT]",
        "[MASKED_POLICY_ACCOUNT]": "[MASKED_ACCOUNT]",
        "[MASKED_PROPERTY_POLICY_ACCOUNT]": "[MASKED_ACCOUNT]",
        "[MASKED_CASE_ID]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_CASE_NUMBER]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_CASE_REF]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_CASE_REFERENCE_NUMBER]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_CONTACT]": "[MASKED_PHONE]",
        "[MASKED_CONTACT_DETAILS]": "[MASKED_PHONE]",
        "[MASKED_CONTACT_PHONE]": "[MASKED_PHONE]",
        "[MASKED_REGISTERED_CONTACT]": "[MASKED_PHONE]",
        "[MASKED_CREATOR_NAME]": "[MASKED_NAME]",
        "[MASKED_MANAGER_NAME]": "[MASKED_NAME]",
        "[MASKED_RECRUITER_NAME]": "[MASKED_NAME]",
        "[MASKED_ID_DOC]": "[MASKED_ID_IMAGE]",
        "[MASKED_ID_DOCUMENT]": "[MASKED_ID_IMAGE]",
        "[MASKED_ID_DOCUMENT_IMAGE]": "[MASKED_ID_IMAGE]",
        "[MASKED_WAITLIST_NUMBER]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_WAITLIST_REF]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_WAITLIST_REFERENCE]": "[MASKED_CASE_REFERENCE]",
        "[MASKED_WAITLIST_REFERENCE_NUMBER]": "[MASKED_CASE_REFERENCE]",
    }
    BRACKET_PLACEHOLDER_ALIASES = {
        "[caller_name]": "[MASKED_NAME]",
        "[customer_name]": "[MASKED_NAME]",
        "[manager_name]": "[MASKED_NAME]",
        "[victim_name]": "[MASKED_NAME]",
    }
    MASKED_PLACEHOLDER_ALIAS_PATTERN = re.compile(r"\[MASKED_[A-Z0-9_]+\]")
    BRACKET_PLACEHOLDER_ALIAS_PATTERN = re.compile(r"\[([a-z][a-z0-9_]{1,80})\]")

    FORBIDDEN_OPERATIONAL_TERMS = (
        "악성코드 설치",
        "피싱 사이트 제작",
        "인증 우회",
        "보안 우회",
        "탈취 링크",
    )

    def mask_text(self, text: str) -> SafetyResult:
        current = self.normalize_placeholder_aliases(text)
        masked_items: list[MaskedItem] = [
            MaskedItem(self.PLACEHOLDER_LABELS[match.group(0)], match.group(0), "")
            for match in self.PLACEHOLDER_PATTERN.finditer(current)
        ]
        for label, placeholder, pattern in self.PATTERNS:
            matches = list(pattern.finditer(current))
            if not matches:
                continue
            originals = [match.group(0) for match in matches]
            current = pattern.sub(placeholder, current)
            masked_items.extend(MaskedItem(label, placeholder, original) for original in originals)
        return SafetyResult(current, masked_items)

    def normalize_placeholder_aliases(self, text: str) -> str:
        current = text
        for alias, replacement in self.PLACEHOLDER_ALIASES.items():
            current = current.replace(alias, replacement)
        for alias, replacement in self.BRACKET_PLACEHOLDER_ALIASES.items():
            current = re.sub(re.escape(alias), replacement, current, flags=re.IGNORECASE)
        current = self.BRACKET_PLACEHOLDER_ALIAS_PATTERN.sub(
            lambda match: self.placeholder_for_lowercase_bracket_token(match.group(1)),
            current,
        )
        current = self.MASKED_PLACEHOLDER_ALIAS_PATTERN.sub(
            lambda match: self.normalize_masked_placeholder_alias(match.group(0)) or match.group(0),
            current,
        )
        return current

    @classmethod
    def normalize_masked_placeholder_alias(cls, placeholder: str) -> str | None:
        if placeholder in cls.PLACEHOLDER_LABELS or placeholder in cls.REVIEWED_NONSTANDARD_PLACEHOLDERS:
            return None
        if placeholder in cls.PLACEHOLDER_ALIASES:
            return cls.PLACEHOLDER_ALIASES[placeholder]
        if not placeholder.startswith("[MASKED_") or not placeholder.endswith("]"):
            return None
        token = placeholder[len("[MASKED_") : -1]
        if any(term in token for term in ("URL", "LINK")):
            return "[MASKED_URL]"
        if any(term in token for term in ("CREDENTIAL", "PASSWORD", "PORTAL_CREDENTIAL")):
            return "[MASKED_CREDENTIAL]"
        if any(term in token for term in ("CONTACT", "PHONE")):
            return "[MASKED_PHONE]"
        if any(term in token for term in ("ACCOUNT", "BANK", "PAYOUT", "PAYMENT")):
            return "[MASKED_ACCOUNT]"
        if any(term in token for term in ("OTP", "VERIFICATION_CODE", "SECURITY_CODE")):
            return "[MASKED_OTP]"
        if token in {"ID_LAST4", "ID_NUMBER", "TAXPAYER_ID"}:
            return "[MASKED_RRN]"
        if any(term in token for term in ("DOCUMENT", "IMAGE", "PHOTO", "TRANSLATION_FILE")):
            return "[MASKED_ID_IMAGE]"
        if any(
            term in token
            for term in ("COMPANY", "BUILDING", "OFFICE", "CENTER", "TEAM", "DEPARTMENT", "CLINIC", "SCHOOL", "HOSPITAL", "INSTITUTION")
        ):
            return "[MASKED_COMPANY]"
        if any(term in token for term in ("NAME", "PET", "STUDENT", "OFFICER", "AGENT")):
            return "[MASKED_NAME]"
        if any(
            term in token
            for term in (
                "REFERENCE",
                "REF",
                "CASE",
                "ORDER",
                "APPOINTMENT",
                "RECALL",
                "LAB",
                "NUMBER",
                "CALLBACK",
                "BOOK_TITLE",
                "FAMILY_INFO",
                "ROLE",
                "INTAKE",
                "PREFERENCE",
            )
        ):
            return "[MASKED_CASE_REFERENCE]"
        return None

    @classmethod
    def placeholder_for_lowercase_bracket_token(cls, token: str) -> str:
        upper = token.upper()
        normalized = cls.normalize_masked_placeholder_alias(f"[MASKED_{upper}]")
        if normalized:
            return normalized
        if any(term in token for term in ("office", "center", "company", "clinic", "school", "hospital", "team", "department")):
            return "[MASKED_COMPANY]"
        if token.endswith("_name") or "name" in token:
            return "[MASKED_NAME]"
        return "[MASKED_CASE_REFERENCE]"

    def contains_forbidden_operational_detail(self, text: str) -> bool:
        return any(term in text for term in self.FORBIDDEN_OPERATIONAL_TERMS)

    def find_sensitive_patterns(self, text: str) -> list[str]:
        labels: list[str] = []
        for label, _placeholder, pattern in self.PATTERNS:
            if pattern.search(text):
                labels.append(label)
        return labels

    def is_residual_pii_clean(self, text: str) -> bool:
        return not self.find_sensitive_patterns(text)

    def masks_to_dicts(self, items: list[MaskedItem]) -> list[dict[str, str]]:
        return [{"label": item.label, "placeholder": item.placeholder} for item in items]
