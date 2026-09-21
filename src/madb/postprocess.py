from __future__ import annotations

import re


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
ROLE_PREFIX_PATTERN = re.compile(r"^\s*(?:공격자|피해자|attacker|victim|assistant)\s*[:：-]\s*", re.IGNORECASE)
STAGE_DIRECTION_PATTERN = re.compile(r"^\s*\[(?!MASKED_)[^\]]{1,80}\]\s*")


def clean_model_utterance(text: str) -> str:
    cleaned = THINK_PATTERN.sub("", text).strip()
    cleaned = cleaned.replace("```", "").strip()
    cleaned = ROLE_PREFIX_PATTERN.sub("", cleaned).strip()
    cleaned = STAGE_DIRECTION_PATTERN.sub("", cleaned).strip()
    cleaned = strip_wrapping_quotes(cleaned)
    if "\n" in cleaned:
        cleaned = first_nonempty_line(cleaned)
        cleaned = ROLE_PREFIX_PATTERN.sub("", cleaned).strip()
        cleaned = STAGE_DIRECTION_PATTERN.sub("", cleaned).strip()
    return cleaned or "공식 경로로 다시 확인하겠습니다."


def strip_wrapping_quotes(text: str) -> str:
    quote_pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
    for left, right in quote_pairs:
        if text.startswith(left) and text.endswith(right) and len(text) >= 2:
            return text[1:-1].strip()
    return text


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def normalized_for_duplicate_check(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())
