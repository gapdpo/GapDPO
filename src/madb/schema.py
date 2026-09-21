from __future__ import annotations

import re
from typing import Any


STANDARD_MASKED_PLACEHOLDER_LABELS = {
    "[MASKED_URL]": "url",
    "[MASKED_RRN]": "rrn",
    "[MASKED_OTP]": "otp",
    "[MASKED_PHONE]": "phone",
    "[MASKED_EMAIL]": "email",
    "[MASKED_ACCOUNT]": "account",
    "[MASKED_CREDENTIAL]": "credential",
}
STANDARD_PLACEHOLDER_BY_LABEL = {
    label: placeholder for placeholder, label in STANDARD_MASKED_PLACEHOLDER_LABELS.items()
}
STANDARD_MASKED_PLACEHOLDERS = frozenset(STANDARD_MASKED_PLACEHOLDER_LABELS)
MASKED_PLACEHOLDER_PATTERN = re.compile(r"\[MASKED_[A-Z0-9_]+\]")
MASKED_SECRET_PATTERN = re.compile(
    "|".join(re.escape(placeholder) for placeholder in sorted(STANDARD_MASKED_PLACEHOLDERS))
)

REQUIRED_RECORD_FIELDS = {
    "id",
    "dataset_name",
    "label",
    "scenario_type",
    "victim_persona",
    "attacker_persona",
    "protected_assets",
    "attacker_goal",
    "outcome",
    "compromised_assets",
    "difficulty",
    "channel",
    "locale",
    "split",
    "context",
    "dialogue",
    "risk_labels",
    "safety_masks",
    "gen_metadata",
    "quality_checks",
}

REQUIRED_TURN_FIELDS = {
    "turn_id",
    "speaker",
    "phase",
    "utterance",
    "intent_label",
    "risk_signal",
    "masked_items",
}

ENVIRONMENTAL_CONTEXT_FIELDS = (
    "channel",
    "timing",
    "trigger_event",
    "claimed_authority",
    "victim_constraint",
    "verification_path",
)


class SchemaError(ValueError):
    pass


def validate_record(
    record: dict[str, Any],
    *,
    max_turns: int,
    allowed_risk_labels: list[str] | set[str] | tuple[str, ...] | None = None,
) -> None:
    missing = REQUIRED_RECORD_FIELDS - set(record)
    if missing:
        raise SchemaError(f"Missing record fields: {sorted(missing)}")

    if not isinstance(record["dialogue"], list) or not record["dialogue"]:
        raise SchemaError("dialogue must be a non-empty list")
    if len(record["dialogue"]) > max_turns:
        raise SchemaError(f"dialogue exceeds max_turns={max_turns}")
    if not isinstance(record["protected_assets"], list) or not record["protected_assets"]:
        raise SchemaError("protected_assets must be a non-empty list")
    if not isinstance(record["risk_labels"], list) or not record["risk_labels"]:
        raise SchemaError("risk_labels must be a non-empty list")
    if not all(isinstance(risk_label, str) and risk_label.strip() for risk_label in record["risk_labels"]):
        raise SchemaError("risk_labels entries must be non-empty strings")
    if allowed_risk_labels is not None:
        allowed = set(allowed_risk_labels)
        unknown = sorted(set(record["risk_labels"]) - allowed)
        if unknown:
            raise SchemaError(f"Unknown risk_labels: {unknown}")
    if record["label"] not in {"attack", "benign", "borderline"}:
        raise SchemaError(f"Invalid label: {record['label']}")
    if record["outcome"] not in {"defended_success", "compromised", "aborted", "inconclusive"}:
        raise SchemaError(f"Invalid outcome: {record['outcome']}")
    if record["split"] not in {"train", "val", "test"}:
        raise SchemaError(f"Invalid split: {record['split']}")
    validate_mask_list(record["safety_masks"], "safety_masks")
    validate_gen_metadata(record["gen_metadata"])

    expected_speaker = "attacker"
    for index, turn in enumerate(record["dialogue"], start=1):
        missing_turn = REQUIRED_TURN_FIELDS - set(turn)
        if missing_turn:
            raise SchemaError(f"Missing turn fields at {index}: {sorted(missing_turn)}")
        if turn["turn_id"] != index:
            raise SchemaError(f"turn_id mismatch at turn {index}")
        if turn["speaker"] not in {"attacker", "victim"}:
            raise SchemaError(f"Invalid speaker at turn {index}: {turn['speaker']}")
        if turn["speaker"] != expected_speaker:
            raise SchemaError(f"Unexpected speaker order at turn {index}: {turn['speaker']}")
        expected_speaker = "victim" if expected_speaker == "attacker" else "attacker"
        validate_mask_list(turn["masked_items"], f"dialogue[{index}].masked_items")

    quality = record["quality_checks"]
    if not isinstance(quality, dict):
        raise SchemaError("quality_checks must be an object")
    required_quality_keys = (
        "turn_count",
        "role_order_valid",
        "safety_passed",
        "no_duplicate_turns",
        "label_consistent",
        "language_consistent",
        "no_refusal_leak",
        "residual_pii_clean",
        "schema_valid",
    )
    for key in required_quality_keys:
        if key not in quality:
            raise SchemaError(f"Missing quality check: {key}")
    if not isinstance(quality["turn_count"], int) or isinstance(quality["turn_count"], bool):
        raise SchemaError("quality_checks.turn_count must be an integer")
    if quality["turn_count"] != len(record["dialogue"]):
        raise SchemaError("quality_checks.turn_count must match dialogue length")
    for key in required_quality_keys[1:]:
        if not isinstance(quality[key], bool):
            raise SchemaError(f"quality_checks.{key} must be a boolean")


def validate_gen_metadata(value: Any) -> None:
    if not isinstance(value, dict):
        raise SchemaError("gen_metadata must be an object")

    if "environmental_context" not in value:
        return

    context = value["environmental_context"]
    if not isinstance(context, dict):
        raise SchemaError("gen_metadata.environmental_context must be an object")
    missing = [key for key in ENVIRONMENTAL_CONTEXT_FIELDS if key not in context]
    if missing:
        raise SchemaError(f"gen_metadata.environmental_context is missing fields: {missing}")
    for key in ENVIRONMENTAL_CONTEXT_FIELDS:
        if not isinstance(context[key], str) or not context[key].strip():
            raise SchemaError(f"gen_metadata.environmental_context.{key} must be a non-empty string")


def validate_mask_list(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise SchemaError(f"{field_name} must be a list")
    for index, mask in enumerate(value):
        if not isinstance(mask, dict):
            raise SchemaError(f"{field_name}[{index}] must be an object")
        if "original" in mask:
            raise SchemaError(f"{field_name} must not expose original sensitive values")
        for key in ("label", "placeholder"):
            if key not in mask:
                raise SchemaError(f"{field_name}[{index}] is missing {key}")
            if not isinstance(mask[key], str) or not mask[key].strip():
                raise SchemaError(f"{field_name}[{index}].{key} must be a non-empty string")
