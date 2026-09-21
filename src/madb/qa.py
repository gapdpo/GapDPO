from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .labeling import has_partial_compliance_signal
from .orchestrator import PAIR_SCENARIO_PREFERENCES, SCENARIO_RISK_LABELS
from .postprocess import normalized_for_duplicate_check
from .safety import SafetyMasker
from .schema import MASKED_PLACEHOLDER_PATTERN, STANDARD_MASKED_PLACEHOLDERS, SchemaError, validate_record


QUALITY_KEYS = (
    "role_order_valid",
    "safety_passed",
    "no_duplicate_turns",
    "label_consistent",
    "language_consistent",
    "no_refusal_leak",
    "residual_pii_clean",
    "schema_valid",
)

ENVIRONMENTAL_CONTEXT_KEYS = (
    "channel",
    "timing",
    "trigger_event",
    "claimed_authority",
    "victim_constraint",
    "verification_path",
)

DUPLICATE_UTTERANCE_METRIC = {
    "scope": "corpus_turn_utterances_after_masking",
    "normalization": "lowercase_without_whitespace",
    "interpretation": (
        "descriptive_only; deterministic mock runs can reflect finite template coverage "
        "rather than real model-output diversity"
    ),
}
FINAL_CORPUS_MAX_DUPLICATE_UTTERANCE_RATE = 0.2
FINAL_CORPUS_MAX_REPAIR_APPLIED_RATE = 0.25
FINAL_CORPUS_MIN_TARGET_OUTCOME_MATCH_RATE = 1.0

ATTACKER_PRESSURE_CUE_PATTERNS = {
    "urgency": re.compile(r"긴급|급하|급해|급한|급히|급해서|지금|바로|즉시|몇\s*분|오늘\s*안|마감|기한|늦으면|서둘"),
    "authority": re.compile(r"담당자|기관|보안팀|관리자|정책|심사|대표|팀장|학교|은행|공식|센터|행정"),
    "consequence": re.compile(r"중단|보류|제한|취소|정지|불이익|문제|지연|거절"),
    "relationship_or_emotional_pressure": re.compile(r"가족|엄마|아빠|자녀|보호자|도와|걱정|부탁|사고|아프|믿"),
    "process_control": re.compile(r"통화.*유지|끊지|확인.*절차|처리.*요구|검증|제출|보완|정산|결제|인증"),
}

STAGE_DIRECTION_PATTERN = re.compile(r"\[(?!MASKED_)[^\]]{1,80}\]")
OTP_TRANSFER_ANOMALY_PATTERN = re.compile(
    r"(?:\[MASKED_OTP\]\s*(?:번)?\s*(?:으로|로)?\s*(?:입금|송금|이체|결제))|"
    r"(?:(?:입금|송금|이체|결제)\s*(?:대상|계좌|번호)?\s*(?:은|는|:)?\s*\[MASKED_OTP\])"
)
UNKNOWN_MASKED_PLACEHOLDER_POLICY: dict[str, dict[str, str]] = {
    "[MASKED_NAME]": {
        "decision": "allowlist_candidate",
        "reason": "generic synthetic person name reference; keep masked and non-operational",
    },
    "[MASKED_COMPANY]": {
        "decision": "allowlist_candidate",
        "reason": "generic synthetic organization reference; useful for official-channel verification turns",
    },
    "[MASKED_CASE_REFERENCE]": {
        "decision": "allowlist_candidate",
        "reason": "generic synthetic case or ticket identifier; useful for defensive verification context",
    },
    "[MASKED_ID_IMAGE]": {
        "decision": "allowlist_candidate",
        "reason": "generic masked identity-document image reference; sensitive but non-operational when masked",
    },
    "[MASKED_VOICE_SAMPLE]": {
        "decision": "allowlist_candidate",
        "reason": "generic masked voice-sample reference; useful for callback or voice-confirmation scenarios",
    },
    "[MASKED_RECRUITER_NAME]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_NAME]",
        "reason": "role-specific person name should collapse to the generic masked name placeholder",
    },
    "[MASKED_CREATOR_NAME]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_NAME]",
        "reason": "role-specific person name should collapse to the generic masked name placeholder",
    },
    "[MASKED_COMPANY_EMAIL]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_EMAIL]",
        "reason": "role-specific email reference should collapse to the generic masked email placeholder",
    },
    "[MASKED_MANAGER_NAME]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_NAME]",
        "reason": "role-specific person name should collapse to the generic masked name placeholder",
    },
    "[MASKED_CONTACT]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_PHONE]",
        "reason": "generic contact detail should collapse to the standard masked phone/contact placeholder",
    },
    "[MASKED_CONTACT_DETAILS]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_PHONE]",
        "reason": "generic contact details should collapse to the standard masked phone/contact placeholder",
    },
    "[MASKED_CASE_NUMBER]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_CASE_REFERENCE]",
        "reason": "case-number wording should collapse to the reviewed case-reference placeholder",
    },
    "[MASKED_ID_DOC]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_ID_IMAGE]",
        "reason": "identity-document shorthand should collapse to the reviewed identity-document image placeholder",
    },
    "[MASKED_WAITLIST_REFERENCE]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_CASE_REFERENCE]",
        "reason": "domain-specific waitlist reference should collapse to the reviewed case-reference placeholder",
    },
    "[MASKED_WAITLIST_NUMBER]": {
        "decision": "normalize_to_existing",
        "normalize_to": "[MASKED_CASE_REFERENCE]",
        "reason": "domain-specific waitlist number should collapse to the reviewed case-reference placeholder",
    },
    "[MASKED_JOB_TITLE]": {
        "decision": "warning_keep_unstandardized",
        "reason": "role/title context is not currently a sensitive-value mask in SafetyMasker",
    },
    "[MASKED_LANGUAGE_SUPPORT_PORTAL]": {
        "decision": "warning_keep_unstandardized",
        "reason": "portal/service context is too domain-specific for the current standard placeholder set",
    },
}

PLACEHOLDER_REVIEWED_SENSITIVE_PATTERNS = (
    "CHILD",
    "MEMBER",
    "PET",
    "CONTACT",
    "HOSPITAL",
    "SCHOOL",
    "BANK",
    "NONPROFIT",
    "CLINIC",
    "INSTITUTION",
    "VOICE_SAMPLE",
    "ADDRESS",
    "BIRTH",
    "VIN",
    "ID_IMAGE",
)
KOREAN_ENTITY_WARNING_PATTERN = re.compile(
    r"(?:[가-힣]{2,4}(?:씨|님))|(?:[가-힣A-Za-z0-9]{2,20}(?:은행|병원|학교|대학교|센터|재단|협회|보험|카드|증권|구청|시청|도서관))"
)


def summarize_records(
    records: list[dict[str, Any]],
    *,
    max_turns: int | None = None,
    allowed_risk_labels: list[str] | set[str] | tuple[str, ...] | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    turn_counts = [len(record.get("dialogue", [])) for record in records]
    utterances = [
        normalized_for_duplicate_check(str(turn.get("utterance", "")))
        for record in records
        for turn in record.get("dialogue", [])
    ]
    duplicate_utterances = sum(count - 1 for count in Counter(utterances).values() if count > 1)
    duplicate_utterance_rate = round(duplicate_utterances / len(utterances), 4) if utterances else 0.0
    duplicate_record_ids = duplicate_record_id_summary(records)
    mask_counts = Counter(
        str(mask.get("label", "unknown"))
        for record in records
        for mask in record.get("safety_masks", [])
    )
    quality_summary = summarize_quality_checks(records)
    corpus_residual_patterns = find_corpus_residual_patterns(records)
    scenario_risk_coverage = scenario_default_attacker_risk_signal_coverage(records)
    quality_signals = quality_signal_summary(records)
    target_alignment = target_outcome_mode_alignment(records)
    summary = {
        "record_count": len(records),
        "source_metadata": source_metadata_summary(records, source_path),
        "attack_only_policy": attack_only_policy_summary(records),
        "scenario_default_attacker_risk_signal_coverage": scenario_risk_coverage,
        "scenario_default_risk_alignment": scenario_risk_coverage,
        "attacker_pressure_cue_diversity": attacker_pressure_cue_diversity(records),
        "assigned_pressure_style_alignment": assigned_pressure_style_alignment(records),
        "environmental_context_coverage": environmental_context_coverage(records),
        "target_outcome_mode_alignment": target_alignment,
        "label_distribution": count_values(records, "label"),
        "outcome_distribution": count_values(records, "outcome"),
        "scenario_distribution": count_values(records, "scenario_type"),
        "split_distribution": count_values(records, "split"),
        "victim_distribution": count_nested_values(records, "victim_persona", "id"),
        "attacker_distribution": count_nested_values(records, "attacker_persona", "id"),
        "mask_counts": dict(sorted(mask_counts.items())),
        "duplicate_utterance_count": duplicate_utterances,
        "duplicate_utterance_rate": duplicate_utterance_rate,
        "duplicate_utterance_metric": DUPLICATE_UTTERANCE_METRIC,
        "duplicate_record_ids": duplicate_record_ids,
        "duplicate_record_id_count": duplicate_record_ids["duplicate_count"],
        "average_turn_count": round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0.0,
        "quality_check_summary": quality_summary,
        "quality_pass_rate": quality_pass_rate(quality_summary),
        "schema_contract_validation": schema_contract_validation_summary(
            records,
            max_turns=max_turns,
            allowed_risk_labels=allowed_risk_labels,
        ),
        "corpus_residual_sensitive_patterns": corpus_residual_patterns,
        "quality_signals": quality_signals,
    }
    summary["final_dialogue_corpus_review"] = final_dialogue_corpus_review(
        duplicate_utterance_rate=duplicate_utterance_rate,
        target_alignment=target_alignment,
        quality_signals=quality_signals,
    )
    return summary


def source_metadata_summary(records: list[dict[str, Any]], source_path: str | Path | None) -> dict[str, Any]:
    if source_path is None:
        return {
            "source_path": None,
            "record_count": len(records),
            "jsonl_line_count": None,
            "filename_count_hint": None,
            "ignored_date_like_tokens": [],
            "count_mismatch_warning": False,
        }
    path = Path(source_path)
    hint = filename_count_hint(path.name)
    line_count = None
    if path.exists():
        line_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    record_count = len(records)
    mismatch = hint["count_hint"] is not None and hint["count_hint"] != record_count
    return {
        "source_path": str(path),
        "record_count": record_count,
        "jsonl_line_count": line_count,
        "filename_count_hint": hint["count_hint"],
        "ignored_date_like_tokens": hint["ignored_date_like_tokens"],
        "numeric_tokens": hint["numeric_tokens"],
        "count_mismatch_warning": mismatch,
        "line_count_mismatch_warning": line_count is not None and line_count != record_count,
    }


def filename_count_hint(filename: str) -> dict[str, Any]:
    numeric_tokens = re.findall(r"(?<!\d)(\d+)(?!\d)", filename)
    ignored_dates = [token for token in numeric_tokens if is_compact_date_token(token)]
    candidates = [token for token in numeric_tokens if token not in ignored_dates]
    count_hint = int(candidates[-1]) if candidates else None
    return {
        "numeric_tokens": numeric_tokens,
        "ignored_date_like_tokens": ignored_dates,
        "count_hint": count_hint,
    }


def is_compact_date_token(token: str) -> bool:
    if not re.fullmatch(r"\d{6}", token):
        return False
    month = int(token[2:4])
    day = int(token[4:6])
    return 1 <= month <= 12 and 1 <= day <= 31


def schema_contract_validation_summary(
    records: list[dict[str, Any]],
    *,
    max_turns: int | None,
    allowed_risk_labels: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if max_turns is None:
        return {
            "enabled": False,
            "pass": 0,
            "fail": 0,
            "missing": len(records),
            "failures": [],
        }

    failures: list[dict[str, Any]] = []
    pass_count = 0
    for index, record in enumerate(records):
        try:
            validate_record(record, max_turns=max_turns, allowed_risk_labels=allowed_risk_labels)
        except SchemaError as exc:
            failures.append(
                {
                    "index": index,
                    "record_id": str(record.get("id", "unknown")),
                    "error": str(exc),
                }
            )
        else:
            pass_count += 1

    return {
        "enabled": True,
        "pass": pass_count,
        "fail": len(failures),
        "missing": 0,
        "failures": failures,
    }


def environmental_context_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    field_summary = {key: {"present": 0, "missing": 0} for key in ENVIRONMENTAL_CONTEXT_KEYS}
    distributions: dict[str, Counter[str]] = {key: Counter() for key in ENVIRONMENTAL_CONTEXT_KEYS}
    variant_distribution: Counter[str] = Counter()
    missing_records: list[dict[str, str]] = []
    with_environmental_context = 0

    for record in records:
        metadata = record.get("gen_metadata", {})
        if isinstance(metadata, dict):
            variant_id = metadata.get("environmental_context_variant_id")
            if isinstance(variant_id, str) and variant_id.strip():
                variant_distribution[variant_id] += 1
        context = environmental_context_from_record(record)
        if context is None:
            missing_records.append({"record_id": str(record.get("id", "unknown")), "missing": "environmental_context"})
            for key in ENVIRONMENTAL_CONTEXT_KEYS:
                field_summary[key]["missing"] += 1
            continue

        with_environmental_context += 1
        missing_fields: list[str] = []
        for key in ENVIRONMENTAL_CONTEXT_KEYS:
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                field_summary[key]["present"] += 1
                distributions[key][value] += 1
            else:
                field_summary[key]["missing"] += 1
                missing_fields.append(key)
        if missing_fields:
            missing_records.append(
                {
                    "record_id": str(record.get("id", "unknown")),
                    "missing": ", ".join(missing_fields),
                }
            )

    return {
        "record_count": len(records),
        "with_environmental_context": with_environmental_context,
        "missing_environmental_context": len(records) - with_environmental_context,
        "field_coverage": field_summary,
        "distributions": {
            key: dict(sorted(values.items()))
            for key, values in distributions.items()
            if values
        },
        "variant_distribution": dict(sorted(variant_distribution.items())),
        "prompt_visible_field_unique_counts": {
            key: len(values)
            for key, values in sorted(distributions.items())
            if values
        },
        "missing_records": missing_records[:10],
    }


def environmental_context_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("gen_metadata", {})
    if not isinstance(metadata, dict):
        return None
    context = metadata.get("environmental_context")
    return context if isinstance(context, dict) else None


def scenario_default_attacker_risk_signal_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    mismatch_count = 0
    missing_attacker_risk_count = 0
    for record in records:
        scenario_type = str(record.get("scenario_type", ""))
        default_risks = set(SCENARIO_RISK_LABELS.get(scenario_type, ["voice_impersonation"]))
        attacker_risks = attacker_risk_signal_set(record)
        if not attacker_risks:
            missing_attacker_risk_count += 1
        if default_risks != attacker_risks:
            mismatch_count += 1
    record_count = len(records)
    return {
        "scope": "record_level_descriptive_scenario_defaults_vs_observed_attacker_turn_risk_signal_set",
        "blocking": False,
        "record_count": record_count,
        "match_count": record_count - mismatch_count,
        "mismatch_count": mismatch_count,
        "mismatch_rate": round(mismatch_count / record_count, 4) if record_count else 0.0,
        "missing_attacker_risk_count": missing_attacker_risk_count,
        "interpretation": (
            "descriptive_only; scenario defaults can intentionally include broader risk context "
            "than a single serialized attacker turn risk_signal"
        ),
    }


def scenario_default_risk_alignment(records: list[dict[str, Any]]) -> dict[str, Any]:
    return scenario_default_attacker_risk_signal_coverage(records)


def attacker_risk_signal_set(record: dict[str, Any]) -> set[str]:
    dialogue = record.get("dialogue", [])
    if not isinstance(dialogue, list):
        return set()
    return {
        str(turn.get("risk_signal", ""))
        for turn in dialogue
        if isinstance(turn, dict)
        and turn.get("speaker") == "attacker"
        and turn.get("risk_signal") not in {"", "generation_fallback"}
    }


def attack_only_policy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    non_attack_count = sum(1 for record in records if record.get("label") != "attack")
    return {
        "expected_label": "attack",
        "pass": non_attack_count == 0,
        "non_attack_count": non_attack_count,
    }


def count_values(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key, "unknown")) for record in records).items()))


def count_nested_values(records: list[dict[str, Any]], parent_key: str, child_key: str) -> dict[str, int]:
    counter = Counter()
    for record in records:
        value = record.get(parent_key, {})
        if isinstance(value, dict):
            counter[str(value.get(child_key, "unknown"))] += 1
        else:
            counter["unknown"] += 1
    return dict(sorted(counter.items()))


def duplicate_record_id_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        record_id
        for record in records
        if isinstance((record_id := record.get("id")), str) and record_id.strip()
    )
    duplicates = {record_id: count for record_id, count in sorted(counts.items()) if count > 1}
    return {
        "scope": "dataset_record_id",
        "blocking": False,
        "duplicate_count": sum(count - 1 for count in duplicates.values()),
        "duplicate_ids": duplicates,
    }


def attacker_pressure_cue_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    attacker_turn_count = 0
    categorized_attacker_turn_count = 0
    urgency_turn_count = 0
    examples: list[dict[str, str]] = []

    for record in records:
        for turn in record.get("dialogue", []):
            if not isinstance(turn, dict) or turn.get("speaker") != "attacker":
                continue
            attacker_turn_count += 1
            utterance = str(turn.get("utterance", ""))
            categories = attacker_pressure_cue_categories(utterance)
            if not categories:
                continue
            categorized_attacker_turn_count += 1
            category_counts.update(categories)
            if "urgency" in categories:
                urgency_turn_count += 1
            examples.append(
                {
                    "record_id": str(record.get("id", "unknown")),
                    "turn_id": str(turn.get("turn_id", "unknown")),
                    "categories": ", ".join(categories),
                    "utterance": truncate_for_report(utterance),
                }
            )

    category_observation_count = sum(category_counts.values())
    dominant_pressure_cue = ""
    dominant_pressure_cue_rate = 0.0
    if category_counts and category_observation_count:
        dominant_pressure_cue, dominant_count = category_counts.most_common(1)[0]
        dominant_pressure_cue_rate = round(dominant_count / category_observation_count, 4)

    urgency_turn_rate = (
        round(urgency_turn_count / categorized_attacker_turn_count, 4)
        if categorized_attacker_turn_count
        else 0.0
    )
    unique_pressure_cue_count = len(category_counts)
    urgency_dominant_warning_only = (
        categorized_attacker_turn_count >= 3
        and urgency_turn_rate >= 0.75
        and unique_pressure_cue_count <= 2
    )

    return {
        "scope": "attacker_turn_text_pressure_cue_categories",
        "blocking": False,
        "attacker_turn_count": attacker_turn_count,
        "categorized_attacker_turn_count": categorized_attacker_turn_count,
        "uncategorized_attacker_turn_count": attacker_turn_count - categorized_attacker_turn_count,
        "category_counts": dict(sorted(category_counts.items())),
        "unique_pressure_cue_count": unique_pressure_cue_count,
        "dominant_pressure_cue": dominant_pressure_cue,
        "dominant_pressure_cue_rate": dominant_pressure_cue_rate,
        "urgency_turn_count": urgency_turn_count,
        "urgency_turn_rate": urgency_turn_rate,
        "urgency_dominant_warning_only": urgency_dominant_warning_only,
        "interpretation": (
            "descriptive_only; high urgency dominance suggests pressure wording may be repetitive "
            "and should be reviewed before expanding generation"
        ),
        "examples": examples[:10],
    }


def attacker_pressure_cue_categories(utterance: str) -> list[str]:
    return [
        category
        for category, pattern in ATTACKER_PRESSURE_CUE_PATTERNS.items()
        if pattern.search(utterance)
    ]


def assigned_pressure_style_alignment(records: list[dict[str, Any]]) -> dict[str, Any]:
    assigned_count = 0
    observed_match_count = 0
    urgency_leakage_count = 0
    assigned_distribution: Counter[str] = Counter()
    observed_distribution: Counter[str] = Counter()
    mismatch_examples: list[dict[str, str]] = []
    leakage_examples: list[dict[str, str]] = []

    for record in records:
        metadata = record.get("gen_metadata", {})
        assigned = metadata.get("pressure_style") if isinstance(metadata, dict) else None
        if not isinstance(assigned, str) or not assigned.strip():
            continue
        assigned_count += 1
        assigned_distribution[assigned] += 1
        attacker_text = " ".join(
            str(turn.get("utterance", ""))
            for turn in record.get("dialogue", [])
            if isinstance(turn, dict) and turn.get("speaker") == "attacker"
        )
        observed = set(attacker_pressure_cue_categories(attacker_text))
        observed_distribution.update(observed)
        if assigned in observed:
            observed_match_count += 1
        elif len(mismatch_examples) < 10:
            mismatch_examples.append(
                {
                    "record_id": str(record.get("id", "unknown")),
                    "assigned": assigned,
                    "observed": ", ".join(sorted(observed)),
                    "utterance": truncate_for_report(attacker_text),
                }
            )
        if assigned != "urgency" and "urgency" in observed:
            urgency_leakage_count += 1
            if len(leakage_examples) < 10:
                leakage_examples.append(
                    {
                        "record_id": str(record.get("id", "unknown")),
                        "assigned": assigned,
                        "utterance": truncate_for_report(attacker_text),
                    }
                )

    return {
        "scope": "record_level_assigned_pressure_style_vs_observed_attacker_cues",
        "blocking": False,
        "assigned_record_count": assigned_count,
        "observed_match_count": observed_match_count,
        "observed_match_rate": round(observed_match_count / assigned_count, 4) if assigned_count else 0.0,
        "urgency_leakage_count": urgency_leakage_count,
        "urgency_leakage_rate": round(urgency_leakage_count / assigned_count, 4) if assigned_count else 0.0,
        "assigned_distribution": dict(sorted(assigned_distribution.items())),
        "observed_distribution": dict(sorted(observed_distribution.items())),
        "mismatch_examples": mismatch_examples,
        "urgency_leakage_examples": leakage_examples,
    }


def target_outcome_mode_alignment(records: list[dict[str, Any]]) -> dict[str, Any]:
    assigned_count = 0
    match_count = 0
    repair_applied_count = 0
    llm_rewrite_applied_count = 0
    retry_success_count = 0
    target_distribution: Counter[str] = Counter()
    observed_distribution: Counter[str] = Counter()
    repair_applied_by_target: Counter[str] = Counter()
    llm_rewrite_applied_by_target: Counter[str] = Counter()
    generation_attempt_distribution: Counter[str] = Counter()
    llm_rewrite_attempt_distribution: Counter[str] = Counter()
    mismatch_examples: list[dict[str, str]] = []
    target_to_observed = {
        "defended_success": "defended_success",
        "inconclusive": "inconclusive",
        "masked_compromise": "compromised",
    }
    for record in records:
        metadata = record.get("gen_metadata", {})
        target = metadata.get("target_outcome_mode") if isinstance(metadata, dict) else None
        if not isinstance(target, str) or not target.strip():
            continue
        observed = str(record.get("outcome", "unknown"))
        assigned_count += 1
        target_distribution[target] += 1
        observed_distribution[observed] += 1
        deterministic_repair_value = metadata.get("target_outcome_deterministic_repair_applied")
        repair_applied = (
            deterministic_repair_value is True
            if isinstance(deterministic_repair_value, bool)
            else metadata.get("target_outcome_repair_applied") is True
        )
        llm_rewrite_applied = metadata.get("target_outcome_llm_rewrite_applied") is True
        generation_attempts = metadata.get("target_outcome_generation_attempts")
        llm_rewrite_attempts = metadata.get("target_outcome_llm_rewrite_attempts")
        if isinstance(generation_attempts, int) and not isinstance(generation_attempts, bool):
            generation_attempt_distribution[str(generation_attempts)] += 1
        if isinstance(llm_rewrite_attempts, int) and not isinstance(llm_rewrite_attempts, bool) and llm_rewrite_attempts > 0:
            llm_rewrite_attempt_distribution[str(llm_rewrite_attempts)] += 1
        if repair_applied:
            repair_applied_count += 1
            repair_applied_by_target[target] += 1
        if llm_rewrite_applied:
            llm_rewrite_applied_count += 1
            llm_rewrite_applied_by_target[target] += 1
        expected = target_to_observed.get(target, target)
        if observed == expected:
            match_count += 1
            if (
                not repair_applied
                and not llm_rewrite_applied
                and isinstance(generation_attempts, int)
                and generation_attempts > 1
            ):
                retry_success_count += 1
        elif len(mismatch_examples) < 10:
            mismatch_examples.append(
                {
                    "record_id": str(record.get("id", "unknown")),
                    "target_outcome_mode": target,
                    "expected_outcome": expected,
                    "observed_outcome": observed,
                }
            )
    return {
        "scope": "record_level_target_outcome_mode_vs_inferred_outcome",
        "blocking": False,
        "assigned_record_count": assigned_count,
        "match_count": match_count,
        "match_rate": round(match_count / assigned_count, 4) if assigned_count else 0.0,
        "alignment_review_required": assigned_count > 0 and match_count < assigned_count,
        "min_final_match_rate": FINAL_CORPUS_MIN_TARGET_OUTCOME_MATCH_RATE,
        "repair_applied_count": repair_applied_count,
        "repair_applied_rate": round(repair_applied_count / assigned_count, 4) if assigned_count else 0.0,
        "repair_review_required": (
            round(repair_applied_count / assigned_count, 4) > FINAL_CORPUS_MAX_REPAIR_APPLIED_RATE
            if assigned_count
            else False
        ),
        "max_final_repair_applied_rate": FINAL_CORPUS_MAX_REPAIR_APPLIED_RATE,
        "repair_applied_by_target": dict(sorted(repair_applied_by_target.items())),
        "llm_rewrite_applied_count": llm_rewrite_applied_count,
        "llm_rewrite_applied_rate": round(llm_rewrite_applied_count / assigned_count, 4) if assigned_count else 0.0,
        "llm_rewrite_applied_by_target": dict(sorted(llm_rewrite_applied_by_target.items())),
        "retry_success_count": retry_success_count,
        "generation_attempt_distribution": dict(sorted(generation_attempt_distribution.items())),
        "llm_rewrite_attempt_distribution": dict(sorted(llm_rewrite_attempt_distribution.items())),
        "target_distribution": dict(sorted(target_distribution.items())),
        "observed_distribution": dict(sorted(observed_distribution.items())),
        "mismatch_examples": mismatch_examples,
    }


def final_dialogue_corpus_review(
    *,
    duplicate_utterance_rate: float,
    target_alignment: dict[str, Any],
    quality_signals: dict[str, Any],
) -> dict[str, Any]:
    target_alignment_review_required = bool(target_alignment.get("alignment_review_required", False))
    repair_review_required = bool(target_alignment.get("repair_review_required", False))
    duplicate_review_required = duplicate_utterance_rate > FINAL_CORPUS_MAX_DUPLICATE_UTTERANCE_RATE
    acceptance_gate = quality_signals.get("acceptance_gate", {})
    stage_direction_review_required = acceptance_gate.get("stage_direction_free") is False
    unknown_policy = quality_signals.get("unknown_masked_placeholder_policy", {})
    unknown_decisions = unknown_policy.get("decision_counts", {})
    unclassified_unknown_count = (
        int(unknown_decisions.get("unclassified_review_required", 0))
        if isinstance(unknown_decisions, dict)
        else 0
    )
    unknown_placeholder_review_required = unclassified_unknown_count > 0
    review_required = (
        target_alignment_review_required
        or repair_review_required
        or duplicate_review_required
        or stage_direction_review_required
        or unknown_placeholder_review_required
    )
    reasons: list[str] = []
    if target_alignment_review_required:
        reasons.append("target_outcome_alignment_below_final_threshold")
    if repair_review_required:
        reasons.append("target_outcome_repair_rate_above_final_threshold")
    if duplicate_review_required:
        reasons.append("duplicate_utterance_rate_above_final_threshold")
    if stage_direction_review_required:
        reasons.append("stage_direction_present")
    if unknown_placeholder_review_required:
        reasons.append("unknown_placeholder_unclassified_review_required")
    return {
        "scope": "final_dialogue_corpus_naturalness_and_label_control",
        "blocking": False,
        "review_required": review_required,
        "reasons": reasons,
        "thresholds": {
            "min_target_outcome_match_rate": FINAL_CORPUS_MIN_TARGET_OUTCOME_MATCH_RATE,
            "max_target_outcome_repair_applied_rate": FINAL_CORPUS_MAX_REPAIR_APPLIED_RATE,
            "max_duplicate_utterance_rate": FINAL_CORPUS_MAX_DUPLICATE_UTTERANCE_RATE,
            "max_stage_direction_count": 0,
            "max_unknown_placeholder_unclassified_count": 0,
        },
        "observed": {
            "target_outcome_match_rate": target_alignment.get("match_rate", 0.0),
            "target_outcome_repair_applied_rate": target_alignment.get("repair_applied_rate", 0.0),
            "duplicate_utterance_rate": duplicate_utterance_rate,
            "stage_direction_count": quality_signals.get("stage_direction_count", 0),
            "unknown_placeholder_unclassified_count": unclassified_unknown_count,
        },
        "interpretation": (
            "review_required means schema/safety may pass while final dialogue corpus quality "
            "still needs inspection or regeneration before being treated as natural final data"
        ),
    }


def summarize_quality_checks(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for key in QUALITY_KEYS:
        passed = 0
        failed = 0
        missing = 0
        for record in records:
            quality = record.get("quality_checks", {})
            if not isinstance(quality, dict) or key not in quality:
                missing += 1
            elif quality[key] is True:
                passed += 1
            else:
                failed += 1
        summary[key] = {"pass": passed, "fail": failed, "missing": missing}
    return summary


def quality_pass_rate(summary: dict[str, dict[str, int]]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for key, counts in summary.items():
        total = counts["pass"] + counts["fail"] + counts["missing"]
        rates[key] = round(counts["pass"] / total, 4) if total else 0.0
    return rates


def find_corpus_residual_patterns(records: list[dict[str, Any]]) -> dict[str, int]:
    masker = SafetyMasker()
    counter: Counter[str] = Counter()
    for record in records:
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        counter.update(masker.find_sensitive_patterns(text))
    return dict(sorted(counter.items()))


def quality_signal_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_direction_count = 0
    mask_usage_anomaly_count = 0
    scenario_persona_mismatch_count = 0
    partial_compliance_signal_count = 0
    unknown_masked_placeholder_counter: Counter[str] = Counter()
    scenario_mismatches: list[dict[str, str]] = []
    partial_compliance_examples: list[dict[str, str]] = []
    unknown_masked_placeholder_examples: list[dict[str, str]] = []
    korean_entity_warning_examples: list[dict[str, str]] = []

    for record in records:
        record_stage_count, record_mask_count = record_quality_signal_counts(record)
        stage_direction_count += record_stage_count
        mask_usage_anomaly_count += record_mask_count
        unknown_placeholders, unknown_examples = record_unknown_masked_placeholders(record)
        unknown_masked_placeholder_counter.update(unknown_placeholders)
        unknown_masked_placeholder_examples.extend(unknown_examples)
        record_partial_count, record_partial_examples = record_partial_compliance_signals(record)
        partial_compliance_signal_count += record_partial_count
        partial_compliance_examples.extend(record_partial_examples)
        korean_entity_warning_examples.extend(record_korean_entity_warnings(record))
        expected = PAIR_SCENARIO_PREFERENCES.get(
            (
                str(record.get("victim_persona", {}).get("id", "")),
                str(record.get("attacker_persona", {}).get("id", "")),
            )
        )
        scenario_type = str(record.get("scenario_type", ""))
        if expected and scenario_type != expected:
            scenario_persona_mismatch_count += 1
            scenario_mismatches.append(
                {
                    "record_id": str(record.get("id", "unknown")),
                    "expected": expected,
                    "observed": scenario_type,
                }
            )

    outcome_distribution = count_values(records, "outcome")
    outcome_diversity_warning = len(outcome_distribution) < 2 if records else False
    return {
        "stage_direction_count": stage_direction_count,
        "mask_usage_anomaly_count": mask_usage_anomaly_count,
        "unknown_masked_placeholder_count": sum(unknown_masked_placeholder_counter.values()),
        "unknown_masked_placeholders": dict(sorted(unknown_masked_placeholder_counter.items())),
        "unknown_masked_placeholder_policy": unknown_masked_placeholder_policy_summary(
            unknown_masked_placeholder_counter
        ),
        "unknown_masked_placeholder_examples": unknown_masked_placeholder_examples[:10],
        "partial_compliance_signal_count": partial_compliance_signal_count,
        "partial_compliance_examples": partial_compliance_examples[:10],
        "scenario_persona_mismatch_count": scenario_persona_mismatch_count,
        "scenario_mismatches": scenario_mismatches[:10],
        "korean_entity_warning_count": len(korean_entity_warning_examples),
        "korean_entity_warnings": korean_entity_warning_examples[:10],
        "outcome_unique_count": len(outcome_distribution),
        "outcome_distribution": outcome_distribution,
        "outcome_diverse_warning_only": outcome_diversity_warning,
        "acceptance_gate": {
            "stage_direction_free": stage_direction_count == 0,
            "mask_usage_anomaly_free": mask_usage_anomaly_count == 0,
            "scenario_persona_match": scenario_persona_mismatch_count == 0,
        },
    }


def unknown_masked_placeholder_policy_summary(counter: Counter[str]) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    placeholders: dict[str, dict[str, Any]] = {}
    for placeholder, count in sorted(counter.items()):
        policy = unknown_masked_placeholder_policy(placeholder)
        decision = str(policy["decision"])
        bucket = placeholder_policy_bucket(placeholder, decision)
        decision_counts[decision] += count
        bucket_counts[bucket] += count
        placeholders[placeholder] = {
            "count": count,
            "decision": decision,
            "bucket": bucket,
            "reason": str(policy["reason"]),
        }
        normalize_to = policy.get("normalize_to")
        if normalize_to:
            placeholders[placeholder]["normalize_to"] = str(normalize_to)
    return {
        "scope": "observed_unknown_masked_placeholders",
        "blocking": False,
        "decision_counts": dict(sorted(decision_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "placeholders": placeholders,
    }


def unknown_masked_placeholder_policy(placeholder: str) -> dict[str, str]:
    explicit = UNKNOWN_MASKED_PLACEHOLDER_POLICY.get(placeholder)
    if explicit:
        return explicit
    normalize_to = SafetyMasker.normalize_masked_placeholder_alias(placeholder)
    if normalize_to:
        return {
            "decision": "normalize_to_existing",
            "normalize_to": normalize_to,
            "reason": "observed placeholder matches the reviewed placeholder normalization heuristic",
        }
    return {
        "decision": "unclassified_review_required",
        "reason": "observed placeholder has not been reviewed",
    }


def placeholder_policy_bucket(placeholder: str, legacy_decision: str) -> str:
    if legacy_decision == "normalize_to_existing":
        return "normalize"
    if legacy_decision == "unclassified_review_required":
        if any(pattern in placeholder for pattern in PLACEHOLDER_REVIEWED_SENSITIVE_PATTERNS):
            return "reviewed_sensitive"
        return "reject_or_escalate"
    if any(pattern in placeholder for pattern in PLACEHOLDER_REVIEWED_SENSITIVE_PATTERNS):
        return "reviewed_sensitive"
    if legacy_decision == "allowlist_candidate":
        return "allowlist"
    return "reject_or_escalate"


def record_quality_signal_counts(record: dict[str, Any]) -> tuple[int, int]:
    stage_direction_count = 0
    mask_usage_anomaly_count = 0
    for turn in record.get("dialogue", []):
        if not isinstance(turn, dict):
            continue
        utterance = str(turn.get("utterance", ""))
        stage_direction_count += len(STAGE_DIRECTION_PATTERN.findall(utterance))
        if OTP_TRANSFER_ANOMALY_PATTERN.search(utterance):
            mask_usage_anomaly_count += 1
    return stage_direction_count, mask_usage_anomaly_count


def record_unknown_masked_placeholders(record: dict[str, Any]) -> tuple[Counter[str], list[dict[str, str]]]:
    counter: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    for turn in record.get("dialogue", []):
        if not isinstance(turn, dict):
            continue
        utterance = str(turn.get("utterance", ""))
        unknowns = [placeholder for placeholder in MASKED_PLACEHOLDER_PATTERN.findall(utterance) if placeholder not in STANDARD_MASKED_PLACEHOLDERS]
        if not unknowns:
            continue
        counter.update(unknowns)
        examples.append(
            {
                "record_id": str(record.get("id", "unknown")),
                "turn_id": str(turn.get("turn_id", "unknown")),
                "placeholders": ", ".join(sorted(set(unknowns))),
                "utterance": truncate_for_report(utterance),
            }
        )
    return counter, examples


def record_partial_compliance_signals(record: dict[str, Any]) -> tuple[int, list[dict[str, str]]]:
    count = 0
    examples: list[dict[str, str]] = []
    for turn in record.get("dialogue", []):
        if not isinstance(turn, dict) or turn.get("speaker") != "victim":
            continue
        utterance = str(turn.get("utterance", ""))
        if not has_partial_compliance_signal(utterance):
            continue
        count += 1
        examples.append(
            {
                "record_id": str(record.get("id", "unknown")),
                "turn_id": str(turn.get("turn_id", "unknown")),
                "utterance": truncate_for_report(utterance),
            }
        )
    return count, examples


def record_korean_entity_warnings(record: dict[str, Any]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for turn in record.get("dialogue", []):
        if not isinstance(turn, dict):
            continue
        utterance = str(turn.get("utterance", ""))
        searchable = MASKED_PLACEHOLDER_PATTERN.sub("", utterance)
        matches = sorted(set(KOREAN_ENTITY_WARNING_PATTERN.findall(searchable)))
        if not matches:
            continue
        examples.append(
            {
                "record_id": str(record.get("id", "unknown")),
                "turn_id": str(turn.get("turn_id", "unknown")),
                "matches": ", ".join(matches[:5]),
                "utterance": truncate_for_report(utterance),
            }
        )
    return examples


def truncate_for_report(text: str, limit: int = 160) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def format_markdown_report(summary: dict[str, Any]) -> str:
    sections = [
        "# QA Report",
        "",
        f"- record_count: {summary['record_count']}",
        f"- average_turn_count: {summary['average_turn_count']}",
        f"- duplicate_utterance_count: {summary['duplicate_utterance_count']}",
        f"- duplicate_utterance_rate: {summary['duplicate_utterance_rate']}",
        f"- duplicate_record_id_count: {summary.get('duplicate_record_id_count', 0)}",
        "",
        "## Source Metadata",
        "",
        render_source_metadata(summary.get("source_metadata", {})),
        "## Duplicate Utterance Metric",
        "",
        render_duplicate_metric(summary.get("duplicate_utterance_metric", DUPLICATE_UTTERANCE_METRIC)),
        "## Duplicate Record Ids",
        "",
        render_duplicate_record_ids(summary.get("duplicate_record_ids", {})),
        "## Quality Checks",
        "",
        render_quality_summary(summary["quality_check_summary"], summary["quality_pass_rate"]),
        "## Schema Contract Validation",
        "",
        render_schema_contract_validation(summary["schema_contract_validation"]),
        "## Quality Signals",
        "",
        render_quality_signals(summary["quality_signals"]),
        "## Attack-Only Policy",
        "",
        render_attack_only_policy(summary["attack_only_policy"]),
        "## Scenario Default Attacker Risk Signal Coverage",
        "",
        render_scenario_risk_alignment(summary["scenario_default_attacker_risk_signal_coverage"]),
        "## Attacker Pressure Cue Diversity",
        "",
        render_attacker_pressure_cue_diversity(summary["attacker_pressure_cue_diversity"]),
        "## Assigned Pressure Style Alignment",
        "",
        render_assigned_pressure_style_alignment(summary["assigned_pressure_style_alignment"]),
        "## Target Outcome Mode Alignment",
        "",
        render_target_outcome_mode_alignment(summary["target_outcome_mode_alignment"]),
        "## Final Dialogue Corpus Review",
        "",
        render_final_dialogue_corpus_review(summary["final_dialogue_corpus_review"]),
        "## Environmental Context Coverage",
        "",
        render_environmental_context_coverage(summary["environmental_context_coverage"]),
        "## Residual Sensitive Patterns",
        "",
        render_distribution("corpus residual patterns", summary["corpus_residual_sensitive_patterns"]),
        "## Distributions",
        "",
        render_distribution("label", summary["label_distribution"]),
        render_distribution("outcome", summary["outcome_distribution"]),
        render_distribution("scenario", summary["scenario_distribution"]),
        render_distribution("split", summary["split_distribution"]),
        render_distribution("victim", summary["victim_distribution"]),
        render_distribution("attacker", summary["attacker_distribution"]),
        render_distribution("masks", summary["mask_counts"]),
    ]
    return "\n".join(sections).rstrip() + "\n"


def render_duplicate_metric(metric: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- scope: {metric.get('scope', DUPLICATE_UTTERANCE_METRIC['scope'])}",
            f"- normalization: {metric.get('normalization', DUPLICATE_UTTERANCE_METRIC['normalization'])}",
            f"- interpretation: {metric.get('interpretation', DUPLICATE_UTTERANCE_METRIC['interpretation'])}",
            "",
        ]
    )


def render_source_metadata(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- source_path: {metadata.get('source_path')}",
            f"- jsonl_line_count: {metadata.get('jsonl_line_count')}",
            f"- filename_count_hint: {metadata.get('filename_count_hint')}",
            f"- ignored_date_like_tokens: {metadata.get('ignored_date_like_tokens', [])}",
            f"- count_mismatch_warning: {str(metadata.get('count_mismatch_warning', False)).lower()}",
            f"- line_count_mismatch_warning: {str(metadata.get('line_count_mismatch_warning', False)).lower()}",
            "",
        ]
    )


def render_duplicate_record_ids(summary: dict[str, Any]) -> str:
    rows = [
        f"- scope: {summary.get('scope', 'dataset_record_id')}",
        f"- blocking: {str(summary.get('blocking', False)).lower()}",
        f"- duplicate_count: {summary.get('duplicate_count', 0)}",
    ]
    duplicate_ids = summary.get("duplicate_ids", {})
    if duplicate_ids:
        rows.append("- duplicate_ids:")
        for record_id, count in sorted(duplicate_ids.items()):
            rows.append(f"  - {record_id}: {count}")
    rows.append("")
    return "\n".join(rows)


def render_quality_summary(summary: dict[str, dict[str, int]], rates: dict[str, float]) -> str:
    if not summary:
        return "- none: 0\n"
    rows = ["| check | pass | fail | missing | pass_rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for key, counts in summary.items():
        rows.append(
            f"| {key} | {counts['pass']} | {counts['fail']} | {counts['missing']} | {rates.get(key, 0.0)} |"
        )
    rows.append("")
    return "\n".join(rows)


def render_schema_contract_validation(validation: dict[str, Any]) -> str:
    rows = [
        f"- enabled: {str(validation.get('enabled', False)).lower()}",
        f"- pass: {validation.get('pass', 0)}",
        f"- fail: {validation.get('fail', 0)}",
        f"- missing: {validation.get('missing', 0)}",
    ]
    failures = validation.get("failures", [])
    if failures:
        rows.append("- failures:")
        for failure in failures[:5]:
            rows.append(
                f"  - index={failure.get('index', 'unknown')} "
                f"id={failure.get('record_id', 'unknown')}: {failure.get('error', '')}"
            )
    rows.append("")
    return "\n".join(rows)


def render_quality_signals(signals: dict[str, Any]) -> str:
    gate = signals.get("acceptance_gate", {})
    rows = [
        f"- stage_direction_count: {signals.get('stage_direction_count', 0)}",
        f"- mask_usage_anomaly_count: {signals.get('mask_usage_anomaly_count', 0)}",
        f"- unknown_masked_placeholder_count: {signals.get('unknown_masked_placeholder_count', 0)}",
        f"- partial_compliance_signal_count: {signals.get('partial_compliance_signal_count', 0)}",
        f"- scenario_persona_mismatch_count: {signals.get('scenario_persona_mismatch_count', 0)}",
        f"- korean_entity_warning_count: {signals.get('korean_entity_warning_count', 0)}",
        f"- outcome_unique_count: {signals.get('outcome_unique_count', 0)}",
        f"- outcome_diverse_warning_only: {str(signals.get('outcome_diverse_warning_only', False)).lower()}",
        f"- gate_stage_direction_free: {str(gate.get('stage_direction_free', False)).lower()}",
        f"- gate_mask_usage_anomaly_free: {str(gate.get('mask_usage_anomaly_free', False)).lower()}",
        f"- gate_scenario_persona_match: {str(gate.get('scenario_persona_match', False)).lower()}",
        "",
    ]
    unknown_placeholders = signals.get("unknown_masked_placeholders", {})
    if unknown_placeholders:
        rows.append("- unknown_masked_placeholders:")
        for placeholder, count in sorted(unknown_placeholders.items()):
            rows.append(f"  - {placeholder}: {count}")
        rows.append("")
    unknown_policy = signals.get("unknown_masked_placeholder_policy", {})
    if unknown_policy and unknown_policy.get("placeholders"):
        rows.append("- unknown_masked_placeholder_policy:")
        rows.append(f"  - scope: {unknown_policy.get('scope', 'observed_unknown_masked_placeholders')}")
        rows.append(f"  - blocking: {str(unknown_policy.get('blocking', False)).lower()}")
        decision_counts = unknown_policy.get("decision_counts", {})
        if decision_counts:
            rows.append("  - decision_counts:")
            for decision, count in sorted(decision_counts.items()):
                rows.append(f"    - {decision}: {count}")
        bucket_counts = unknown_policy.get("bucket_counts", {})
        if bucket_counts:
            rows.append("  - bucket_counts:")
            for bucket, count in sorted(bucket_counts.items()):
                rows.append(f"    - {bucket}: {count}")
        placeholders = unknown_policy.get("placeholders", {})
        if placeholders:
            rows.append("  - placeholders:")
            for placeholder, policy in sorted(placeholders.items()):
                normalize_to = policy.get("normalize_to")
                suffix = f" normalize_to={normalize_to}" if normalize_to else ""
                rows.append(
                    f"    - {placeholder}: count={policy.get('count', 0)} "
                    f"decision={policy.get('decision', '')} bucket={policy.get('bucket', '')}{suffix}"
                )
        rows.append("")
    unknown_examples = signals.get("unknown_masked_placeholder_examples", [])
    if unknown_examples:
        rows.append("- unknown_masked_placeholder_examples:")
        for example in unknown_examples[:5]:
            rows.append(
                f"  - id={example.get('record_id', 'unknown')} turn={example.get('turn_id', 'unknown')} "
                f"placeholders={example.get('placeholders', '')}: {example.get('utterance', '')}"
            )
        rows.append("")
    partial_examples = signals.get("partial_compliance_examples", [])
    if partial_examples:
        rows.append("- partial_compliance_examples:")
        for example in partial_examples[:5]:
            rows.append(
                f"  - id={example.get('record_id', 'unknown')} turn={example.get('turn_id', 'unknown')}: "
                f"{example.get('utterance', '')}"
            )
        rows.append("")
    mismatches = signals.get("scenario_mismatches", [])
    if mismatches:
        rows.append("- scenario_mismatches:")
        for mismatch in mismatches[:5]:
            rows.append(
                f"  - id={mismatch.get('record_id', 'unknown')} "
                f"expected={mismatch.get('expected', '')} observed={mismatch.get('observed', '')}"
            )
        rows.append("")
    korean_warnings = signals.get("korean_entity_warnings", [])
    if korean_warnings:
        rows.append("- korean_entity_warnings:")
        for warning in korean_warnings[:5]:
            rows.append(
                f"  - id={warning.get('record_id', 'unknown')} turn={warning.get('turn_id', 'unknown')} "
                f"matches={warning.get('matches', '')}: {warning.get('utterance', '')}"
            )
        rows.append("")
    return "\n".join(rows)


def render_attack_only_policy(policy: dict[str, Any]) -> str:
    status = "pass" if policy.get("pass") is True else "fail"
    return "\n".join(
        [
            f"- expected_label: {policy.get('expected_label', 'attack')}",
            f"- status: {status}",
            f"- non_attack_count: {policy.get('non_attack_count', 0)}",
            "",
        ]
    )


def render_scenario_risk_alignment(alignment: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- scope: {alignment.get('scope', 'record_level_descriptive_scenario_defaults_vs_observed_attacker_turn_risk_signal_set')}",
            f"- blocking: {str(alignment.get('blocking', False)).lower()}",
            f"- record_count: {alignment.get('record_count', 0)}",
            f"- match_count: {alignment.get('match_count', 0)}",
            f"- mismatch_count: {alignment.get('mismatch_count', 0)}",
            f"- mismatch_rate: {alignment.get('mismatch_rate', 0.0)}",
            f"- missing_attacker_risk_count: {alignment.get('missing_attacker_risk_count', 0)}",
            f"- interpretation: {alignment.get('interpretation', '')}",
            "",
        ]
    )


def render_attacker_pressure_cue_diversity(summary: dict[str, Any]) -> str:
    rows = [
        f"- scope: {summary.get('scope', 'attacker_turn_text_pressure_cue_categories')}",
        f"- blocking: {str(summary.get('blocking', False)).lower()}",
        f"- attacker_turn_count: {summary.get('attacker_turn_count', 0)}",
        f"- categorized_attacker_turn_count: {summary.get('categorized_attacker_turn_count', 0)}",
        f"- uncategorized_attacker_turn_count: {summary.get('uncategorized_attacker_turn_count', 0)}",
        f"- unique_pressure_cue_count: {summary.get('unique_pressure_cue_count', 0)}",
        f"- dominant_pressure_cue: {summary.get('dominant_pressure_cue', '')}",
        f"- dominant_pressure_cue_rate: {summary.get('dominant_pressure_cue_rate', 0.0)}",
        f"- urgency_turn_count: {summary.get('urgency_turn_count', 0)}",
        f"- urgency_turn_rate: {summary.get('urgency_turn_rate', 0.0)}",
        f"- urgency_dominant_warning_only: {str(summary.get('urgency_dominant_warning_only', False)).lower()}",
        f"- interpretation: {summary.get('interpretation', '')}",
        "",
    ]
    category_counts = summary.get("category_counts", {})
    if category_counts:
        rows.append("- category_counts:")
        for category, count in sorted(category_counts.items()):
            rows.append(f"  - {category}: {count}")
        rows.append("")
    examples = summary.get("examples", [])
    if examples:
        rows.append("- examples:")
        for example in examples[:5]:
            rows.append(
                f"  - id={example.get('record_id', 'unknown')} turn={example.get('turn_id', 'unknown')} "
                f"categories={example.get('categories', '')}: {example.get('utterance', '')}"
            )
        rows.append("")
    return "\n".join(rows)


def render_assigned_pressure_style_alignment(summary: dict[str, Any]) -> str:
    rows = [
        f"- scope: {summary.get('scope', 'record_level_assigned_pressure_style_vs_observed_attacker_cues')}",
        f"- blocking: {str(summary.get('blocking', False)).lower()}",
        f"- assigned_record_count: {summary.get('assigned_record_count', 0)}",
        f"- observed_match_count: {summary.get('observed_match_count', 0)}",
        f"- observed_match_rate: {summary.get('observed_match_rate', 0.0)}",
        f"- urgency_leakage_count: {summary.get('urgency_leakage_count', 0)}",
        f"- urgency_leakage_rate: {summary.get('urgency_leakage_rate', 0.0)}",
        "",
    ]
    for name in ("assigned_distribution", "observed_distribution"):
        values = summary.get(name, {})
        if values:
            rows.append(f"- {name}:")
            for key, count in sorted(values.items()):
                rows.append(f"  - {key}: {count}")
            rows.append("")
    examples = summary.get("mismatch_examples", [])
    if examples:
        rows.append("- mismatch_examples:")
        for example in examples[:5]:
            rows.append(
                f"  - id={example.get('record_id', 'unknown')} assigned={example.get('assigned', '')} "
                f"observed={example.get('observed', '')}: {example.get('utterance', '')}"
            )
        rows.append("")
    return "\n".join(rows)


def render_target_outcome_mode_alignment(summary: dict[str, Any]) -> str:
    rows = [
        f"- scope: {summary.get('scope', 'record_level_target_outcome_mode_vs_inferred_outcome')}",
        f"- blocking: {str(summary.get('blocking', False)).lower()}",
        f"- assigned_record_count: {summary.get('assigned_record_count', 0)}",
        f"- match_count: {summary.get('match_count', 0)}",
        f"- match_rate: {summary.get('match_rate', 0.0)}",
        f"- alignment_review_required: {str(summary.get('alignment_review_required', False)).lower()}",
        f"- repair_applied_count: {summary.get('repair_applied_count', 0)}",
        f"- repair_applied_rate: {summary.get('repair_applied_rate', 0.0)}",
        f"- repair_review_required: {str(summary.get('repair_review_required', False)).lower()}",
        f"- llm_rewrite_applied_count: {summary.get('llm_rewrite_applied_count', 0)}",
        f"- llm_rewrite_applied_rate: {summary.get('llm_rewrite_applied_rate', 0.0)}",
        f"- retry_success_count: {summary.get('retry_success_count', 0)}",
        "",
    ]
    for name in (
        "target_distribution",
        "observed_distribution",
        "repair_applied_by_target",
        "llm_rewrite_applied_by_target",
        "generation_attempt_distribution",
        "llm_rewrite_attempt_distribution",
    ):
        values = summary.get(name, {})
        if values:
            rows.append(f"- {name}:")
            for key, count in sorted(values.items()):
                rows.append(f"  - {key}: {count}")
            rows.append("")
    examples = summary.get("mismatch_examples", [])
    if examples:
        rows.append("- mismatch_examples:")
        for example in examples[:5]:
            rows.append(
                f"  - id={example.get('record_id', 'unknown')} "
                f"target={example.get('target_outcome_mode', '')} "
                f"expected={example.get('expected_outcome', '')} observed={example.get('observed_outcome', '')}"
            )
        rows.append("")
    return "\n".join(rows)


def render_final_dialogue_corpus_review(review: dict[str, Any]) -> str:
    rows = [
        f"- scope: {review.get('scope', 'final_dialogue_corpus_naturalness_and_label_control')}",
        f"- blocking: {str(review.get('blocking', False)).lower()}",
        f"- review_required: {str(review.get('review_required', False)).lower()}",
    ]
    reasons = review.get("reasons", [])
    if reasons:
        rows.append("- reasons:")
        for reason in reasons:
            rows.append(f"  - {reason}")
    thresholds = review.get("thresholds", {})
    if thresholds:
        rows.append("- thresholds:")
        for key, value in sorted(thresholds.items()):
            rows.append(f"  - {key}: {value}")
    observed = review.get("observed", {})
    if observed:
        rows.append("- observed:")
        for key, value in sorted(observed.items()):
            rows.append(f"  - {key}: {value}")
    interpretation = review.get("interpretation")
    if interpretation:
        rows.append(f"- interpretation: {interpretation}")
    return "\n".join(rows)


def render_environmental_context_coverage(coverage: dict[str, Any]) -> str:
    rows = [
        f"- record_count: {coverage.get('record_count', 0)}",
        f"- with_environmental_context: {coverage.get('with_environmental_context', 0)}",
        f"- missing_environmental_context: {coverage.get('missing_environmental_context', 0)}",
        "",
        "| field | present | missing |",
        "| --- | ---: | ---: |",
    ]
    field_coverage = coverage.get("field_coverage", {})
    for key in ENVIRONMENTAL_CONTEXT_KEYS:
        counts = field_coverage.get(key, {})
        rows.append(f"| {key} | {counts.get('present', 0)} | {counts.get('missing', 0)} |")
    rows.append("")

    missing_records = coverage.get("missing_records", [])
    if missing_records:
        rows.append("- missing_records:")
        for missing in missing_records[:5]:
            rows.append(f"  - id={missing.get('record_id', 'unknown')}: {missing.get('missing', '')}")
        rows.append("")

    distributions = coverage.get("distributions", {})
    variant_distribution = coverage.get("variant_distribution", {})
    if variant_distribution:
        rows.append(render_distribution("environmental_context_variant_id", variant_distribution))
    for key in ENVIRONMENTAL_CONTEXT_KEYS:
        values = distributions.get(key)
        if values:
            rows.append(render_distribution(key, values))
    rows.append("")
    return "\n".join(rows)


def render_distribution(name: str, values: dict[str, int]) -> str:
    if not values:
        return f"### {name}\n\n- none: 0\n"
    rows = [f"### {name}", ""]
    rows.extend(f"- {key}: {value}" for key, value in values.items())
    rows.append("")
    return "\n".join(rows)


def write_markdown_report(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    max_turns: int | None = None,
    allowed_risk_labels: list[str] | set[str] | tuple[str, ...] | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = summarize_records(
        records,
        max_turns=max_turns,
        allowed_risk_labels=allowed_risk_labels,
        source_path=source_path,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown_report(summary), encoding="utf-8")
    return summary


def write_json_report(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    max_turns: int | None = None,
    allowed_risk_labels: list[str] | set[str] | tuple[str, ...] | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    summary = summarize_records(
        records,
        max_turns=max_turns,
        allowed_risk_labels=allowed_risk_labels,
        source_path=source_path,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
