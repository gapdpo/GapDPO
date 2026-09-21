from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

REPEATED_NOUN_RE = re.compile(r"(알림|안내|확인|처리|경로|절차|요청)(\s+\1)+")
PLACEHOLDER_RE = re.compile(r"\[MASKED_([A-Z0-9_]+)\]")
MASKED_TOKEN_RE = re.compile(r"\[MASKED_[A-Z0-9_]+\]")

SURFACE_LEXICON = (
    "공식",
    "저장된",
    "직접",
    "링크",
    "경로",
    "접속",
    "제공",
    "입력",
    "말씀",
    "전달",
    "공유",
    "확인",
    "필요",
    "긴급",
    "즉시",
    "오늘",
    "마감",
    "지연",
    "보류",
    "제한",
    "중단",
    "신고",
    "기록",
    "추가",
    "같은 방식",
    "새로운",
    "아직",
    "설명",
    "기다리",
    "처리",
)
CATEGORY_LEAKAGE_TERMS = (
    "계정 보안",
    "계정 확인",
    "업무 처리",
    "가족 연락",
    "가족 긴급",
    "보안 알림",
    "접수 상태",
    "정산",
)
INSTRUCTION_RESIDUE_MARKERS = (
    "보류하세요",
    "답하세요",
    "설명하세요",
    "요구하세요",
    "마무리하세요",
    "선택하세요",
    "제시하세요",
    "말하세요",
    "망설이세요",
    "끝내세요",
)
VERY_SHORT_UTTERANCES = (
    "기다리겠습니다",
    "보류하겠습니다",
    "설명만 듣겠습니다",
)


def build_shortcut_report(
    public_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_by_id = {row.get("benchmark_id"): row for row in manifest_rows}
    public_by_id = {row.get("benchmark_id"): row for row in public_rows}

    profile_position_pairs: list[tuple[tuple[str, int], str]] = []
    position_pairs: list[tuple[int, str]] = []
    stream_events: dict[str, list[tuple[int, str]]] = defaultdict(list)
    stream_profile: dict[str, str] = {}
    event_actions: dict[str, Counter[str]] = defaultdict(Counter)
    action_examples_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    variant_surface_examples: list[dict[str, Any]] = []
    fallback_pairs: list[tuple[str, str]] = []
    fallback_strings: Counter[str] = Counter()
    fallback_events_by_string: dict[str, Counter[str]] = defaultdict(Counter)
    non_fallback_pairs: list[tuple[str, str]] = []
    non_fallback_strings: Counter[str] = Counter()
    non_fallback_events_by_string: dict[str, Counter[str]] = defaultdict(Counter)
    category_leakage = []
    placeholder_only_turns = []
    very_short_turns = []
    instruction_residue_turns = []

    for gold in gold_rows:
        benchmark_id = gold.get("benchmark_id")
        manifest = manifest_by_id.get(benchmark_id, {})
        public = public_by_id.get(benchmark_id, {})
        event = str(gold.get("event", ""))
        action = str(gold.get("action", ""))
        position = int(manifest.get("stream_position") or public.get("stream_position") or 0)
        profile = str(manifest.get("stream_profile") or "unknown")
        stream_id = str(gold.get("stream_id", ""))
        current_text = current_turn_text(public)
        is_fallback = bool(gold.get("is_fallback_turn")) or bool(manifest.get("is_fallback_turn"))
        surface_feature = public_surface_feature(public, manifest)

        profile_position_pairs.append(((profile, position), event))
        position_pairs.append((position, event))
        stream_events[stream_id].append((position, event))
        stream_profile.setdefault(stream_id, profile)
        event_actions[event][action] += 1
        action_examples_by_event[event].append(
            {"feature": action_surface_feature(public, manifest), "label": action, "split_key": stream_id}
        )
        variant = manifest.get("trajectory_variant")
        if not is_fallback and variant not in (None, "", "canonical"):
            variant_surface_examples.append(
                {"feature": surface_feature, "label": str(variant), "split_key": stream_id}
            )
        if is_fallback:
            fallback_pairs.append((current_text, event))
            fallback_strings[current_text] += 1
            fallback_events_by_string[current_text][event] += 1
        else:
            non_fallback_pairs.append((current_text, event))
            non_fallback_strings[current_text] += 1
            non_fallback_events_by_string[current_text][event] += 1
        artifact = utterance_artifact(current_text)
        if artifact == "placeholder_only_utterance":
            placeholder_only_turns.append(artifact_row(benchmark_id, current_text, event))
        elif artifact == "very_short_utterance":
            very_short_turns.append(artifact_row(benchmark_id, current_text, event))
        elif artifact == "instruction_residue_in_utterance":
            instruction_residue_turns.append(artifact_row(benchmark_id, current_text, event))
        leakage_terms = sorted(term for term in CATEGORY_LEAKAGE_TERMS if term in current_text)
        if leakage_terms:
            category_leakage.append(
                {
                    "benchmark_id": benchmark_id,
                    "terms": leakage_terms,
                    "utterance": current_text,
                }
            )

    sequences_by_profile: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for stream_id, rows in stream_events.items():
        sequence = tuple(event for _, event in sorted(rows))
        sequences_by_profile[stream_profile.get(stream_id, "unknown")].add(sequence)

    event_action_dominance = {
        event: {
            "count": sum(actions.values()),
            "dominant_action": actions.most_common(1)[0][0],
            "dominant_action_count": actions.most_common(1)[0][1],
            "dominant_action_ratio": actions.most_common(1)[0][1] / sum(actions.values()),
            "action_distribution": dict(sorted(actions.items())),
        }
        for event, actions in sorted(event_actions.items())
        if actions
    }
    event_action_is_fully_deterministic = all(
        row["dominant_action_ratio"] == 1.0 for row in event_action_dominance.values()
    )
    max_event_action_dominance = max(
        (row["dominant_action_ratio"] for row in event_action_dominance.values()),
        default=0.0,
    )
    repeated_artifacts = repeated_noun_artifacts(public_rows)
    profile_unique_counts = {
        profile: len(sequences)
        for profile, sequences in sorted(sequences_by_profile.items())
    }
    min_profile_unique_sequences = min(profile_unique_counts.values()) if profile_unique_counts else 0
    profile_position_accuracy = majority_accuracy(profile_position_pairs)
    position_accuracy = majority_accuracy(position_pairs)
    fallback_accuracy = majority_accuracy(fallback_pairs)
    repeated_fallback_pairs = [
        (text, event)
        for text, event in fallback_pairs
        if fallback_strings[text] >= 2
    ]
    repeated_fallback_accuracy = majority_accuracy(repeated_fallback_pairs)
    high_count_single_event_fallback_strings = [
        {
            "utterance": text,
            "count": fallback_strings[text],
            "event": events.most_common(1)[0][0],
        }
        for text, events in fallback_events_by_string.items()
        if fallback_strings[text] >= 5 and len(events) == 1
    ]
    high_count_single_event_fallback_strings.sort(key=lambda row: (-int(row["count"]), row["utterance"]))
    repeated_non_fallback_pairs = [
        (text, event)
        for text, event in non_fallback_pairs
        if non_fallback_strings[text] >= 2
    ]
    high_count_single_event_non_fallback_strings = [
        {
            "utterance": text,
            "count": non_fallback_strings[text],
            "event": events.most_common(1)[0][0],
        }
        for text, events in non_fallback_events_by_string.items()
        if non_fallback_strings[text] >= 5 and len(events) == 1
    ]
    high_count_single_event_non_fallback_strings.sort(key=lambda row: (-int(row["count"]), row["utterance"]))
    fallback_turn_rate = len(fallback_pairs) / len(gold_rows) if gold_rows else 0.0
    fallback_quality_gate_applicable = 0 < len(fallback_pairs) < len(gold_rows)
    repeated_fallback_gate_applicable = (
        fallback_quality_gate_applicable and len(repeated_fallback_pairs) >= 20
    )
    variant_surface_model = train_test_majority_accuracy(variant_surface_examples)
    variant_surface_gate_applicable = (
        variant_surface_model["support"] >= 200 and variant_surface_model["unique_label_count"] >= 2
    )
    action_branch_report = {
        event: train_test_majority_accuracy(examples)
        for event, examples in sorted(action_examples_by_event.items())
        if len({example["label"] for example in examples}) >= 2 and len(examples) >= 30
    }
    low_action_branch_signal_events = [
        event
        for event, row in action_branch_report.items()
        if row["test_count"] >= 10 and row["accuracy"] < 0.60
    ]

    gate_checks = {
        "min_profile_unique_event_sequences_at_least_4": min_profile_unique_sequences >= 4,
        "profile_position_event_accuracy_below_0_90": profile_position_accuracy < 0.90,
        "event_action_not_fully_deterministic": not event_action_is_fully_deterministic,
        "max_event_action_dominance_below_0_85": max_event_action_dominance < 0.85,
        "low_action_branch_signal_events_is_empty": len(low_action_branch_signal_events) == 0,
        "variant_surface_event_accuracy_below_0_45_or_skipped": (
            not variant_surface_gate_applicable or variant_surface_model["accuracy"] < 0.45
        ),
        "fallback_turn_rate_below_0_08_or_skipped": (
            not fallback_quality_gate_applicable or fallback_turn_rate < 0.08
        ),
        "repeated_fallback_string_event_accuracy_below_0_80_or_skipped": (
            not repeated_fallback_gate_applicable or repeated_fallback_accuracy < 0.80
        ),
        "high_count_single_event_fallback_strings_is_zero_or_skipped": (
            not fallback_quality_gate_applicable or len(high_count_single_event_fallback_strings) == 0
        ),
        "placeholder_only_turn_count_is_zero": len(placeholder_only_turns) == 0,
        "very_short_turn_count_is_zero": len(very_short_turns) == 0,
        "instruction_residue_turn_count_is_zero": len(instruction_residue_turns) == 0,
        "high_count_single_event_non_fallback_strings_is_zero": (
            len(high_count_single_event_non_fallback_strings) == 0
        ),
        "repeated_noun_artifact_count_is_zero": len(repeated_artifacts) == 0,
        "rejected_count_is_zero": len(rejected_rows) == 0,
    }

    return {
        "scope": "stream_v2_shortcut_diagnostics",
        "profile_position_majority_event_accuracy": profile_position_accuracy,
        "position_only_majority_event_accuracy": position_accuracy,
        "profile_unique_event_sequence_counts": profile_unique_counts,
        "min_profile_unique_event_sequences": min_profile_unique_sequences,
        "event_action_is_fully_deterministic": event_action_is_fully_deterministic,
        "max_event_action_dominance": max_event_action_dominance,
        "event_action_dominance": event_action_dominance,
        "action_branch_surface_accuracy_by_event": action_branch_report,
        "low_action_branch_signal_events": low_action_branch_signal_events,
        "variant_surface_model": variant_surface_model,
        "variant_surface_gate_applicable": variant_surface_gate_applicable,
        "fallback_turn_count": len(fallback_pairs),
        "fallback_turn_rate": fallback_turn_rate,
        "fallback_quality_gate_applicable": fallback_quality_gate_applicable,
        "fallback_string_majority_event_accuracy": fallback_accuracy,
        "fallback_unique_string_count": len(fallback_strings),
        "repeated_fallback_string_count": sum(1 for count in fallback_strings.values() if count >= 2),
        "repeated_fallback_pair_count": len(repeated_fallback_pairs),
        "repeated_fallback_string_majority_event_accuracy": repeated_fallback_accuracy,
        "repeated_fallback_gate_applicable": repeated_fallback_gate_applicable,
        "high_count_single_event_fallback_string_count": len(high_count_single_event_fallback_strings),
        "high_count_single_event_fallback_strings": high_count_single_event_fallback_strings[:20],
        "top_fallback_strings": [
            {"utterance": text, "count": count}
            for text, count in fallback_strings.most_common(20)
        ],
        "repeated_non_fallback_string_count": sum(1 for count in non_fallback_strings.values() if count >= 2),
        "repeated_non_fallback_pair_count": len(repeated_non_fallback_pairs),
        "high_count_single_event_non_fallback_string_count": len(high_count_single_event_non_fallback_strings),
        "high_count_single_event_non_fallback_strings": high_count_single_event_non_fallback_strings[:20],
        "placeholder_only_turn_count": len(placeholder_only_turns),
        "placeholder_only_turn_examples": placeholder_only_turns[:20],
        "very_short_turn_count": len(very_short_turns),
        "very_short_turn_examples": very_short_turns[:20],
        "instruction_residue_turn_count": len(instruction_residue_turns),
        "instruction_residue_turn_examples": instruction_residue_turns[:20],
        "category_leakage_turn_count": len(category_leakage),
        "category_leakage_examples": category_leakage[:20],
        "repeated_noun_artifact_count": len(repeated_artifacts),
        "repeated_noun_artifact_examples": repeated_artifacts[:20],
        "gate_checks": gate_checks,
        "shortcut_gate_passed": all(gate_checks.values()),
    }


def artifact_row(benchmark_id: Any, utterance: str, event: str) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "event": event,
        "utterance": utterance,
    }


def strip_masked_tokens(text: str) -> str:
    return MASKED_TOKEN_RE.sub(" ", text)


def substantive_korean_length(text: str) -> int:
    return len(re.findall(r"[가-힣]", strip_masked_tokens(text)))


def is_placeholder_only(text: str) -> bool:
    return not re.search(r"[가-힣A-Za-z0-9]", strip_masked_tokens(text))


def utterance_artifact(text: str) -> str | None:
    if is_placeholder_only(text):
        return "placeholder_only_utterance"
    if any(marker in text for marker in INSTRUCTION_RESIDUE_MARKERS):
        return "instruction_residue_in_utterance"
    if any(marker in text for marker in VERY_SHORT_UTTERANCES) or substantive_korean_length(text) < 9:
        return "very_short_utterance"
    return None


def current_turn_text(public: dict[str, Any]) -> str:
    window = public.get("input", {}).get("dialogue_window", [])
    if not window:
        return ""
    return str(window[-1].get("utterance", ""))


def public_surface_feature(public: dict[str, Any], manifest: dict[str, Any]) -> tuple[Any, ...]:
    window = public.get("input", {}).get("dialogue_window", [])
    current = window[-1] if window else {}
    text = " ".join(str(turn.get("utterance", "")) for turn in window)
    current_text = str(current.get("utterance", ""))
    position = int(manifest.get("stream_position") or public.get("stream_position") or 0)
    speaker = str(manifest.get("current_speaker") or current.get("speaker") or "")
    placeholder_types = tuple(sorted(set(PLACEHOLDER_RE.findall(text))))
    current_placeholder_types = tuple(sorted(set(PLACEHOLDER_RE.findall(current_text))))
    lexicon_hits = tuple(term for term in SURFACE_LEXICON if term in text)
    current_hits = tuple(term for term in SURFACE_LEXICON if term in current_text)
    length_bucket = min(len(current_text) // 18, 5)
    window_turns = len(window)
    return (
        position,
        speaker,
        placeholder_types,
        current_placeholder_types,
        lexicon_hits,
        current_hits,
        length_bucket,
        window_turns,
    )


def action_surface_feature(public: dict[str, Any], manifest: dict[str, Any]) -> tuple[Any, ...]:
    window = public.get("input", {}).get("dialogue_window", [])
    current = window[-1] if window else {}
    current_text = str(current.get("utterance", ""))
    position = int(manifest.get("stream_position") or public.get("stream_position") or 0)
    speaker = str(manifest.get("current_speaker") or current.get("speaker") or "")
    current_placeholder_types = tuple(sorted(set(PLACEHOLDER_RE.findall(current_text))))
    cue_flags = (
        "url_or_path" if any(term in current_text for term in ("[MASKED_URL]", "링크", "경로", "포털", "접속")) else "",
        "pressure" if any(term in current_text for term in ("긴급", "즉시", "오늘", "마감", "지연", "보류", "제한", "차질")) else "",
        "refusal" if any(term in current_text for term in ("제공하지", "입력하지", "사용하지", "접속하지", "열지", "진행하지")) else "",
        "official" if any(term in current_text for term in ("공식", "저장된", "직접")) else "",
        "disclosure" if any(term in current_text for term in ("전달하겠습니다", "제공하겠습니다", "말씀드리겠습니다", "입력하겠습니다", "공유하겠습니다")) else "",
        "recovery" if any(term in current_text for term in ("중단", "보존", "신고", "기록", "복구")) else "",
    )
    return (
        position,
        speaker,
        current_placeholder_types,
        tuple(flag for flag in cue_flags if flag),
    )


def majority_accuracy(pairs: list[tuple[Any, str]]) -> float:
    if not pairs:
        return 0.0
    grouped: dict[Any, Counter[str]] = defaultdict(Counter)
    for feature, label in pairs:
        grouped[feature][label] += 1
    correct = sum(counter.most_common(1)[0][1] for counter in grouped.values())
    return correct / len(pairs)


def train_test_majority_accuracy(examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(example["label"]) for example in examples)
    if not examples:
        return {
            "accuracy": 0.0,
            "support": 0,
            "test_count": 0,
            "unique_label_count": 0,
            "label_distribution": {},
            "unseen_feature_rate": 0.0,
        }
    train = [example for example in examples if stable_bucket(example.get("split_key")) == 0]
    test = [example for example in examples if stable_bucket(example.get("split_key")) == 1]
    if not train or not test:
        midpoint = len(examples) // 2
        train = examples[:midpoint] or examples
        test = examples[midpoint:] or examples
    global_majority = Counter(str(example["label"]) for example in train).most_common(1)[0][0]
    feature_to_labels: dict[Any, Counter[str]] = defaultdict(Counter)
    for example in train:
        feature_to_labels[example["feature"]][str(example["label"])] += 1

    correct = 0
    unseen = 0
    for example in test:
        feature = example["feature"]
        if feature in feature_to_labels:
            prediction = feature_to_labels[feature].most_common(1)[0][0]
        else:
            prediction = global_majority
            unseen += 1
        correct += int(prediction == str(example["label"]))

    return {
        "accuracy": correct / len(test) if test else 0.0,
        "support": len(examples),
        "test_count": len(test),
        "unique_label_count": len(labels),
        "label_distribution": dict(sorted(labels.items())),
        "unseen_feature_rate": unseen / len(test) if test else 0.0,
    }


def stable_bucket(value: Any) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def repeated_noun_artifacts(public_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for public in public_rows:
        for turn in public.get("input", {}).get("dialogue_window", []):
            text = str(turn.get("utterance", ""))
            match = REPEATED_NOUN_RE.search(text)
            if match:
                artifacts.append(
                    {
                        "benchmark_id": public.get("benchmark_id"),
                        "turn_id": turn.get("turn_id"),
                        "artifact": match.group(0),
                        "utterance": text,
                    }
                )
                break
    return artifacts
