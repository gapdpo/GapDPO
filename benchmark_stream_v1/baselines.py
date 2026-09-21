from __future__ import annotations

import json
from typing import Any

from benchmark_test.benchmark_lib import extract_placeholders, pressure_categories, request_state

from benchmark_stream_v1.constants import EVENT_TO_ACTION, EVENT_TO_POLICY_RULE_IDS
from benchmark_stream_v1.score_stream_predictions import score_predictions

BASELINES = (
    "always_active",
    "never_active",
    "caller_turn_only",
    "position_only",
    "current_keyword",
    "prefix_keyword",
    "placeholder_count_or_type",
    "repeat_prior_state",
    "static_verify",
    "static_refuse",
    "static_preserve_on_any_mask",
    "window_length_only",
    "shuffled_dialogue",
)


def build_baseline_report(
    public_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for name in BASELINES:
        predictions = baseline_predictions(public_rows, name)
        score = score_predictions(
            public_rows=public_rows,
            gold_rows=gold_rows,
            predictions=predictions,
            manifest_rows=manifest_rows,
        )
        baselines[name] = {
            "prediction_count": len(predictions),
            "SSU": score["SSU"],
            "weighted_cost_per_100": score["weighted_cost_per_100"],
            "missed_actionable_rate": score["missed_actionable_rate"],
            "false_active_rate": score["false_active_rate"],
            "duplicate_alert_rate": score["duplicate_alert_rate"],
            "event_macro_f1_diagnostic": score["event_macro_f1_diagnostic"],
            "action_macro_f1": score["action_macro_f1"],
        }
    return {
        "scope": "selective_stream_trivial_baselines",
        "scale": "SSU_0_to_100_higher_is_better",
        "baselines": baselines,
        "official_blocking_threshold": {
            "baseline_ssu_at_or_above": 60,
            "baseline_within_reference_model_ssu_points": 15,
        },
    }


def baseline_predictions(public_rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    previous_event_by_stream: dict[str, str] = {}
    for public in sorted(public_rows, key=lambda row: (str(row.get("stream_id", "")), int(row.get("stream_position", 0)))):
        event = baseline_event(public, name, previous_event_by_stream.get(str(public.get("stream_id", ""))))
        previous_event_by_stream[str(public.get("stream_id", ""))] = event
        predictions.append(prediction(public, event))
    return predictions


def baseline_event(public: dict[str, Any], name: str, previous_event: str | None = None) -> str:
    window = public.get("input", {}).get("dialogue_window", [])
    current = window[-1] if window else {}
    current_text = str(current.get("utterance", ""))
    prefix_text = json.dumps(window, ensure_ascii=False)
    current_speaker = current.get("speaker")
    position = int(public.get("stream_position", 0))
    if name == "always_active":
        return "new_sensitive_request"
    if name == "never_active":
        return "no_new_actionable_event"
    if name == "caller_turn_only":
        return "new_sensitive_request" if current_speaker == "caller" else "no_new_actionable_event"
    if name == "position_only":
        return "new_sensitive_request" if position >= 3 else "needs_more_context"
    if name == "current_keyword":
        if request_state(current_text) != "none_or_context_only":
            return "new_sensitive_request"
        if pressure_categories(current_text):
            return "pressure_only_monitor"
        return "needs_more_context"
    if name == "prefix_keyword":
        return "new_sensitive_request" if request_state(prefix_text) != "none_or_context_only" else "needs_more_context"
    if name == "placeholder_count_or_type":
        placeholders = extract_placeholders(prefix_text)
        if any(item in {"[MASKED_OTP]", "[MASKED_CREDENTIAL]", "[MASKED_ACCOUNT]"} for item in placeholders):
            return "new_sensitive_request"
        if placeholders:
            return "pressure_only_monitor"
        return "needs_more_context"
    if name == "repeat_prior_state":
        return previous_event or "needs_more_context"
    if name == "static_verify":
        return "pressure_escalation"
    if name == "static_refuse":
        return "new_sensitive_request"
    if name == "static_preserve_on_any_mask":
        return "recovery_followup_due" if extract_placeholders(prefix_text) else "needs_more_context"
    if name == "window_length_only":
        return "new_sensitive_request" if len(window) >= 4 else "needs_more_context"
    if name == "shuffled_dialogue":
        shuffled_text = " ".join(str(turn.get("utterance", "")) for turn in sorted(window, key=lambda row: str(row.get("utterance", ""))))
        return "new_sensitive_request" if request_state(shuffled_text) != "none_or_context_only" else "needs_more_context"
    raise ValueError(f"Unknown baseline: {name}")


def prediction(public: dict[str, Any], event: str) -> dict[str, Any]:
    window = public.get("input", {}).get("dialogue_window", [])
    evidence_turn_id = int((window[-1] if window else {}).get("turn_id", public.get("stream_position", 0)))
    return {
        "benchmark_id": public["benchmark_id"],
        "event": event,
        "action": EVENT_TO_ACTION[event],
        "policy_rule_ids": list(EVENT_TO_POLICY_RULE_IDS[event]),
        "evidence_turn_ids": [evidence_turn_id],
    }

