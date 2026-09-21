from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from benchmark_test.benchmark_lib import (
    GeneratedItem,
    RejectedItem,
    disclosure_state,
    has_stage_direction_or_unmasked_bracket,
    item_placeholder_risk,
    pressure_categories,
    private_record_summary,
    public_turns,
    recipient_defense_action,
    request_states,
    stable_id,
)

from benchmark_stream_v1.constants import (
    DIAGNOSTIC_FAMILY,
    ACTIONS,
    EVENTS,
    EVENT_TO_ACTION,
    EVENT_TO_POLICY_RULE_IDS,
    POLICY_CARD,
    SCHEMA_VERSION,
)

REQUEST_NONE = "none_or_context_only"
CALLER_DIRECTED_PATH = "caller_directed_link_or_path"


@dataclass(frozen=True)
class StreamTransition:
    event: str
    action: str
    policy_rule_ids: list[str]
    evidence_turn_ids: list[int]
    oracle_state: dict[str, Any]


def generate_stream_items(
    records: list[dict[str, Any]],
    placeholder_disposition: dict[str, Any],
    *,
    family: str = DIAGNOSTIC_FAMILY,
    surface: str = "typed_raw",
    window_size: int = 6,
    exclude_risky_placeholders: bool = True,
) -> tuple[list[GeneratedItem], list[RejectedItem]]:
    items: list[GeneratedItem] = []
    rejected: list[RejectedItem] = []
    for record in records:
        source_id = str(record.get("id", "unknown"))
        if has_stage_direction_or_unmasked_bracket(record):
            rejected.append(RejectedItem(family, source_id, "stage_direction_or_unmasked_bracket"))
            continue
        if not record.get("dialogue"):
            rejected.append(RejectedItem(family, source_id, "no_dialogue_turns"))
            continue
        stream_id = stable_id("stream", family, source_id, surface)
        split_group_id = stable_id("stream_group", source_id)
        for turn in record.get("dialogue", []):
            turn_id = int(turn.get("turn_id", 0))
            public = public_stream_item(
                record,
                turn_id,
                family=family,
                stream_id=stream_id,
                surface=surface,
                window_size=window_size,
            )
            placeholder_status = item_placeholder_risk(public, placeholder_disposition)
            if exclude_risky_placeholders and placeholder_status == "diagnostic_only":
                rejected.append(RejectedItem(family, source_id, "placeholder_reject_or_escalate", str(turn_id)))
                continue
            transition = classify_transition(record, turn_id)
            evidence_turn_ids = [
                evidence_turn_id
                for evidence_turn_id in transition.evidence_turn_ids
                if evidence_turn_id in valid_public_turn_ids(public)
            ]
            if not evidence_turn_ids:
                evidence_turn_ids = [turn_id] if turn_id in valid_public_turn_ids(public) else []
            gold_status = "diagnostic_only" if family == DIAGNOSTIC_FAMILY else placeholder_status
            gold = {
                "benchmark_id": public["benchmark_id"],
                "stream_id": stream_id,
                "family": family,
                "gold_status": gold_status,
                "event": transition.event,
                "action": transition.action,
                "policy_rule_ids": transition.policy_rule_ids,
                "evidence_turn_ids": evidence_turn_ids,
                "oracle_state": transition.oracle_state,
            }
            manifest = {
                "benchmark_id": public["benchmark_id"],
                "stream_id": stream_id,
                "family": family,
                "source": private_record_summary(record),
                "surface": surface,
                "schema_version": SCHEMA_VERSION,
                "stream_position": public["stream_position"],
                "current_turn_id": turn_id,
                "current_speaker": public["input"]["dialogue_window"][-1]["speaker"],
                "split_group_id": split_group_id,
                "gold_status": gold_status,
                "placeholder_status": placeholder_status,
                "release_stance": "diagnostic_only" if family == DIAGNOSTIC_FAMILY else "official_candidate_pending_gates",
            }
            items.append(GeneratedItem(public=public, manifest=manifest, gold=gold))
    return items, rejected


def public_stream_item(
    record: dict[str, Any],
    current_turn_id: int,
    *,
    family: str,
    stream_id: str,
    surface: str,
    window_size: int,
) -> dict[str, Any]:
    source_id = str(record.get("id", "unknown"))
    prefix = public_turns(record, surface=surface, max_turn_id=current_turn_id)
    window = prefix[-window_size:]
    return {
        "benchmark_id": stable_id("ssi", family, source_id, current_turn_id, surface),
        "stream_id": stream_id,
        "family": family,
        "mode": "stream_rollout",
        "schema_version": SCHEMA_VERSION,
        "locale": record.get("locale", "ko-KR"),
        "surface": surface,
        "stream_position": current_turn_id,
        "input": {
            "dialogue_window": window,
            "policy_card": POLICY_CARD,
        },
    }


def classify_transition(record: dict[str, Any], current_turn_id: int) -> StreamTransition:
    turns = sorted(record.get("dialogue", []), key=lambda row: int(row.get("turn_id", 0)))
    prior_turns = [turn for turn in turns if int(turn.get("turn_id", 0)) < current_turn_id]
    current = next((turn for turn in turns if int(turn.get("turn_id", 0)) == current_turn_id), {})
    speaker = str(current.get("speaker", ""))
    text = str(current.get("utterance", ""))

    prior_requests = request_type_set(prior_turns)
    current_requests = request_type_set([current]) if speaker == "attacker" else set()
    prior_path = CALLER_DIRECTED_PATH in prior_requests
    current_path = CALLER_DIRECTED_PATH in current_requests or caller_path_text(text)
    prior_pressures = pressure_type_set(prior_turns)
    current_pressures = set(pressure_categories(text)) if speaker == "attacker" else set()
    prior_direct_disclosure_ids = direct_disclosure_turn_ids(prior_turns)
    current_disclosure = stream_disclosure_state(text) if speaker == "victim" else "none"
    current_defense = recipient_defense_action(text) if speaker == "victim" else "none"
    new_requests = current_requests - prior_requests
    new_pressures = current_pressures - prior_pressures

    event = "no_new_actionable_event"
    evidence_turn_ids = [current_turn_id]
    if current_disclosure == "direct_masked_disclosure_or_entry":
        event = "recipient_partial_or_direct_disclosure"
    elif prior_direct_disclosure_ids:
        event = "recovery_followup_due"
        evidence_turn_ids = [prior_direct_disclosure_ids[-1]]
    elif speaker == "attacker" and current_path and not prior_path:
        event = "new_caller_directed_path"
    elif speaker == "attacker" and new_requests:
        event = "new_sensitive_request"
    elif speaker == "attacker" and new_pressures and (prior_requests or current_requests or prior_path):
        event = "pressure_escalation"
    elif speaker == "victim" and current_defense == "safe_verify_or_refuse":
        event = "recipient_containment"
    elif speaker == "attacker" and current_pressures and not current_requests and not prior_requests and not prior_path:
        event = "pressure_only_monitor"
    elif not prior_requests and not current_requests and not prior_pressures and not current_pressures:
        event = "needs_more_context"

    action = EVENT_TO_ACTION[event]
    return StreamTransition(
        event=event,
        action=action,
        policy_rule_ids=list(EVENT_TO_POLICY_RULE_IDS[event]),
        evidence_turn_ids=evidence_turn_ids,
        oracle_state={
            "prior_request_types": sorted(prior_requests),
            "current_request_types": sorted(current_requests),
            "prior_pressure_categories": sorted(prior_pressures),
            "current_pressure_categories": sorted(current_pressures),
            "current_disclosure_state": current_disclosure,
            "current_recipient_defense_action": current_defense,
            "prior_direct_disclosure_turn_ids": prior_direct_disclosure_ids,
        },
    )


def request_type_set(turns: list[dict[str, Any]]) -> set[str]:
    states: set[str] = set()
    for turn in turns:
        if turn.get("speaker") != "attacker":
            continue
        for state in request_states(str(turn.get("utterance", ""))):
            if state != REQUEST_NONE:
                states.add(state)
    return states


def pressure_type_set(turns: list[dict[str, Any]]) -> set[str]:
    states: set[str] = set()
    for turn in turns:
        if turn.get("speaker") == "attacker":
            states.update(pressure_categories(str(turn.get("utterance", ""))))
    return states


def direct_disclosure_turn_ids(turns: list[dict[str, Any]]) -> list[int]:
    return [
        int(turn.get("turn_id", 0))
        for turn in turns
        if turn.get("speaker") == "victim"
        and stream_disclosure_state(str(turn.get("utterance", ""))) == "direct_masked_disclosure_or_entry"
    ]


def stream_disclosure_state(text: str) -> str:
    state = disclosure_state(text)
    if state != "none":
        return state
    has_mask = "[MASKED_" in text
    if not has_mask:
        return "none"
    if "공식" in text and ("후" in text or "확인" in text):
        return "conditional_masked_action"
    direct_verbs = (
        "안내드리겠습니다",
        "말씀드리겠습니다",
        "공유하겠습니다",
        "알려 드리겠습니다",
        "알려드릴게요",
    )
    if any(verb in text for verb in direct_verbs):
        return "direct_masked_disclosure_or_entry"
    return "none"


def caller_path_text(text: str) -> bool:
    return "[MASKED_URL]" in text or "링크" in text or "접속" in text or "임시 포털" in text


def valid_public_turn_ids(public: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for turn in public.get("input", {}).get("dialogue_window", []):
        try:
            result.add(int(turn.get("turn_id", 0)))
        except (TypeError, ValueError):
            continue
    return result


def build_yield_report(
    public_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    event_counts = Counter(row.get("event") for row in gold_rows)
    action_counts = Counter(row.get("action") for row in gold_rows)
    return {
        "scope": "selective_stream_intervention_yield",
        "public_item_count": len(public_rows),
        "stream_count": len({row.get("stream_id") for row in public_rows}),
        "event_distribution": {event: event_counts.get(event, 0) for event in EVENTS},
        "action_distribution": {action: action_counts.get(action, 0) for action in ACTIONS},
        "gold_status_distribution": dict(sorted(Counter(row.get("gold_status") for row in gold_rows).items())),
        "surface_distribution": dict(sorted(Counter(row.get("surface") for row in public_rows).items())),
        "placeholder_status_distribution": dict(
            sorted(Counter(row.get("placeholder_status") for row in manifest_rows).items())
        ),
        "rejected_count": len(rejected_rows),
        "rejected_by_reason": dict(sorted(Counter(row.get("reason") for row in rejected_rows).items())),
        "release_floor_30_by_event": {
            event: event_counts.get(event, 0) >= 30
            for event in EVENTS
        },
        "release_floor_30_by_action": {
            action: action_counts.get(action, 0) >= 30
            for action in ACTIONS
        },
    }
