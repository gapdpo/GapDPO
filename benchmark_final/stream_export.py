from __future__ import annotations

from typing import Any

from benchmark_test.benchmark_lib import (
    GeneratedItem,
    RejectedItem,
    has_stage_direction_or_unmasked_bracket,
    item_placeholder_risk,
    private_record_summary,
    stable_id,
)
from benchmark_stream_v1.constants import OFFICIAL_CANDIDATE_FAMILY, SCHEMA_VERSION
from benchmark_stream_v1.stream_logic import public_stream_item, valid_public_turn_ids


def generate_stream_items_v2(
    records: list[dict[str, Any]],
    placeholder_disposition: dict[str, Any],
    *,
    family: str = OFFICIAL_CANDIDATE_FAMILY,
    surface: str = "typed_raw",
    window_size: int = 6,
) -> tuple[list[GeneratedItem], list[RejectedItem]]:
    items: list[GeneratedItem] = []
    rejected: list[RejectedItem] = []
    for record in records:
        source_id = str(record.get("id", "unknown"))
        if has_stage_direction_or_unmasked_bracket(record):
            rejected.append(RejectedItem(family, source_id, "stage_direction_or_unmasked_bracket"))
            continue
        stream_id = stable_id("streamv2", source_id, surface)
        split_group_id = stable_id("streamv2_group", source_id)
        metadata = record.get("gen_metadata", {})
        fallback_turn_ids = {
            int(turn_id)
            for turn_id in metadata.get("fallback_turn_ids", [])
            if isinstance(turn_id, int) or str(turn_id).isdigit()
        }
        for turn in record.get("dialogue", []):
            turn_id = int(turn.get("turn_id", 0))
            if not turn.get("stream_event"):
                rejected.append(RejectedItem(family, source_id, "missing_stream_turn_target", str(turn_id)))
                continue
            public = public_stream_item(
                record,
                turn_id,
                family=family,
                stream_id=stream_id,
                surface=surface,
                window_size=window_size,
            )
            visible_turn_ids = valid_public_turn_ids(public)
            evidence_ids = []
            for raw_evidence_id in turn.get("stream_evidence_turn_ids", [turn_id]):
                try:
                    evidence_id = int(raw_evidence_id)
                except (TypeError, ValueError):
                    continue
                if evidence_id in visible_turn_ids:
                    evidence_ids.append(evidence_id)
            evidence_ids = evidence_ids or ([turn_id] if turn_id in visible_turn_ids else [])
            placeholder_status = item_placeholder_risk(public, placeholder_disposition)
            gold_status = "programmatic_validated" if placeholder_status == "programmatic_validated" else "diagnostic_only"
            gold = {
                "benchmark_id": public["benchmark_id"],
                "stream_id": stream_id,
                "family": family,
                "gold_status": gold_status,
                "event": turn["stream_event"],
                "action": turn["stream_action"],
                "policy_rule_ids": list(turn["stream_policy_rule_ids"]),
                "evidence_turn_ids": evidence_ids,
                "is_fallback_turn": turn_id in fallback_turn_ids,
            }
            manifest = {
                "benchmark_id": public["benchmark_id"],
                "stream_id": stream_id,
                "family": family,
                "schema_version": SCHEMA_VERSION,
                "source": private_record_summary(record),
                "surface": surface,
                "stream_position": public["stream_position"],
                "current_turn_id": turn_id,
                "current_speaker": public["input"]["dialogue_window"][-1]["speaker"],
                "split_group_id": split_group_id,
                "gold_status": gold_status,
                "placeholder_status": placeholder_status,
                "stream_profile": metadata.get("stream_profile"),
                "trajectory_variant": metadata.get("trajectory_variant"),
                "canonical_pair_index": metadata.get("canonical_pair_index"),
                "is_fallback_turn": turn_id in fallback_turn_ids,
                "release_stance": "official_candidate_pending_gates",
            }
            items.append(GeneratedItem(public=public, manifest=manifest, gold=gold))
    return items, rejected
