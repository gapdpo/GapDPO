from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_test.benchmark_lib import output_safety_violations, read_jsonl, write_json

from benchmark_stream_v1.constants import (
    ACTIONS,
    ACTIVE_ACTIONS,
    BAD_EVIDENCE_ID_PENALTY,
    DUPLICATE_ACTIVE_ALERT_PENALTY,
    EVENTS,
    FALSE_ACTIVE_PENALTY,
    INVALID_PREDICTION_PENALTY,
    LATE_FIRST_ACTION_PENALTY_CAP,
    LATE_FIRST_ACTION_PENALTY_PER_TURN,
    MISS_PENALTY_BY_EVENT,
    VALID_POLICY_RULE_IDS,
    WRONG_ACTIVE_ACTION_PENALTY,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score selective stream-intervention predictions.")
    parser.add_argument("--public", required=True, help="Stream benchmark public_items.jsonl.")
    parser.add_argument("--gold", required=True, help="Stream benchmark gold_stream.jsonl.")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL keyed by benchmark_id.")
    parser.add_argument("--out", required=True, help="Output JSON score report.")
    parser.add_argument("--manifest", help="Optional private_manifest.jsonl.")
    args = parser.parse_args()

    report = score_predictions(
        public_rows=read_jsonl(args.public),
        gold_rows=read_jsonl(args.gold),
        predictions=read_jsonl(args.predictions),
        manifest_rows=read_jsonl(args.manifest) if args.manifest else [],
    )
    write_json(args.out, report)
    print(f"Wrote stream score report to {args.out}")
    return 0


def score_predictions(
    *,
    public_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    public_by_id = {row["benchmark_id"]: row for row in public_rows}
    gold_by_id = {row["benchmark_id"]: row for row in gold_rows}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    duplicate_prediction_ids: set[str] = set()
    for row in predictions:
        benchmark_id = row.get("benchmark_id")
        if not benchmark_id:
            continue
        if benchmark_id in prediction_by_id:
            duplicate_prediction_ids.add(str(benchmark_id))
            continue
        prediction_by_id[str(benchmark_id)] = row

    gold_ids = set(gold_by_id)
    predicted_ids = set(prediction_by_id)
    missing_ids = gold_ids - predicted_ids
    extra_ids = predicted_ids - gold_ids
    ordered_ids = sorted(
        gold_ids,
        key=lambda benchmark_id: (
            str(public_by_id.get(benchmark_id, {}).get("stream_id", "")),
            int(public_by_id.get(benchmark_id, {}).get("stream_position", 0)),
        ),
    )

    total_penalty = 0.0
    invalid_predictions: list[dict[str, Any]] = []
    per_item_penalties: list[dict[str, Any]] = []
    event_gold: list[str] = []
    event_pred: list[str] = []
    action_gold: list[str] = []
    action_pred: list[str] = []
    evidence_scores: list[float] = []
    policy_scores: list[float] = []
    missed_actionable = 0
    actionable_count = 0
    false_active = 0
    inactive_count = 0
    duplicate_active_alert = 0
    missed_recovery = 0
    recovery_count = 0

    stream_active_alert_seen: dict[str, bool] = defaultdict(bool)
    first_gold_active_by_stream: dict[str, int] = {}
    first_pred_active_by_stream: dict[str, int] = {}

    for benchmark_id in ordered_ids:
        public = public_by_id[benchmark_id]
        gold = gold_by_id[benchmark_id]
        stream_id = str(public.get("stream_id", ""))
        position = int(public.get("stream_position", 0))
        gold_event = str(gold.get("event"))
        gold_action = str(gold.get("action"))
        gold_active = gold_action in ACTIVE_ACTIONS
        if gold_active:
            actionable_count += 1
            first_gold_active_by_stream.setdefault(stream_id, position)
        else:
            inactive_count += 1
        if gold_event == "recovery_followup_due":
            recovery_count += 1

        pred = prediction_by_id.get(benchmark_id)
        item_penalty = 0.0
        item_reasons: list[str] = []
        if pred is None:
            item_penalty += INVALID_PREDICTION_PENALTY
            item_reasons.append("missing_prediction")
            if gold_active:
                item_penalty += miss_penalty(gold_event)
                missed_actionable += 1
                if gold_event == "recovery_followup_due":
                    missed_recovery += 1
            total_penalty += item_penalty
            per_item_penalties.append({"benchmark_id": benchmark_id, "penalty": item_penalty, "reasons": item_reasons})
            continue

        errors = prediction_errors(pred, valid_evidence_ids(public))
        if errors:
            item_penalty += INVALID_PREDICTION_PENALTY
            item_reasons.extend(errors)
            invalid_predictions.append({"benchmark_id": benchmark_id, "errors": errors})
        pred_event = str(pred.get("event"))
        pred_action = str(pred.get("action"))
        pred_active = pred_action in ACTIVE_ACTIONS
        if pred_active:
            first_pred_active_by_stream.setdefault(stream_id, position)

        event_gold.append(gold_event)
        event_pred.append(pred_event)
        action_gold.append(gold_action)
        action_pred.append(pred_action)
        evidence_scores.append(_set_f1(_as_int_set(pred.get("evidence_turn_ids")), _as_int_set(gold.get("evidence_turn_ids"))))
        policy_scores.append(_set_f1(_as_str_set(pred.get("policy_rule_ids")), _as_str_set(gold.get("policy_rule_ids"))))

        if gold_active and not pred_active:
            item_penalty += miss_penalty(gold_event)
            item_reasons.append("missed_actionable")
            missed_actionable += 1
            if gold_event == "recovery_followup_due":
                missed_recovery += 1
        elif not gold_active and pred_active:
            item_penalty += FALSE_ACTIVE_PENALTY
            item_reasons.append("false_active")
            false_active += 1
            if stream_active_alert_seen[stream_id]:
                item_penalty += DUPLICATE_ACTIVE_ALERT_PENALTY
                item_reasons.append("duplicate_active_alert")
                duplicate_active_alert += 1
        elif gold_active and pred_active and pred_action != gold_action:
            item_penalty += WRONG_ACTIVE_ACTION_PENALTY
            item_reasons.append("wrong_active_action_family")

        invalid_evidence = [eid for eid in _as_int_set(pred.get("evidence_turn_ids")) if eid not in valid_evidence_ids(public)]
        if invalid_evidence:
            item_penalty += BAD_EVIDENCE_ID_PENALTY
            item_reasons.append("out_of_window_evidence")

        if pred_active:
            stream_active_alert_seen[stream_id] = True
        total_penalty += item_penalty
        if item_penalty:
            per_item_penalties.append({"benchmark_id": benchmark_id, "penalty": item_penalty, "reasons": item_reasons})

    late_penalties: list[float] = []
    for stream_id, first_gold_position in first_gold_active_by_stream.items():
        first_pred_position = first_pred_active_by_stream.get(stream_id)
        if first_pred_position is None or first_pred_position <= first_gold_position:
            continue
        penalty = min(
            LATE_FIRST_ACTION_PENALTY_CAP,
            (first_pred_position - first_gold_position) * LATE_FIRST_ACTION_PENALTY_PER_TURN,
        )
        late_penalties.append(penalty)
        total_penalty += penalty

    total_penalty += len(extra_ids) * INVALID_PREDICTION_PENALTY
    total_penalty += len(duplicate_prediction_ids) * INVALID_PREDICTION_PENALTY
    scored_count = len(gold_rows)
    weighted_cost_per_100 = round(100 * total_penalty / scored_count, 4) if scored_count else 0.0
    ssu = round(100 - min(100.0, weighted_cost_per_100), 4)
    return {
        "scope": "selective_stream_intervention_ssu_lite",
        "item_count": scored_count,
        "prediction_count": len(predictions),
        "matched_prediction_count": len(gold_ids & predicted_ids),
        "missing_prediction_count": len(missing_ids),
        "extra_prediction_count": len(extra_ids),
        "duplicate_prediction_count": len(duplicate_prediction_ids),
        "weighted_cost_per_100": weighted_cost_per_100,
        "SSU": ssu,
        "total_penalty": round(total_penalty, 4),
        "missed_actionable_rate": _safe_rate(missed_actionable, actionable_count),
        "false_active_rate": _safe_rate(false_active, inactive_count),
        "duplicate_alert_rate": _safe_rate(duplicate_active_alert, max(1, len(gold_rows))),
        "missed_recovery_rate": _safe_rate(missed_recovery, recovery_count),
        "latency": latency_report(first_gold_active_by_stream, first_pred_active_by_stream),
        "event_macro_f1_diagnostic": macro_f1(event_pred, event_gold, EVENTS),
        "action_macro_f1": macro_f1(action_pred, action_gold, ACTIONS),
        "evidence_turn_id_set_f1": _average(evidence_scores),
        "policy_rule_set_f1": _average(policy_scores),
        "invalid_prediction_count": len(invalid_predictions),
        "invalid_prediction_rate": _safe_rate(len(invalid_predictions), len(gold_rows)),
        "invalid_prediction_examples": invalid_predictions[:20],
        "penalty_examples": per_item_penalties[:20],
        "manifest_stream_count": len({row.get("stream_id") for row in manifest_rows or []}) or None,
    }


def prediction_errors(prediction: dict[str, Any], valid_turn_ids: set[int]) -> list[str]:
    errors: list[str] = []
    if prediction.get("event") not in EVENTS:
        errors.append("invalid_event")
    if prediction.get("action") not in ACTIONS:
        errors.append("invalid_action")
    policy_rule_ids = prediction.get("policy_rule_ids")
    if not isinstance(policy_rule_ids, list) or not all(str(rule_id) in VALID_POLICY_RULE_IDS for rule_id in policy_rule_ids):
        errors.append("invalid_policy_rule_ids")
    evidence_turn_ids = prediction.get("evidence_turn_ids")
    if not isinstance(evidence_turn_ids, list) or not evidence_turn_ids:
        errors.append("invalid_evidence_turn_ids")
    elif any(not isinstance(turn_id, int) or isinstance(turn_id, bool) for turn_id in evidence_turn_ids):
        errors.append("non_integer_evidence_turn_id")
    elif any(turn_id not in valid_turn_ids for turn_id in evidence_turn_ids):
        errors.append("out_of_window_evidence")
    if "confidence" in prediction:
        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append("invalid_confidence")
    for violation in output_safety_violations(prediction):
        errors.append(f"unsafe_output:{violation}")
    return errors


def valid_evidence_ids(public: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for turn in public.get("input", {}).get("dialogue_window", []):
        try:
            result.add(int(turn.get("turn_id", 0)))
        except (TypeError, ValueError):
            continue
    return result


def miss_penalty(event: str) -> float:
    return MISS_PENALTY_BY_EVENT.get(event, 2.0)


def latency_report(gold_first: dict[str, int], pred_first: dict[str, int]) -> dict[str, Any]:
    latencies = [
        pred_position - gold_position
        for stream_id, gold_position in gold_first.items()
        if (pred_position := pred_first.get(stream_id)) is not None and pred_position >= gold_position
    ]
    missed = sum(1 for stream_id in gold_first if stream_id not in pred_first)
    return {
        "scored_stream_count": len(gold_first),
        "missed_stream_count": missed,
        "mean_turn_latency": round(statistics.mean(latencies), 4) if latencies else None,
        "median_turn_latency": round(statistics.median(latencies), 4) if latencies else None,
        "p95_turn_latency": percentile(latencies, 0.95),
        "max_turn_latency": max(latencies) if latencies else None,
    }


def macro_f1(predictions: list[str], gold: list[str], labels: tuple[str, ...]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for pred, truth in zip(predictions, gold) if pred == label and truth == label)
        fp = sum(1 for pred, truth in zip(predictions, gold) if pred == label and truth != label)
        fn = sum(1 for pred, truth in zip(predictions, gold) if pred != label and truth == label)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return round(_average(scores), 4)


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def _as_int_set(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    result: set[int] = set()
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            result.add(item)
    return result


def _as_str_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def _set_f1(pred: set[Any], gold: set[Any]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 0.0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 4)


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

