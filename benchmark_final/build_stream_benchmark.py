from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_test.benchmark_lib import (
    find_forbidden_public_keys,
    placeholder_disposition_from_qa,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from benchmark_stream_v1.baselines import build_baseline_report
from benchmark_stream_v1.build_stream_benchmark import (
    render_baseline_markdown,
    render_sample_fixtures,
    render_yield_markdown,
)
from benchmark_stream_v1.constants import OFFICIAL_CANDIDATE_FAMILY
from benchmark_stream_v1.stream_logic import build_yield_report
from benchmark_final.diagnostics import build_shortcut_report
from benchmark_final.stream_export import generate_stream_items_v2


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v2 persona-grounded stream benchmark artifacts.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qa")
    parser.add_argument("--surface", default="typed_raw", choices=["typed_raw", "neutral_mask", "placeholder_ablated"])
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--shortcut-gate", action="store_true", help="Fail if shortcut diagnostics do not pass.")
    args = parser.parse_args()

    qa = read_json(args.qa) if args.qa else {"quality_signals": {"unknown_masked_placeholder_policy": {"placeholders": {}}}}
    placeholder_disposition = placeholder_disposition_from_qa(qa)
    records = read_jsonl(args.source)
    items, rejected = generate_stream_items_v2(
        records,
        placeholder_disposition,
        family=OFFICIAL_CANDIDATE_FAMILY,
        surface=args.surface,
        window_size=args.window_size,
    )
    public_rows = [item.public for item in items]
    manifest_rows = [item.manifest for item in items]
    gold_rows = [item.gold for item in items]
    rejected_rows = [row.to_dict() for row in rejected]
    whitelist_violations = [
        {"benchmark_id": row.get("benchmark_id"), "keys": keys}
        for row in public_rows
        if (keys := find_forbidden_public_keys(row))
    ]
    yield_report = build_yield_report(public_rows, manifest_rows, gold_rows, rejected_rows)
    shortcut_report = build_shortcut_report(public_rows, manifest_rows, gold_rows, rejected_rows)
    shortcut_report["gate_checks"]["public_whitelist_violation_count_is_zero"] = len(whitelist_violations) == 0
    shortcut_report["shortcut_gate_passed"] = all(shortcut_report["gate_checks"].values())
    yield_report.update(
        {
            "status": "official_candidate_pending_gates",
            "source_path": args.source,
            "qa_path": args.qa,
            "family": OFFICIAL_CANDIDATE_FAMILY,
            "surface": args.surface,
            "window_size": args.window_size,
            "public_whitelist_violation_count": len(whitelist_violations),
            "public_whitelist_violations": whitelist_violations[:20],
            "shortcut_diagnostics_summary": {
                "shortcut_gate_passed": shortcut_report["shortcut_gate_passed"],
                "profile_position_majority_event_accuracy": shortcut_report[
                    "profile_position_majority_event_accuracy"
                ],
                "position_only_majority_event_accuracy": shortcut_report["position_only_majority_event_accuracy"],
                "min_profile_unique_event_sequences": shortcut_report["min_profile_unique_event_sequences"],
                "event_action_is_fully_deterministic": shortcut_report["event_action_is_fully_deterministic"],
                "max_event_action_dominance": shortcut_report["max_event_action_dominance"],
                "low_action_branch_signal_events": shortcut_report["low_action_branch_signal_events"],
                "variant_surface_accuracy": shortcut_report["variant_surface_model"]["accuracy"],
                "variant_surface_support": shortcut_report["variant_surface_model"]["support"],
                "variant_surface_gate_applicable": shortcut_report["variant_surface_gate_applicable"],
                "fallback_turn_count": shortcut_report["fallback_turn_count"],
                "fallback_turn_rate": shortcut_report["fallback_turn_rate"],
                "fallback_quality_gate_applicable": shortcut_report["fallback_quality_gate_applicable"],
                "repeated_fallback_string_majority_event_accuracy": shortcut_report[
                    "repeated_fallback_string_majority_event_accuracy"
                ],
                "high_count_single_event_fallback_string_count": shortcut_report[
                    "high_count_single_event_fallback_string_count"
                ],
                "high_count_single_event_non_fallback_string_count": shortcut_report[
                    "high_count_single_event_non_fallback_string_count"
                ],
                "placeholder_only_turn_count": shortcut_report["placeholder_only_turn_count"],
                "very_short_turn_count": shortcut_report["very_short_turn_count"],
                "instruction_residue_turn_count": shortcut_report["instruction_residue_turn_count"],
                "category_leakage_turn_count": shortcut_report["category_leakage_turn_count"],
                "repeated_noun_artifact_count": shortcut_report["repeated_noun_artifact_count"],
            },
        }
    )
    baseline_report = build_baseline_report(public_rows, gold_rows, manifest_rows)
    out_dir = Path(args.out)
    write_jsonl(out_dir / "public_items.jsonl", public_rows)
    write_jsonl(out_dir / "private_manifest.jsonl", manifest_rows)
    write_jsonl(out_dir / "gold_stream.jsonl", gold_rows)
    write_jsonl(out_dir / "rejected_items.jsonl", rejected_rows)
    write_json(out_dir / "yield_report.json", yield_report)
    write_json(out_dir / "shortcut_report.json", shortcut_report)
    write_json(out_dir / "baseline_report.json", baseline_report)
    write_json(out_dir / "placeholder_disposition_v0.json", placeholder_disposition)
    (out_dir / "yield_report.md").write_text(render_yield_markdown(yield_report), encoding="utf-8")
    (out_dir / "baseline_report.md").write_text(render_baseline_markdown(baseline_report), encoding="utf-8")
    (out_dir / "sample_fixtures.md").write_text(render_sample_fixtures(public_rows, manifest_rows, gold_rows), encoding="utf-8")
    print(f"Wrote v2 stream benchmark artifacts to {out_dir}")
    print(f"public_items={len(public_rows)} rejected_items={len(rejected_rows)}")
    if args.shortcut_gate:
        print(f"shortcut_gate_passed={shortcut_report['shortcut_gate_passed']}")
        if not shortcut_report["shortcut_gate_passed"]:
            failed = [key for key, passed in shortcut_report["gate_checks"].items() if not passed]
            print(f"shortcut_gate_failed_checks={failed}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
