from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from benchmark_stream_v1.constants import DIAGNOSTIC_FAMILY, OFFICIAL_CANDIDATE_FAMILY
from benchmark_stream_v1.stream_logic import build_yield_report, generate_stream_items


def main() -> int:
    parser = argparse.ArgumentParser(description="Build selective stream-intervention benchmark artifacts.")
    parser.add_argument("--source", required=True, help="Source JSONL dialogue records.")
    parser.add_argument("--qa", help="Optional QA JSON for placeholder disposition.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--family",
        default=DIAGNOSTIC_FAMILY,
        choices=[DIAGNOSTIC_FAMILY, OFFICIAL_CANDIDATE_FAMILY],
        help="Stream benchmark family name.",
    )
    parser.add_argument(
        "--surface",
        default="typed_raw",
        choices=["typed_raw", "neutral_mask", "placeholder_ablated"],
        help="Public text surface.",
    )
    parser.add_argument("--window-size", type=int, default=6, help="Maximum dialogue turns visible per item.")
    parser.add_argument(
        "--keep-risky-placeholders",
        action="store_true",
        help="Keep rows with unreviewed placeholder risk as diagnostic instead of rejecting them.",
    )
    args = parser.parse_args()

    if args.window_size < 1:
        raise SystemExit("--window-size must be positive")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.source)
    qa = read_json(args.qa) if args.qa else {"quality_signals": {"unknown_masked_placeholder_policy": {"placeholders": {}}}}
    placeholder_disposition = placeholder_disposition_from_qa(qa)
    items, rejected = generate_stream_items(
        records,
        placeholder_disposition,
        family=args.family,
        surface=args.surface,
        window_size=args.window_size,
        exclude_risky_placeholders=not args.keep_risky_placeholders,
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
    yield_report.update(
        {
            "status": "diagnostic_pilot" if args.family == DIAGNOSTIC_FAMILY else "official_candidate_pending_gates",
            "source_path": str(args.source),
            "qa_path": str(args.qa) if args.qa else None,
            "family": args.family,
            "surface": args.surface,
            "window_size": args.window_size,
            "public_whitelist_violation_count": len(whitelist_violations),
            "public_whitelist_violations": whitelist_violations[:20],
        }
    )
    baseline_report = build_baseline_report(public_rows, gold_rows, manifest_rows)

    write_jsonl(out_dir / "public_items.jsonl", public_rows)
    write_jsonl(out_dir / "private_manifest.jsonl", manifest_rows)
    write_jsonl(out_dir / "gold_stream.jsonl", gold_rows)
    write_jsonl(out_dir / "rejected_items.jsonl", rejected_rows)
    write_json(out_dir / "yield_report.json", yield_report)
    write_json(out_dir / "baseline_report.json", baseline_report)
    write_json(out_dir / "placeholder_disposition_v0.json", placeholder_disposition)
    (out_dir / "yield_report.md").write_text(render_yield_markdown(yield_report), encoding="utf-8")
    (out_dir / "baseline_report.md").write_text(render_baseline_markdown(baseline_report), encoding="utf-8")
    (out_dir / "sample_fixtures.md").write_text(render_sample_fixtures(public_rows, manifest_rows, gold_rows), encoding="utf-8")

    print(f"Wrote stream benchmark artifacts to {out_dir}")
    print(f"public_items={len(public_rows)} rejected_items={len(rejected_rows)}")
    if whitelist_violations:
        print(f"WARNING: public whitelist violations={len(whitelist_violations)}")
    return 0


def render_yield_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stream Benchmark Yield Report",
        "",
        f"- status: {report.get('status')}",
        f"- family: {report.get('family')}",
        f"- source_path: {report.get('source_path')}",
        f"- surface: {report.get('surface')}",
        f"- window_size: {report.get('window_size')}",
        f"- public_item_count: {report.get('public_item_count')}",
        f"- stream_count: {report.get('stream_count')}",
        f"- rejected_count: {report.get('rejected_count')}",
        f"- public_whitelist_violation_count: {report.get('public_whitelist_violation_count')}",
        "",
        "## Event Distribution",
        "",
        *render_mapping(report.get("event_distribution", {})),
        "",
        "## Action Distribution",
        "",
        *render_mapping(report.get("action_distribution", {})),
        "",
        "## Release Floors",
        "",
        "### Events",
        "",
        *render_mapping(report.get("release_floor_30_by_event", {})),
        "",
        "### Actions",
        "",
        *render_mapping(report.get("release_floor_30_by_action", {})),
        "",
        "## Rejections",
        "",
        *render_mapping(report.get("rejected_by_reason", {})),
        "",
    ]
    return "\n".join(lines)


def render_baseline_markdown(report: dict[str, Any]) -> str:
    lines = ["# Stream Benchmark Baseline Report", "", f"- scope: {report.get('scope')}", ""]
    for name, detail in report.get("baselines", {}).items():
        lines.extend([f"## {name}", ""])
        lines.extend(render_mapping(detail))
        lines.append("")
    return "\n".join(lines)


def render_sample_fixtures(
    public_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> str:
    manifest_by_id = {row["benchmark_id"]: row for row in manifest_rows}
    gold_by_id = {row["benchmark_id"]: row for row in gold_rows}
    lines = ["# Stream Benchmark Sample Fixtures", ""]
    for public in public_rows[:3]:
        benchmark_id = public["benchmark_id"]
        lines.extend(
            [
                f"## {benchmark_id}",
                "",
                "Public:",
                "",
                "```json",
                json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
                "Manifest:",
                "",
                "```json",
                json.dumps(manifest_by_id[benchmark_id], ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
                "Gold:",
                "",
                "```json",
                json.dumps(gold_by_id[benchmark_id], ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def render_mapping(mapping: dict[str, Any]) -> list[str]:
    if not mapping:
        return ["- none"]
    return [f"- {key}: {value}" for key, value in sorted(mapping.items())]


if __name__ == "__main__":
    raise SystemExit(main())
