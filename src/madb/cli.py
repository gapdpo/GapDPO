from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from .agents import AttackerAgent, VictimAgent
from .config import GenerationConfig, load_json_list, validate_persona_catalog_capacity, validate_personas
from .llm_client import MockLLMClient, VLLMClient
from .orchestrator import DatasetOrchestrator
from .qa import write_json_report, write_markdown_report
from .safety import SafetyMasker
from .writer import append_jsonl_record, read_existing_ids, read_jsonl, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-agent deepvoice phishing defensive JSONL data.")
    parser.add_argument("--config", default="configs/default.json", help="Path to generation config JSON.")
    parser.add_argument("--output", default="data/generated/run.jsonl", help="Output JSONL path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Record count target. With --append, generate this many new records; with --resume, top up output to this total count.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based generation index to start from. Use 12 with --limit 12 to generate pairs 13-24 only.",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=("rotating", "canonical_repeat"),
        default=None,
        help="Generation schedule. Defaults to config; canonical_repeat repeats canonical victim/attacker pairs for repair runs.",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock clients instead of vLLM endpoints.")
    append_mode = parser.add_mutually_exclusive_group()
    append_mode.add_argument(
        "--append",
        action="store_true",
        help="Append new records to output JSONL while skipping record ids already present.",
    )
    append_mode.add_argument(
        "--resume",
        action="store_true",
        help="Skip record ids already present in output and append only the records needed to reach --limit.",
    )
    parser.add_argument("--errors-output", default=None, help="JSONL path for per-record generation errors.")
    parser.add_argument("--qa-report", default=None, help="Write a Markdown QA summary for the output records.")
    parser.add_argument("--qa-report-json", default=None, help="Write a JSON QA summary for the output records.")
    progress_mode = parser.add_mutually_exclusive_group()
    progress_mode.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=False,
        help="Write generation progress and ETA to stderr.",
    )
    progress_mode.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable generation progress output.",
    )
    parser.add_argument(
        "--fail-on-final-review",
        action="store_true",
        help="Return exit code 1 when final_dialogue_corpus_review.review_required is true in a QA report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    config = GenerationConfig.from_file(args.config)
    if args.schedule_mode is not None:
        config = replace(config, schedule_mode=args.schedule_mode)
    victims = load_json_list(config.victims_path)
    attackers = load_json_list(config.attackers_path)
    validate_personas(victims, kind="victim")
    validate_personas(attackers, kind="attacker")
    validate_persona_catalog_capacity(victims, attackers, limit=config.persona_catalog_limit)

    total_target = args.limit if args.limit is not None else config.default_limit
    existing_records = read_jsonl(args.output) if args.resume and Path(args.output).exists() else []
    if args.resume:
        existing_ids = {
            record["id"]
            for record in existing_records
            if isinstance(record.get("id"), str) and record["id"]
        }
    elif args.append:
        existing_ids = read_existing_ids(args.output)
    else:
        existing_ids = set()
    generation_limit = max(total_target - len(existing_records), 0) if args.resume else total_target
    errors_output = args.errors_output or default_errors_path(args.output)
    records: list[dict] = []
    error_count = 0

    if generation_limit > 0:
        orchestrator = build_orchestrator(config, use_mock=args.mock)
        progress = ProgressReporter(generation_limit, enabled=args.progress)

        try:
            if args.append or args.resume:
                for event in orchestrator.iter_generate(
                    victims,
                    attackers,
                    limit=generation_limit,
                    start_index=args.start_index,
                    skip_record_ids=existing_ids,
                ):
                    if event.record is not None:
                        append_jsonl_record(event.record, args.output)
                        records.append(event.record)
                        progress.record()
                    elif event.error is not None:
                        append_jsonl_record(event.error, errors_output)
                        error_count += 1
                        progress.error()
            else:
                for event in orchestrator.iter_generate(
                    victims,
                    attackers,
                    limit=generation_limit,
                    start_index=args.start_index,
                ):
                    if event.record is not None:
                        records.append(event.record)
                        progress.record()
                    elif event.error is not None:
                        append_jsonl_record(event.error, errors_output)
                        error_count += 1
                        progress.error()
                write_jsonl(records, args.output)
        finally:
            progress.finish()
    else:
        if not (args.append or args.resume):
            write_jsonl(records, args.output)

    report_records = read_jsonl(args.output) if Path(args.output).exists() else records
    qa_summaries: list[dict] = []
    if args.qa_report:
        qa_summaries.append(
            write_markdown_report(
                report_records,
                args.qa_report,
                max_turns=config.max_turns,
                allowed_risk_labels=config.risk_label_set,
                source_path=args.output,
            )
        )
    if args.qa_report_json:
        qa_summaries.append(
            write_json_report(
                report_records,
                args.qa_report_json,
                max_turns=config.max_turns,
                allowed_risk_labels=config.risk_label_set,
                source_path=args.output,
            )
        )

    print(f"Wrote {len(records)} records to {args.output}")
    if error_count:
        print(f"Wrote {error_count} generation errors to {errors_output}")
    if has_schema_contract_failures(qa_summaries):
        print("QA schema contract validation failed; see the generated QA report.")
        return 1
    if args.fail_on_final_review and has_final_dialogue_corpus_review_failures(qa_summaries):
        print("QA final dialogue corpus review failed; see the generated QA report.")
        return 1
    return 0


@dataclass
class ProgressReporter:
    target: int
    enabled: bool
    stream: TextIO | None = None

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stderr
        self.started_at = time.monotonic()
        self.completed = 0
        self.errors = 0
        self.printed = False
        self.last_width = 0

    def record(self) -> None:
        self.completed += 1
        self.render()

    def error(self) -> None:
        self.errors += 1
        self.render()

    def finish(self) -> None:
        if not self.enabled or not self.printed:
            return
        if self.stream is not None and self.stream.isatty():
            self.stream.write("\n")
            self.stream.flush()

    def render(self) -> None:
        if not self.enabled:
            return
        if self.stream is None:
            return
        elapsed = max(time.monotonic() - self.started_at, 0.0)
        average_seconds = elapsed / self.completed if self.completed else 0.0
        remaining = max(self.target - self.completed, 0)
        eta = format_duration(remaining * average_seconds) if self.completed else "calculating"
        line = (
            f"Generated {self.completed}/{self.target} records | errors {self.errors} | "
            f"elapsed {format_duration(elapsed)} | avg {average_seconds:.1f}s/record | ETA {eta}"
        )
        if self.stream.isatty():
            padding = " " * max(self.last_width - len(line), 0)
            self.stream.write(f"\r{line}{padding}")
            self.last_width = len(line)
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()
        self.printed = True


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def build_orchestrator(config: GenerationConfig, *, use_mock: bool) -> DatasetOrchestrator:
    if use_mock:
        attacker_client = MockLLMClient()
        victim_client = MockLLMClient()
    else:
        attacker_endpoint = os.getenv("MODEL_ATTACKER_ENDPOINT")
        victim_endpoint = os.getenv("MODEL_VICTIM_ENDPOINT")
        if not attacker_endpoint or not victim_endpoint:
            raise SystemExit(
                "MODEL_ATTACKER_ENDPOINT and MODEL_VICTIM_ENDPOINT must be set unless --mock is used."
            )
        attacker_client = VLLMClient(attacker_endpoint)
        victim_client = VLLMClient(victim_endpoint)

    return DatasetOrchestrator(
        config=config,
        attacker_agent=AttackerAgent(
            client=attacker_client,
            model=config.attacker_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ),
        victim_agent=VictimAgent(
            client=victim_client,
            model=config.victim_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ),
        safety=SafetyMasker(),
    )


def has_schema_contract_failures(summaries: list[dict]) -> bool:
    for summary in summaries:
        validation = summary.get("schema_contract_validation", {})
        if validation.get("enabled") is True and validation.get("fail", 0) > 0:
            return True
    return False


def has_final_dialogue_corpus_review_failures(summaries: list[dict]) -> bool:
    return any(
        summary.get("final_dialogue_corpus_review", {}).get("review_required") is True
        for summary in summaries
    )


def default_errors_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}.errors.jsonl"))


if __name__ == "__main__":
    raise SystemExit(main())
