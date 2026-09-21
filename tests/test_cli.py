import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from madb.cli import build_parser, main
from madb.config import GenerationConfig


class CliTest(unittest.TestCase):
    def test_help_describes_append_and_resume_limit_semantics(self):
        help_text = " ".join(build_parser().format_help().split())

        self.assertIn("With --append, generate this many new records", help_text)
        self.assertIn("with --resume, top up output to this total count", help_text)
        self.assertIn("records needed to reach --limit", help_text)
        self.assertIn("generate pairs 13-24 only", help_text)

    def test_rejects_negative_limit_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "negative.jsonl"

            with self.assertRaises(SystemExit) as raised:
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "-1",
                    "--mock",
                ])

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_rejects_negative_start_index_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "negative_start.jsonl"

            with self.assertRaises(SystemExit) as raised:
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--start-index",
                    "-1",
                    "--mock",
                ])

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_start_index_generates_batch2_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "batch2.jsonl"
            qa_json = Path(tmpdir) / "qa.json"

            exit_code = main(
                [
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "12",
                    "--start-index",
                    "12",
                    "--mock",
                    "--qa-report-json",
                    str(qa_json),
                ]
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            summary = json.loads(qa_json.read_text(encoding="utf-8"))

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(records), 12)
            self.assertEqual(records[0]["victim_persona"]["id"], "seasonal_tax_filer")
            self.assertEqual(records[0]["scenario_type"], "tax_refund_notice")
            self.assertEqual(records[-1]["victim_persona"]["id"], "infant_childcare_applicant")
            self.assertEqual(records[-1]["scenario_type"], "childcare_service_notice")
            self.assertEqual(summary["record_count"], 12)
            self.assertEqual(summary["schema_contract_validation"]["fail"], 0)

    def test_rejects_persona_catalog_above_capacity_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            victims = tmpdir_path / "victims.json"
            attackers = tmpdir_path / "attackers.json"
            config = tmpdir_path / "config.json"
            output = tmpdir_path / "records.jsonl"

            victim = {
                "id": "victim_0",
                "name": "Victim 0",
                "description": "Synthetic defensive victim persona.",
                "vulnerabilities": ["urgency"],
                "resistance_style": "Verify through official channels.",
                "protected_assets": ["account_password"],
            }
            victims.write_text(
                json.dumps(
                    [
                        victim,
                        {**victim, "id": "victim_1", "name": "Victim 1"},
                    ]
                ),
                encoding="utf-8",
            )
            attackers.write_text(
                json.dumps(
                    [
                        {
                            "id": "attacker_0",
                            "name": "Attacker 0",
                            "description": "Synthetic defensive attacker persona.",
                            "default_goals": ["credential_capture"],
                            "pressure_methods": ["authority"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            config.write_text(
                json.dumps(
                    {
                        "dataset_name": "capacity_test",
                        "max_turns": 4,
                        "default_limit": 1,
                        "persona_catalog_limit": 1,
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "attacker_model": "mock-attacker",
                        "victim_model": "mock-victim",
                        "victims_path": str(victims),
                        "attackers_path": str(attackers),
                        "scenario_types": ["bank_fraud_alert"],
                        "risk_label_set": ["authority_impersonation"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "victim persona catalog size 2 exceeds persona_catalog_limit=1",
            ):
                main([
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                ])

            self.assertFalse(output.exists())

    def test_mock_generation_writes_qa_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"
            qa_markdown = Path(tmpdir) / "qa.md"
            qa_json = Path(tmpdir) / "qa.json"

            exit_code = main(
                [
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "2",
                    "--mock",
                    "--qa-report",
                    str(qa_markdown),
                    "--qa-report-json",
                    str(qa_json),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
            self.assertIn(
                "## Scenario Default Attacker Risk Signal Coverage",
                qa_markdown.read_text(encoding="utf-8"),
            )
            self.assertIn("### corpus residual patterns", qa_markdown.read_text(encoding="utf-8"))
            summary = json.loads(qa_json.read_text(encoding="utf-8"))
            self.assertIn("scenario_default_attacker_risk_signal_coverage", summary)
            self.assertIn("scenario_default_risk_alignment", summary)
            self.assertNotIn("unknown_risk_labels", summary)

    def test_progress_writes_eta_to_stderr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"
            stderr = io.StringIO()

            with patch("sys.stderr", stderr):
                exit_code = main(
                    [
                        "--config",
                        "configs/default.json",
                        "--output",
                        str(output),
                        "--limit",
                        "2",
                        "--mock",
                        "--progress",
                    ]
                )

            self.assertEqual(exit_code, 0)
            progress = stderr.getvalue()
            self.assertIn("Generated 1/2 records", progress)
            self.assertIn("Generated 2/2 records", progress)
            self.assertIn("ETA", progress)

    def test_fail_on_final_review_returns_nonzero_for_review_required_qa(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"
            qa_json = Path(tmpdir) / "qa.json"

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                ]),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            record["gen_metadata"]["target_outcome_mode"] = "masked_compromise"
            record["outcome"] = "inconclusive"
            record["compromised_assets"] = []
            output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(
                [
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--resume",
                    "--qa-report-json",
                    str(qa_json),
                    "--fail-on-final-review",
                ]
            )

            summary = json.loads(qa_json.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertTrue(summary["final_dialogue_corpus_review"]["review_required"])
            self.assertIn(
                "target_outcome_alignment_below_final_threshold",
                summary["final_dialogue_corpus_review"]["reasons"],
            )

    def test_mock_generation_limit_can_exceed_persona_catalog_limit(self):
        config = GenerationConfig.from_file("configs/default.json")
        requested_records = config.persona_catalog_limit + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"

            exit_code = main(
                [
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    str(requested_records),
                    "--mock",
                ]
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [record["id"] for record in records]

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(records), requested_records)
            self.assertEqual(len(set(ids)), requested_records)

    def test_append_skips_existing_record_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                ]),
                0,
            )
            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                    "--append",
                ]),
                0,
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [record["id"] for record in records]
            self.assertEqual(len(ids), 2)
            self.assertEqual(len(set(ids)), 2)

    def test_resume_tops_up_existing_output_to_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                ]),
                0,
            )
            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "3",
                    "--mock",
                    "--resume",
                ]),
                0,
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [record["id"] for record in records]
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(set(ids)), 3)

    def test_resume_counts_existing_records_not_unique_ids_for_top_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                ]),
                0,
            )
            first_line = output.read_text(encoding="utf-8")
            output.write_text(first_line + first_line, encoding="utf-8")

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "3",
                    "--mock",
                    "--resume",
                ]),
                0,
            )

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            ids = [record["id"] for record in records]
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(set(ids)), 2)

    def test_append_and_resume_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"

            with self.assertRaises(SystemExit) as raised:
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                    "--append",
                    "--resume",
                ])

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_resume_qa_revalidates_existing_records_against_config_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"
            qa_json = Path(tmpdir) / "qa.json"

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                ]),
                0,
            )
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            record["risk_labels"].append("not_configured")
            record["quality_checks"]["schema_valid"] = True
            output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            self.assertEqual(
                main([
                    "--config",
                    "configs/default.json",
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                    "--mock",
                    "--resume",
                    "--qa-report-json",
                    str(qa_json),
                ]),
                1,
            )

            summary = json.loads(qa_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["quality_check_summary"]["schema_valid"]["pass"], 1)
            self.assertEqual(summary["schema_contract_validation"]["fail"], 1)
            self.assertIn("Unknown risk_labels", summary["schema_contract_validation"]["failures"][0]["error"])
            self.assertNotIn("unknown_risk_labels", summary)

    def test_resume_qa_without_missing_records_does_not_require_model_endpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "records.jsonl"
            qa_json = Path(tmpdir) / "qa.json"

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(
                    main([
                        "--config",
                        "configs/default.json",
                        "--output",
                        str(output),
                        "--limit",
                        "1",
                        "--mock",
                    ]),
                    0,
                )

                self.assertEqual(
                    main([
                        "--config",
                        "configs/default.json",
                        "--output",
                        str(output),
                        "--limit",
                        "1",
                        "--resume",
                        "--qa-report-json",
                        str(qa_json),
                    ]),
                    0,
                )

            summary = json.loads(qa_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["schema_contract_validation"]["fail"], 0)


if __name__ == "__main__":
    unittest.main()
