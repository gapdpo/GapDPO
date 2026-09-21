from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def append_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def append_jsonl_record(record: dict[str, Any], output_path: str | Path) -> None:
    append_jsonl([record], output_path)


def read_existing_ids(path: str | Path) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(output_path):
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id:
            ids.add(record_id)
    return ids


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records
