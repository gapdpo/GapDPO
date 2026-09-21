import tempfile
import unittest
from pathlib import Path

from madb.writer import append_jsonl_record, read_existing_ids, read_jsonl, write_jsonl


class WriterTest(unittest.TestCase):
    def test_append_and_read_existing_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"

            count = write_jsonl([{"id": "a"}], path)
            append_jsonl_record({"id": "b"}, path)

            self.assertEqual(count, 1)
            self.assertEqual(read_existing_ids(path), {"a", "b"})
            self.assertEqual([record["id"] for record in read_jsonl(path)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
