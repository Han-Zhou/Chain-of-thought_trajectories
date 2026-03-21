"""CS1QA dataloader."""

from pathlib import Path

from dataloader.base import BaseBenchmarkDataset

_CS1QA_DIR = Path("/storage/backup/han/cot/cs1qa")


class CS1QADataLoader(BaseBenchmarkDataset):
    """QA dataset for introductory programming courses (Lee et al., 2022).

    Loads the test split from local disk.
    Source: https://github.com/cyoon47/CS1QA
    """

    def _load_all(self) -> list[dict]:
        path = _CS1QA_DIR / "test_cleaned.jsonl"
        rows = self._load_json_or_jsonl(path)
        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=str(i),
                question=row.get("question", ""),
                answer=str(row.get("answer", "")),
                context=row.get("code", None),
                source="cs1qa",
                question_type=row.get("questionType", None),
            ))
        print(f"[cs1qa] {len(entries)} entries")
        return entries