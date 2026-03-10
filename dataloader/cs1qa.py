"""CS1QA dataloader."""

import os
from pathlib import Path

from dataloader.base import BaseBenchmarkDataset


class CS1QADataLoader(BaseBenchmarkDataset):
    """QA dataset for introductory programming courses (Lee et al., 2022).

    The dataset is not on HuggingFace. Set CS1QA_PATH to a local JSON/JSONL
    file with fields: id, question, answer, code (optional).

    Download from: https://github.com/cs1qa/cs1qa
    """

    def _load_all(self) -> list[dict]:
        local = os.environ.get("CS1QA_PATH")
        if not local:
            raise RuntimeError(
                "CS1QA is not available on HuggingFace. "
                "Download it from https://github.com/cs1qa/cs1qa "
                "and set CS1QA_PATH=/path/to/cs1qa_test.json"
            )

        rows = self._load_json_or_jsonl(Path(local))
        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=str(row.get("id", i)),
                question=row.get("question", ""),
                answer=str(row.get("answer", "")),
                context=row.get("code", None),
                source="cs1qa",
            ))
        print(f"[cs1qa] {len(entries)} entries")
        return entries
