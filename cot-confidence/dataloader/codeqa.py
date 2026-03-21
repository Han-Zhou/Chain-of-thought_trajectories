"""CodeQA dataloader."""

import os
from pathlib import Path

from dataloader.base import BaseBenchmarkDataset


class CodeQADataLoader(BaseBenchmarkDataset):
    """Question answering over source code (lissadesu/codeqa_v2).

    Set CODEQA_PATH env var to a local JSON/JSONL file to skip the HF download.
    """

    def _load_all(self) -> list[dict]:
        local = os.environ.get("CODEQA_PATH")
        if local:
            rows = self._load_json_or_jsonl(Path(local))
        else:
            rows = list(self._hf("lissadesu/codeqa_v2", "train"))

        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=str(row.get("id", i)),
                question=row["question"],
                answer=row.get("answer", ""),
                context=row.get("code") or row.get("code_processed") or None,
                source="codeqa",
                question_type=row.get("questionType", ""),
            ))
        print(f"[codeqa] {len(entries)} entries")
        return entries
