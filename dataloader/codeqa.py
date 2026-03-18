"""CodeQA dataloader."""

import os
from pathlib import Path

from dataloader.base import BaseBenchmarkDataset


class CodeQADataLoader(BaseBenchmarkDataset):
    """Question answering over source code (vm2825/codeqa-dataset).

    Set CODEQA_PATH env var to a local JSON/JSONL file to skip the HF download.
    """

    def _load_all(self) -> list[dict]:
        local = os.environ.get("CODEQA_PATH")
        if local:
            rows = self._load_json_or_jsonl(Path(local))
        else:
            rows = list(self._hf("vm2825/codeqa-dataset", "train"))

        entries = []
        for i, row in enumerate(rows):
            entries.append(self._entry(
                id_=str(row.get("id", i)),
                question=row["Instruction"],
                answer=row.get("output_code", ""),
                context=row.get("input_code") or None,
                source="codeqa",
            ))
        print(f"[codeqa] {len(entries)} entries")
        return entries
