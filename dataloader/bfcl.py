"""BFCL v4 dataloader."""

import json
from pathlib import Path

from dataloader.base import BaseBenchmarkDataset

_BFCL_DIR = Path("/storage/backup/han/cot/bfcl")

_TYPES = ("simple_python", "multiple", "parallel", "parallel_multiple")


class BFCLDataLoader(BaseBenchmarkDataset):
    """Load all four BFCL v4 categories from local disk.

    File-naming note: for parallel_multiple the question/answer files are
    stored swapped relative to the other categories:
      v4_parallel_multiple.json         → ground truth  (answers)
      v4_parallel_multiple_answer.json  → questions + function specs
    All other types follow:
      v4_{type}.json         → questions + function specs
      v4_{type}_answer.json  → ground truth
    """

    def _load_all(self) -> list[dict]:
        all_entries: list[dict] = []
        for bfcl_type in _TYPES:
            entries = self._load_type(bfcl_type)
            print(f"  [bfcl/{bfcl_type}] {len(entries)} entries")
            all_entries.extend(entries)
        print(f"[bfcl] {len(all_entries)} total entries")
        return all_entries

    def _load_type(self, bfcl_type: str) -> list[dict]:
        if bfcl_type == "parallel_multiple":
            q_path = _BFCL_DIR / "v4_parallel_multiple_answer.json"
            a_path = _BFCL_DIR / "v4_parallel_multiple.json"
        else:
            q_path = _BFCL_DIR / f"v4_{bfcl_type}.json"
            a_path = _BFCL_DIR / f"v4_{bfcl_type}_answer.json"

        questions = self._load_json_or_jsonl(q_path)
        answers = {
            row["id"]: row["ground_truth"]
            for row in self._load_json_or_jsonl(a_path)
        }

        entries = []
        for q in questions:
            qid = q["id"]
            turns = q["question"][0] if q["question"] else []
            user_msg = next(
                (t["content"] for t in turns if t.get("role") == "user"), ""
            )
            ground_truth = answers.get(qid, [])
            entries.append(self._entry(
                id_=qid,
                question=user_msg,
                answer=json.dumps(ground_truth),
                functions=q.get("function", []),
                ground_truth=ground_truth,
                bfcl_type=bfcl_type,
            ))
        return entries
