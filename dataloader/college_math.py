"""College Math Test dataloader."""

from dataloader.base import BaseBenchmarkDataset


class CollegeMathDataLoader(BaseBenchmarkDataset):
    """College-level math problems (di-zhang-fdu/College_Math_Test)."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("di-zhang-fdu/College_Math_Test", "test")
        entries = []
        for i, row in enumerate(ds):
            entries.append(self._entry(
                id_=str(row.get("question_number", i)),
                question=row["question"],
                answer=str(row.get("answer", "")),
                source=row.get("data_source", ""),
                data_topic=row.get("data_topic", ""),
            ))
        print(f"[college_math] {len(entries)} entries")
        return entries
