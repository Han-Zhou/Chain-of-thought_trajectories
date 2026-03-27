"""College Math Test dataloader."""

from dataloader.base import BaseBenchmarkDataset


class CollegeMathDataLoader(BaseBenchmarkDataset):
    """Math questions from di-zhang-fdu/College_Math_Test."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("di-zhang-fdu/College_Math_Test", "test")
        entries = []
        for i, row in enumerate(ds):
            if i < 2:
                continue
            entries.append(self._entry(
                id_=str(row.get("question_number", i)),
                question=row["question"],
                answer=str(row.get("answer", "")),
                source=row.get("data_source", "college_math"),
                category=row.get("data_topic", ""),
            ))
        print(f"[college_math] {len(entries)} entries")
        return entries
