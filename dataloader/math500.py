"""MATH-500 dataloader."""

from dataloader.base import BaseBenchmarkDataset


class Math500DataLoader(BaseBenchmarkDataset):
    """MATH-500 benchmark (HuggingFaceH4/MATH-500)."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("HuggingFaceH4/MATH-500", "test")
        entries = []
        for i, row in enumerate(ds):
            if i < 2:
                continue
            entries.append(self._entry(
                id_=str(row.get("unique_id", i)),
                question=row["problem"],
                answer=row.get("answer", row.get("solution", "")),
                source="math500",
                solution=row.get("solution", ""),
                level=row.get("level", ""),
                subject=row.get("subject", ""),
            ))
        print(f"[math500] {len(entries)} entries")
        return entries