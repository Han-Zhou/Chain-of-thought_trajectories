"""College Math Test dataloader."""

from dataloader.base import BaseBenchmarkDataset


class CollegeMathDataLoader(BaseBenchmarkDataset):
    """Math questions from MMLU-Pro (TIGER-Lab/MMLU-Pro), math category only."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("TIGER-Lab/MMLU-Pro", "test")
        entries = []
        for i, row in enumerate(ds):
            cat = (row.get("category") or "").lower()
            if cat not in ("math", "mathematics"):
                continue
            options = row.get("options") or []
            choices = [f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)]
            entries.append(self._entry(
                id_=str(row.get("question_id", i)),
                question=row["question"],
                answer=str(row.get("answer", "")),
                choices=choices or None,
                source="mmlu_pro/math",
                category=row.get("category", ""),
                cot_content=row.get("cot_content", ""),
            ))
        print(f"[college_math] {len(entries)} entries")
        return entries
