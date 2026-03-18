"""OlympiadBench dataloader."""

from dataloader.base import BaseBenchmarkDataset


class OlympiadBenchDataLoader(BaseBenchmarkDataset):
    """English math Olympiad problems, text-only (Hothan/OlympiadBench)."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("Hothan/OlympiadBench", "train", name="OE_TO_maths_en_COMP")
        entries = []
        for i, row in enumerate(ds):
            raw_ans = row.get("final_answer") or []
            answer = "; ".join(str(a) for a in raw_ans) if raw_ans else ""
            entries.append(self._entry(
                id_=str(row.get("id", i)),
                question=row["question"],
                answer=answer,
                context=row.get("context") or None,
                source="olympiadbench",
                subfield=row.get("subfield", ""),
                answer_type=row.get("answer_type", ""),
                is_multiple_answer=row.get("is_multiple_answer", False),
                unit=row.get("unit", ""),
                difficulty=row.get("difficulty", ""),
            ))
        print(f"[olympiadbench] {len(entries)} entries")
        return entries
