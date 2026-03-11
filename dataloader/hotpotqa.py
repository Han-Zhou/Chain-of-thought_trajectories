"""HotPotQA dataloader."""

from dataloader.base import BaseBenchmarkDataset


class HotPotQADataLoader(BaseBenchmarkDataset):
    """HotPotQA full-wiki validation split (hotpotqa/hotpot_qa)."""

    def _load_all(self) -> list[dict]:
        ds = self._hf("hotpotqa/hotpot_qa", "validation", name="fullwiki")
        entries = []
        for i, row in enumerate(ds):
            ctx_dict = row.get("context") or {}
            titles = ctx_dict.get("title") or []
            sent_lists = ctx_dict.get("sentences") or []
            context = "\n\n".join(
                f"{t}: {' '.join(s)}" for t, s in zip(titles, sent_lists)
            ) or None

            entries.append(self._entry(
                id_=row.get("id", f"hotpotqa_{i}"),
                question=row["question"],
                answer=row["answer"],
                context=context,
                answer_type=row.get("type", ""),
                level=row.get("level", ""),
                source="hotpotqa/fullwiki",
            ))
        print(f"[hotpotqa] {len(entries)} entries")
        return entries
