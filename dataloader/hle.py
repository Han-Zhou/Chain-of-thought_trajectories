"""HLE (Humanity's Last Exam) dataloader."""

import os

from dataloader.base import BaseBenchmarkDataset


class HLEDataLoader(BaseBenchmarkDataset):
    """Humanity's Last Exam (cais/hle).

    This is a gated dataset. Accept the terms on HuggingFace and set
    HF_TOKEN before loading:
      export HF_TOKEN=hf_...
    """

    def _load_all(self) -> list[dict]:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HLE (Humanity's Last Exam) is a gated dataset. "
                "Accept the terms at https://huggingface.co/datasets/cais/hle "
                "and set HF_TOKEN=<your_token>."
            )

        from datasets import load_dataset as hf_load
        ds = hf_load("cais/hle", split="test", token=token)
        entries = []
        for i, row in enumerate(ds):
            if row.get("image") is not None:
                continue
            choices = None
            if row.get("answer_type") == "multiple_choice":
                raw = row.get("choices")
                choices = raw if isinstance(raw, list) else None
            entries.append(self._entry(
                id_=str(row.get("id", i)),
                question=row["question"],
                answer=str(row.get("answer", "")),
                choices=choices,
                source="hle",
                category=row.get("category", ""),
                answer_type=row.get("answer_type", ""),
            ))
        print(f"[hle] {len(entries)} entries")
        return entries
