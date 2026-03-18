"""Base class for all benchmark dataset loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import json

from torch.utils.data import Dataset


class BaseBenchmarkDataset(Dataset, ABC):
    """PyTorch Dataset base for benchmark loaders."""

    def __init__(self) -> None:
        self._data: list[dict] = self._load_all()

    @abstractmethod
    def _load_all(self) -> list[dict]:
        """Fetch and normalize all dataset entries."""
        ...

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    # ------------------------------------------------------------------
    # Shared helpers – available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _entry(
        id_: str,
        question: str,
        answer: Any,
        *,
        context: str | None = None,
        choices: list | None = None,
        **metadata,
    ) -> dict:
        """Build a normalized dataset entry."""
        return {
            "id": id_,
            "question": question,
            "answer": answer,
            "context": context,
            "choices": choices,
            "metadata": metadata,
        }

    @staticmethod
    def _load_json_or_jsonl(path: Path) -> list[dict]:
        """Read a file that is either a JSON array or JSONL."""
        with open(path, encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _hf(repo: str, split: str, *, name: str | None = None, token: str | None = None) -> Any:
        """Load a HuggingFace dataset split."""
        from datasets import load_dataset
        kwargs: dict = {}
        if name:
            kwargs["name"] = name
        if token:
            kwargs["token"] = token
        return load_dataset(repo, split=split, **kwargs)
