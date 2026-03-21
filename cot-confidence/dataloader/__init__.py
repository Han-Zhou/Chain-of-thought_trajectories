"""Dataloader registry.

Usage:
    from dataloader import load_dataset, make_dataloader
    dataset = load_dataset("logiqa")          # BaseBenchmarkDataset
    loader  = make_dataloader(dataset, n=100) # torch DataLoader
"""

from torch.utils.data import DataLoader, Subset

from dataloader.base import BaseBenchmarkDataset
from dataloader.bfcl import BFCLDataLoader
from dataloader.bigbench import BigBenchMovieDataLoader, BigBenchCausalDataLoader
from dataloader.logiqa import LogiQADataLoader
from dataloader.codeqa import CodeQADataLoader
from dataloader.cs1qa import CS1QADataLoader
from dataloader.hotpotqa import HotPotQADataLoader
from dataloader.college_math import CollegeMathDataLoader
from dataloader.olympiadbench import OlympiadBenchDataLoader
from dataloader.math500 import Math500DataLoader
from dataloader.hle import HLEDataLoader

_REGISTRY: dict[str, type[BaseBenchmarkDataset]] = {
    "bfcl":            BFCLDataLoader,
    "bigbench_movie":  BigBenchMovieDataLoader,
    "bigbench_causal": BigBenchCausalDataLoader,
    "logiqa":          LogiQADataLoader,
    "codeqa":          CodeQADataLoader,
    "cs1qa":           CS1QADataLoader,
    "hotpotqa":        HotPotQADataLoader,
    "college_math":    CollegeMathDataLoader,
    "olympiadbench":   OlympiadBenchDataLoader,
    "math500":         Math500DataLoader,
    "hle":             HLEDataLoader,
}


def load_dataset(dataset_name: str) -> BaseBenchmarkDataset:
    """Instantiate and return the dataset for the given name."""
    name = dataset_name.strip().lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid names: {valid}"
        )
    return cls()


def make_dataloader(
    dataset: BaseBenchmarkDataset,
    n: int | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a dataset (optionally truncated to n samples) in a DataLoader.

    Each item is a plain dict; the default collate_fn is overridden to avoid
    PyTorch's tensor-stacking logic on heterogeneous dict values.
    """
    source = Subset(dataset, range(min(n, len(dataset)))) if n is not None else dataset
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0] if batch_size == 1 else batch,
    )


__all__ = [
    "BaseBenchmarkDataset",
    "BFCLDataLoader",
    "BigBenchMovieDataLoader",
    "BigBenchCausalDataLoader",
    "LogiQADataLoader",
    "CodeQADataLoader",
    "CS1QADataLoader",
    "HotPotQADataLoader",
    "CollegeMathDataLoader",
    "OlympiadBenchDataLoader",
    "Math500DataLoader",
    "HLEDataLoader",
    "load_dataset",
    "make_dataloader",
]
