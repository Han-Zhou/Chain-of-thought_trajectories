"""Load datasets via the dataloader and save them as pickle files.

For datasets with > 1000 entries, a randomized subset of 500 is saved.
"""

import os
import pickle
import random
import argparse

from dataloader import load_dataset

# Datasets supported by the eval registry (from i28_eval-changes.md)
DATASETS = [
    "logiqa",
    "college_math",
    "bigbench_causal",
    "bigbench_movie",
    "hotpotqa",
    "math500",
    "olympiadbench",
]

MAX_SIZE = 1000
SUBSET_SIZE = 500
SEED = 42


def dump_dataset(name: str, out_dir: str):
    dataset = load_dataset(name)
    entries = [dataset[i] for i in range(len(dataset))]
    total = len(entries)

    if total > MAX_SIZE:
        random.seed(SEED)
        entries = random.sample(entries, SUBSET_SIZE)
        print(f"  {name}: {total} entries -> sampled {len(entries)}")
    else:
        print(f"  {name}: {total} entries (kept all)")

    out_path = os.path.join(out_dir, f"{name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(entries, f)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir", type=str, default="pickles",
        help="Directory to save pickle files (default: pickles/)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to dump. Defaults to all: {DATASETS}",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = args.datasets or DATASETS

    for name in datasets:
        print(f"Loading {name}...")
        try:
            dump_dataset(name, args.out_dir)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
