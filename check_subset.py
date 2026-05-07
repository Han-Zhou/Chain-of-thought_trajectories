#!/usr/bin/env python3
import os
from pathlib import Path

traj_dir = Path("trajectories")
test_426_dirs = sorted([d for d in traj_dir.iterdir() if d.is_dir() and d.name.startswith("test-426")])

# Group by base name
base_to_dirs = {}
for d in test_426_dirs:
    if "_sidx" in d.name:
        base = d.name.split("_sidx")[0]
        sidx = d.name.split("_sidx")[1]
    else:
        base = d.name
        sidx = "none"

    if base not in base_to_dirs:
        base_to_dirs[base] = {}
    base_to_dirs[base][sidx] = d

# Check if non-suffixed is subset of sidx1
for base, dirs in sorted(base_to_dirs.items()):
    if "none" in dirs and "1" in dirs:
        no_suffix_dir = dirs["none"]
        sidx1_dir = dirs["1"]

        no_suffix_files = set(f.name for f in no_suffix_dir.glob("traj_*.json"))
        sidx1_files = set(f.name for f in sidx1_dir.glob("traj_*.json"))

        is_subset = no_suffix_files.issubset(sidx1_files)

        print(f"\nBase: {base.split('/')[-1][:60]}...")
        print(f"  No suffix: {len(no_suffix_files)} files")
        print(f"  sidx1:     {len(sidx1_files)} files")
        print(f"  Is subset: {is_subset}")

        if not is_subset:
            missing = no_suffix_files - sidx1_files
            print(f"  Files in no-suffix but NOT in sidx1: {sorted(missing)[:5]}")
