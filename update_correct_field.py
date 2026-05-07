#!/usr/bin/env python3
"""
Update all trajectory files in the specified directory:
Copy correct_llm_judge value to correct field.
"""

import json
import os
from pathlib import Path

def update_trajectory_file(filepath):
    """Update a single trajectory file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check if trajectory has correct_llm_judge
    if 'trajectory' in data and 'correct_llm_judge' in data['trajectory']:
        # Copy correct_llm_judge to correct
        data['trajectory']['correct'] = data['trajectory']['correct_llm_judge']

        # Write back
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return True
    return False

def main():
    directory = "trajectories/1_full_trajectories/cots-422-llama-codeqa-32_llama_regular_codeqa_zero_0_500_type2_conf_vanillaonly_mod-none_samples32"

    # Find all trajectory files
    traj_files = sorted(Path(directory).glob("traj_*.json"))

    print(f"Found {len(traj_files)} trajectory files")

    updated_count = 0
    for filepath in traj_files:
        if update_trajectory_file(filepath):
            updated_count += 1

    print(f"Updated {updated_count} files")

if __name__ == "__main__":
    main()
