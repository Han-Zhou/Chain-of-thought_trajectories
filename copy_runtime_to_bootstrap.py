#!/usr/bin/env python3
"""
Copy runtime_seconds from full trajectories to bootstrap trajectories as runtime_vanilla_seconds.
Full trajectories: traj_N_sample_0.json  ->  Bootstrap: traj_N.json
"""

import json
import os
from pathlib import Path

# Mapping from bootstrap subdirectory names (under 1_true_bootstrap) to full trajectory
# subdirectory names (under 1_full_trajectories). Omits bootstrap-bm-430-qwen and qwen
# cs1qa_span MERGED runs (not produced yet). Skips "prev" and non-data dirs.
BOOTSTRAP_TO_FULL_MAPPING = {
    # "bootstrap-hotpotqa-505-llama_llama_regular_hotpotqa_zero_0_7405_type2_conf_bootstrap_mod-none" : "cots-427-llama-hotpotqa-32_llama_regular_hotpotqa_zero_0_500_type2_conf_vanillaonly_mod-none_samples32",
    # "bootstrap-bfcl-504-llama_llama_regular_bfcl_zero_0_1000_type2_conf_bootstrap_mod-none" : "cots-420-llama-bfcl-32_llama_regular_bfcl_zero_0_500_type2_conf_vanillaonly_mod-none_samples32",
    # "bootstrap-cs1qa-span-505-llama_llama_regular_cs1qa_span_zero_0_989_type2_conf_bootstrap_mod-none" : "cots-427-llama-cs1qa_span-32_llama_regular_cs1qa_span_zero_0_500_type2_conf_vanillaonly_mod-none_samples32",
    # "bootstrap-hotpotqa-503-qwen_qwen_regular_hotpotqa_zero_0_7405_type1_conf_bootstrap_mod-none" : "MERGED-cots-503-qwen-hotpotqa-8"
    # "bootstrap-bm-504-qwen_qwen_regular_bigbench_movie_zero_0_250_type1_conf_bootstrap_mod-none" : "MERGED-cots-430-qwen-bm-8"


    # "bootstrap-codeqa-505-llama_llama_regular_coeqa_zero_0_35358_type2_conf_bootstrap_mod-none" : "cots-422-llama-codeqa-32_llama_regular_codeqad_zero_0_500_type2_conf_vanillaonly_mod-none_samples32"

    # "bootstrap-bfcl-505-qwen_qwen_regular_bfcl_zero_0_1000_type1_conf_bootstrap_mod-none" : "MERGED-cots-503-qwen-bfcl-8" ,
    # "bootstrap-logiqa-505-qwen_qwen_regular_logiqa_zero_0_651_type2_conf_bootstrap_mod-none" : "MERGED-cots-503-qwen-logiqa-8"

    "bootstrap-codeqa-505-qwen_qwen_regular_codeqa_zero_0_35358_type1_conf_bootstrap_mod-none" : "MERGED-cots-503-qwen-codeqa-8",
    "bootstrap-cs1qa-span-505-qwen_qwen_regular_cs1qa_span_zero_0_989_type1_conf_bootstrap_mod-none" : "MERGED-cots-503-qwen-cs1qa_span-8",
    
    "bootstrap-hotpotqa-505-qwen_qwen_regular_hotpotqa_zero_0_7405_type1_conf_bootstrap_mod-none" : "MERGED-cots-503-qwen-hotpotqa-8"
}

BASE_DIR = Path("/storage/backup/han/backup_workspace/cot-eval/trajectories")
BOOTSTRAP_DIR = BASE_DIR / "1_true_bootstrap"
FULL_DIR = BASE_DIR / "1_full_trajectories"


def main():
    total_updated = 0
    total_missing = 0
    total_no_runtime = 0

    for bootstrap_name, full_name in BOOTSTRAP_TO_FULL_MAPPING.items():
        bootstrap_path = BOOTSTRAP_DIR / bootstrap_name
        full_path = FULL_DIR / full_name

        if not bootstrap_path.exists():
            print(f"Warning: Bootstrap directory not found: {bootstrap_path}")
            continue

        if not full_path.exists():
            print(f"Warning: Full trajectory directory not found: {full_path}")
            continue

        print(f"\nProcessing: {bootstrap_name}")
        print(f"  -> {full_name}")

        # Get all trajectory files in bootstrap directory
        bootstrap_files = sorted(bootstrap_path.glob("traj_*.json"))

        updated_count = 0
        missing_count = 0
        no_runtime_count = 0

        for bootstrap_file in bootstrap_files:
            # Extract trajectory number: traj_0 -> traj_N_sample_0
            traj_num = bootstrap_file.stem  # e.g., "traj_0"
            # Try both _sample_0 and _sample_00 patterns
            full_file_sample0 = full_path / f"{traj_num}_sample_0.json"
            if not full_file_sample0.exists():
                # Try without trailing _0: traj_0 might be the same file
                full_file_sample0 = full_path / f"{traj_num}.json"

            if not full_file_sample0.exists():
                print(f"  Missing full trajectory: {full_file_sample0.name}")
                missing_count += 1
                continue

            # Read runtime_seconds from full trajectory
            try:
                with open(full_file_sample0, 'r') as f:
                    full_data = json.load(f)

                # runtime_seconds is inside the 'trajectory' field
                if "trajectory" not in full_data:
                    print(f"  No trajectory field in {full_file_sample0.name}")
                    no_runtime_count += 1
                    continue

                trajectory = full_data["trajectory"]
                if "runtime_seconds" not in trajectory:
                    print(f"  No runtime_seconds in {full_file_sample0.name}")
                    no_runtime_count += 1
                    continue

                runtime_seconds = trajectory["runtime_seconds"]

                # Read bootstrap trajectory
                with open(bootstrap_file, 'r') as f:
                    bootstrap_data = json.load(f)

                # Add runtime_vanilla_seconds field
                bootstrap_data["runtime_vanilla_seconds"] = runtime_seconds

                # Sync trajectory.correct from full to bootstrap (overwrite if present)
                if "correct" in trajectory and "trajectory" in bootstrap_data:
                    bootstrap_data["trajectory"]["correct"] = trajectory["correct"]

                # Write back to bootstrap file
                with open(bootstrap_file, 'w') as f:
                    json.dump(bootstrap_data, f, indent=2)

                updated_count += 1

            except Exception as e:
                print(f"  Error processing {bootstrap_file.name}: {e}")
                missing_count += 1

        print(f"  Updated: {updated_count}, Missing: {missing_count}, No runtime: {no_runtime_count}")
        total_updated += updated_count
        total_missing += missing_count
        total_no_runtime += no_runtime_count

    print(f"\n{'='*60}")
    print(f"Total updated: {total_updated}")
    print(f"Total missing: {total_missing}")
    print(f"Total no runtime field: {total_no_runtime}")


if __name__ == "__main__":
    main()
