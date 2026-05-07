#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

def merge_trajectories(input_configs, output_dir):
    """
    Merge trajectory files from multiple directories.

    Args:
        input_configs: List of dicts with keys 'dir', 'start', 'end'
        output_dir: Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for config in input_configs:
        src_dir = Path(config['dir'])
        start_idx = config['start']
        end_idx = config['end']

        num_files = end_idx - start_idx + 1

        for i in range(num_files):
            # Check for multi-sample format first
            sample_files = list(src_dir.glob(f"traj_{i}_sample_*.json"))

            if sample_files:
                for sample_file in sample_files:
                    sample_suffix = sample_file.name.split('_sample_')[1]
                    dst_file = output_path / f"traj_{start_idx + i}_sample_{sample_suffix}"
                    shutil.copy2(sample_file, dst_file)
            else:
                # Fall back to single file format
                src_file = src_dir / f"traj_{i}.json"
                dst_file = output_path / f"traj_{start_idx + i}.json"

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Warning: {src_file} not found")

if __name__ == "__main__":
    # Example usage
    configs = [
        # {'dir': '', 'start': , 'end': },
        
       {'dir': 'trajectories/cots-430-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_0_500_type1_conf_vanillaonly_mod-none_samples8', 'start': 0, 'end': 71} ,
       
       {'dir': 'trajectories/cots-sup-430-p2-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_72_143_type1_conf_vanillaonly_mod-none_samples8', 'start': 72, 'end': 142} ,
       
       {'dir': 'trajectories/cots-sup-430-p3-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_143_214_type1_conf_vanillaonly_mod-none_samples8', 'start': 143, 'end': 213} ,
       
       {'dir': 'trajectories/cots-sup-430-p4-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_214_285_type1_conf_vanillaonly_mod-none_samples8', 'start': 214, 'end': 284} ,
       
       {'dir': 'trajectories/cots-sup-430-p5-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_285_356_type1_conf_vanillaonly_mod-none_samples8', 'start': 285, 'end': 355} ,
       
       {'dir': 'trajectories/cots-sup-430-p6-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_356_427_type1_conf_vanillaonly_mod-none_samples8', 'start': 356, 'end': 426} ,
       
       {'dir': 'trajectories/cots-sup-430-p7-qwen-hotpotqa-8_qwen_thinking_hotpotqa_zero_427_500_type1_conf_vanillaonly_mod-none_samples8', 'start': 427, 'end': 499} ,
       
    ]
    output_dir = 'trajectories/1_full_trajectories/MERGED-cots-503-qwen-hotpotqa-8'

    merge_trajectories(configs, output_dir)










