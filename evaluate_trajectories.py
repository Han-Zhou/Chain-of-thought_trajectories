import json
import os
from huggingface_hub import snapshot_download
import argparse


RUN = "qwen_logiqa_zero_1"

# place 'trajectories' before the last underscore in RUN
if "_" in RUN:
    prefix, suffix = RUN.rsplit("_", 1)
    file_name = f"{prefix}_trajectories_{suffix}.json"
else:
    file_name = f"{RUN}_trajectories.json"

FILE_PATH = f"./trajectories/{RUN}/{file_name}"

def main():
    with open(FILE_PATH, "r") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    trajectories = []
    idx = 0
    while idx < len(content):
        # skip whitespace
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        obj, end = decoder.raw_decode(content, idx)
        trajectories.append(obj)
        idx = end

        # dict_keys(['id', 'question', 'ground_truth', 'cot_steps', 'raw_cot_block', 'generated_text', 'prompt_token_positions', 'generated_token_positions'])
        t = trajectories[0]["trajectory"]

        breakpoint()



if __name__ == "__main__":
    main()




