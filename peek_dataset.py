import json
import os
from huggingface_hub import snapshot_download
import argparse

# 1. Fetch the entire dataset repository to a local directory
# print("Downloading BFCL dataset files...")
# local_dir = "./bfcl_data"
# snapshot_download(
#     repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
#     repo_type="dataset",
#     local_dir=local_dir
# )

# # 2. Target the specific category file you want to evaluate
# # Example: Loading the "Simple" function calling scenarios
# file_path = os.path.join(local_dir, "BFCL_v3_simple.json")

# # 3. Parse the data into a list of dictionaries
# dataset = []
# with open(file_path, 'r', encoding='utf-8') as f:
#     # Some BFCL files are standard JSON, others might be JSONL (JSON-lines). 
#     # This block safely handles both.
#     try:
#         dataset = json.load(f)
#     except json.JSONDecodeError:
#         f.seek(0)
#         for line in f:
#             if line.strip():
#                 dataset.append(json.loads(line))

# print(f"\nSuccessfully loaded {len(dataset)} evaluation cases!")

# # View the structure of the first test case
# # print("\nSample entry:")

# # for i, entry in enumerate(dataset):
# #     if "ground_truth" in entry:
# #         print(f"\nEntry {i}:")
# #         print(json.dumps(entry, indent=2))


# # Collect all unique columns across all entries
# all_columns = set()
# for entry in dataset:
#     all_columns.update(entry.keys())

# print(f"\nAll columns ({len(all_columns)} total):")
# for col in sorted(all_columns):
#     print(f"  - {col}")

# # Show a sample value for each column from the first entry that has it
# print("\nSample values per column:")
# for col in sorted(all_columns):
#     for entry in dataset:
#         if col in entry:
#             val = entry[col]
#             preview = json.dumps(val)
#             if len(preview) > 200:
#                 preview = preview[:200] + "..."
#             print(f"\n  [{col}]")
#             print(f"  {preview}")
#             break



args = argparse.ArgumentParser()
args.add_argument("--type", type=str, default="parallel", help="The type of dataset to process (e.g., 'parallel' or 'multiple').")

args = args.parse_args()


input_file1 = f"/shared_work/han/beam_search/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_{args.type}.json"
# 
input_file2 = f"/shared_work/han/beam_search/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_{args.type}.json"

output_path1 = f"/shared_work/han/beam_search/v4_{args.type}_answer.json"

output_path2 = f"/shared_work/han/beam_search/v4_{args.type}.json"

# 3. Parse the data into a list of dictionaries
dataset1 = []
with open(input_file1, 'r', encoding='utf-8') as f:
    # Some BFCL files are standard JSON, others might be JSONL (JSON-lines). 
    # This block safely handles both.
    try:
        dataset1 = json.load(f)
    except json.JSONDecodeError:
        f.seek(0)
        for line in f:
            if line.strip():
                dataset1.append(json.loads(line))

dataset2 = []   
with open(input_file2, 'r', encoding='utf-8') as f:
    # Some BFCL files are standard JSON, others might be JSONL (JSON-lines). 
    # This block safely handles both.
    try:
        dataset2 = json.load(f)
    except json.JSONDecodeError:
        f.seek(0)
        for line in f:
            if line.strip():
                dataset2.append(json.loads(line))


with open(output_path1, 'w') as f:
    json.dump(dataset1, f, indent=2)

with open(output_path2, 'w') as f:
    json.dump(dataset2, f, indent=2)