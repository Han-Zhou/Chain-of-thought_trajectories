import os
import json
import sys

folders = [
    "trajectories/llama_regular_logiqa_zero_0_300_type2_conf_jackknife_0331",
    "trajectories/llama_regular_logiqa_zero_300_651_type2_conf_jackknife_0331",
]

def count_final_answers(folder_paths):
    total = 0
    non_empty_count = 0
    empty_count = 0
    empty_trajectories = []
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid directory.", file=sys.stderr)
            continue
        for filename in os.listdir(folder_path):
            if filename.startswith("traj_") and filename.endswith(".json"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    trajectory = data.get("trajectory", {})
                    if "final_answer" in trajectory:
                        total += 1
                        if trajectory["final_answer"] == "":
                            empty_count += 1
                            empty_trajectories.append(filepath)
                        else:
                            non_empty_count += 1
                    else:
                        print(
                            f"Warning: 'final_answer' field missing in {filename} in {folder_path}",
                            file=sys.stderr,
                        )
    if total == 0:
        print("No traj_*.json files with 'final_answer' field found in provided folders.")
        return
    print(f"Total: {total}")
    print(f"Non-empty: {non_empty_count}")
    print(f"Empty: {empty_count}")
    print(f"Non-empty rate: {non_empty_count / total:.4f}")
    if empty_trajectories:
        print("\nTrajectories with empty final_answer:")
        for path in empty_trajectories:
            print(f"  {path}")

if __name__ == "__main__":
    count_final_answers(folders)
