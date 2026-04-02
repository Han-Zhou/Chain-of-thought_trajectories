import os
import json
import sys

folders = [
    "trajectories/qwen_thinking_logiqa_few_0_300_type2_conf_0330_jacknife",
    "trajectories/qwen_thinking_logiqa_few_300_651_type2_conf_0330_jacknife",
]

def count_accuracies(folder_paths):
    total = 0
    correct_count = 0
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid directory.", file=sys.stderr)
            continue
        for filename in os.listdir(folder_path):
            if filename.startswith("traj_") and filename.endswith(".json"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                    trajectory = data.get("trajectory", [])
                    if "correct" in trajectory and isinstance(trajectory["correct"], bool):
                        total += 1
                        if trajectory["correct"]:
                            correct_count += 1
                    else:
                        print(
                            f"Warning: 'correct' field missing or not bool in {filename} in {folder_path}",
                            file=sys.stderr,
                        )
    if total == 0:
        print("No traj_*.json files with 'correct' field found in provided folders.")
        return
    accuracy = correct_count / total
    print(f"Total: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    count_accuracies(folders)