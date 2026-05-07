#!/usr/bin/env python3
import json
import os
from pathlib import Path

def check_trajectory_dir(dir_path):
    """Check if trajectory directory is complete and not corrupted."""
    issues = []

    # Get all trajectory files
    traj_files = sorted(Path(dir_path).glob("traj_*.json"))

    if not traj_files:
        return {"status": "EMPTY", "issues": ["No trajectory files found"]}

    # Check each trajectory file
    for traj_file in traj_files:
        try:
            with open(traj_file, 'r') as f:
                data = json.load(f)

            # Check for required fields
            required_fields = ['question', 'ground_truth', 'generated_text']
            missing = [f for f in required_fields if f not in data]
            if missing:
                issues.append(f"{traj_file.name}: missing fields {missing}")

            # Check if generated_text is empty or None
            if not data.get('generated_text'):
                issues.append(f"{traj_file.name}: empty generated_text")

        except json.JSONDecodeError:
            issues.append(f"{traj_file.name}: JSON decode error (corrupted)")
        except Exception as e:
            issues.append(f"{traj_file.name}: {str(e)}")

    status = "OK" if not issues else "ISSUES"
    return {"status": status, "count": len(traj_files), "issues": issues}

# Find all test-426 directories with sidx suffix
traj_base = Path("trajectories")
dirs = sorted([d for d in traj_base.iterdir() if d.is_dir() and d.name.startswith("test-426") and "sidx" in d.name])

print(f"Found {len(dirs)} test-426 directories with sidx suffix\n")

results = {}
for d in dirs:
    result = check_trajectory_dir(d)
    results[d.name] = result

    status_symbol = "✓" if result["status"] == "OK" else "✗"
    print(f"{status_symbol} {d.name}: {result['count']} files - {result['status']}")
    if result["issues"]:
        for issue in result["issues"][:3]:  # Show first 3 issues
            print(f"    - {issue}")
        if len(result["issues"]) > 3:
            print(f"    ... and {len(result['issues']) - 3} more issues")

# Summary
ok_count = sum(1 for r in results.values() if r["status"] == "OK")
issue_count = len(results) - ok_count
print(f"\n{'='*60}")
print(f"Summary: {ok_count} OK, {issue_count} with issues")
