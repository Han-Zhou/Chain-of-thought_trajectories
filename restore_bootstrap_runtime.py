"""
restore_bootstrap_runtime.py

Restore runtime_seconds in bootstrap trajectory files from commit 93425d89e0
(before they were overwritten by the re-run at f6f8783014).

This script:
1. Reads runtime_seconds from the old commit (93425d89e0)
2. Updates the current trajectory files with the original runtime_seconds
3. Preserves all other fields (confidence scores, debug_info, etc.)
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def get_old_runtime(commit: str, file_path: str) -> float:
    """Extract runtime_seconds from a file at a specific git commit."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        capture_output=True,
        text=True,
        check=True
    )
    data = json.loads(result.stdout)
    return data["trajectory"]["runtime_seconds"]


def atomic_write_json(path: str, data: dict) -> None:
    """Atomically write JSON to a file."""
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".restore_", suffix=".json", dir=dir_)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    parser = argparse.ArgumentParser(description="Restore runtime_seconds from commit 93425d89e0")
    parser.add_argument("--commit", default="93425d89e0", help="Commit to restore from")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("bootstrap_dir", help="Bootstrap directory to restore (e.g., trajectories/1_true_bootstrap/bootstrap-bm-430-qwen_...)")
    args = parser.parse_args()

    bootstrap_dir = Path(args.bootstrap_dir)
    if not bootstrap_dir.exists():
        raise SystemExit(f"Directory not found: {bootstrap_dir}")

    traj_files = sorted(bootstrap_dir.glob("traj_*.json"))
    print(f"Found {len(traj_files)} trajectory files in {bootstrap_dir}")

    n_updated = 0
    n_skipped = 0
    n_failed = 0

    for traj_file in traj_files:
        rel_path = traj_file.relative_to(".")

        try:
            # Get old runtime_seconds from git
            old_runtime = get_old_runtime(args.commit, str(rel_path))

            # Read current file
            with open(traj_file) as f:
                current_data = json.load(f)

            current_runtime = current_data["trajectory"]["runtime_seconds"]

            if abs(current_runtime - old_runtime) < 0.001:
                n_skipped += 1
                continue

            print(f"{traj_file.name}: {current_runtime:.2f}s -> {old_runtime:.2f}s")

            if not args.dry_run:
                current_data["trajectory"]["runtime_seconds"] = old_runtime
                atomic_write_json(str(traj_file), current_data)

            n_updated += 1

        except subprocess.CalledProcessError:
            print(f"WARNING: {traj_file.name} not found in commit {args.commit}")
            n_failed += 1
        except Exception as e:
            print(f"ERROR processing {traj_file.name}: {e}")
            n_failed += 1

    print(f"\nDone. updated={n_updated} skipped={n_skipped} failed={n_failed}")
    if args.dry_run:
        print("(dry run - no files were modified)")


if __name__ == "__main__":
    main()
