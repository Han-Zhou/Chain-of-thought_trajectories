#!/usr/bin/env python3
"""Check which trajectory directories contain exact same questions as pickle files."""

import json
import pickle
from pathlib import Path
from collections import defaultdict

def extract_dataset_name(dir_name):
    """Extract dataset name from directory name."""
    # Handle MERGED directories: MERGED-cots-{tag}-{model}-{dataset}-{samples}
    if dir_name.startswith('MERGED-'):
        parts = dir_name.split('-')
        if len(parts) >= 5:
            dataset = parts[4]
            return dataset

    # Handle regular directories: cots-{tag}-{model}-{dataset}-{samples}_{full_name}
    # The full name follows pattern: {tag}_{model}_{thinking|regular}_{dataset}_...
    if dir_name.startswith('cots-'):
        # First try to get from the underscore-delimited part
        if '_' in dir_name:
            underscore_parts = dir_name.split('_')
            if len(underscore_parts) >= 4:
                dataset = underscore_parts[3]
                # Handle bigbench special case
                if dataset == 'bigbench' and len(underscore_parts) >= 5:
                    return f"{dataset}_{underscore_parts[4]}"
                return dataset

        # Fallback: try dash-delimited
        dash_parts = dir_name.split('-')
        if len(dash_parts) >= 4:
            dataset = dash_parts[3]
            return dataset

    return None

def load_pickle_questions(dataset_name):
    """Load questions from pickle file for a dataset."""
    pickle_map = {
        'bfcl': 'pickles/bfcl.pkl',
        'bigbench_causal': 'pickles/bigbench_causal.pkl',
        'bigbench_movie': 'pickles/bigbench_movie.pkl',
        'logiqa': 'pickles/logiqa.pkl',
        'codeqa': 'pickles/codeqa.pkl',
        'cs1qa_span': 'pickles/cs1qa_span.pkl',
        'cs1qa_type': 'pickles/cs1qa_type.pkl',
        'hotpotqa': 'pickles/hotpotqa.pkl',
        'm500': 'pickles/math500.pkl',
        'math500': 'pickles/math500.pkl',
        'bm': 'pickles/bigbench_movie.pkl',
        'bc': 'pickles/bigbench_causal.pkl',
    }

    pickle_path = pickle_map.get(dataset_name)
    if not pickle_path or not Path(pickle_path).exists():
        return None

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Extract questions - handle different data structures
    questions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                q = item.get('question') or item.get('input') or item.get('prompt')
                questions.append(q)
    elif isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict):
                q = item.get('question') or item.get('input') or item.get('prompt')
                questions.append(q)

    return questions

def load_trajectory_questions(traj_dir, num_samples):
    """Load questions from sample 0 trajectory files only."""
    questions = []
    traj_path = Path(traj_dir)

    if not traj_path.exists():
        return None

    # Load all traj_*.json files
    traj_files = sorted(traj_path.glob('traj_*.json'), key=lambda x: int(x.stem.split('_')[1]))

    # Only take every num_samples-th file (sample 0 for each question)
    for i, traj_file in enumerate(traj_files):
        if i % num_samples == 0:  # Only sample 0
            try:
                with open(traj_file, 'r') as f:
                    data = json.load(f)
                    q = data.get('question')
                    questions.append(q)
            except Exception as e:
                print(f"  Error loading {traj_file}: {e}")

    return questions

def extract_num_samples(dir_name):
    """Extract number of samples from directory name."""
    # MERGED directories: MERGED-cots-{tag}-{model}-{dataset}-{samples}
    if dir_name.startswith('MERGED-'):
        parts = dir_name.split('-')
        if len(parts) >= 6:
            try:
                return int(parts[5])
            except ValueError:
                pass

    # Regular directories: look for _samples{N} pattern
    if '_samples' in dir_name:
        import re
        match = re.search(r'_samples(\d+)', dir_name)
        if match:
            return int(match.group(1))

    return None

def main():
    base_dir = Path('trajectories/1_full_trajectories')

    # Get all directories except FULL-prefixed ones
    dirs = [d for d in base_dir.iterdir()
            if d.is_dir() and not d.name.startswith('FULL')]

    results = {}

    for traj_dir in sorted(dirs):
        dir_name = traj_dir.name
        print(f"\nChecking: {dir_name}")

        # Extract dataset name
        dataset_name = extract_dataset_name(dir_name)
        if not dataset_name:
            print(f"  Could not extract dataset name")
            continue

        print(f"  Dataset: {dataset_name}")

        # Extract number of samples
        num_samples = extract_num_samples(dir_name)
        if not num_samples:
            print(f"  Could not extract number of samples")
            continue

        print(f"  Samples per question: {num_samples}")

        # Load pickle questions
        pickle_questions = load_pickle_questions(dataset_name)
        if pickle_questions is None:
            print(f"  No pickle file found or could not load")
            continue

        # Load trajectory questions (sample 0 only)
        traj_questions = load_trajectory_questions(traj_dir, num_samples)
        if traj_questions is None:
            print(f"  Could not load trajectory files")
            continue

        print(f"  Pickle questions: {len(pickle_questions)}")
        print(f"  Trajectory questions (sample 0): {len(traj_questions)}")

        # Compare questions
        if len(pickle_questions) != len(traj_questions):
            print(f"  ❌ Different number of questions")
            results[dir_name] = False
            continue

        # Check if questions match exactly
        all_match = True
        for i, (pq, tq) in enumerate(zip(pickle_questions, traj_questions)):
            if pq != tq:
                all_match = False
                print(f"  ❌ Question {i} differs")
                print(f"     Pickle: {pq[:100]}...")
                print(f"     Traj:   {tq[:100]}...")
                break

        if all_match:
            print(f"  ✅ All questions match exactly")
            results[dir_name] = True
        else:
            results[dir_name] = False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    matching = [k for k, v in results.items() if v]
    not_matching = [k for k, v in results.items() if not v]

    print(f"\n✅ Matching ({len(matching)}):")
    for d in matching:
        print(f"  - {d}")

    print(f"\n❌ Not matching ({len(not_matching)}):")
    for d in not_matching:
        print(f"  - {d}")

if __name__ == '__main__':
    main()
