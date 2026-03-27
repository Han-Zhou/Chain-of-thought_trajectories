"""CLI entry point for evaluating CoT trajectory files.

Usage:
    python evaluate_trajectories.py --trajectory_dir trajectories/qwen_thinking_logiqa_few_1_type1_vanilla
    python evaluate_trajectories.py --trajectory_file <path.json> --dataset logiqa
    python evaluate_trajectories.py --trajectory_dir <dir> --output results.json
"""

import argparse
import json
import os
import sys
import statistics

from eval.registry import evaluate_one, EVAL_REGISTRY, UNSUPPORTED


def read_trajectories(file_path: str) -> list[dict]:
    """Read concatenated JSON objects from a trajectory file."""
    with open(file_path, "r") as f:
        content = f.read()

    decoder = json.JSONDecoder()
    trajectories = []
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx].isspace():
            idx += 1
        if idx >= len(content):
            break
        obj, end = decoder.raw_decode(content, idx)
        trajectories.append(obj)
        idx = end

    return trajectories


def detect_dataset(dir_name: str) -> str | None:
    """Detect dataset name from trajectory directory name.

    Directory names follow: {model}_{thinking/regular}_{dataset}_{shot}_{n}[_type{t}][_{conf/vanilla}]
    The dataset is the 3rd underscore-delimited segment.
    """
    parts = os.path.basename(dir_name.rstrip(os.sep)).split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


def find_trajectory_file(trajectory_dir: str) -> str | None:
    """Find the trajectory JSON file inside a directory."""
    for fname in os.listdir(trajectory_dir):
        if fname.endswith(".json") and "trajectories" in fname:
            return os.path.join(trajectory_dir, fname)
    # Fallback: any .json file
    for fname in os.listdir(trajectory_dir):
        if fname.endswith(".json"):
            return os.path.join(trajectory_dir, fname)
    return None


def evaluate_file(file_path: str, dataset_name: str) -> dict:
    """Evaluate all trajectories in a file. Returns aggregated results."""
    trajectories = read_trajectories(file_path)

    results = []
    extraction_failures = 0

    for entry in trajectories:
        traj = entry.get("trajectory", entry)
        result = evaluate_one(traj, dataset_name)
        if result is None:
            continue
        if result["extraction_failed"]:
            extraction_failures += 1
        result["index"] = entry.get("index")
        result["id"] = traj.get("id")
        results.append(result)

    if not results:
        return {"dataset": dataset_name, "error": "No evaluable entries found"}

    metric = results[0]["metric"]
    scores = [r["score"] for r in results]

    output = {
        "dataset": dataset_name,
        "metric": metric,
        "total": len(results),
        "extraction_failures": extraction_failures,
        "results": results,
    }

    if metric == "accuracy":
        correct = sum(1 for s in scores if s == 1.0)
        output["correct"] = correct
        output["aggregate_score"] = correct / len(results)
    elif metric == "f1":
        output["aggregate_score"] = statistics.mean(scores)
        output["median_score"] = statistics.median(scores)
        if len(scores) > 1:
            output["stdev"] = statistics.stdev(scores)

    return output


def print_summary(result: dict):
    """Print a summary of evaluation results to stdout."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    dataset = result["dataset"]
    metric = result["metric"]
    total = result["total"]
    agg = result["aggregate_score"]
    failures = result["extraction_failures"]

    print(f"\n{'=' * 50}")
    print(f"Dataset:     {dataset}")
    print(f"Metric:      {metric}")
    print(f"Total:       {total}")
    print(f"Extraction failures: {failures}")

    if metric == "accuracy":
        correct = result["correct"]
        print(f"Correct:     {correct}/{total}")
        print(f"Accuracy:    {agg:.2%}")
    elif metric == "f1":
        print(f"Mean F1:     {agg:.4f}")
        if "median_score" in result:
            print(f"Median F1:   {result['median_score']:.4f}")
        if "stdev" in result:
            print(f"Stdev:       {result['stdev']:.4f}")

    print(f"{'=' * 50}")

    # Per-entry detail
    for r in result["results"]:
        idx = r.get("index", "?")
        pred = r["prediction"]
        gt = r["ground_truth"]
        score = r["score"]
        flag = "FAIL" if r["extraction_failed"] else ("OK" if score == 1.0 else "WRONG")
        if metric == "f1":
            flag = f"F1={score:.3f}"
        print(f"  [{idx}] pred={pred!r:30s} gt={gt!r:30s}  {flag}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoT trajectory files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trajectory_file", type=str, help="Path to trajectory JSON file")
    group.add_argument("--trajectory_dir", type=str, help="Path to trajectory directory")
    parser.add_argument("--dataset", type=str, help="Dataset name (auto-detected if using --trajectory_dir)")
    parser.add_argument("--output", type=str, help="Path to write JSON results")
    args = parser.parse_args()

    # Resolve file and dataset
    if args.trajectory_dir:
        file_path = find_trajectory_file(args.trajectory_dir)
        if file_path is None:
            print(f"Error: No trajectory JSON file found in {args.trajectory_dir}", file=sys.stderr)
            sys.exit(1)
        dataset_name = args.dataset or detect_dataset(args.trajectory_dir)
        if dataset_name is None:
            print("Error: Could not auto-detect dataset name. Use --dataset.", file=sys.stderr)
            sys.exit(1)
        output_path = args.output or os.path.join(args.trajectory_dir, "eval_results.json")
    else:
        file_path = args.trajectory_file
        dataset_name = args.dataset
        if dataset_name is None:
            print("Error: --dataset is required when using --trajectory_file.", file=sys.stderr)
            sys.exit(1)
        output_path = args.output

    dataset_name = dataset_name.lower()

    if dataset_name in UNSUPPORTED:
        print(f"Dataset '{dataset_name}' is not supported for automatic evaluation.")
        print(f"Unsupported datasets: {', '.join(sorted(UNSUPPORTED))}")
        sys.exit(0)

    if dataset_name not in EVAL_REGISTRY:
        print(f"Error: Unknown dataset '{dataset_name}'.", file=sys.stderr)
        print(f"Supported: {', '.join(sorted(EVAL_REGISTRY.keys()))}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating: {file_path}")
    print(f"Dataset:    {dataset_name}")

    result = evaluate_file(file_path, dataset_name)
    print_summary(result)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nDetailed results written to: {output_path}")


if __name__ == "__main__":
    main()
