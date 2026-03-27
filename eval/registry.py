"""Dataset-to-evaluator dispatcher.

Maps each supported dataset to its (extractor, comparator, metric_name) tuple
and provides `evaluate_one()` for single-trajectory evaluation.
"""

import warnings

from eval.extractors import extract_mcq_letter, extract_text_answer, extract_boxed_or_text
from eval.comparators import (
    exact_match,
    normalized_text_match,
    math_equal,
    qa_f1_score,
)


# (extractor_fn, comparator_fn, metric_name)
EVAL_REGISTRY = {
    "logiqa":           (extract_mcq_letter,    exact_match,           "accuracy"),
    "college_math":     (extract_boxed_or_text, math_equal,            "accuracy"),
    "bigbench_causal":  (extract_text_answer,   exact_match,           "accuracy"),
    "bigbench_movie":   (extract_mcq_letter,    exact_match,           "accuracy"),
    "hotpotqa":         (extract_text_answer,   qa_f1_score,           "f1"),
    "math500":          (extract_boxed_or_text, math_equal,            "accuracy"),
    "olympiadbench":    (extract_boxed_or_text, math_equal,            "accuracy"),
}

UNSUPPORTED = {"bfcl", "codeqa", "cs1qa", "hle"}


def evaluate_one(trajectory: dict, dataset_name: str) -> dict | None:
    """Evaluate a single trajectory entry.

    Args:
        trajectory: The trajectory dict (must contain 'generated_text' and 'ground_truth').
        dataset_name: Name of the dataset (lowercase, as used in EVAL_REGISTRY).

    Returns:
        Dict with keys: prediction, ground_truth, score, metric, extraction_failed.
        Returns None for unsupported datasets.
    """
    dataset_name = dataset_name.lower()

    if dataset_name in UNSUPPORTED:
        warnings.warn(f"Dataset '{dataset_name}' is not supported for evaluation. Skipping.")
        return None

    if dataset_name not in EVAL_REGISTRY:
        warnings.warn(f"Dataset '{dataset_name}' not found in evaluation registry. Skipping.")
        return None

    extractor, comparator, metric = EVAL_REGISTRY[dataset_name]

    generated_text = trajectory.get("generated_text", "")
    ground_truth = trajectory.get("ground_truth", "")

    # Extract prediction from model output
    prediction = extractor(generated_text)

    if prediction is None:
        return {
            "prediction": None,
            "ground_truth": ground_truth,
            "score": 0.0,
            "metric": metric,
            "extraction_failed": True,
        }

    # Special case: OlympiadBench multi-answer (ground_truth joined by "; ")
    if dataset_name == "olympiadbench" and "; " in str(ground_truth):
        gt_parts = str(ground_truth).split("; ")
        pred_parts = str(prediction).split("; ")
        if len(pred_parts) == len(gt_parts):
            all_correct = all(
                comparator(p, g) for p, g in zip(pred_parts, gt_parts)
            )
            score = 1.0 if all_correct else 0.0
        else:
            score = 0.0
    else:
        result = comparator(prediction, str(ground_truth))
        # comparator returns bool for accuracy, float for f1
        score = float(result) if isinstance(result, (bool, int)) else result

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "score": score,
        "metric": metric,
        "extraction_failed": False,
    }
