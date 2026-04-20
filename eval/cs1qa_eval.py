"""CS1QA evaluation: type classification, span prediction, and answer generation.

Ported from https://github.com/cyoon47/CS1QA evaluation pipelines.

Type classification: accuracy, F1 macro/weighted, per-class P/R/F1.
Span prediction:     line-set F1 and exact match (EM).
Answer generation:   BLEU-1 (unigram), token F1, ROUGE-L, with per-type breakdown.
"""

import re
import numpy as np
from collections import Counter

from eval.extractors import extract_text_answer
from eval.codeqa_eval import (
    normalize_answer,
    _f1_score,
    _compute_bleu_sentence,
    _rouge_l_sentence,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CS1QA_TYPE_LABELS = [
    "code_understanding", "logical", "error", "usage", "algorithm",
    "task", "comparison", "reasoning", "code_explain", "variable", "guiding",
]

# 9-type subset used for answer retrieval BLEU breakdown (no comparison/guiding)
CS1QA_ANSWER_LABELS = [
    "code_understanding", "logical", "error", "usage", "algorithm",
    "task", "reasoning", "code_explain", "variable",
]

# ST (student-oriented) = first 6, TA = last 3
CS1QA_ST_TYPES = CS1QA_ANSWER_LABELS[:6]
CS1QA_TA_TYPES = CS1QA_ANSWER_LABELS[6:]

# Aliases for fuzzy matching type labels
_TYPE_ALIASES = {}
for _label in CS1QA_TYPE_LABELS:
    _TYPE_ALIASES[_label] = _label
    _TYPE_ALIASES[_label.replace("_", " ")] = _label
    _TYPE_ALIASES[_label.replace("_", "")] = _label
_TYPE_ALIASES["code explain"] = "code_explain"
_TYPE_ALIASES["code understanding"] = "code_understanding"
_TYPE_ALIASES["code_understanding"] = "code_understanding"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_type_label(generated_text: str) -> str | None:
    """Extract a question type label from generated text."""
    boxed = extract_text_answer(generated_text)
    if boxed is not None:
        candidate = boxed.strip().lower().strip("'\"` ")
        if candidate in _TYPE_ALIASES:
            return _TYPE_ALIASES[candidate]

    # Fallback: scan the full text for any known label (last occurrence wins)
    text_lower = generated_text.lower()
    found = None
    for label in CS1QA_TYPE_LABELS:
        # Search for the label as a standalone word
        pattern = r'\b' + re.escape(label) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            found = label
        # Also try space-separated variant
        pattern_sp = r'\b' + re.escape(label.replace("_", " ")) + r'\b'
        matches_sp = list(re.finditer(pattern_sp, text_lower))
        if matches_sp:
            found = label
    return found


def _extract_span(generated_text: str) -> tuple[int, int] | None:
    """Extract a (startLine, endLine) span from generated text.

    Handles formats: 24-30, 24 - 30, 24 to 30, lines 24-30,
    startLine: 24, endLine: 30, (24, 30).
    """
    boxed = extract_text_answer(generated_text)
    text = boxed if boxed is not None else generated_text

    # Try "startLine: N, endLine: M" or "start: N, end: M"
    m = re.search(r'(?:start(?:Line)?)\s*[:=]\s*(\d+)\s*[,;]\s*(?:end(?:Line)?)\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Try "N-M", "N - M", "N to M", "lines N-M", "lines N to M"
    m = re.search(r'(\d+)\s*[-–—]\s*(\d+)', text)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r'(\d+)\s+to\s+(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Try "(N, M)" or "N, M"
    m = re.search(r'(\d+)\s*,\s*(\d+)', text)
    if m:
        s, e = int(m.group(1)), int(m.group(2))
        if s <= e:
            return s, e

    return None


# ---------------------------------------------------------------------------
# Core metric: line-set F1 and EM (ported from cs1qa/src/span_prediction/utils.py:557-587)
# ---------------------------------------------------------------------------

def calculate_F1_EM(
    selected_intervals: list[tuple[int, int]],
    label: tuple[int, int],
) -> tuple[float, float]:
    """Compute line-set F1 and exact match for a single span prediction.

    Args:
        selected_intervals: List of (start_line, end_line) predicted intervals.
        label: Ground-truth (start_line, end_line) tuple.

    Returns:
        (F1, EM) where EM = 1 if F1 == 1 else 0.
    """
    label_set = set(range(label[0], label[1] + 1))

    selection_set: set[int] = set()
    for start, end in selected_intervals:
        selection_set |= set(range(start, end + 1))

    tp = len(label_set & selection_set)
    if tp == 0:
        return 0.0, 0.0

    fp = len(selection_set - label_set)
    fn = len(label_set - selection_set)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    em = 1.0 if f1 == 1.0 else 0.0

    return f1, em


# ---------------------------------------------------------------------------
# Per-trajectory evaluators
# ---------------------------------------------------------------------------

def evaluate_cs1qa_type(trajectory: dict) -> dict:
    """Evaluate a single CS1QA type-classification trajectory.

    Returns dict with score (0 or 1), metric="accuracy".
    """
    generated_text = trajectory.get("generated_text", "")
    ground_truth = trajectory.get("ground_truth", "")

    prediction = _extract_type_label(generated_text)

    if prediction is None:
        return {
            "prediction": None,
            "ground_truth": ground_truth,
            "score": 0.0,
            "metric": "accuracy",
            "extraction_failed": True,
        }

    score = 1.0 if prediction == ground_truth else 0.0
    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "score": score,
        "metric": "accuracy",
        "extraction_failed": False,
    }


def evaluate_cs1qa_span(trajectory: dict) -> dict:
    """Evaluate a single CS1QA span-prediction trajectory.

    ground_truth is a list of (startLine, endLine) tuples (multiple valid annotations).
    Returns dict with score = best F1 across annotations, metric="f1".
    """
    generated_text = trajectory.get("generated_text", "")
    ground_truth = trajectory.get("ground_truth", [])

    span = _extract_span(generated_text)

    if span is None:
        return {
            "prediction": None,
            "ground_truth": ground_truth,
            "score": 0.0,
            "metric": "f1",
            "scores": {"f1": 0.0, "em": 0.0},
            "extraction_failed": True,
        }

    # Compare against each annotation, take best F1
    best_f1, best_em = 0.0, 0.0
    for label in ground_truth:
        label_tuple = (label[0], label[1]) if isinstance(label, (list, tuple)) else label
        f1, em = calculate_F1_EM([span], label_tuple)
        if f1 > best_f1:
            best_f1, best_em = f1, em

    return {
        "prediction": f"{span[0]}-{span[1]}",
        "ground_truth": ground_truth,
        "score": best_f1,
        "metric": "f1",
        "scores": {"f1": round(best_f1 * 100, 4), "em": round(best_em * 100, 4)},
        "extraction_failed": False,
    }


def evaluate_cs1qa_answer(trajectory: dict) -> dict:
    """Evaluate a single CS1QA answer-generation trajectory.

    Computes BLEU-1 (unigram), token F1, and ROUGE-L.
    Returns dict with score = BLEU-1, metric="bleu1".
    """
    generated_text = trajectory.get("generated_text", "")
    ground_truth = trajectory.get("ground_truth", "")

    # Use full normalized text as prediction (no boxed extraction), same as codeqa
    prediction = normalize_answer(generated_text)
    gt_normalized = normalize_answer(ground_truth)

    pred_tokens = prediction.split()
    gt_tokens = gt_normalized.split()

    # BLEU-1: unigram only, matching cs1qa/src/answer_retrieval/get_bleu.py
    bleu1 = _compute_bleu_sentence([gt_tokens], pred_tokens, max_order=1, smooth=False)

    # Token F1
    _, _, token_f1 = _f1_score(pred_tokens, gt_tokens)

    # ROUGE-L
    rouge_l = _rouge_l_sentence(pred_tokens, gt_tokens, beta=1.2)

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "score": bleu1,
        "metric": "bleu1",
        "scores": {
            "bleu1": round(bleu1 * 100, 4),
            "token_f1": round(token_f1 * 100, 4),
            "rouge_l": round(rouge_l * 100, 4),
        },
        "extraction_failed": False,
    }


# ---------------------------------------------------------------------------
# Batch aggregation helpers (for evaluate_trajectories.py)
# ---------------------------------------------------------------------------

def compute_cs1qa_type_metrics(results: list[dict]) -> dict:
    """Compute aggregate type-classification metrics from per-trajectory results.

    Mirrors acc_and_f1() from cs1qa/src/type_classification/utils.py:361-389.
    Returns dict with accuracy, f1_macro, f1_weighted, per-class P/R/F1, confusion matrix.
    """
    from sklearn.metrics import f1_score, confusion_matrix

    preds = []
    labels = []
    label_to_idx = {label: i for i, label in enumerate(CS1QA_TYPE_LABELS)}

    for r in results:
        gt = r.get("ground_truth", "")
        pred = r.get("prediction")
        if gt not in label_to_idx:
            continue
        labels.append(label_to_idx[gt])
        preds.append(label_to_idx.get(pred, -1) if pred else -1)

    preds_arr = np.array(preds)
    labels_arr = np.array(labels)

    # For entries where extraction failed, prediction index is -1 (always wrong)
    acc = float((preds_arr == labels_arr).mean())
    n_classes = len(CS1QA_TYPE_LABELS)

    # For sklearn metrics, clamp -1 to 0 (will be wrong anyway)
    preds_clamped = np.clip(preds_arr, 0, n_classes - 1)
    f1_macro = float(f1_score(labels_arr, preds_clamped, average="macro", zero_division=0))
    f1_weighted = float(f1_score(labels_arr, preds_clamped, average="weighted", zero_division=0))

    conf = confusion_matrix(labels_arr, preds_clamped, labels=list(range(n_classes)))

    # Per-class precision, recall, F1 from confusion matrix
    class_p, class_r, class_f1 = [], [], []
    for i in range(n_classes):
        col_sum = conf[:, i].sum()
        row_sum = conf[i, :].sum()
        p = conf[i, i] / col_sum if col_sum > 0 else 0.0
        r = conf[i, i] / row_sum if row_sum > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_p.append(round(p, 4))
        class_r.append(round(r, 4))
        class_f1.append(round(f, 4))

    return {
        "accuracy": round(acc * 100, 4),
        "f1_macro": round(f1_macro * 100, 4),
        "f1_weighted": round(f1_weighted * 100, 4),
        "acc_and_f1_macro": round((acc + f1_macro) / 2 * 100, 4),
        "acc_and_f1_weighted": round((acc + f1_weighted) / 2 * 100, 4),
        "per_class": {
            label: {"precision": class_p[i], "recall": class_r[i], "f1": class_f1[i]}
            for i, label in enumerate(CS1QA_TYPE_LABELS)
        },
        "confusion_matrix": conf.tolist(),
    }


def compute_cs1qa_answer_metrics(results: list[dict]) -> dict:
    """Compute aggregate answer-generation metrics with per-type BLEU breakdown.

    Mirrors get_bleu.py from cs1qa/src/answer_retrieval/get_bleu.py.
    """
    all_bleu = []
    all_f1 = []
    all_rouge = []
    type_bleu: dict[str, list[float]] = {t: [] for t in CS1QA_ANSWER_LABELS}

    for r in results:
        scores = r.get("scores", {})
        bleu = scores.get("bleu1", 0.0)
        all_bleu.append(bleu)
        all_f1.append(scores.get("token_f1", 0.0))
        all_rouge.append(scores.get("rouge_l", 0.0))

        qtype = r.get("question_type") or (r.get("metadata") or {}).get("question_type")
        if qtype and qtype in type_bleu:
            type_bleu[qtype].append(bleu)

    overall = {
        "bleu1_mean": round(float(np.mean(all_bleu)), 4) if all_bleu else 0.0,
        "token_f1_mean": round(float(np.mean(all_f1)), 4) if all_f1 else 0.0,
        "rouge_l_mean": round(float(np.mean(all_rouge)), 4) if all_rouge else 0.0,
    }

    # Per-type BLEU breakdown
    per_type = {}
    for t, scores in type_bleu.items():
        per_type[t] = round(float(np.mean(scores)), 4) if scores else None

    # ST / TA grouping
    st_scores = [s for t in CS1QA_ST_TYPES for s in type_bleu.get(t, [])]
    ta_scores = [s for t in CS1QA_TA_TYPES for s in type_bleu.get(t, [])]

    overall["bleu1_ST"] = round(float(np.mean(st_scores)), 4) if st_scores else None
    overall["bleu1_TA"] = round(float(np.mean(ta_scores)), 4) if ta_scores else None
    overall["per_type_bleu1"] = per_type

    return overall
