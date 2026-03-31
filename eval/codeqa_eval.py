"""CodeQA evaluation: 6 metrics (EM, BLEU-4, ROUGE-L, Precision, Recall, F1).

Ported from codeqa/codeBERT/code/eval/ and run.py.
- BLEU: Google's smoothed BLEU (average of per-sentence scores)
- ROUGE-L: LCS-based F-score with beta=1.2
- Token-level: Precision, Recall, F1 via Counter intersection
- EM: exact match after normalization
"""

import collections
import math
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase and collapse whitespace. No punctuation removal."""
    def white_space_fix(text):
        return ' '.join(text.split())
    def lower(text):
        return text.lower()
    return white_space_fix(lower(s))


def _f1_score(prediction_tokens, ground_truth_tokens):
    """Token-level F1 via Counter intersection.

    Returns: (precision, recall, f1) as floats 0-1.
    """
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def _best_score(prediction, ground_truths):
    """Find best EM, precision, recall, F1 across all ground truths.

    Returns: (em, precision, recall, f1) as floats 0-1.
    """
    best_em = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_f1 = 0.0

    for gt in ground_truths:
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(gt)

        em = 1.0 if norm_pred == norm_gt else 0.0
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()
        prec, rec, f1 = _f1_score(pred_tokens, gt_tokens)

        if f1 > best_f1:
            best_em = em
            best_prec = prec
            best_rec = rec
            best_f1 = f1

    return best_em, best_prec, best_rec, best_f1


def _get_ngrams(segment, max_order):
    """Extract n-grams from token list."""
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu_sentence(reference_corpus, translation_corpus, max_order=4, smooth=True):
    """Compute smoothed BLEU for a single sentence pair.

    Args:
        reference_corpus: list of reference token lists
        translation_corpus: hypothesis token list
        max_order: max n-gram order (default 4)
        smooth: apply Lin et al. 2004 smoothing (default True)

    Returns: BLEU score as float 0-1.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = len(translation_corpus)

    # Use shortest reference length (matches codeqa google_bleu.py:65)
    reference_length = min(len(r) for r in reference_corpus) if reference_corpus else 0

    merged_ref_ngram_counts = collections.Counter()
    for reference in reference_corpus:
        merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

    translation_ngram_counts = _get_ngrams(translation_corpus, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts

    for ngram in overlap:
        matches_by_order[len(ngram) - 1] += overlap[ngram]

    for order in range(1, max_order + 1):
        possible_matches = len(translation_corpus) - order + 1
        if possible_matches > 0:
            possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                               possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length if reference_length > 0 else 0

    if ratio > 1.0:
        bp = 1.0
    else:
        if ratio < 1e-6:
            bp = 0
        else:
            bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp
    return bleu


def _lcs_length(x, y):
    """Compute longest common subsequence length via DP."""
    if len(x) < len(y):
        x, y = y, x

    lengths = [[0 for _ in range(len(y) + 1)] for _ in range(len(x) + 1)]

    for j in range(1, len(y) + 1):
        for i in range(1, len(x) + 1):
            if x[i - 1] == y[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(x)][len(y)]


def _rouge_l_sentence(hypothesis_tokens, reference_tokens, beta=1.2):
    """Compute ROUGE-L F-score for a single sentence pair.

    Args:
        hypothesis_tokens: list of hypothesis tokens
        reference_tokens: list of reference tokens
        beta: recall weighting (default 1.2)

    Returns: ROUGE-L F-score as float 0-1.
    """
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)

    prec = lcs / len(hypothesis_tokens) if hypothesis_tokens else 0.0
    rec = lcs / len(reference_tokens) if reference_tokens else 0.0

    if prec == 0 or rec == 0:
        return 0.0

    score = ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec)
    return score


def evaluate_codeqa(trajectory: dict) -> dict:
    """Evaluate a CodeQA trajectory.

    Args:
        trajectory: dict with 'generated_text' and 'ground_truth'

    Returns:
        dict with prediction, ground_truth, score (F1 0-1), metric, scores dict, extraction_failed
    """
    generated_text = trajectory.get("generated_text", "")
    ground_truth = trajectory.get("ground_truth", "")

    # Handle ground_truth as str or list
    ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]

    # Prediction is the full normalized text (no extraction)
    prediction = normalize_answer(generated_text)

    # Compute best scores across all ground truths
    em, precision, recall, f1 = _best_score(prediction, ground_truths)

    # BLEU: average of per-sentence smoothed BLEU
    # Normalize all ground truths and split into tokens
    gt_token_lists = [normalize_answer(gt).split() for gt in ground_truths]
    pred_tokens = prediction.split()
    bleu = _compute_bleu_sentence(gt_token_lists, pred_tokens, max_order=4, smooth=True)

    # ROUGE-L: against the best reference (the one that maximized F1)
    # Find which reference gave best F1
    best_gt_idx = 0
    best_f1_check = 0.0
    for i, gt in enumerate(ground_truths):
        norm_gt = normalize_answer(gt)
        _, _, f1_check = _f1_score(prediction.split(), norm_gt.split())
        if f1_check > best_f1_check:
            best_f1_check = f1_check
            best_gt_idx = i

    best_ref_tokens = normalize_answer(ground_truths[best_gt_idx]).split()
    rouge_l = _rouge_l_sentence(pred_tokens, best_ref_tokens, beta=1.2)

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "score": f1,  # primary score for aggregation (0-1)
        "metric": "f1",
        "scores": {
            "em":        round(em * 100, 4),
            "bleu":      round(bleu * 100, 4),
            "rouge_l":   round(rouge_l * 100, 4),
            "precision": round(precision * 100, 4),
            "recall":    round(recall * 100, 4),
            "f1":        round(f1 * 100, 4),
        },
        "extraction_failed": False,
    }
