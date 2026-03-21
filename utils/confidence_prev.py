"""confidence.py

Confidence metric implementations evaluated on the logits of the final-answer tokens.

Three metrics:
  perplexity  – exp(-1/N * Σ log p(tᵢ)) on chosen tokens
  entropy     – mean Shannon entropy over token distributions
  min-entropy – mean min-entropy (-log max_v p(v)) over token distributions
"""

import torch
import torch.nn.functional as F


def _get_answer_scores(
    scores: tuple,
    answer_token_start: int,
    prompt_len: int,
) -> torch.Tensor:
    """Return logit tensors for the final-answer tokens only.

    Args:
        scores: Tuple of (1, vocab_size) tensors, index 0 = first generated token.
        answer_token_start: Absolute position in full sequence where final answer begins.
        prompt_len: Number of prompt tokens.

    Returns:
        Tensor of shape (n_answer_tokens, vocab_size).
    """
    # first_idx is the index of the first answer token inside scores
    first_idx = answer_token_start - prompt_len
    if first_idx < 0 or first_idx >= len(scores):
        raise ValueError(
            f"answer_token_start={answer_token_start} out of range: "
            f"prompt_len={prompt_len}, {len(scores)} generated tokens."
        )
    return torch.stack([scores[i].squeeze(0) for i in range(first_idx, len(scores))])


def compute_perplexity(
    scores: tuple,
    generated_ids: torch.Tensor,
    answer_token_start: int,
    prompt_len: int,
) -> float:
    answer_logits = _get_answer_scores(scores, answer_token_start, prompt_len)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    first_idx = answer_token_start - prompt_len
    answer_ids = generated_ids[first_idx:]
    token_log_probs = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    return (-token_log_probs.mean()).exp().item()


def compute_entropy(
    scores: tuple,
    answer_token_start: int,
    prompt_len: int,
) -> float:
    answer_logits = _get_answer_scores(scores, answer_token_start, prompt_len)
    probs = F.softmax(answer_logits, dim=-1)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.mean().item()


def compute_min_entropy(
    scores: tuple,
    answer_token_start: int,
    prompt_len: int,
) -> float:
    answer_logits = _get_answer_scores(scores, answer_token_start, prompt_len)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    min_entropy = -log_probs.max(dim=-1).values
    return min_entropy.mean().item()


def compute_confidence_metrics(
    metric: str,
    scores: tuple,
    generated_ids: torch.Tensor,
    answer_token_start: int,
    prompt_len: int,
) -> float:
    """Dispatch to the requested confidence metric.

    Args:
        metric: One of "perplexity", "entropy", "min-entropy".
        scores: Per-step logits from model.generate(output_scores=True).
        generated_ids: 1-D tensor of generated token IDs (prompt excluded).
        answer_token_start: Absolute position where final answer begins.
        prompt_len: Number of prompt tokens.

    Returns:
        Scalar metric value (float).
    """
    if metric == "perplexity":
        return compute_perplexity(scores, generated_ids, answer_token_start, prompt_len)
    elif metric == "entropy":
        return compute_entropy(scores, answer_token_start, prompt_len)
    elif metric == "min-entropy":
        return compute_min_entropy(scores, answer_token_start, prompt_len)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Choose from perplexity, entropy, min-entropy.")
