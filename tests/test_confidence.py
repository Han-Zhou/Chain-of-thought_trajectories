import math
import pytest
import torch
from utils.confidence_prev import compute_confidence_metrics, compute_perplexity, compute_entropy, compute_min_entropy


def make_uniform_scores(vocab_size, n_tokens):
    """All logits = 0.0 -> uniform distribution."""
    return tuple(torch.zeros(1, vocab_size) for _ in range(n_tokens))


def make_peaked_scores(vocab_size, n_tokens, hot_idx=0):
    """One logit = 100.0, rest = 0.0 -> near-certain at hot_idx."""
    scores = []
    for _ in range(n_tokens):
        t = torch.zeros(1, vocab_size)
        t[0, hot_idx] = 100.0
        scores.append(t)
    return tuple(scores)


class TestPerplexity:
    def test_uniform_perplexity(self):
        """Uniform -> perplexity ~ vocab_size."""
        vocab_size = 10
        scores = make_uniform_scores(vocab_size, n_tokens=3)
        generated_ids = torch.zeros(3, dtype=torch.long)
        # All answer tokens: prompt_len=5, answer_token_start=5 (all 3 generated tokens)
        ppl = compute_perplexity(scores, generated_ids, answer_token_start=5, prompt_len=5)
        assert abs(ppl - vocab_size) < 0.1

    def test_peaked_perplexity(self):
        """Near-certain distribution -> perplexity ~ 1.0."""
        vocab_size = 10
        hot_idx = 2
        scores = make_peaked_scores(vocab_size, n_tokens=2, hot_idx=hot_idx)
        generated_ids = torch.full((2,), hot_idx, dtype=torch.long)
        ppl = compute_perplexity(scores, generated_ids, answer_token_start=5, prompt_len=5)
        assert ppl < 1.01

    def test_perplexity_via_dispatch(self):
        vocab_size = 8
        scores = make_uniform_scores(vocab_size, n_tokens=2)
        generated_ids = torch.zeros(2, dtype=torch.long)
        result = compute_confidence_metrics("perplexity", scores, generated_ids, 5, 5)
        assert abs(result - vocab_size) < 0.1


class TestEntropy:
    def test_uniform_entropy(self):
        """Uniform -> entropy = log(vocab_size)."""
        vocab_size = 16
        scores = make_uniform_scores(vocab_size, n_tokens=2)
        entropy = compute_entropy(scores, answer_token_start=4, prompt_len=4)
        assert abs(entropy - math.log(vocab_size)) < 0.01

    def test_peaked_entropy(self):
        """Near-certain -> entropy ~ 0."""
        vocab_size = 16
        scores = make_peaked_scores(vocab_size, n_tokens=2)
        entropy = compute_entropy(scores, answer_token_start=4, prompt_len=4)
        assert entropy < 0.01

    def test_entropy_via_dispatch(self):
        vocab_size = 8
        scores = make_uniform_scores(vocab_size, n_tokens=3)
        generated_ids = torch.zeros(3, dtype=torch.long)
        result = compute_confidence_metrics("entropy", scores, generated_ids, 5, 5)
        assert abs(result - math.log(vocab_size)) < 0.01


class TestMinEntropy:
    def test_uniform_min_entropy(self):
        """Uniform -> min-entropy = log(vocab_size)."""
        vocab_size = 16
        scores = make_uniform_scores(vocab_size, n_tokens=2)
        me = compute_min_entropy(scores, answer_token_start=4, prompt_len=4)
        assert abs(me - math.log(vocab_size)) < 0.01

    def test_peaked_min_entropy(self):
        """Near-certain -> min-entropy ~ 0."""
        vocab_size = 16
        scores = make_peaked_scores(vocab_size, n_tokens=2)
        me = compute_min_entropy(scores, answer_token_start=4, prompt_len=4)
        assert me < 0.01

    def test_min_entropy_via_dispatch(self):
        vocab_size = 4
        scores = make_uniform_scores(vocab_size, n_tokens=1)
        generated_ids = torch.zeros(1, dtype=torch.long)
        result = compute_confidence_metrics("min-entropy", scores, generated_ids, 3, 3)
        assert abs(result - math.log(vocab_size)) < 0.01


class TestTokenIndexing:
    def test_only_answer_tokens_used(self):
        """Tokens before answer_token_start should be ignored.

        Use 3 generated tokens: first 2 are uniform, last 1 is peaked.
        answer_token_start = prompt_len + 2 -> only last token is answer.
        Entropy of last (peaked) token should be ~ 0, not log(vocab_size).
        """
        vocab_size = 16
        uniform = torch.zeros(1, vocab_size)
        peaked = torch.zeros(1, vocab_size)
        peaked[0, 0] = 100.0
        scores = (uniform, uniform, peaked)
        entropy = compute_entropy(scores, answer_token_start=7, prompt_len=5)  # first_idx=2
        assert entropy < 0.01

    def test_invalid_answer_start_raises(self):
        """answer_token_start < prompt_len should raise ValueError."""
        vocab_size = 4
        scores = make_uniform_scores(vocab_size, n_tokens=2)
        with pytest.raises(ValueError):
            compute_entropy(scores, answer_token_start=3, prompt_len=5)

    def test_invalid_metric_raises(self):
        vocab_size = 4
        scores = make_uniform_scores(vocab_size, n_tokens=2)
        generated_ids = torch.zeros(2, dtype=torch.long)
        with pytest.raises(ValueError):
            compute_confidence_metrics("bad-metric", scores, generated_ids, 5, 5)
