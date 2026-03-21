# i11: Confidence Metrics for Final Answer Tokens

## Overview

This feature adds confidence metrics to the CoT trajectory pipeline, quantifying model uncertainty on the tokens it generates after `"Final Answer: "`. Three metrics are supported: perplexity, entropy, and min-entropy — all computed directly on the logits from `model.generate()`.

## Metrics

Let $N$ be the number of final-answer tokens and $p_i(v)$ the softmax probability of token $v$ at step $i$.

### Perplexity

$$\text{perplexity} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \log p_i(t_i)\right)$$

where $t_i$ is the token actually generated at step $i$. Lower perplexity → model is more certain about its answer tokens.

### Entropy (Shannon)

$$\text{entropy} = \frac{1}{N}\sum_{i=1}^{N} \left(-\sum_v p_i(v) \log p_i(v)\right)$$

Mean Shannon entropy over the answer token distributions, in nats. Lower entropy → more concentrated distribution.

### Min-Entropy

$$\text{min-entropy} = \frac{1}{N}\sum_{i=1}^{N} \left(-\log \max_v p_i(v)\right)$$

Mean min-entropy, which focuses on the single most likely token. Tighter worst-case bound on uncertainty than Shannon entropy.

## Implementation

### New file: `confidence.py`

Contains:
- `_get_answer_scores(scores, answer_token_start, prompt_len)` — extracts logit tensors for answer tokens only
- `compute_perplexity(scores, generated_ids, answer_token_start, prompt_len)`
- `compute_entropy(scores, answer_token_start, prompt_len)`
- `compute_min_entropy(scores, answer_token_start, prompt_len)`
- `compute_confidence_metrics(metric, scores, generated_ids, answer_token_start, prompt_len)` — dispatch function

### Modified: `utils/structures.py`

`GenerationResult` gains two optional fields:
- `scores: tuple | None` — per-step logit tuples from `model.generate(output_scores=True)`
- `generated_ids: torch.Tensor | None` — 1-D tensor of generated token IDs (prompt excluded)

### Modified: `main.py`

- **`parse()`**: New `--confidence` argument (`choices=["perplexity", "entropy", "min-entropy"]`, default `None`)
- **`generate_one()`**: New `output_scores: bool = False` parameter; passes `output_scores=output_scores` to `model.generate()` and returns scores + generated_ids in `GenerationResult`
- **`generate_trajectories()`**: New `confidence: str | None = None` parameter; computes metric and stores `confidence_metric` and `confidence_score` in the trajectory dict
- **`main()`**: Passes `confidence=args.confidence` to `generate_trajectories()`

## Token Indexing

Finding which scores correspond to final-answer tokens:

- `outputs.scores` is a tuple of `(1, vocab_size)` tensors, one per generated token
- `scores[k]` = logits for the token at absolute sequence position `prompt_len + k`
- `answer_token_start_position` = absolute position where `"Final Answer: "` text begins (computed in `generate_trajectories` by tokenizing the prefix)
- **First answer token index in scores** = `answer_token_start_position - prompt_len`

## Usage

```bash
# Perplexity
python main.py --dataset logiqa --model llama --confidence perplexity

# Shannon entropy
python main.py --dataset logiqa --model llama --confidence entropy

# Min-entropy
python main.py --dataset logiqa --model llama --confidence min-entropy

# No confidence metric (original behavior)
python main.py --dataset logiqa --model llama
```

## Output Format

When `--confidence` is set, each trajectory in the output JSON gains two fields:

```json
{
  "confidence_metric": "perplexity",
  "confidence_score": 1.042
}
```

When `--confidence` is not set (or answer was not found), both fields are `null`.

## Testing

Unit tests are in `tests/test_confidence.py`. They use synthetic CPU tensors — no GPU or real model required.

```bash
pytest tests/test_confidence.py -v
```

Test coverage:
- Uniform distribution → expected metric values (perplexity = vocab_size, entropy = log(vocab_size), min-entropy = log(vocab_size))
- Peaked distribution → near-zero entropy/min-entropy, perplexity ≈ 1
- Token indexing: only tokens from `answer_token_start` onwards are used
- Error handling: invalid `answer_token_start`, invalid metric name
