# KV Cache Reuse in Confidence Extraction

## Problem

The confidence pipeline (`confidence.py::dropout_forward`) was called 4 times per sample (answer_logits, ptrue1, ptrue2, verbconf). Each call:

1. Re-tokenized the entire conversation from scratch via `_tokenize_for_confidence()`
2. Ran an **expensive early forward pass** over all prompt + CoT tokens to build `early_cache`

The early portion is typically 80-90% of the sequence, so this was the dominant cost. Meanwhile, `model.generate()` had already computed a KV cache (`gen.past_key_values`) covering all prompt + generated tokens — but it was never passed to the confidence code.

## Solution

Three tiers of optimization, all backward-compatible (`gen_cache=None` falls back to the old behavior).

### Tier 1: Eliminate early forward pass (all 4 calls)

Instead of running `llm.model(input_ids=early_tokens, past_key_values=empty_cache)`, we crop a deep copy of the generation cache to `early_late_split` tokens:

```python
early_cache = copy.deepcopy(gen_cache)
crop_cache(early_cache, early_late_split)
```

This replaces 4 full forward passes with 4 tensor slice operations.

### Tier 2: Overlap-based cache reuse for suffix calls (3 of 4 calls)

When a suffix is appended (ptrue1, ptrue2, verbconf), we find the maximum token overlap between the base tokenization (no suffix) and the suffix tokenization:

```python
overlap_len = find_token_overlap(base_tokens[0], suffix_tokens[0])
```

The overlap typically extends well past `early_late_split` (covering all prompt + CoT + answer tokens). We crop the generation cache to `overlap_len` and forward pass only the remaining suffix tokens — usually just a handful.

### Tier 3: Compute base_tokens once

`_tokenize_for_confidence` with no suffix is now called once in `compute_all_confidence_scores` and shared across all 4 confidence methods, eliminating 3 redundant tokenizations.

## Files Changed

### `utils/text_utils.py`
- Added `find_token_overlap(base_ids, suffix_ids)`: returns the length of the longest common prefix between two 1D token ID tensors.

### `confidence.py`
- Added `crop_cache(cache, max_length)`: handles both `DynamicCache` (has `.crop()`) and `Qwen3_5DynamicCache` (which does NOT inherit from `DynamicCache` and lacks `.crop()` — we manually slice `key_cache[idx][:, :, :max_length, :]` on attention layers).
- `compute_all_confidence_scores`: new `gen_cache` param; computes `base_tokens` once and passes both downstream.
- `dropout_answerlogits`, `dropout_indirectlogits`, `dropout_verbalconf`: accept and forward `gen_cache` + `base_tokens`.
- `dropout_forward`: rewritten core logic — see "Solution" above. Logs overlap length and number of discarded cache tokens.

### `main.py`
- Passes `gen.past_key_values` as `gen_cache` to `compute_all_confidence_scores`.

## Discarded Token Logging

`dropout_forward` logs via `logger.info`:
- **Suffix case**: `KV cache reuse: overlap=N, discarded=M (of T)` — M tokens from the generation cache couldn't be reused because re-tokenization changed them at the suffix boundary.
- **No-suffix case**: `KV cache reuse (no suffix): reused=N, discarded=M (of T)` — M tokens are the answer region beyond the early/late split.

## Qwen3_5DynamicCache Caveat

`Qwen3_5DynamicCache` inherits from `object`, not `DynamicCache`. It has no `crop()` method. The `crop_cache()` helper handles this by directly slicing the 4D KV tensors (`[batch, heads, seq_len, head_dim]`) on attention layers and leaving linear attention states (`conv_states`, `recurrent_states`) untouched (they are sequence-length-independent).

## Verification

1. Run with `gen_cache=None` to confirm identical behavior to the old code path.
2. Run with `--confidence --debug_conf` on a small sample and compare scores.
3. Check log output for overlap/discarded counts — overlap should be close to `len(base_tokens)`.
