# Fusing Confidence Forward Passes — Optimization Notes

**Branch:** `i28/eval`
**Date:** 2026-03-25
**Scope:** `confidence.py:compute_all_confidence_scores()` and all functions it calls — the confidence computation path only. Complements `i28_cot_optimization.md` (which covers the generation path).

---

## Table of Contents

1. [Current Architecture](#1-current-architecture)
2. [The Redundancy Problem](#2-the-redundancy-problem)
3. [Proposed Solution: Fused Confidence Computation](#3-proposed-solution-fused-confidence-computation)
4. [Implementation Plan](#4-implementation-plan)
5. [Expected Savings](#5-expected-savings)
6. [Risks and Caveats](#6-risks-and-caveats)

---

## 1. Current Architecture

### 1.1 Entry Point

`compute_all_confidence_scores()` (`confidence.py:52-112`) is called once per instance from `main.py:204-214`. It receives the LLM, the conversation messages, the generated text, the parsed output, and the generation KV cache (`gen_cache`).

### 1.2 Three Confidence Methods, Four `dropout_forward()` Calls

The function calls three confidence methods sequentially:

```
compute_all_confidence_scores()
 ├── dropout_answerlogits()      →  dropout_forward(suffix="")                         # Call A
 ├── dropout_indirectlogits()    →  dropout_forward(suffix="\nTrue/False:")             # Call B
 │                               →  dropout_forward(suffix="\nIs the answer X correct?") # Call C
 └── dropout_verbalconf()        →  dropout_forward(suffix="\nPlease respond...")       # Call D
```

Each `dropout_forward()` (`confidence.py:371-506`) performs:

1. **Tokenize** the full assistant content + suffix → `tokens`
2. **Locate** the answer region within `tokens` via `find_token_indices_from_end()`
3. **Split** into `early_tokens` and `late_tokens` at `early_late_split`
4. **Build `early_cache`** by deep-copying `gen_cache` and cropping to `early_late_split`
5. **Vanilla forward** on `late_tokens` using a deep copy of `early_cache`
6. **Dropout forward** (batched, `nb_dropout_samples=3`) on `late_tokens` using another deep copy of `early_cache`

### 1.3 What Each Method Extracts

| Method | Reads from forward output | Suffix tokens |
|--------|--------------------------|---------------|
| `dropout_answerlogits` | Logits at answer token positions → per-token P(token) and entropy | None |
| `dropout_indirectlogits` (ptrue1) | Last-position logit → P(True) vs P(False) | `\nTrue/False:` (~4 tokens) |
| `dropout_indirectlogits` (ptrue2) | Last-position logit → P(Yes) vs P(No) | `\nIs the answer X correct?` (~8 tokens) |
| `dropout_verbalconf` | Last-position logit → distribution over 0–100, then multi-token joint probs via `_compute_verbconf_joint_probs()` | `\nPlease respond...` (~20 tokens) |

### 1.4 `_compute_verbconf_joint_probs()` — Additional Forward Passes

After the main `dropout_forward()`, `dropout_verbalconf` calls `_compute_verbconf_joint_probs()` (`confidence.py:224-287`). This function handles multi-token number sequences (e.g., "42" tokenizes as ["4", "2"]). For each unique first token among the 101 numbers (0–100), it:

1. Deep-copies the KV cache from the forward output
2. Runs an additional `llm.model.forward()` call with that single token
3. Reads the next-token logits to get P(second digit | first digit)

In practice, there are ~10–15 unique first tokens, so this adds **10–15 extra forward passes** for vanilla, and another 10–15 for dropout. Each uses `copy.deepcopy` on the cache.

---

## 2. The Redundancy Problem

### 2.1 Identical `early_late_split` Across All Calls

The split point is computed from `parsed_output.answer_fullstring_start` (`confidence.py:405-409`):

```python
full_text = (assistant_prefill + generated_text).strip()
fullstring_text = full_text[parsed_output.answer_fullstring_start:]
fs_start, _ = find_token_indices_from_end(llm.tokenizer, tokens[0], fullstring_text)
early_late_split = fs_start - 1
```

`fullstring_text` is the same across all 4 calls (it depends only on the generated answer, not the suffix). Adding suffix tokens extends `tokens` beyond the base, but the search from the end for `fullstring_text` finds the same position relative to the start. Therefore `early_late_split` is identical for all 4 calls.

### 2.2 Redundancy: 4x `deepcopy(gen_cache)`

Each call independently deep-copies the full generation KV cache and crops it to the same split point (`confidence.py:428-430`):

```python
early_cache = copy.deepcopy(gen_cache)         # ← happens 4 times
crop_cache(early_cache, early_late_split)       # ← same split every time
```

For Qwen3.5-27B with a long CoT sequence, `gen_cache` contains KV tensors for all 36+ layers. Each deep copy duplicates every tensor. Estimated cost per copy: **4–8 GB of GPU memory allocation + copy**, depending on sequence length. Total waste: **3 unnecessary copies** (12–24 GB of redundant GPU memcpy).

### 2.3 Redundancy: 8x `deepcopy(early_cache)`

Inside `dropout_forward()`, the early cache is deep-copied twice per call — once for the vanilla forward (`confidence.py:448, 468, 483`) and once for the dropout forward (`confidence.py:497`):

```python
# Vanilla (one of three code paths, but all deepcopy):
vanilla_output = llm.model.forward(
    input_ids=...,
    past_key_values=copy.deepcopy(early_cache),    # ← happens 4 times
    ...
)

# Dropout:
dropout_late_forward(
    ...,
    copy.deepcopy(early_cache),                    # ← happens 4 times
    ...
)
```

Total: **8 deep copies** of the (cropped) early cache. 6 of these are unnecessary.

### 2.4 Redundancy: Call A's Vanilla Forward is a Subset of Calls B/C/D

Call A runs the vanilla forward on `late_tokens` (no suffix). Calls B, C, D run the vanilla forward on `late_tokens + suffix_tokens`. Because the model is causal, the logits at the base `late_tokens` positions are identical regardless of whether suffix tokens are appended. Call A's entire vanilla forward output is contained within any suffix call's output.

More concretely: if Call D (the longest suffix) is run, its output logits at positions `0:len(late_tokens)` are exactly what Call A computes. Call A is pure waste when any suffix call is also being run.

### 2.5 Redundancy: Separate Vanilla Forwards for B, C, D

Calls B, C, D differ only in the suffix appended after the generated text. When `gen_cache` is available and the overlap is large enough (the common case — see `confidence.py:438-462`), the vanilla forward only needs to process the non-overlapping tail tokens. But each call deep-copies `gen_cache` again to crop to the overlap point.

These three suffixes could be batched into a single forward pass (batch=3) sharing the same prefix cache, eliminating 2 redundant cache copies and 2 redundant forward passes.

### 2.6 Redundancy: Separate Dropout Forwards for A, B, C, D

All 4 dropout forwards:
- Use the same `early_cache`
- Apply the same CoT step masking pattern (since the steps are in the early region)
- Differ only in the `late_tokens` (base vs. base+suffix)

These could be batched into a single forward pass (batch = 4 variants × 3 dropout samples = 12) with appropriate padding.

### 2.7 Summary of All Forward Passes Per Instance

| Component | Vanilla | Dropout (batch=3) | verbconf extra | Total `model.forward()` |
|-----------|---------|-------------------|----------------|------------------------|
| `dropout_answerlogits` | 1 | 1 | — | 2 |
| `dropout_indirectlogits` (ptrue1) | 1 | 1 | — | 2 |
| `dropout_indirectlogits` (ptrue2) | 1 | 1 | — | 2 |
| `dropout_verbalconf` | 1 | 1 | ~10–15 (vanilla) + ~10–15 (dropout) | 22–32 |
| **Total** | **4** | **4** | **~20–30** | **~28–36** |

Plus **12 `deepcopy` calls** on large KV caches (4 on `gen_cache`, 8 on `early_cache`), plus ~20–30 more deep copies inside `_compute_verbconf_joint_probs`.

---

## 3. Proposed Solution: Fused Confidence Computation

### 3.1 Overview

Replace the 4 independent `dropout_forward()` calls with a single fused function that:

1. Builds the early cache **once**
2. Runs **one batched vanilla forward** covering all suffix variants
3. Runs **one batched dropout forward** covering all suffix variants × dropout samples
4. Dispatches the outputs to each confidence method for post-processing

### 3.2 Phase 1: Deduplicate Cache Construction

```python
def compute_all_confidence_scores_fused(llm, messages, generated_text, parsed_output, ...):
    # ---- Step 1: Build early_cache ONCE ----
    base_content = (assistant_prefill + generated_text).strip()
    base_tokens = _tokenize_for_confidence(llm, messages, base_content)

    fullstring_text = base_content[parsed_output.answer_fullstring_start:]
    fs_start, _ = find_token_indices_from_end(llm.tokenizer, base_tokens[0], fullstring_text)
    early_late_split = fs_start - 1

    early_cache = copy.deepcopy(gen_cache)       # ONE deep copy, not 4
    crop_cache(early_cache, early_late_split)

    base_late_tokens = base_tokens[:, early_late_split:]
    # ... reuse early_cache for all subsequent work
```

**Savings:** 3 eliminated `deepcopy(gen_cache)` calls.

### 3.3 Phase 2: Batch the Vanilla Forwards

The 4 vanilla forwards differ only in suffix tokens. Build a padded batch:

```python
    suffixes = [
        "",                                                    # answerlogits
        "\nTrue/False:",                                       # ptrue1
        f"\nIs the answer {parsed_output.final_answer} correct?",  # ptrue2
        "\nPlease respond with a score from 0 to 100 ...",     # verbconf
    ]

    # Tokenize each suffix variant
    suffix_token_lists = []
    for suffix in suffixes:
        if suffix:
            content = (assistant_prefill + generated_text + suffix).strip()
            toks = _tokenize_for_confidence(llm, messages, content)
            suffix_token_lists.append(toks[:, early_late_split:])
        else:
            suffix_token_lists.append(base_late_tokens)

    # Pad to max length, build attention mask
    max_late_len = max(t.shape[1] for t in suffix_token_lists)
    padded_late = torch.full((4, max_late_len), llm.tokenizer.pad_token_id, device=device)
    attention_masks = torch.zeros(4, max_late_len, dtype=torch.bool, device=device)
    for i, t in enumerate(suffix_token_lists):
        L = t.shape[1]
        padded_late[i, :L] = t[0]
        attention_masks[i, :L] = True

    # ONE vanilla forward (batch=4), ONE deepcopy of early_cache
    vanilla_cache = copy.deepcopy(early_cache)
    vanilla_cache.reorder_cache(torch.tensor([0, 0, 0, 0]))

    # Build causal + padding attention mask for the batch
    # (batch=4, 1, max_late_len, early_late_split + max_late_len)
    vanilla_attn_mask = _build_causal_mask_with_padding(
        attention_masks, early_late_split, max_late_len, device)

    with torch.no_grad():
        vanilla_output = llm.model.forward(
            input_ids=padded_late,
            attention_mask=vanilla_attn_mask,
            past_key_values=vanilla_cache,
            output_hidden_states=True,
        )

    # Slice outputs per suffix variant
    v_out_answer   = _slice_output(vanilla_output, 0, suffix_token_lists[0].shape[1])
    v_out_ptrue1   = _slice_output(vanilla_output, 1, suffix_token_lists[1].shape[1])
    v_out_ptrue2   = _slice_output(vanilla_output, 2, suffix_token_lists[2].shape[1])
    v_out_verbconf = _slice_output(vanilla_output, 3, suffix_token_lists[3].shape[1])
```

**Savings:** 4 vanilla forwards → 1. 4 `deepcopy(early_cache)` for vanilla → 1.

### 3.4 Phase 3: Batch the Dropout Forwards

The same approach, but expanded for dropout samples:

```python
    nb_dropout = 3
    total_batch = 4 * nb_dropout  # = 12

    # Expand late_tokens: each suffix variant × nb_dropout samples
    dropout_late = padded_late.repeat_interleave(nb_dropout, dim=0)  # [12, max_late_len]

    # Build dropout masks: same CoT step masking, different suffix lengths
    # The masking applies to the early region (CoT steps), which is shared
    dropout_masks = _build_dropout_masks(
        llm, parsed_output.cot_steps,
        early_tokens=base_tokens[:, :early_late_split],
        late_lens=[t.shape[1] for t in suffix_token_lists],
        nb_dropout=nb_dropout,
        threshold=0.5,
        early_late_split=early_late_split,
        max_late_len=max_late_len,
    )

    dropout_cache = copy.deepcopy(early_cache)   # 1 copy, not 4
    dropout_cache.reorder_cache(torch.tensor([0] * total_batch))

    with torch.no_grad():
        dropout_output = llm.model.forward(
            input_ids=dropout_late,
            attention_mask=dropout_masks,
            past_key_values=dropout_cache,
            output_hidden_states=True,
        )

    # Reshape: [12, ...] → [4, 3, ...] → slice per method
    d_out_answer   = _slice_dropout(dropout_output, 0, nb_dropout, ...)
    d_out_ptrue1   = _slice_dropout(dropout_output, 1, nb_dropout, ...)
    d_out_ptrue2   = _slice_dropout(dropout_output, 2, nb_dropout, ...)
    d_out_verbconf = _slice_dropout(dropout_output, 3, nb_dropout, ...)
```

**Savings:** 4 dropout forwards → 1. 4 `deepcopy(early_cache)` for dropout → 1.

### 3.5 Phase 4: Post-Processing (No Change to Logic)

Each confidence method's post-processing remains the same — it just receives sliced outputs instead of calling `dropout_forward()` itself:

```python
    # Extract confidence scores from sliced outputs
    vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy = \
        _extract_answerlogits(llm, base_late_tokens, v_out_answer, d_out_answer, ans_start, ans_end)

    vanilla_ptrue1, vanilla_ptrue2, dropout_ptrue1, dropout_ptrue2 = \
        _extract_indirectlogits(llm, v_out_ptrue1, v_out_ptrue2, d_out_ptrue1, d_out_ptrue2)

    vanilla_verbconf, dropout_verbconf, ... = \
        _extract_verbalconf(llm, v_out_verbconf, d_out_verbconf, token_seqs, ...)
```

### 3.6 Phase 5: Optimize `_compute_verbconf_joint_probs()`

This function currently does ~10–15 individual forward passes per output (vanilla and dropout separately). Two approaches to reduce this:

**Option A — Batch all unique first tokens into one forward pass:**

Currently the function loops over unique prefixes at each depth level, running a separate `llm.model.forward()` for each. Instead, batch all unique prefixes at a given depth into a single forward pass:

```python
# Current: one forward per unique prefix
for prefix, entries in groups.items():
    kv = copy.deepcopy(parent_kv)
    out = llm.model.forward(input_ids=torch.tensor([[feed_token]]), past_key_values=kv)

# Proposed: one forward for all prefixes at this depth
all_feed_tokens = torch.tensor([[tok] for tok in unique_tokens])  # [N, 1]
batched_kv = copy.deepcopy(parent_kv)
batched_kv.reorder_cache(torch.tensor([0] * len(unique_tokens)))
out = llm.model.forward(input_ids=all_feed_tokens, past_key_values=batched_kv)
```

This collapses ~10–15 forward passes per depth level into 1. Since max depth for numbers 0–100 is 3, this reduces the verbconf extra passes from ~20–30 to ~2–3.

**Option B — Pre-compute all joint probabilities from a single extended forward:**

Instead of conditioning one token at a time, append all 101 candidate tokens after the suffix and run a single forward pass with appropriate masking. This is more complex but eliminates the loop entirely.

---

## 4. Implementation Plan

### Step 1 (Low effort, high impact): Deduplicate `early_cache`

- Compute `early_late_split` once in `compute_all_confidence_scores()`
- Build `early_cache` once (single `deepcopy(gen_cache)` + `crop_cache`)
- Pass `early_cache` and `early_late_split` to each confidence method

This is a **drop-in refactor** that doesn't change the call structure. Each method still calls `dropout_forward()`, but `dropout_forward()` accepts a pre-built `early_cache` parameter instead of rebuilding it.

**Expected savings:** ~60% of the `deepcopy` cost (3 of 4 `gen_cache` copies eliminated).

### Step 2 (Medium effort, high impact): Batch vanilla forwards

- Tokenize all 4 suffix variants upfront
- Pad to max length and build a combined attention mask
- Run one `model.forward()` with batch=4
- Slice outputs per method

**Expected savings:** 4 vanilla forwards → 1, 3 more `deepcopy(early_cache)` eliminated.

### Step 3 (Medium effort, medium impact): Batch dropout forwards

- Expand the batch=4 late tokens × 3 dropout samples = batch=12
- Build all dropout masks in one tensor operation
- Run one `model.forward()` with batch=12
- Reshape outputs per method × sample

**Expected savings:** 4 dropout forwards → 1, 3 more `deepcopy(early_cache)` eliminated.

### Step 4 (Medium effort, medium impact): Batch `_compute_verbconf_joint_probs()`

- At each depth level, batch all unique prefix tokens into a single forward pass
- Reduces ~10–15 forwards per depth to 1 per depth

**Expected savings:** ~20–30 extra forwards → 2–3.

---

## 5. Expected Savings

### 5.1 `deepcopy` Calls

| Resource | Current | After Step 1 | After Steps 2+3 |
|----------|---------|--------------|------------------|
| `deepcopy(gen_cache)` | 4 | **1** | **1** |
| `deepcopy(early_cache)` for vanilla | 4 | 4 | **1** |
| `deepcopy(early_cache)` for dropout | 4 | 4 | **1** |
| `deepcopy` in verbconf joint probs | ~20–30 | ~20–30 | ~20–30 (or ~2–3 after Step 4) |
| **Total deep copies** | **~32–42** | **~29–39** | **~3** (after all steps) |

### 5.2 `model.forward()` Calls

| Resource | Current | After Steps 2+3 | After Step 4 |
|----------|---------|------------------|--------------|
| Vanilla forwards (main) | 4 | **1** (batch=4) | **1** |
| Dropout forwards (main) | 4 (each batch=3) | **1** (batch=12) | **1** |
| Verbconf extra forwards | ~20–30 | ~20–30 | **~2–3** |
| **Total forward calls** | **~28–38** | **~22–32** | **~4–5** |

### 5.3 Estimated Wall-Clock Improvement

Assuming Qwen3.5-27B on a single A100-80GB with a ~2000-token CoT sequence:

| Component | Current est. | After all steps |
|-----------|-------------|-----------------|
| `deepcopy` overhead | 20–50s | 2–5s |
| Main forward passes (vanilla + dropout) | 30–80s | 8–20s |
| Verbconf joint prob forwards | 20–60s | 3–8s |
| **Confidence total** | **~70–190s** | **~13–33s** |

Combined with generation-path optimizations from `i28_cot_optimization.md` (early stopping, removing unnecessary `deepcopy` in `generate_one()`), the full per-instance runtime could drop from **200–400s to ~50–100s**.

---

## 6. Risks and Caveats

### 6.1 GPU Memory Pressure from Larger Batches

Batching 4 vanilla forwards means 4× the activation memory. Batching 12 dropout forwards means 12× the activation memory for the late region. For very long CoT sequences, this could cause OOM on GPUs with limited memory.

**Mitigation:** Make the batch size configurable. Fall back to sequential processing if a CUDA OOM is caught:

```python
try:
    out = llm.model.forward(input_ids=batched_late, ...)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Fall back to sequential
    out = [llm.model.forward(input_ids=late[i:i+1], ...) for i in range(batch)]
```

### 6.2 Padding Affects Logit Values

When padding shorter suffix variants to the max length, the padding tokens participate in the forward pass. The attention mask must correctly prevent real tokens from attending to padding positions. Use a 4D attention mask with `-inf` (or `-10000`) at padding positions.

**Verification:** After implementation, compare the fused outputs against the current sequential outputs on a few instances to confirm numerical equivalence (up to floating-point tolerance).

### 6.3 Dropout Mask Construction Complexity

The current `dropout_late_forward()` builds masks relative to per-call `late_tokens`. Fusing requires building masks for all 4 variants simultaneously, accounting for different `late_tokens` lengths and ensuring the CoT step masking positions in the early region are consistent.

**Mitigation:** The CoT step positions are in the early region, which is identical across all calls. Only the `modify_start_late` and `modify_end_late` vary per suffix variant and need per-variant indexing in the mask tensor.

### 6.4 `_compute_verbconf_joint_probs()` Cache Sharing

The verbconf joint probability computation deep-copies the KV cache from the forward output to branch on different first tokens. When batching, the KV cache from the forward output already has batch dimension = 4 (or 12 for dropout). The deep copy and branching logic needs to index into the correct batch element.

**Mitigation:** Extract the single relevant batch element's KV cache before passing to `_compute_verbconf_joint_probs()`, so its internal logic remains unchanged.

### 6.5 Backward Compatibility

The post-processing functions (`dropout_answerlogits`, `dropout_indirectlogits`, `dropout_verbalconf`) currently receive full `model.forward()` output objects. After fusing, they'll receive sliced sub-tensors. The slicing must preserve the expected shapes (batch dim, sequence dim, vocab dim).

**Mitigation:** Refactor the post-processing into pure functions that accept logit tensors rather than full model output objects. This also improves testability.
