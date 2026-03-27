# Confidence & LLM Speed Optimizations

**Branch:** `i28/eval`
**Files:** `llm.py`, `confidence.py`, `utils/text_utils.py`
**Date:** 2026-03-25

---

## Table of Contents

1. [Eliminate redundant KV cache deep copies across confidence methods](#1-eliminate-redundant-kv-cache-deep-copies-across-confidence-methods)
2. [Batch verbconf prefix forwards instead of per-prefix deep copy](#2-batch-verbconf-prefix-forwards-instead-of-per-prefix-deep-copy)
3. [Make KV cache return optional in `generate_one`](#3-make-kv-cache-return-optional-in-generate_one)
4. [Pre-compute static token sequences](#4-pre-compute-static-token-sequences)
5. [Optimize `find_token_indices_from_end`](#5-optimize-find_token_indices_from_end)
6. [`torch.compile` for repeated short forwards](#6-torchcompile-for-repeated-short-forwards)

---

## 1. Eliminate redundant KV cache deep copies across confidence methods

### Problem

`compute_all_confidence_scores` (confidence.py:52) calls three methods sequentially:

```
dropout_answerlogits   → calls dropout_forward → deep copies gen_cache (line 428)
dropout_indirectlogits → calls dropout_forward twice → deep copies gen_cache twice (lines 428 x2)
dropout_verbalconf     → calls dropout_forward → deep copies gen_cache (line 428)
```

Each `dropout_forward` call independently:
1. Deep copies the **entire** generation KV cache (`copy.deepcopy(gen_cache)` at line 428)
2. Crops it to `early_late_split` (line 430)
3. Then deep copies the cropped cache again for the vanilla forward (line 468) and the dropout forward (line 497)

For a 7B model with 2048+ token context, a single KV cache can be **2–4 GB**. `copy.deepcopy` on GPU tensors triggers a full device-to-device copy for every tensor in every layer. With 4 calls to `dropout_forward`, that is **4 full cache deep copies** just to produce the same `early_cache`, plus additional copies inside each call.

### Current flow (simplified)

```
compute_all_confidence_scores
├── dropout_answerlogits
│   └── dropout_forward(gen_cache)
│       ├── early_cache = deepcopy(gen_cache)   # EXPENSIVE
│       ├── crop_cache(early_cache, split)
│       ├── vanilla: deepcopy(early_cache)      # EXPENSIVE
│       └── dropout: deepcopy(early_cache)      # EXPENSIVE
├── dropout_indirectlogits
│   ├── dropout_forward(gen_cache, suffix="True/False:")
│   │   ├── early_cache = deepcopy(gen_cache)   # REDUNDANT — same split
│   │   ├── crop_cache(early_cache, split)
│   │   ├── vanilla: deepcopy(early_cache) or deepcopy(gen_cache)
│   │   └── dropout: deepcopy(early_cache)
│   └── dropout_forward(gen_cache, suffix="Is the answer...")
│       ├── early_cache = deepcopy(gen_cache)   # REDUNDANT — same split
│       └── ...
└── dropout_verbalconf
    └── dropout_forward(gen_cache)
        ├── early_cache = deepcopy(gen_cache)   # REDUNDANT — same split
        └── ...
```

Total deep copies of the full `gen_cache`: **4**
Total deep copies of `early_cache`: **up to 8** (vanilla + dropout per call)

### Fix

Compute `early_cache` **once** in `compute_all_confidence_scores` and pass it to all methods. The `early_late_split` point is determined by `parsed_output.answer_fullstring_start`, which is the same across all three methods.

```python
def compute_all_confidence_scores(llm, messages, generated_text, parsed_output,
                                   nb_dropout_samples=10, use_fullstring=False,
                                   assistant_prefill="", debug_conf=False,
                                   gen_cache=None) -> AllConfidenceData:
    # Compute base tokenization once (already done)
    base_content = (assistant_prefill + generated_text).strip()
    base_tokens = _tokenize_for_confidence(llm, messages, base_content)

    # NEW: compute early_cache once
    early_cache, early_late_split = _build_early_cache(
        llm, gen_cache, base_tokens, parsed_output
    )

    # Pass early_cache + split to each method
    dropout_answerlogits(..., early_cache=early_cache, split=early_late_split, ...)
    dropout_indirectlogits(..., early_cache=early_cache, split=early_late_split, ...)
    dropout_verbalconf(..., early_cache=early_cache, split=early_late_split, ...)
```

Where `_build_early_cache` encapsulates the one-time work:

```python
def _build_early_cache(llm, gen_cache, base_tokens, parsed_output):
    """Compute the early KV cache once for reuse across confidence methods."""
    full_text = ...  # same as in dropout_forward
    fullstring_text = full_text[parsed_output.answer_fullstring_start:]
    fs_start, _ = find_token_indices_from_end(llm.tokenizer, base_tokens[0], fullstring_text)
    early_late_split = fs_start - 1

    if gen_cache is not None:
        early_cache = copy.deepcopy(gen_cache)  # ONE deep copy
        crop_cache(early_cache, early_late_split)
    else:
        device = next(llm.model.parameters()).device
        early_tokens = base_tokens[:, :early_late_split].to(device)
        with torch.no_grad():
            out = llm.model(input_ids=early_tokens, past_key_values=DynamicCache())
        early_cache = out.past_key_values

    return early_cache, early_late_split
```

Then inside `dropout_forward`, the existing logic that builds `early_cache` is skipped when it's passed in. Each consumer only deep copies `early_cache` (which is already cropped and much smaller than the full `gen_cache`) when it needs a mutable copy for vanilla/dropout forwards.

### Expected savings

| Before | After |
|--------|-------|
| 4x deepcopy of full `gen_cache` | 1x deepcopy of full `gen_cache` |
| ~8x deepcopy of `early_cache` | ~8x deepcopy of **smaller** `early_cache` |

The full `gen_cache` includes all prompt + CoT + answer tokens. `early_cache` only covers prompt + CoT (excluding the answer region). The savings scale with answer length and suffix length, but eliminating 3 redundant full-cache copies is the main win.

For a 7B model with 2048-token context, each deep copy is ~500ms–2s depending on GPU. Eliminating 3 saves **1.5–6 seconds per question**.

---

## 2. Batch verbconf prefix forwards instead of per-prefix deep copy

### Problem

`_compute_verbconf_joint_probs` (confidence.py:224) handles multi-token number sequences (e.g., "42" tokenized as `[tok_4, tok_2]`). For each depth level > 0, it iterates over unique prefixes and does:

```python
for prefix, entries in groups.items():
    kv = copy.deepcopy(parent_kv)                          # line 271 — EXPENSIVE
    tok_input = torch.full((batch_size, 1), feed_token, ...)
    out = llm.model.forward(input_ids=tok_input, past_key_values=kv)  # line 276
```

For numbers 0–100, the tokenizer typically produces:
- Single-token: 0–9 (10 numbers) — no extra forwards needed
- Two-token: 10–99 (90 numbers) — grouped by first digit, ~9 unique prefixes at depth 1
- Three-token: 100 (1 number) — 1 prefix at depth 1, 1 at depth 2

So at depth 1, there are ~10 unique prefixes, each getting a deep copy + a separate forward pass. With dropout (batch_size = nb_dropout_samples), each deep copy duplicates the cache across all samples.

### Fix

Group all prefixes sharing the same parent and batch them into a single forward pass using `reorder_cache`:

```python
for d in range(1, max_depth):
    # Group sequences by their prefix at depth d
    groups = defaultdict(list)
    for i, seq in enumerate(token_seqs):
        if len(seq) > d:
            prefix = tuple(seq[:d])
            groups[prefix].append((i, seq[d]))

    # Group prefixes by their PARENT — these share the same cache state
    parent_groups = defaultdict(list)
    for prefix in groups:
        parent = prefix[:-1]
        parent_groups[parent].append(prefix)

    for parent, child_prefixes in parent_groups.items():
        parent_kv = prefix_cache[parent]
        nb_children = len(child_prefixes)

        # Expand the parent cache to nb_children copies in one operation
        expand_indices = torch.tensor([0] * (nb_children * batch_size))
        batched_kv = copy.deepcopy(parent_kv)  # ONE copy for all children
        batched_kv.reorder_cache(expand_indices)

        # Stack all feed tokens into a single batched input
        feed_tokens = [child[-1] for child in child_prefixes]
        # Shape: [nb_children * batch_size, 1]
        tok_input = torch.tensor(
            [[ft] for ft in feed_tokens for _ in range(batch_size)],
            device=device
        )

        with torch.no_grad():
            out = llm.model.forward(input_ids=tok_input, past_key_values=batched_kv)

        # Scatter logprobs back to the right sequence indices
        logprobs_d = out.logits[:, -1, :].float().log_softmax(-1).cpu()
        for child_idx, prefix in enumerate(child_prefixes):
            start = child_idx * batch_size
            end = start + batch_size
            child_logprobs = logprobs_d[start:end]
            for seq_idx, next_tok in groups[prefix]:
                joint_logprobs[:, seq_idx] += child_logprobs[:, next_tok]
```

### Expected savings

| Before | After |
|--------|-------|
| ~10 deep copies at depth 1 | 1 deep copy + 1 reorder_cache |
| ~10 forward passes at depth 1 | 1 batched forward pass |
| ~1 deep copy at depth 2 | 1 deep copy at depth 2 (same) |

Forward pass throughput scales well with batch size on GPUs, so batching 10 single-token forwards into 1 is nearly free compared to 10 sequential passes. The bigger win is eliminating 9 deep copies.

---

## 3. Make KV cache return optional in `generate_one`

### Problem

`llm.py:202` unconditionally deep copies the entire generation cache:

```python
return GenerationResult(
    ...
    past_key_values=copy.deepcopy(outputs.past_key_values),
    ...
)
```

The `GenerationResult.past_key_values` field is only used when confidence scoring is enabled. When running vanilla evaluation without confidence (or when the caller immediately discards the result), this deep copy is pure waste.

### Fix

Add a `return_cache` parameter:

```python
def generate_one(self, ..., return_cache: bool = False) -> GenerationResult:
    ...
    return GenerationResult(
        ...
        past_key_values=outputs.past_key_values if return_cache else None,
        ...
    )
```

Note: when `return_cache=True`, we return the **original** cache object (no deep copy). This is safe because `outputs` goes out of scope after `generate_one` returns — the caller owns the only reference. If the caller needs multiple independent copies (e.g., for branching), they can deep copy at their call site.

The `GenerationResult.past_key_values` type annotation in `structures.py` should be updated to allow `None`:

```python
past_key_values: DynamicCache | Qwen3_5DynamicCache | None = None
```

### Expected savings

When confidence is disabled: eliminates 1 full cache deep copy (~500ms–2s per question for 7B+ models). When confidence is enabled: eliminates 1 deep copy by returning the live reference instead.

---

## 4. Pre-compute static token sequences

### Problem

Several tokenizer lookups are repeated identically on every call:

1. **`dropout_verbalconf`** (confidence.py:306): encodes strings "0" through "100" into token sequences every time:
   ```python
   token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
   score_values = torch.FloatTensor([int(s) / 100 for s in verbconf_strings])
   ```

2. **`dropout_indirectlogits`** (confidence.py:162–165): resolves True/False/Yes/No token IDs every time:
   ```python
   positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
   negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
   positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
   negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])
   ```

These are deterministic for a given tokenizer and never change between calls.

### Fix

Cache them on the `LLM` instance after first use:

```python
class LLM:
    def __init__(self, model_name, thinking):
        ...
        # Pre-compute static token lookups
        self._verbconf_token_seqs = [
            self.tokenizer.encode(str(i), add_special_tokens=False)
            for i in range(101)
        ]
        self._verbconf_score_values = torch.FloatTensor([i / 100 for i in range(101)])

        self._true_ids = get_token_ids(self.tokenizer, ANSWER_TOKENS[' True'])
        self._false_ids = get_token_ids(self.tokenizer, ANSWER_TOKENS[' False'])
        self._yes_ids = get_token_ids(self.tokenizer, ANSWER_TOKENS[' Yes'])
        self._no_ids = get_token_ids(self.tokenizer, ANSWER_TOKENS[' No'])
```

Then pass `llm._verbconf_token_seqs` etc. instead of recomputing. This also avoids 101 `tokenizer.encode` calls per question.

### Expected savings

Minor (~5–20ms per question), but it's free to implement and removes unnecessary work from the hot path.

---

## 5. Optimize `find_token_indices_from_end`

### Problem

`find_token_indices_from_end` (text_utils.py:15) uses a linear scan from the end of the token sequence, calling `tokenizer.convert_ids_to_tokens` in a loop:

```python
start = len(token_ids) - 1
while start > 0:
    text = "".join(tokenizer.convert_ids_to_tokens(token_ids[start:]))  # O(n) per iteration
    if string in text:
        break
    start -= 1
```

This is **O(n^2)** in the worst case — for each candidate `start`, it converts all tokens from `start` to end into strings and joins them. For a 2048-token sequence searching for a string near the beginning, that's ~2048 iterations each doing ~1024 token conversions on average.

This function is called:
- Once per `dropout_forward` call for `fullstring_text` (confidence.py:407)
- Once per `dropout_forward` call for `final_answer` (confidence.py:414–415)
- Once per reasoning step in `dropout_late_forward` (confidence.py:547–548), inside a loop over all `cot_steps`

For a generation with 10 CoT steps, that's ~12+ calls to this function per `dropout_forward`, and `dropout_forward` is called 4 times, so ~48 calls per question.

### Fix

Pre-compute the full decoded string once and use `rfind` to locate the substring:

```python
def find_token_indices_from_end(tokenizer, token_ids, string):
    """Find the token index range [start, end) that spans `string` in `token_ids`.

    Searches from the end of the sequence for the last occurrence.
    """
    # Encode the search string the same way for consistent comparison
    string_encoded = tokenizer.encode(string, add_special_tokens=False)
    string_normalized = "".join(tokenizer.convert_ids_to_tokens(string_encoded))

    # Convert all tokens once
    all_token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    # Build a cumulative string from the end and use rfind
    full_str = "".join(all_token_strs)
    match_pos = full_str.rfind(string_normalized)
    if match_pos == -1:
        raise ValueError(f"Cannot find '{string_normalized}' in token sequence")

    # Map character position back to token index
    cum_len = 0
    start = None
    end = None
    for i, tok_str in enumerate(all_token_strs):
        if start is None and cum_len + len(tok_str) > match_pos:
            start = i
        cum_len += len(tok_str)
        if start is not None and cum_len >= match_pos + len(string_normalized):
            end = i + 1
            break

    return start, end
```

This is **O(n)** — one pass to join, one `rfind`, one pass to map back to indices.

### Expected savings

For 2048-token sequences: reduces from ~2M token conversions to ~2048. With 48 calls per question, this can save **seconds** of pure Python overhead.

---

## 6. `torch.compile` for repeated short forwards

### Problem

The confidence scoring pipeline makes many short forward passes:
- 4 vanilla forwards (1 answer_logits, 2 indirect_logits, 1 verbconf)
- 4 dropout forwards (same)
- ~10 prefix forwards for verbconf multi-token scoring

Each forward pass through the model involves Python overhead for dispatching operations. `torch.compile` can fuse operations and eliminate this overhead.

### Fix

In `LLM.__init__`:

```python
self.model = AutoModelForCausalLM.from_pretrained(...)

# Compile for faster repeated forwards
if torch.cuda.is_available():
    self.model = torch.compile(self.model, mode="reduce-overhead")
```

`mode="reduce-overhead"` uses CUDA graphs which are ideal for fixed-shape or similar-shape forward passes (like the many short confidence forwards).

### Caveats

- First forward pass is slower (compilation overhead). Amortized over many questions.
- `torch.compile` can conflict with custom attention masks (used in `dropout_late_forward`). Test with `mode="default"` first if `reduce-overhead` causes issues.
- Requires PyTorch >= 2.0.
- Dynamic shapes (varying sequence lengths) can cause recompilation. The `dynamic=True` flag can help but reduces optimization potential.

### Expected savings

20–40% speedup on all forward passes after warmup. Most impactful for the many short (1-token) forwards in `_compute_verbconf_joint_probs`.

---

## Priority Order

| Priority | Optimization | Effort | Impact |
|----------|-------------|--------|--------|
| **P0** | 1. Share early_cache across methods | Medium | High — eliminates 3 full cache deep copies |
| **P0** | 3. Optional cache return in generate_one | Low | High — eliminates 1 full cache deep copy |
| **P1** | 2. Batch verbconf prefix forwards | Medium-High | Medium-High — eliminates ~9 deep copies + ~9 forwards |
| **P1** | 5. Optimize find_token_indices_from_end | Low | Medium — O(n^2) → O(n), ~48 calls per question |
| **P2** | 4. Pre-compute static token sequences | Low | Low — removes redundant tokenizer calls |
| **P2** | 6. torch.compile | Low | Medium — 20-40% on all forwards, but caveats |
