# CoT Trajectory Generation — Speed Optimization Notes

**Branch:** `i28/eval`
**Date:** 2026-03-25
**Scope:** `main.py:generate_trajectories()` and `llm.py:LLM.generate_one()` — generation path only (confidence computation excluded).

---

## Table of Contents

1. [Remove Wasteful `deepcopy` of KV Cache](#1-remove-wasteful-deepcopy-of-kv-cache)
2. [Batched Generation](#2-batched-generation)
3. [Early Stopping at `\boxed{}`](#3-early-stopping-at-boxed)
4. [Prefix Caching / Shared Few-Shot KV Cache](#4-prefix-caching--shared-few-shot-kv-cache)
5. [Redundant Tokenization for Answer Token Position](#5-redundant-tokenization-for-answer-token-position)
6. [Double String Processing of Prompt Template](#6-double-string-processing-of-prompt-template)
7. [Debug Pickle Includes KV Caches](#7-debug-pickle-includes-kv-caches)
8. [Prompt Template Rebuilt From Scratch Every Iteration](#8-prompt-template-rebuilt-from-scratch-every-iteration)
9. [Scores Tuple Kept in GPU Memory Unnecessarily](#9-scores-tuple-kept-in-gpu-memory-unnecessarily)
10. [`torch.inference_mode()` Scope Is Too Narrow](#10-torchinference_mode-scope-is-too-narrow)
11. [`torch.compile()` / Speculative Decoding](#11-torchcompile--speculative-decoding)

---

## 1. Remove Wasteful `deepcopy` of KV Cache

**File:** `llm.py:202`
**Severity:** High — immediate free win

### Problem

Every call to `generate_one()` executes:

```python
past_key_values=copy.deepcopy(outputs.past_key_values),  # llm.py:202
```

This deep-copies the **entire** KV cache — all transformer layers, all attention heads, all key/value GPU tensors — after every single generation. For a large model this is easily hundreds of megabytes of GPU memory being duplicated.

However, in `main.py:294-298`, the cache is immediately discarded:

```python
cache = traj.pop("past_key_values")   # main.py:294
# torch.save(cache, ...)              # main.py:298  ← commented out
```

The deep copy serves no purpose. The copied cache is never used and is garbage-collected shortly after.

### Impact

- **Memory:** Eliminates a temporary allocation of the full KV cache per sample (can be 200MB–1GB+ depending on model size and sequence length).
- **Time:** `deepcopy` on GPU tensors triggers CUDA memcpy operations. Removing it saves a non-trivial constant per sample.

### Proposed Fix

Option A — **Return `None` when cache is not needed** (cleanest):

```python
# llm.py — add a parameter
def generate_one(self, ..., return_cache: bool = False) -> GenerationResult:
    ...
    return GenerationResult(
        ...
        past_key_values=copy.deepcopy(outputs.past_key_values) if return_cache else None,
        ...
    )
```

Then in `main.py`, only pass `return_cache=True` when the cache is actually consumed downstream (e.g., confidence computation).

Option B — **Return the raw reference without copying** (simpler, but the caller must not mutate it):

```python
past_key_values=outputs.past_key_values,  # no deepcopy
```

This is safe as long as nothing modifies the cache in-place before the next `generate()` call. Since `generate()` creates a fresh cache each call, the old reference is safe to hold.

---

## 2. Batched Generation

**File:** `main.py:157`, `llm.py:91-206`
**Severity:** High — largest throughput win, medium-high implementation effort

### Problem

The generation loop in `main.py:157` processes one sample at a time:

```python
for i, entry in enumerate(dataloader):
    ...
    gen: GenerationResult = llm.generate_one(messages, ...)
```

Each call to `model.generate()` runs a full forward pass for a single sequence. On modern GPUs with large batch capacity, this severely underutilizes the hardware — the GPU spends much of its time waiting for memory transfers rather than computing.

### Impact

Batching `N` prompts together in a single `model.generate()` call can provide near-linear speedup up to the point where GPU memory or compute becomes the bottleneck. For short sequences (which are common in CoT eval where many questions produce <200 tokens), batch sizes of 4–16 are often feasible even on a single A100.

### Proposed Fix

1. **Left-pad prompts** to the same length (HuggingFace convention for decoder-only models):

```python
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
```

2. **Collect a batch of tokenized prompts**, pad them, and create an attention mask:

```python
batch_inputs = tokenizer(
    [prompt_text_1, prompt_text_2, ...],
    return_tensors="pt",
    padding=True,
).to(device)
```

3. **Call `model.generate()` once** for the batch:

```python
outputs = model.generate(
    **batch_inputs,
    max_new_tokens=max_new_tokens,
    ...
)
```

4. **Post-process each sequence** in the batch to extract `generated_text`, `prompt_end_position`, etc.

### Caveats

- Prompts with very different lengths waste padding tokens — consider grouping prompts by approximate length (bucket batching).
- `output_scores=True` with batching returns scores for all sequences, increasing memory usage multiplicatively.
- The `past_key_values` returned is for the entire batch; extracting per-sequence caches requires slicing.
- The current code's per-sample `StopStringCriteria` / early stopping logic needs adaptation for batched generation (HuggingFace handles per-sequence EOS, but custom stopping criteria need batch-aware logic).

---

## 3. Early Stopping at `\boxed{}`

**File:** `llm.py:32-54` (commented-out `_StopAfterBoxedAnswer` class)
**Severity:** High — low effort, high reward

### Problem

The model generates up to `max_new_tokens` (default 512) tokens regardless of whether it has already produced a complete answer. Once the model outputs `\boxed{...}` with balanced braces, all subsequent tokens are wasted compute — they're typically trailing whitespace, repetition, or EOS tokens.

For easy questions where the CoT is short (e.g., 100 tokens of reasoning + 20 tokens for the boxed answer), this wastes 392 tokens worth of autoregressive decoding.

### Impact

- On average, if answers complete at ~60% of `max_new_tokens`, early stopping saves ~40% of generation time per sample.
- Effect is multiplicative with sample count: 500 samples × 200 saved tokens each = 100,000 fewer forward passes.

### Proposed Fix

The code for this already exists but is commented out at `llm.py:32-54`. Uncomment and wire it in:

```python
# llm.py:167-169 — uncomment
stop_criteria = StoppingCriteriaList([
    _StopAfterBoxedAnswer(self.tokenizer, prompt_len),
])

# llm.py:182 — uncomment
outputs = self.model.generate(
    ...
    stopping_criteria=stop_criteria,
)
```

### How `_StopAfterBoxedAnswer` works

The class (already implemented at `llm.py:32-54`) works as follows:

1. After each generated token, it decodes only the newly generated tokens (from `prompt_len` onward).
2. It searches for `\boxed{` in the decoded text.
3. If found, it walks character-by-character counting `{` and `}` to check brace balance.
4. Once braces balance (i.e., the `\boxed{...}` is complete), it returns `True` to stop generation.

### Performance Concern

The stopping criteria decodes the full generated sequence at every step, which is O(n²) in the number of generated tokens. For very long generations this could become a bottleneck. Two mitigations:

- **Only check every K tokens** (e.g., every 10 tokens) by tracking a step counter. The `\boxed{` string is unlikely to span exactly a check boundary.
- **Use `StopStringCriteria`** from HuggingFace (available in recent transformers versions) which is implemented more efficiently at the C level. However, this only matches a fixed string and can't do brace-matching, so you'd stop at the first `}` after `\boxed{` — which fails for nested braces like `\boxed{\frac{1}{2}}`. For datasets where answers are always simple (single letter, number), `StopStringCriteria` suffices.

---

## 4. Prefix Caching / Shared Few-Shot KV Cache

**File:** `main.py:157-168`, `llm.py:91-206`, `prompts/load.py:33-109`
**Severity:** High — medium implementation effort, very high reward for few-shot

### Problem

In few-shot mode, every sample's prompt shares an **identical prefix**: the system message + all few-shot exemplars. Only the final user turn differs. Despite this, every call to `generate_one()`:

1. Re-formats the full message list via `load_messages()` (`prompts/load.py:33`)
2. Re-applies the chat template via `apply_chat_template()` (`llm.py:125-137`)
3. Re-tokenizes the entire prompt string (`llm.py:156`)
4. Re-runs the model's prefill forward pass over the entire prompt, including the shared prefix

For few-shot prompts, the shared prefix can be 1000–3000 tokens. The per-sample unique suffix (user question) is typically 50–200 tokens. So ~90% of the prefill computation is redundant.

### Impact

If the shared prefix is `P` tokens and the unique suffix is `S` tokens, the prefill cost per sample drops from O(P + S) to O(S) — a potential **5–15x reduction in prefill time** for few-shot evaluation.

### Proposed Fix

**Step 1 — Compute the shared prefix once before the loop:**

```python
# Build messages for a dummy entry, identify the split point
shared_messages = [system_msg] + few_shot_messages  # everything before the final user turn
shared_prompt_text = tokenizer.apply_chat_template(shared_messages, tokenize=False, ...)
shared_input_ids = tokenizer(shared_prompt_text, return_tensors="pt").to(device)

# Run a single prefill forward pass to get the KV cache for the shared prefix
with torch.inference_mode():
    shared_outputs = model(
        **shared_input_ids,
        use_cache=True,
    )
    shared_kv_cache = shared_outputs.past_key_values
```

**Step 2 — For each sample, clone the shared cache and only process the unique suffix:**

```python
for entry in dataloader:
    suffix_text = format_user_turn(entry)  # only the unique part
    suffix_ids = tokenizer(suffix_text, return_tensors="pt").to(device)

    # Clone the shared cache (shallow clone of tensors is fine since generate() doesn't mutate in-place)
    sample_cache = clone_kv_cache(shared_kv_cache)

    outputs = model.generate(
        input_ids=suffix_ids,
        past_key_values=sample_cache,
        ...
    )
```

**Step 3 — Handle the chat template split carefully:**

The tricky part is that `apply_chat_template` produces a single string for the entire conversation. You need to ensure that the token boundary between the shared prefix and the per-sample suffix is clean. The safest approach:

1. Apply the template to the full message list.
2. Apply the template to only the shared prefix messages.
3. The suffix is `full_template[len(shared_template):]`.
4. Verify that tokenizing them separately and concatenating produces the same IDs as tokenizing the full string (tokenizer boundary effects can cause mismatches — if so, add a token overlap margin).

### Caveats

- **Zero-shot mode:** The system prompt alone is the only shared part, which is relatively short. Prefix caching still helps but the gain is smaller.
- **Position IDs:** When passing `past_key_values` to `generate()`, you must also pass correct `position_ids` starting from `len(shared_prefix)` so that rotary embeddings are computed correctly.
- **Attention mask:** The attention mask must cover the full sequence (shared prefix + suffix), not just the suffix.

---

## 5. Redundant Tokenization for Answer Token Position

**File:** `main.py:191-198`
**Severity:** Low — easy fix

### Problem

To find the token position where the `\boxed{...}` answer starts, the code re-tokenizes a substring:

```python
generated_prefix = full_generated_text[len(assistant_prefill):parsed.answer_fullstring_start]
prefix_ids = tokenizer(generated_prefix, add_special_tokens=False)["input_ids"]
answer_token_start_position = gen.prompt_end_position + len(prefix_ids)
```

This calls the tokenizer's encode path on a potentially long string, which is unnecessary because `gen.generated_ids` already contains the full token ID sequence for the generation.

### Additional Risk

Re-tokenizing a substring can produce **different token IDs** than the original tokenization due to BPE merge boundaries. For example, if the original tokenization merged characters across the split point, re-tokenizing the substring will produce a different segmentation. This means `len(prefix_ids)` might not accurately reflect the true token boundary.

### Proposed Fix

Use the existing `generated_ids` and decode token-by-token to find the character→token mapping:

```python
# Build a character offset → token index mapping
generated_ids = gen.generated_ids  # already available
char_offset = len(assistant_prefill)  # offset into full_generated_text
target_char = parsed.answer_fullstring_start

cumulative_chars = char_offset
for tok_idx, tok_id in enumerate(generated_ids):
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
    cumulative_chars += len(tok_str)
    if cumulative_chars >= target_char:
        answer_token_start_position = gen.prompt_end_position + tok_idx
        break
```

Alternatively, use `tokenizer.decode(generated_ids[:k])` with binary search on `k` to find the token index corresponding to the target character offset — O(log n) decode calls instead of O(n).

---

## 6. Double String Processing of Prompt Template

**File:** `llm.py:125-156`
**Severity:** Low-Medium

### Problem

The prompt construction involves two passes:

1. `apply_chat_template(tokenize=False)` → produces a full prompt string (`llm.py:125-137`)
2. Regex post-processing on the string: Qwen `<think>` stripping (`llm.py:141-142`), GPT channel swap (`llm.py:146-150`)
3. `self.tokenizer(prompt_text, return_tensors="pt")` → tokenizes the string (`llm.py:156`)

The first pass generates a string, the second pass modifies it, and the third pass tokenizes it. If the post-processing could be done at the token level (or eliminated), `apply_chat_template(tokenize=True)` could be used directly, avoiding the string→tokens round-trip.

### Proposed Fix

**For Qwen `<think>` stripping:**

The regex `re.sub(r"<think>\s*</think>\s*", "", prompt_text)` removes empty think blocks that Qwen's chat template injects. Instead:

```python
# Tokenize directly
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, ...)

# Find and remove the <think></think> token subsequence
think_open_id = tokenizer.convert_tokens_to_ids("<think>")
think_close_id = tokenizer.convert_tokens_to_ids("</think>")
# Remove the subsequence [think_open_id, ..., think_close_id] if it contains only whitespace tokens
```

**For GPT channel swap:**

```python
# Find the token ID for "<|channel|>final" and "<|channel|>analysis"
# Replace the last occurrence at the token level
```

This is more complex and may not be worth the effort unless profiling shows tokenization is a real bottleneck. **Recommended only if samples number in the thousands.**

---

## 7. Debug Pickle Includes KV Caches

**File:** `main.py:189, 256-260`
**Severity:** Medium — causes disk I/O stalls and bloated files

### Problem

The debug cache saves the full `GenerationResult` object per sample:

```python
entries_to_save.append((gen, parsed, assistant_prefill, full_generated_text, messages))
```

`gen` is a `GenerationResult` which includes `past_key_values` (the full KV cache) and `scores` (per-token logits). When pickled at `main.py:259`:

```python
with open(cache_file, "wb") as f:
    pickle.dump(entries_to_save, f)
```

This serializes massive GPU tensors to disk. For a model with 32 layers and 2048-token sequences, a single KV cache can be hundreds of MB. The pickle file can easily reach tens of GB, causing long I/O stalls at the end of the run and eating disk space.

### Proposed Fix

Null out the heavy fields before appending to `entries_to_save`:

```python
import copy as _copy

gen_for_save = _copy.copy(gen)  # shallow copy of the dataclass
gen_for_save.past_key_values = None
gen_for_save.scores = None
entries_to_save.append((gen_for_save, parsed, assistant_prefill, full_generated_text, messages))
```

Or, more directly, only save the fields needed for debug replay:

```python
entries_to_save.append({
    "generated_text": gen.generated_text,
    "prompt_end_position": gen.prompt_end_position,
    "generated_end_position": gen.generated_end_position,
    "generated_ids": gen.generated_ids.cpu(),
    "parsed": parsed,
    "assistant_prefill": assistant_prefill,
    "full_generated_text": full_generated_text,
    "messages": messages,
})
```

---

## 8. Prompt Template Rebuilt From Scratch Every Iteration

**File:** `prompts/load.py:33-109`, called from `main.py:164`
**Severity:** Low-Medium

### Problem

`load_messages()` is called inside the generation loop for every sample:

```python
messages = load_messages(dataset_name, few_shot=(shot_mode=="few"), entry=entry, ...)
```

Inside `load_messages()` (`prompts/load.py:33`), the following work is repeated every iteration:

1. `load_prompt_from_registry(dataset)` — dictionary lookup + string operations
2. `THINKING_TOKENS` lookup and `KeyError` handling
3. For few-shot: `load_few_shot_prompt_from_registry(dataset)` — another dict lookup
4. For few-shot: iterating over all few-shot messages and doing string `.replace()` on `{thinking_token_open}` / `{thinking_token_close}` for every exemplar

The few-shot message formatting (step 4) is pure string manipulation that produces the **same result** every iteration since it depends only on `model_name`, `thinking`, and `prompt_type` — not on the per-sample `entry`.

### Proposed Fix

Pre-compute the shared message components once before the loop:

```python
# Before the loop
base_messages = load_messages_shared(dataset_name, few_shot, model_name, thinking, prompt_type)
# base_messages = [system_msg, few_shot_user_1, few_shot_assistant_1, ..., few_shot_user_N, few_shot_assistant_N]

for entry in dataloader:
    user_msg = format_user_message(entry)  # only the per-sample part
    messages = base_messages + [user_msg]
    if prompt_type == 1:
        messages.append({"role": "assistant", "content": assistant_start})
```

This avoids redundant string replacements and dictionary lookups per sample.

---

## 9. Scores Tuple Kept in GPU Memory Unnecessarily

**File:** `llm.py:204`, `main.py:166`
**Severity:** Medium — reduces peak GPU memory, prevents OOM on long runs

### Problem

When `output_scores=True` (triggered by `--confidence`), `model.generate()` returns a tuple of logit tensors — one per generated token, each of shape `(1, vocab_size)`. For a 32K vocab and 512 generated tokens, this is `512 × 32,000 × 4 bytes ≈ 62 MB` per sample, sitting on the GPU.

These scores are stored in `GenerationResult.scores` and passed through the trajectory dict. If confidence is computed later (not inline), all scores from all samples accumulate in memory.

Even when confidence IS computed inline, the scores from previous samples remain referenced in the `trajectories` list until the loop ends.

### Proposed Fix

**Option A — Move scores to CPU immediately after generation:**

```python
# llm.py:204
scores=tuple(s.cpu() for s in outputs.scores) if output_scores else None,
```

**Option B — Discard scores after confidence is computed (in main.py):**

```python
# After confidence computation in main.py
gen.scores = None  # free the reference
```

**Option C — Don't store scores in GenerationResult at all; pass them directly to confidence:**

Refactor so that confidence is computed inside `generate_one()` or immediately after, and scores are never stored.

---

## 10. `torch.inference_mode()` Scope Is Too Narrow

**File:** `llm.py:171`
**Severity:** Low

### Problem

The `torch.inference_mode()` context manager only wraps `model.generate()`:

```python
with torch.inference_mode():
    outputs = self.model.generate(...)
```

But tensor operations happen outside this scope — in `main.py`, tokenization creates tensors, and the answer token position computation involves tensor operations. Without inference mode, PyTorch tracks autograd metadata on these tensors unnecessarily.

### Proposed Fix

Wrap the entire generation loop in `torch.inference_mode()`:

```python
# main.py — wrap the outer loop
with torch.inference_mode():
    for i, entry in enumerate(dataloader):
        ...
```

Or apply it as a decorator on `generate_trajectories()`:

```python
@torch.inference_mode()
def generate_trajectories(...):
    ...
```

This is a minor optimization but is free to implement and follows best practice.

---

## 11. `torch.compile()` / Speculative Decoding

**File:** `llm.py:77-82` (model initialization)
**Severity:** Medium — one-time setup, ongoing benefit

### Problem

The model runs in eager execution mode. For large-scale evaluation runs (hundreds or thousands of samples), the per-token overhead of eager PyTorch adds up.

### `torch.compile()`

PyTorch 2.x's `torch.compile()` traces the model and generates optimized CUDA kernels (via Triton). This eliminates Python overhead and enables kernel fusion.

```python
self.model = AutoModelForCausalLM.from_pretrained(model_name, ...)
self.model = torch.compile(self.model, mode="reduce-overhead")
```

- **First call:** slow (compilation overhead, 30s–5min depending on model size).
- **Subsequent calls:** 10–30% faster per token due to fused kernels and reduced overhead.
- **Caveat:** `torch.compile` + `model.generate()` has known compatibility issues in some transformers versions. Test thoroughly. Using `torch.compile` on just the model's `forward()` method (not the full `generate()` loop) is safer.

### Speculative Decoding

Use a smaller "draft" model to propose multiple tokens, then verify them in parallel with the large model. HuggingFace supports this natively:

```python
from transformers import AutoModelForCausalLM

# Main model (e.g., Qwen-72B)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-72B-Instruct", ...)

# Draft model (e.g., Qwen-0.5B, same tokenizer family)
assistant_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", ...)

outputs = model.generate(
    **inputs,
    assistant_model=assistant_model,
    max_new_tokens=512,
)
```

- **Speedup:** Typically 2–3x for greedy decoding when the draft model's acceptance rate is high.
- **Caveat:** The draft model must share the same tokenizer. Memory increases since two models are loaded.

---

## Summary — Priority Matrix

| # | Optimization | Effort | Impact | Risk |
|---|-------------|--------|--------|------|
| 1 | Remove `deepcopy` of KV cache | Trivial (1 line) | Medium | None |
| 3 | Early stopping at `\boxed{}` | Low (uncomment) | High | Low (edge cases with nested braces) |
| 7 | Strip KV cache from debug pickle | Low (3 lines) | Medium | None |
| 10 | Widen `inference_mode()` scope | Trivial | Low | None |
| 9 | Move scores to CPU / discard early | Low | Medium | None |
| 8 | Pre-compute shared prompt components | Low | Low-Medium | None |
| 5 | Use `generated_ids` for answer position | Low | Low | Low (decode boundary) |
| 6 | Token-level template post-processing | Medium | Low-Medium | Medium (tokenizer edge cases) |
| 4 | Prefix caching (shared few-shot KV) | Medium | Very High (few-shot) | Medium (position IDs, attention mask) |
| 2 | Batched generation | Medium-High | Very High | Medium (padding, per-seq stopping) |
| 11 | `torch.compile()` / speculative decoding | Low-Medium | Medium-High | Medium (compatibility) |

**Recommended implementation order:** Start with the trivial/low-effort wins (1, 3, 7, 10, 9) to get immediate gains with no risk, then tackle the high-impact structural changes (4, 2, 3) for major throughput improvements.
