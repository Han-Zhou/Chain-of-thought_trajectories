# Memory Optimizations (no speed regression)

Analysis of memory usage across the pipeline with concrete fixes.
All changes are speed-neutral or faster — none will regress throughput.

---

## 1. `output_scores=True` is dead code (~1-3 GB GPU per sample)

**Location:** `main.py:165`

```python
gen = llm.generate_one(messages, max_new_tokens=max_new_tokens,
                       output_scores=bool(confidence), ...)
```

**Problem:**
When `--confidence` is enabled, `output_scores=True` causes `model.generate()` to
store the full logit tensor at every decoding step. For a 5,000-token generation with
vocab_size ~152K (Qwen), that is:

    5000 steps × 152064 vocab × 4 bytes/float ≈ 3.04 GB on GPU

These scores are stored in `gen.scores` (a tuple of `[1, vocab_size]` float tensors).

**Why it's dead code:**
`gen.scores` is never consumed anywhere downstream:
- `confidence.py` runs its own forward passes to get logits — it never reads `gen.scores`.
- The trajectory dict (`main.py:236-252`) does not include scores.
- The only reader would be the debug pickle, but even that path doesn't use scores.

Grep confirms no downstream usage:
- `structures.py:17`: field definition
- `llm.py:241`: assignment from `outputs.scores`
- `main.py:165`: the only call site that sets `output_scores`

**Fix:**
```python
# main.py:165
gen = llm.generate_one(messages, max_new_tokens=max_new_tokens,
                       output_scores=False, ...)
```

**Savings:** ~1-3 GB GPU per sample. Also slightly faster generation (HF skips
collecting logits when `output_scores=False`).

---

## 2. Unnecessary `copy.deepcopy(outputs.past_key_values)` in `llm.py:239` (~100-300 MB per sample)

**Location:** `llm.py:239`

```python
return GenerationResult(
    ...
    past_key_values=copy.deepcopy(outputs.past_key_values),
    ...
)
```

**Problem:**
`model.generate()` creates a fresh `DynamicCache` internally when `use_cache=True`
is passed without an explicit `past_key_values` argument. The returned
`outputs.past_key_values` is a reference to that internally-created cache. After
`generate()` returns, the model does not retain a reference to this cache — it is
local to the generation loop. The `outputs` local variable is garbage-collected when
`generate_one()` returns (no other reference holds it).

The `deepcopy` allocates a full duplicate of the KV cache on GPU for no reason. For
Qwen-27B with a 5,000-token sequence:

    28 layers × 2 (K+V) × 1 × 32 heads × 5000 seq × 128 head_dim × 2 bytes (bf16)
    ≈ 229 MB (duplicated pointlessly)

**Fix:**
```python
past_key_values=outputs.past_key_values,  # take ownership directly
```

**Savings:** ~100-300 MB GPU per sample + eliminates the deepcopy latency (~0.1-0.5s).

---

## 3. KV cache accumulates in `trajectories` list until write time (~100-300 MB × N)

**Location:** `main.py:251` (stored), `main.py:317` (freed)

```python
# Line 251: stored in trajectory dict
trajectories.append({
    ...
    "past_key_values": gen.past_key_values,
})

# Line 317: only freed during the write loop at the very end
cache = traj.pop("past_key_values")
```

**Problem:**
Every sample's KV cache is held in the `trajectories` list until the entire generation
loop finishes and the write loop begins. For N samples, that is N × 100-300 MB of GPU
memory pinned simultaneously. For 100 samples at 200 MB each = **20 GB** held uselessly
on GPU.

The cache is needed for:
1. Confidence scoring (`main.py:210`): `gen_cache=gen.past_key_values`
2. Nothing else — it is popped and discarded during write (line 317), the
   `torch.save` for cache is commented out (line 321).

**Fix:**
Free immediately after confidence (or immediately if no confidence):

```python
# After confidence scoring (or after the if/else block if no confidence):
if hasattr(gen, 'past_key_values'):
    del gen.past_key_values
past_key_values_ref = None  # store None in trajectory

trajectories.append({
    ...
    "past_key_values": None,  # already freed
})
```

Or more minimally, just delete from gen right after the confidence block:

```python
# After line 218 (end of confidence if/else)
kv_cache = gen.past_key_values
gen.past_key_values = None  # release reference from GenerationResult

# ... build trajectory dict ...
trajectories.append({
    ...
    "past_key_values": None,
})
del kv_cache
torch.cuda.empty_cache()
```

**Savings:** Prevents accumulation of N caches. Peak usage drops from N×200 MB to
1×200 MB (only current sample's cache).

---

## 4. `early_cache` is recomputed 4x identically (saves 3 full-size deepcopies)

**Location:** `confidence.py:434` inside `dropout_forward()`, called from:
- `dropout_answerlogits()` (1 call)
- `dropout_indirectlogits()` (2 calls: True/False suffix, Is answer correct suffix)
- `dropout_verbalconf()` (1 call)

```python
# confidence.py:434 — executed 4 times per sample
early_cache = copy.deepcopy(gen_cache)
crop_cache(early_cache, early_late_split)
```

**Problem:**
`early_late_split` depends on `parsed_output.answer_fullstring_start` and the token
sequence, which are identical across all 4 calls (same `parsed_output`, same
`generated_text`, same `messages`). So all 4 `early_cache` objects are byte-for-byte
identical. Three of the four deepcopy+crop operations are pure waste.

Each deepcopy copies the FULL gen_cache (~200 MB) before cropping. So:

    3 redundant copies × 200 MB = 600 MB wasted per sample

**Fix:**
Compute `early_cache` once in `compute_all_confidence_scores()` and pass it as a
parameter to `dropout_forward()`:

```python
def compute_all_confidence_scores(...):
    # Compute early_cache once
    base_tokens = _tokenize_for_confidence(llm, messages, base_content)

    # Determine early_late_split (same logic as dropout_forward lines 409-414)
    full_text = (assistant_prefill + generated_text).strip()
    fullstring_text = full_text[parsed_output.answer_fullstring_start:]
    tokens = base_tokens  # for no-suffix case
    fs_start, _ = find_token_indices_from_end(llm.tokenizer, tokens[0], fullstring_text)
    early_late_split = fs_start - 1

    early_cache = copy.deepcopy(gen_cache)
    crop_cache(early_cache, early_late_split)

    # Pass early_cache and early_late_split to all calls
    dropout_answerlogits(..., early_cache=early_cache, early_late_split=early_late_split)
    dropout_indirectlogits(..., early_cache=early_cache, early_late_split=early_late_split)
    dropout_verbalconf(..., early_cache=early_cache, early_late_split=early_late_split)
```

Then modify `dropout_forward()` to accept an optional `early_cache` parameter and skip
the deepcopy+crop when provided:

```python
def dropout_forward(..., early_cache=None, early_late_split=None):
    ...
    if early_cache is not None:
        # Reuse pre-computed early cache — still need copies for vanilla/dropout
        # since model.forward() mutates the cache in-place
        pass
    else:
        early_cache = copy.deepcopy(gen_cache)
        crop_cache(early_cache, early_late_split)
```

**Savings:** 3 full deepcopy(gen_cache) operations eliminated (~600 MB + ~0.3-1.5s
deepcopy time per sample).

**Caveat:** Each `dropout_forward()` call still needs its own copies of `early_cache`
for the vanilla and dropout forward passes (since `model.forward()` mutates the cache
in-place). So the inner copies remain — but the 3 redundant outer copies are gone.

---

## 5. One `early_cache` copy per `dropout_forward()` is eliminable (saves 4 copies total)

**Location:** `confidence.py:495-510`

```python
# Vanilla forward (no-suffix path, lines 496-503):
_early_cache_copy = copy.deepcopy(early_cache)       # copy A
vanilla_output = llm.model.forward(
    input_ids=late_tokens, past_key_values=_early_cache_copy, ...)

# Dropout forward (lines 508-522):
_early_cache_copy = copy.deepcopy(early_cache)       # copy B
dropout_output = dropout_late_forward(
    ..., _early_cache_copy, late_tokens, ...)
```

**Problem:**
Both vanilla and dropout need their own copy because `model.forward()` mutates the
cache in-place. Two copies are made; one could be avoided by passing the original
`early_cache` to the last consumer.

**Fix:**
Run vanilla with a deepcopy, then pass the original `early_cache` directly to dropout
(which is the last consumer and can consume it destructively):

```python
# Vanilla — needs a copy (early_cache still needed after)
_early_cache_copy = copy.deepcopy(early_cache)
vanilla_output = llm.model.forward(
    input_ids=late_tokens, past_key_values=_early_cache_copy, ...)

# Dropout — last consumer, pass original directly (no copy)
dropout_output = dropout_late_forward(
    ..., early_cache, late_tokens, ...)  # early_cache consumed here
```

For the suffix + overlap path (lines 444-482), vanilla uses `vanilla_cache` from
`gen_cache` and doesn't touch `early_cache`, so the same pattern applies: dropout
can consume `early_cache` directly.

**Savings:** 1 deepcopy(early_cache) per `dropout_forward()` call × 4 calls = 4 copies
eliminated. At ~100-150 MB per cropped cache, that's ~400-600 MB saved per sample.

**Combined with #4:** Total savings from cache deduplication = 3 full gen_cache copies +
4 cropped early_cache copies = 7 deepcopies eliminated per sample.

---

## 6. Debug pickle stores unnecessary data

**Location:** `main.py:188, 280-283`

```python
# Line 188: append full GenerationResult
entries_to_save.append((gen, parsed, assistant_prefill, full_generated_text, messages))

# Lines 280-283: pickle everything
pickle.dump(entries_to_save, f)
```

**Problem:**
`entries_to_save` accumulates the full `GenerationResult` for every sample, including:
- `gen.past_key_values` — the full KV cache (100-300 MB per sample, on GPU → moved to CPU during pickle)
- `gen.scores` — full logit tensors (1-3 GB per sample if output_scores=True)
- `gen.generated_ids` — token ID tensor (small)
- `gen.prompt_tail_ids` — small tensor

For 100 samples at 20K tokens each with output_scores, this pickle file can reach
**tens of GB on disk** and requires equivalent RAM to serialize.

**What the debug loader actually needs** (line 161):
```python
gen, parsed, assistant_prefill, full_generated_text, messages = cached_entries[i]
```

Downstream usage of `gen`:
- `gen.past_key_values` — YES, needed for confidence cache reuse
- `gen.generated_text` — YES, used directly
- `gen.prompt_end_position` — YES, used in trajectory
- `gen.generated_end_position` — YES, used in trajectory
- `gen.scores` — **NO**, never read
- `gen.generated_ids` — **NO**, never read
- `gen.prompt_tail_ids` — **NO**, never read
- `gen.prompt_text` — **NO**, never read

**Fix:**
Strip unnecessary fields before appending:

```python
# Before appending to entries_to_save
gen_for_cache = GenerationResult(
    prompt_text="",                        # not needed
    generated_text=gen.generated_text,
    prompt_end_position=gen.prompt_end_position,
    generated_end_position=gen.generated_end_position,
    past_key_values=gen.past_key_values,   # needed for confidence
    scores=None,                           # not needed — saves 1-3 GB
    generated_ids=None,                    # not needed
    prompt_tail_ids=torch.tensor([]),      # not needed
)
entries_to_save.append((gen_for_cache, parsed, assistant_prefill, full_generated_text, messages))
```

Or more simply, just null out the unneeded fields on the existing object before pickle:

```python
# Before pickle.dump:
for entry in entries_to_save:
    gen = entry[0]
    gen.scores = None
    gen.generated_ids = None
    gen.prompt_text = ""
```

**Savings:** Eliminates 1-3 GB per sample from the pickle (scores). Also reduces
in-memory accumulation during the generation loop.

---

## 7. `copy.deepcopy(messages)` in `_tokenize_for_confidence()` is overkill

**Location:** `confidence.py:354`

```python
def _tokenize_for_confidence(llm, messages, full_assistant_content):
    conf_messages = copy.deepcopy(messages)  # <-- here
    if conf_messages[-1]["role"] == "assistant":
        conf_messages[-1]["content"] = full_assistant_content
    else:
        conf_messages.append({"role": "assistant", "content": full_assistant_content})
    ...
```

**Problem:**
`messages` is a `list[dict[str, str]]` — a flat structure of immutable strings.
`copy.deepcopy` is designed for complex object graphs with cycles and shared
references. For a list of string-keyed, string-valued dicts, it's unnecessary overhead.

The only mutation is replacing `content` on the last dict. A shallow copy of the list
+ a shallow copy of the modified dict is sufficient:

**Fix:**
```python
conf_messages = [dict(m) for m in messages]
```

This creates a new list with new dict objects (so mutating `conf_messages[-1]["content"]`
doesn't affect the original), but shares the string values (which are immutable in Python
anyway).

**Called:** 5+ times per sample (once for `base_tokens`, once per suffix variant).

**Savings:** Marginal memory (~KB), but removes deepcopy's internal bookkeeping overhead
(~0.01-0.1ms per call). Mainly a cleanliness improvement.

---

## Summary Table

| # | Fix | Memory Saved (per sample) | Speed Impact | Complexity |
|---|-----|--------------------------|--------------|------------|
| 1 | `output_scores=False` | ~1-3 GB GPU | Faster | 1 line |
| 2 | Remove deepcopy in `llm.py:239` | ~100-300 MB GPU | Faster | 1 line |
| 3 | Free KV cache after confidence | ~100-300 MB × N (cumulative) | Neutral | ~5 lines |
| 4 | Hoist `early_cache` (1 instead of 4) | ~600 MB GPU | Faster | ~20 lines |
| 5 | Pass `early_cache` directly to dropout | ~400-600 MB GPU | Faster | ~5 lines |
| 6 | Strip scores from debug pickle | ~1-3 GB × N (RAM + disk) | Faster pickle | ~3 lines |
| 7 | Shallow copy messages | Marginal | Faster | 1 line |

**Total peak memory reduction per sample (with confidence):**
- Items 1+2+4+5: ~2.2-4.2 GB GPU
- Item 3: prevents N × 200 MB accumulation
- Item 6: prevents N × 1-3 GB RAM accumulation in debug pickle

**For a 10-sample run with Qwen-27B at max_new_tokens=5000 with --confidence:**
- Before: peak ~65 GB (model 50GB + 10 caches + scores + confidence copies)
- After: peak ~52-54 GB (model 50GB + 1 cache + confidence copies, no scores)
