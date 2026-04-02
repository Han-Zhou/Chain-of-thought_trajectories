# Cache Copy Redundancy in `confidence.py`

## Call Graph

`compute_all_confidence_scores` (L52) is the orchestrator. It dispatches to three methods, each of which calls `dropout_forward`:

| Method | `dropout_forward` calls | `suffix_text` |
|---|---|---|
| `dropout_answerlogits` | 1 | `""` (none) |
| `dropout_indirectlogits` | 2 | `"\nTrue/False:"` and `"\nIs the answer X correct?"` |
| `dropout_verbalconf` | 1 | `"\nPlease respond with a score..."` |

**Total: 4 `dropout_forward` invocations per question**, all sharing the same `gen_cache`.

---

## Every `copy.deepcopy` on a KV Cache

### Inside `dropout_forward` (called 4x)

| Line | What's copied | When it triggers | Purpose |
|---|---|---|---|
| **L428** | `copy.deepcopy(gen_cache)` | Always (when gen_cache given) | Build `early_cache` by copy-then-crop |
| **L448** | `copy.deepcopy(gen_cache)` | Suffix path, overlap > split | Build `vanilla_cache` for overlap reuse |
| **L468** | `copy.deepcopy(early_cache)` | Suffix path, overlap <= split (fallback) | Vanilla forward needs its own cache |
| **L486** | `copy.deepcopy(early_cache)` | No-suffix path | Vanilla forward needs its own cache |
| **L497** | `copy.deepcopy(early_cache)` | Always (when dropout runs) | Dropout forward needs its own cache |

### Inside `_compute_verbconf_joint_probs` (called from `dropout_verbalconf`)

| Line | What's copied | When it triggers | Purpose |
|---|---|---|---|
| **L271** | `copy.deepcopy(parent_kv)` | Per unique token prefix, per depth level | Condition on different prefix tokens for multi-token numbers |

---

## Deep Copy Tally Per `dropout_forward` Call

**Call 1 -- `answerlogits` (no suffix):**
- L428: 1x `deepcopy(gen_cache)` -> crop -> `early_cache`
- L486: 1x `deepcopy(early_cache)` -> vanilla forward
- L497: 1x `deepcopy(early_cache)` -> dropout forward
- **Subtotal: 1 full-size + 2 cropped-size**

**Calls 2 & 3 -- `indirectlogits` (suffix, overlap typically > split):**
- L428: 1x `deepcopy(gen_cache)` -> crop -> `early_cache`
- L448: 1x `deepcopy(gen_cache)` -> crop -> `vanilla_cache`
- L497: 1x `deepcopy(early_cache)` -> dropout forward
- **Subtotal per call: 2 full-size + 1 cropped-size**
- **Subtotal both: 4 full-size + 2 cropped-size**

**Call 4 -- `verbconf` (suffix, overlap typically > split):**
- L428: 1x `deepcopy(gen_cache)` -> crop -> `early_cache`
- L448: 1x `deepcopy(gen_cache)` -> crop -> `vanilla_cache`
- L497: 1x `deepcopy(early_cache)` -> dropout forward
- **Subtotal: 2 full-size + 1 cropped-size**

### Grand Total Across All 4 Calls

| Cache size | Count | Description |
|---|---|---|
| **Full `gen_cache`** | **7** | Contains the entire generation sequence -- longest, most expensive |
| **Cropped `early_cache`** | **5** | Cropped to `early_late_split` -- smaller but still large |
| **Total deep copies** | **12** | Plus additional copies inside `_compute_verbconf_joint_probs` |

---

## Why This Is Redundant

### Redundancy 1: `early_cache` is identical across all 4 calls

`early_late_split` depends only on `base_tokens` and `fullstring_text` -- both are constant
across all calls. The suffix tokens are appended *after* the fullstring region, so
`find_token_indices_from_end` (L407) resolves `fs_start` to the same position regardless
of suffix.

This means L428 (`deepcopy(gen_cache)` + `crop_cache`) produces **the exact same
`early_cache`** 4 times. Three of those are pure waste. This is the costliest redundancy
because `gen_cache` is the largest object.

### Redundancy 2: Last consumer copies unnecessarily

Within each `dropout_forward` call, `early_cache` is consumed by two things: the vanilla
forward and the dropout forward. Both do `copy.deepcopy(early_cache)` because
`model.forward()` mutates the cache in-place (extends it). But whichever runs **last**
could consume `early_cache` directly -- nothing reads it afterward. That eliminates 1
cropped copy per call (4 total).

### Redundancy 3: Suffix path double-copies `gen_cache`

In calls 2-4 (suffix path), both L428 and L448 deep-copy `gen_cache`:
- L428 copies it to build `early_cache` (crop to `early_late_split`)
- L448 copies it again to build `vanilla_cache` (crop to `overlap_len`)

If `early_cache` is hoisted (fix #1), L428 disappears, but L448 remains -- 3 full copies
still happen for the vanilla overlap path. These could potentially be reduced further if
`overlap_len` could be precomputed, but each suffix has a different `overlap_len` so the
savings are more limited.

### Redundancy 4: `_compute_verbconf_joint_probs` prefix copies

For numbers 0-100, multi-token numbers (e.g., "42" -> `["4", "2"]`) require conditioning
forward passes. At each depth level, every unique prefix triggers a
`copy.deepcopy(parent_kv)`. The count depends on tokenization but is typically 10-20
copies of caches that are ~full generation length + suffix. This is a secondary concern
compared to the main path but adds up.

---

## Summary of Potential Savings

| Fix | Copies eliminated | Relative cost |
|---|---|---|
| Hoist `early_cache` out of `dropout_forward` | 3 full `gen_cache` copies | Highest -- removes the 3 costliest redundant copies |
| Last consumer skips deepcopy | 4 cropped `early_cache` copies | Medium -- saves ~30-40% of an early_cache per call |
| Precompute suffix vanilla caches or slice instead of copy | Up to 3 full `gen_cache` copies | Medium -- but harder since each suffix differs |
| Optimize verbconf prefix tree | 10-20 medium-size copies | Lower -- harder to eliminate structurally |

The **minimum-effort, maximum-impact** fix is hoisting `early_cache`: compute it once in
`compute_all_confidence_scores`, pass it into each method, and have `dropout_forward`
accept it as a parameter instead of rebuilding it from `gen_cache` every time. This turns
7 full-size deep copies into 4 (or fewer with further work on the suffix path).
