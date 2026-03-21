# i11: Dropout-Based Confidence Metrics (KV-Cache Adaptation)

## Overview

This update adapts the dropout-based confidence extraction methods from `reasoning-testing-clean/dropout.py` into `confidence.py`, replacing the old `\boxed{answer}` format with the new `"Final Answer: A"` MCQ format. The key architectural change is **reusing the KV cache from `GenerationResult`** instead of re-tokenizing and re-computing it from scratch.

## Background

The original `reasoning-testing-clean/dropout.py` (used by `02-compute-confidences.py`) worked by:
1. Reconstructing the full prompt (system + user + assistant CoT + answer) as chat messages
2. Tokenizing everything via `apply_chat_template`
3. Running a fresh early forward pass to build the KV cache
4. Splitting into early/late tokens and running vanilla + dropout forward passes

This was wasteful because we already have a KV cache from the generation step.

## New Architecture

`confidence.py` now takes `GenerationResult` (which carries `past_key_values`, `generated_ids`, `prompt_end_position`) and `ParsedOutput` (which carries `cot_steps`, `final_answer`, `answer_fullstring_start`, `answer_start`).

### Flow

```
GenerationResult.past_key_values
         │
         ▼
    crop(early_late_split)     ◄── one token before "Final Answer"
         │
    ┌────┴────┐
    ▼         ▼
 vanilla    dropout (batched, with per-sample attention masks)
```

### Early/Late Split

The sequence is split one token before `"Final Answer"` in the generated tokens:
- **Early cache**: prompt tokens + all generated tokens up to (but not including) "Final Answer"
- **Late tokens**: from that split point to end of generation + optional suffix tokens

This mirrors the old code's `boxed_start - 1` split.

## Confidence Methods

### 1. `dropout_answerlogits`
- **No suffix** — uses the generation as-is
- Extracts logits at the answer letter token position(s)
- Returns per-token probabilities and top-10 distributions for both vanilla and dropout
- Logit indexing: position `t-1` predicts token `t`, so answer logits are at `[ans_start-1 : ans_end-1]`

### 2. `dropout_indirectlogits`
- **ptrue1**: appends `"\nTrue/False:"` suffix → reads P(True) vs P(False) at last position
- **ptrue2**: appends `"\nIs the answer <X> correct?"` suffix → reads P(Yes) vs P(No) at last position
- Positive/negative token sets match the old `ANSWER_TOKENS` dictionary

### 3. `dropout_verbalconf`
- Appends confidence elicitation prompt as suffix
- Reads logits over tokens for integers 0–100 at last position
- Returns weighted mean confidence (score / 100)

## Dropout Mechanism

`dropout_late_forward` implements per-sample attention masking:

1. Creates a base causal mask for late → [early, late] attention
2. For each CoT step (skipping step 1, which is always kept):
   - Finds its token range in the early portion using `find_token_indices_from_end`
   - For each dropout sample, randomly masks or keeps attention from answer-region rows to step columns
3. Uses `reorder_cache` to replicate the early cache for the batch
4. Runs a single batched forward pass with the 4D attention mask

### Attention Mask Layout

```
              ┌─── early tokens ───┬── late tokens ──┐
              │  prompt  │  steps  │ FinalAns│ suffix │
late token 0  │    0     │    0    │    0    │ -10000 │  (causal)
late token 1  │    0     │  0/-1e4 │    0    │ -10000 │  (dropout on steps)
...           │          │         │         │        │
```

Values: `0` = attend, `-10000` = mask.

### Step Position Mapping

Steps are found in `generated_ids[:early_late_split_gen]` using backward search. Their absolute positions in the mask = `prompt_end + position_in_generated_ids`.

## `use_fullstring` Flag

- `False` (default): only the answer letter token(s) have their step-attention modified by dropout
- `True`: all tokens from "Final Answer" onwards (including any suffix) are modified — stronger perturbation

## Important Limitations

- **4D attention masks require `sdpa` attention**. Flash Attention 2 does not support 4D masks. The `LLM` class uses `flash_attention_2` for GPT/LLaMA models — these will need to switch to `sdpa` for dropout to work.
- `DynamicCache.crop()` and `reorder_cache()` must be supported by the cache implementation (standard `DynamicCache` and `Qwen3_5DynamicCache` should both work).

## API

```python
from confidence import compute_all_confidence_scores
from utils.structures import AllConfidenceData

result: AllConfidenceData = compute_all_confidence_scores(
    llm,
    generation_result,   # from llm.generate_one()
    parsed_output,       # from parsing.parse_output()
    nb_dropout_samples=10,
    use_fullstring=False,
)

# Access vanilla / dropout scores
result.vanilla_confidences.answer_probabilities      # list[float]
result.dropout_confidences.indirect_ptrue1_probabilities  # list[float] per sample
```
