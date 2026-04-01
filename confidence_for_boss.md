# Confidence Estimation Module: Technical Summary

## Overview

`confidence.py` implements a comprehensive confidence estimation framework for evaluating Large Language Model (LLM) outputs. It measures how confident a model is in its generated answers using three complementary approaches, each computed under both "vanilla" (full context) and "dropout" (perturbed context) conditions.

The main entry point is `compute_all_confidence_scores()` (line 54), which orchestrates all three methods and returns an `AllConfidenceData` object (defined in `utils/structures.py:45`).

---

## Core Confidence Methods

### 1. Answer Logits Confidence (`dropout_answerlogits`, line 200+)

Measures the probability the model assigns to its own answer tokens during generation.

**How it works:**
- Re-runs a forward pass on the generated answer text
- For each token in the answer, extracts the probability the model assigned to that specific token
- Also computes entropy at each position (higher entropy = more uncertainty)
- Returns both vanilla (full context) and dropout (perturbed) versions

**Code reference:**
- Line 89-92: Called from `compute_all_confidence_scores()` with `base_tokens` and `precomputed_early_cache`
- Line 200-250: Core implementation that tokenizes the answer and runs forward passes
- Returns: `vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy`

**Output:**
- Per-token probabilities: `[{" 42": 0.85}, {" is": 0.92}, ...]`
- Per-token entropy values

### 2. Indirect Logits / P(True) Probing (`dropout_indirectlogits`, line 300+)

Asks the model follow-up questions to probe its confidence indirectly.

**Two probing strategies:**
1. **ptrue1**: Appends `"\nTrue/False:"` after the answer and measures P(True) vs P(False)
2. **ptrue2**: Appends `"\nIs the answer {X} correct?"` and measures P(Yes) vs P(No)

**Code reference:**
- Line 96-101: Called from `compute_all_confidence_scores()` 
- Line 300-400: Implements both ptrue1 and ptrue2 suffixes
- Line 36-42: `ANSWER_TOKENS` dict defines token variants (e.g., `' Yes': [' Yes', ' yes', ' YES', ' Yeah', ...]`)
- Line 45-51: `get_token_ids()` converts token strings to model-specific token IDs

**How it works:**
- Runs forward pass with the suffix appended
- Extracts logits for positive tokens (True, Yes, yeah, etc.) and negative tokens (False, No, nope, etc.)
- Normalizes via softmax to get calibrated probabilities

**Output:**
- `{"True": 0.87, "False": 0.13}` for ptrue1
- `{"Yes": 0.91, "No": 0.09}` for ptrue2

### 3. Verbalized Confidence (`dropout_verbalconf`, line 450+)

Explicitly asks the model to rate its confidence on a 0-100 scale.

**Prompt suffix:**
```
Please respond with a score from 0 to 100 in <confidence> </confidence> tags.
How confident are you in your previous answer?
<confidence>
```

**Code reference:**
- Line 103-112: Called from `compute_all_confidence_scores()`
- Line 450-550: Core implementation that handles multi-token numbers (e.g., "42" = tokens ["4", "2"])
- Line 700+: `_compute_verbconf_joint_probs()` computes joint probabilities for all 0-100 scores
- Line 825-831: Returns normalized probabilities for positive/negative tokens

**How it works:**
- Computes joint probabilities for all numbers 0-100 (handling multi-token numbers like "42" = ["4", "2"])
- Uses a tree-based approach with KV cache cloning for efficient multi-token probability computation
- Returns weighted average: `Σ (score/100) × P(score)`

**Output:**
- Expected confidence score (0.0 to 1.0)
- Full distribution over 0-100
- Top predicted score and its probability

---

## Dropout Perturbation Framework

The key innovation is measuring confidence under **reasoning step dropout** — randomly masking portions of the chain-of-thought (CoT) to see how robust the model's confidence is.

### Coin-Flip Dropout (Default)

Each reasoning step (except the first) is independently kept with probability 0.5.

**Code reference** (`dropout_late_forward`, line 682-761):
```python
# Line 723-725: Coin-flip selection
is_step_selected = (np.random.random((nb_dropout_samples, len(steps))) <= threshold)
```

```
Sample 1: Keep steps [1, 3, 5], mask steps [2, 4]
Sample 2: Keep steps [2, 4], mask steps [1, 3, 5]
...
```

### Jackknife Dropout (Experimental)

Keeps exactly `ceil(log(k))` steps where k = total steps - 1. More structured than coin-flip.

**Code reference** (line 714-722):
```python
# Jackknife: keep ceil(log(k)) steps, mask the rest
k = len(steps)
nb_keep = math.ceil(math.log(k)) if k >= 1 else 0
nb_mask = k - nb_keep
is_step_selected = np.ones((nb_dropout_samples, k), dtype=bool)
for i in range(nb_dropout_samples):
    masked_indices = np.random.choice(k, size=nb_mask, replace=False)
    is_step_selected[i, masked_indices] = False
```

### Implementation Details

The dropout is implemented via **attention masking**, not by removing tokens:

1. **Early/Late Split** (line 540-541): Tokens are split at the answer region boundary
   ```python
   early_tokens = tokens[:, :early_late_split].to(device)
   late_tokens = tokens[:, early_late_split:].to(device)
   ```
   - Early tokens: prompt + reasoning steps (cached)
   - Late tokens: answer + any suffix

2. **Attention Mask Modification** (line 705-710, 738-740): For masked steps, the attention mask is set to `-10000` (effectively zero after softmax), preventing answer tokens from attending to those reasoning steps
   ```python
   # Build base causal mask
   late_mask = torch.cat([
       torch.ones(nb_late, nb_early_tokens),
       torch.ones(nb_late, nb_late).tril(),
   ], dim=1)
   late_mask[late_mask == 0] = -10000.
   late_mask[late_mask == 1] = 0.
   
   # Apply step masking (line 738-740)
   late_mask[i, 0, modify_start_late - 1:modify_end_late - 1, step_start:step_end] = \
       0. if is_step_selected[i, step_id] else -10000.
   ```

3. **Batched Forward Pass** (line 749-756): All dropout samples run in a single batched forward pass for efficiency
   ```python
   late_tokens_batch = late_tokens.expand(nb_dropout_samples, -1)
   early_cache.reorder_cache(torch.tensor([0] * nb_dropout_samples))
   
   with torch.no_grad():
       late_output = llm.model.forward(
           input_ids=late_tokens_batch,
           attention_mask=late_mask,
           past_key_values=early_cache,
       )
   ```

---

## KV Cache Optimization

The module heavily optimizes computation by reusing the KV cache from generation. This is critical for performance since re-computing attention over the full prompt + CoT for each confidence method would be prohibitively expensive.

### Cache Reuse Strategy

**Code reference** (line 79-86 in `compute_all_confidence_scores`):
```python
# Pre-compute early_cache once — avoids 3 redundant deepcopy(gen_cache) + crop
precomputed_early_cache = None
if gen_cache is not None:
    full_text = (assistant_prefill + generated_text).strip()
    fullstring_text = full_text[parsed_output.answer_fullstring_start:]
    fs_start, _ = find_token_indices_from_end(llm.tokenizer, base_tokens[0], fullstring_text, llm.model_name)
    early_late_split = fs_start - 1
    precomputed_early_cache = copy.deepcopy(gen_cache)
    crop_cache(precomputed_early_cache, early_late_split)
```

```
Generation Cache (full response)
        │
        ├──► Crop to early_late_split ──► early_cache (shared)
        │
        └──► For suffix calls: find overlap with base tokens
             └──► Reuse up to overlap_len tokens
```

### Key Optimizations

1. **Precomputed Early Cache**: Computed once (line 79-86) and shared across all three confidence methods

2. **Overlap Detection** (line 578-597): For suffix calls, finds how many tokens match between base and suffix tokenizations to maximize cache reuse
   ```python
   overlap_len = min(find_token_overlap(base_tokens[0], tokens[0]), gen_cache_len)
   # ...
   if overlap_len > early_late_split:
       # Reuse more of the cache via overlap — forward only the tail
       vanilla_cache = copy.deepcopy(gen_cache)
       crop_cache(vanilla_cache, overlap_len)
       remaining_tokens = tokens[:, overlap_len:].to(device)
   ```

3. **Consume Flag** (line 558, 608-614, 637-652): The last consumer of a cache can use it directly without copying
   ```python
   _owns_early_cache = (precomputed_early_cache is None) or consume_early_cache
   # ...
   if _owns_early_cache and not will_run_dropout:
       logger.info("consuming early_cache directly for vanilla (no suffix)")
       vanilla_output = llm.model.forward(
           input_ids=late_tokens,
           past_key_values=early_cache,  # No deepcopy needed
       )
   ```

4. **`crop_cache()` function** (line 19-33): Handles both standard `DynamicCache` and Qwen's custom `Qwen3_5DynamicCache`
   ```python
   def crop_cache(cache, max_length):
       if hasattr(cache, 'crop'):
           cache.crop(max_length)
       else:
           # Qwen3_5DynamicCache: crop only attention layers
           for idx in range(len(cache.key_cache)):
               if cache.key_cache[idx] is not None and cache.key_cache[idx].dim() == 4:
                   cache.key_cache[idx] = cache.key_cache[idx][:, :, :max_length, :]
                   cache.value_cache[idx] = cache.value_cache[idx][:, :, :max_length, :]
   ```

---

## Data Flow

The main orchestration happens in `compute_all_confidence_scores()` (line 54-179):

```
compute_all_confidence_scores()  [line 54]
        │
        ├──► _tokenize_for_confidence()  [line 76]  ──► base_tokens (shared)
        │
        ├──► Precompute early_cache  [line 79-86]  ──► precomputed_early_cache (shared)
        │
        ├──► dropout_answerlogits()  [line 89-92]  ──► vanilla + dropout answer probs
        │
        ├──► dropout_indirectlogits() [line 96-101] ──► vanilla + dropout P(True) scores
        │
        └──► dropout_verbalconf()     [line 103-112] ──► vanilla + dropout verbalized scores
        
        [If experimental_jackknife=True, repeat all three with use_jackknife=True, lines 117-153]
        
        └──► AllConfidenceData(  [line 155-179]
               vanilla_confidences,
               dropout_confidences,
               jackknife_confidences,
               debug_info
             )
```

Each confidence method internally calls `dropout_forward()` (line 493-679), which:
1. Tokenizes the full content (with optional suffix)
2. Splits into early/late tokens at the answer boundary
3. Runs vanilla forward pass (reusing cache when possible)
4. Runs batched dropout forward pass via `dropout_late_forward()` (line 682-761)

---

## Output Structure

### ConfidenceScores (per condition)

Defined in `utils/structures.py:31-42`:

| Field | Type | Description |
|-------|------|-------------|
| `answer_probabilities` | `list[dict[str, float]]` | Per-token probs for the answer |
| `answer_entropy` | `list[dict[str, float]]` | Per-token entropy values |
| `indirect_ptrue1_probabilities` | `list[dict[str, float]]` | P(True/False) scores |
| `indirect_ptrue2_probabilities` | `list[dict[str, float]]` | P(Yes/No) scores |
| `verbconf_probabilities` | `list[float]` | Expected verbalized confidence (0.0-1.0) |
| `verbconf_distribution` | `list[float] \| None` | Full 0-100 distribution (length 101) |
| `verbconf_top_score` | `int \| None` | Most likely confidence score (0-100) |
| `verbconf_top_prob` | `float \| None` | Probability of top score |
| `step_masks` | `list[list[int]] \| None` | Binary list per sample indicating masked steps |

### AllConfidenceData

Defined in `utils/structures.py:45-50`:

```python
@dataclass
class AllConfidenceData:
    vanilla_confidences: ConfidenceScores      # Full context (no dropout)
    dropout_confidences: ConfidenceScores      # List of scores under coin-flip dropout
    jackknife_confidences: ConfidenceScores | None = None  # Optional jackknife dropout
    debug_info: dict = field(default_factory=dict)  # Detailed debugging info
```

**Example output structure:**
```python
{
  "vanilla_confidences": {
    "answer_probabilities": [{"42": 0.95}, {"is": 0.87}],
    "verbconf_probabilities": [0.82],
    "verbconf_distribution": [0.001, 0.002, ..., 0.082, ...],  # 101 values
    "verbconf_top_score": 82,
    "verbconf_top_prob": 0.082
  },
  "dropout_confidences": {
    "answer_probabilities": [[...], [...], ...],  # 10 samples
    "verbconf_probabilities": [0.71, 0.79, 0.68, ...],  # 10 samples
    "step_masks": [[1, 0, 1, 0], [0, 1, 1, 0], ...]  # which steps kept per sample
  },
  "debug_info": {
    "answer_logits": {...},
    "indirect_logits": {...},
    "verbconf": {...}
  }
}
```

---

## Key Helper Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `crop_cache()` | `confidence.py:19-33` | Truncates KV cache to specified length; handles both `DynamicCache` (via `.crop()`) and `Qwen3_5DynamicCache` (manual tensor slicing) |
| `get_token_ids()` | `confidence.py:45-51` | Converts token strings like `" Yes"` to model-specific token IDs, filtering to single-token entries only |
| `_tokenize_for_confidence()` | `confidence.py:447-490` | Tokenizes conversation with assistant response, applying model-specific post-processing (strips `<think></think>` for Qwen, extracts channel tags for GPT) |
| `_extract_thinking_and_content()` | `confidence.py:421-444` | Parses GPT-OSS channel-tag format to separate thinking and final content |
| `_compute_verbconf_joint_probs()` | `confidence.py:296-361` | Computes joint probabilities for multi-token numbers (e.g., `P("42") = P("4") * P("2"|"4")`) using a tree of KV cache clones |
| `find_token_indices_from_end()` | `utils/text_utils.py:15-52` | Locates text spans in token sequences by searching backwards from the end |
| `find_token_overlap()` | `utils/text_utils.py:55-66` | Finds longest common prefix between two token ID tensors (used for cache reuse) |

---

## Supporting Data Structures

### ParsedOutput (`utils/structures.py:21-28`)

Represents the parsed structure of a model's generated response:

```python
@dataclass
class ParsedOutput:
    cot_steps: list           # individual reasoning steps as strings
    final_answer: str         # the extracted answer (e.g., "42", "A")
    raw_cot_block: str        # the full CoT text before the final answer
    answer_fullstring_start: int  # char offset where "\boxed{" begins
    answer_start: int         # char offset of first character inside \boxed{...}
```

- `cot_steps` is central to the dropout mechanism: `cot_steps[1:]` are the steps eligible for masking (the first step is always kept, see `confidence.py:700`).
- `answer_fullstring_start` determines where the early/late token split happens (`confidence.py:535-538`).

### LLM (`llm.py:74-229`)

The model wrapper that provides:
- `llm.model`: The HuggingFace `AutoModelForCausalLM` instance (loaded at `llm.py:84-89`)
- `llm.tokenizer`: The `AutoTokenizer` instance (`llm.py:91`)
- `llm.model_name`: Model identifier used for model-specific branching
- `llm.switch_attn_implementation()` (`llm.py:96-110`): Switches between attention backends; confidence computation uses `"sdpa"` or `"eager"` (for custom attention masks), while generation uses Flash Attention

### GenerationResult (`utils/structures.py:9-18`)

Returned by `llm.generate_one()` (`llm.py:114-229`), provides:
- `past_key_values`: The KV cache from generation, passed as `gen_cache` into confidence computation
- `generated_text`: The raw generated text to analyze

---

## Model-Specific Handling

- **Qwen models** (`confidence.py:484-485`): Uses `Qwen3_5DynamicCache` for KV cache, removes empty `<think></think>` tags via regex during tokenization
  ```python
  if "qwen" in llm.model_name.lower():
      prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)
  ```

- **GPT-OSS models** (`confidence.py:421-444`): Responses use a channel-tag structure for thinking vs. final content. The `_extract_thinking_and_content()` function parses:
  ```
  <|channel|>analysis<|message|>...thinking...<|end|>
  <|start|>assistant<|channel|>final<|message|>...answer...<|return|>
  ```
  This is essential because confidence scores should be computed on the final answer content, not the model's internal reasoning channel.

- **Attention backend switching** (`llm.py:96-110`): During generation, Flash Attention is used for speed. For confidence computation, the code switches to `sdpa`/`eager` because custom attention masks (needed for dropout) are not compatible with Flash Attention.

---

## Multi-Token Verbalized Confidence (`_compute_verbconf_joint_probs`, line 296-361)

This function deserves special attention as it solves a non-trivial problem: computing P(42) when "42" is tokenized as two separate tokens ["4", "2"].

**Algorithm:**
1. **Depth 0** (line 316-318): Extract first-token log-probs directly from the existing forward pass output
2. **Depth >= 1** (line 327-359): For each unique prefix, clone the KV cache, run a single-token forward pass, and accumulate conditional log-probs
   - E.g., to get P("42"): first get P("4"), then feed "4" into the model to get P("2"|"4"), and compute P("42") = P("4") * P("2"|"4")
3. **Normalization** (line 361): Softmax over all 101 joint log-probs to get a proper probability distribution

---

## Debug Infrastructure (lines 764-869)

When `debug_conf=True` is passed, each confidence method returns detailed diagnostic information via helper functions:

| Debug Function | Line | What it captures |
|----------------|------|-----------------|
| `_debug_answer_logit_tokens()` | 785-809 | Per-token probabilities with top-5 alternatives for each answer token |
| `_debug_indirect_logit_tokens()` | 812-831 | Logit values and probabilities for all positive/negative probe tokens |
| `_debug_verbconf_tokens()` | 834-852 | Top-10 predicted confidence scores with their probabilities |
| `_debug_masked_text()` | 855-868 | Decoded text of kept vs. masked reasoning steps per dropout sample |

Debug output is stored in `AllConfidenceData.debug_info` and serialized to JSON for offline analysis.

---

## Usage Example

```python
from confidence import compute_all_confidence_scores

# After generating a response and parsing it:
confidence_data = compute_all_confidence_scores(
    llm=llm_instance,                       # LLM wrapper (llm.py)
    messages=conversation_messages,          # Chat history
    generated_text=model_output,             # Raw generated text
    parsed_output=parsed_response,           # ParsedOutput with cot_steps + final_answer
    nb_dropout_samples=10,                   # Number of dropout perturbations
    use_fullstring=False,                    # Whether to mask entire answer string or just final answer
    assistant_prefill="",                    # Any assistant prefix used during generation
    debug_conf=True,                         # Enable debug output
    gen_cache=generation_kv_cache,           # KV cache from generate_one() for speedup
    experimental_jackknife=False,            # Enable jackknife dropout variant
)

# Access results
vanilla_verbconf = confidence_data.vanilla_confidences.verbconf_probabilities  # [0.82]
dropout_verbconfs = confidence_data.dropout_confidences.verbconf_probabilities  # [0.71, 0.79, ...]
vanilla_ptrue = confidence_data.vanilla_confidences.indirect_ptrue1_probabilities  # [{"True": 0.87, ...}]
step_masks = confidence_data.dropout_confidences.step_masks  # [[1, 0, 1], [0, 1, 0], ...]
```

---

## Research Applications

This module enables research into:

1. **Confidence Calibration**: Are model confidence scores well-calibrated to actual accuracy?
2. **Reasoning Robustness**: How much does confidence drop when reasoning steps are masked? (Comparing vanilla vs. dropout scores)
3. **Faithfulness**: Do models actually use their reasoning, or is confidence independent of CoT? (If dropout barely changes confidence, the model may not rely on its stated reasoning)
4. **Uncertainty Quantification**: Comparing answer logits vs. P(True) probing vs. verbalized confidence -- which is most reliable?
5. **Step Importance**: By tracking `step_masks`, individual reasoning steps can be correlated with confidence changes to identify which steps are load-bearing
