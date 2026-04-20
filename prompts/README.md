# prompts/

Prompt construction for the CoT evaluation pipeline. Assembles system, user, and assistant messages for each dataset.

## Files

| File | Purpose |
|---|---|
| `cot_prompt.py` | Zero-shot prompt constants: system instructions, per-dataset user templates, answer-format suffixes, and the `PROMPT_REGISTRY` / `DATASET_ANSWER_FORMAT` mappings |
| `few_shot_prompt.py` | Few-shot example messages for each dataset, the `FEW_SHOT_PROMPT_REGISTRY`, and `THINKING_TOKENS` (per-model think delimiters) |
| `load.py` | `load_messages()` — the single entry point that assembles the final message list from dataset, model, prompt type, and flags |

## How messages are built

`load_messages()` in `load.py` is the only function callers need. It:

1. Looks up `(system_prompt, user_template, assistant_start)` from `PROMPT_REGISTRY` in `cot_prompt.py`.
2. Fills the user template with entry-specific fields (question, functions, code, context, etc.).
3. Applies flag-based system prompt modifications:
   - `experimental_extra_box=True` swaps `SYSTEM` for `SYSTEM_WITHOUT_BOXED` (strips `\boxed{}` instruction).
   - `thinking=True` + Qwen model swaps `SYSTEM` for `SYSTEM_QWEN_THINKING`.
4. Appends the dataset-specific answer-format suffix from `DATASET_ANSWER_FORMAT` (e.g. `SYSTEM_MCQ`, `SYSTEM_MATH`, `SYSTEM_BINARY`).
5. If `experimental_extra_box=True`, appends `SYSTEM_COT_ONLY` (instructs model not to reveal the answer during reasoning).
6. Wraps everything into a `[system, user, assistant]` message list, optionally inserting few-shot examples.

## System prompt constants

| Constant | Used by | Purpose |
|---|---|---|
| `SYSTEM` | All datasets (default) | Base CoT instruction with `\boxed{}` format |
| `SYSTEM_WITHOUT_BOXED` | extra-box mode | Same as `SYSTEM` minus `\boxed{}` instruction |
| `SYSTEM_QWEN_THINKING` | Qwen + thinking | Adds `<think>`/`</think>` delimiters |
| `SYSTEM_MCQ` | bigbench_movie, logiqa | Output single letter only |
| `SYSTEM_BFCL` | bfcl | Output function calls in `[func(...)]` format |
| `SYSTEM_MATH` | college_math, math500, olympiadbench | Output exact numerical/symbolic answer |
| `SYSTEM_BINARY` | bigbench_causal | Output Yes/No only |
| `SYSTEM_OTHER` | hotpotqa | Output short string answer |
| `SYSTEM_COT_ONLY` | All (when extra-box) | Suppress answer during reasoning steps |

## Adding a new dataset

1. Add a user prompt template in `cot_prompt.py` with `{question}` (and any extra placeholders).
2. Register it in `PROMPT_REGISTRY`.
3. Add an answer-format entry in `DATASET_ANSWER_FORMAT` if the dataset needs one.
4. Add the dataset name to `_SUPPORTED_MESSAGE_DATASETS` in `load.py`.
5. If the dataset needs custom field handling (like BFCL's `{functions}`), add a branch in `load_messages()`.
6. Optionally add few-shot examples in `few_shot_prompt.py` and register in `FEW_SHOT_PROMPT_REGISTRY`.
