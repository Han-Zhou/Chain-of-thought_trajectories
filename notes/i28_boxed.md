# i28: Migrate answer extraction from "Final Answer:" to `\boxed{}`

## What changed

Replaced the `"Final Answer: <answer>"` extraction pattern with `\boxed{answer}` across the entire pipeline.

## Files modified

### Core logic
- **`parsing.py`** — New `_extract_boxed_with_positions()` using brace-matching. `parse_output()` now splits CoT/answer at the last `\boxed{...}` instead of `"Final Answer:"`.
- **`eval/extractors.py`** — Removed `_FINAL_ANSWER_MCQ_RE` and `_FINAL_ANSWER_MARKER` regexes. All three extractors (`extract_mcq_letter`, `extract_text_answer`, `extract_boxed_or_text`) now funnel through the existing `_extract_boxed()` brace-matcher.
- **`llm.py`** — `_StopAfterFinalAnswer` → `_StopAfterBoxedAnswer`. Stops generation when `\boxed{...}` braces are balanced (handles nested braces like `\boxed{\frac{1}{2}}`).

### Prompts
- **`prompts/cot_prompt.py`** — System prompt and all 12 dataset templates now instruct `\boxed{}` format.
- **`prompts/few_shot_prompt.py`** — All ~25 assistant examples changed from `"Final Answer: X"` to `"\\boxed{X}"`. Math answers stripped of `$...$` wrapping.
- **`prompts/load.py`** — GPT thinking-field split: finds `\\boxed{` to separate thinking from answer content.

### Docs / comments
- **`utils/structures.py`** — `ParsedOutput` field docstrings updated.
- **`main.py`** — Comments updated.
- **`confidence.py`** — One comment updated. No logic changes; offset-based indexing works identically with `\boxed{}` positions.

### Tests
- **`tests/test_eval.py`** — All test inputs use `\boxed{}` format. 39/39 passing.

## How `ParsedOutput` fields map to `\boxed{}`

| Field | Before | After |
|---|---|---|
| `answer_fullstring_start` | char offset of `"Final Answer:"` | char offset of `\` in `\boxed{...}` |
| `answer_start` | char offset of first answer char after `"Final Answer: "` | char offset of first char inside braces |
| `final_answer` | text after `"Final Answer: "` | content inside `\boxed{...}` |
