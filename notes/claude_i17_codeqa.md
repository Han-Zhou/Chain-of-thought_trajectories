# i17 – CodeQA Support

## Overview

This iteration adds full CodeQA support to the CoT evaluation pipeline, which was previously only wired up for LogiQA. Five files were modified.

| File | Change type |
|---|---|
| `parsing.py` | Unified `parse_output(dataset_name=)` + `_PARSING_REGEX` dispatch dict |
| `prompts/load.py` | New branch — CodeQA in `load_messages()` |
| `prompts/few_shot_prompt.py` | New examples — `CODEQA_FEW_SHOT_MESSAGES` + registry entry |
| `main.py` | Import + dataset-aware parser dispatch + doc/arg updates |
| `evaluate_trajectories.py` | Full rewrite — CLI args, CodeQA evaluator, no breakpoint |

---

## Background: what was already in place

The `dataloader/codeqa.py` (`CodeQADataLoader`) and the `codeqa` prompt template in `prompts/cot_prompt.py` (`CODEQA`) were already implemented but never wired into the generation or evaluation paths. The old `load_messages()` raised `NotImplementedError` for any dataset other than `logiqa`, and `evaluate_trajectories.py` was a throwaway script with a hardcoded run name and a `breakpoint()`.

---

## CodeQA vs LogiQA — key differences

| Dimension | LogiQA | CodeQA |
|---|---|---|
| Answer type | Single letter (A–D) | Free-form text |
| Entry `context` | Natural-language passage | Source code snippet |
| Entry `choices` | 4 labeled options | `None` |
| Parser | `parse_output(dataset_name="logiqa")` — letter extraction | `parse_output(dataset_name="codeqa")` — full text extraction |
| Evaluator metric | Exact-match on letter | Case-insensitive exact-match + coverage |

---

## `parsing.py` — unified `parse_output()` with `_PARSING_REGEX` dispatch

Instead of a separate `parse_output_open_ended()` function, a single `parse_output(generated_text, dataset_name="logiqa")` handles all datasets by dispatching through two module-level dicts.

**`_PARSING_REGEX`** maps a mode name to a `(compiled_pattern, extractor_lambda)` tuple:
```python
_PARSING_REGEX = {
    "mcq": (
        re.compile(r"final\s+answer\s*:?\s*[\(\[]?\s*([A-Da-d])\s*[\)\]]?", re.IGNORECASE),
        lambda m: m.group(1).upper(),
    ),
    "open_ended": (
        re.compile(r"final\s+answer\s*:\s*([\s\S]+)", re.IGNORECASE),
        lambda m: m.group(1).strip(),
    ),
}
```
- `"mcq"` captures a single letter and uppercases it.
- `"open_ended"` captures all remaining text (including newlines) and strips it.

**`_DATASET_PARSING_MODE`** maps dataset names to a mode key. Unlisted datasets default to `"mcq"`:
```python
_DATASET_PARSING_MODE = {
    "logiqa":  "mcq",
    "codeqa":  "open_ended",
}
```

**Dispatch inside `parse_output()`:**
```python
mode = _DATASET_PARSING_MODE.get(dataset_name, "mcq")
pattern, extract = _PARSING_REGEX[mode]
...
final_answer = extract(m)
```

Adding support for a new answer type requires only a new entry in `_PARSING_REGEX` and a line in `_DATASET_PARSING_MODE` — no new functions.

---

## `prompts/load.py` — CodeQA branch in `load_messages()`

Added `elif dataset == "codeqa":` block. Key decisions:

**System message:** Uses only `SYSTEM` (the generic CoT instruction). Unlike LogiQA, the `CODEQA` template from `cot_prompt.py` is not appended to the system turn — it was designed as a user-turn template with `{code}` and `{question}` placeholders.

**User message construction:**
```python
user_prompt = (
    f"Code snippet:\n```python\n{entry['context']}\n```\n\n"
    f"Question:\n{entry['question']}"
)
```
`entry["context"]` holds the source code (set by `CodeQADataLoader` from `row["code"]` or `row["code_processed"]`). A fenced Python block is used so models treat the content as code.

**Few-shot insertion** follows the identical pattern as LogiQA: `messages[0:1] + few_shot_messages + messages[1:]` — system first, then few-shot pairs, then the test user/assistant turns.

---

## `prompts/few_shot_prompt.py` — `CODEQA_FEW_SHOT_MESSAGES`

Two demonstrations covering distinct CodeQA question types:

### Example 1 — Return value tracing (`find_duplicates`)
- **Code:** set-based duplicate detector iterating over a list.
- **Question:** "What does `find_duplicates([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])` return?"
- **CoT:** 4 steps — understand purpose, trace all 10 iterations explicitly showing `seen` and `duplicates` evolving, note the return, discuss the edge case for elements appearing 3+ times.
- **Final Answer:** `[1, 5, 3]`

### Example 2 — Complexity analysis (`flatten`)
- **Code:** recursive list flattener using `extend`.
- **Question:** Time complexity in terms of N (total elements across all nesting levels).
- **CoT:** 5 steps — per-call work, cost of `extend`, worst-case derivation (deeply nested chain → O(N²)), best-case flat list (O(N)), summary.
- **Final Answer:** `O(N^2) worst case (due to repeated extend calls propagating elements up each level of nesting), O(N) best case for a flat list.`

**Registry entry added:**
```python
FEW_SHOT_PROMPT_REGISTRY["codeqa"] = CODEQA_FEW_SHOT_MESSAGES
```

---

## `main.py` — dataset-aware parser dispatch

**Import:**
```python
from parsing import parse_output
```

**Call inside `generate_trajectories()`:**
```python
parsed: ParsedOutput = parse_output(full_generated_text, dataset_name=dataset_name)
```
The dispatch is fully encapsulated in `parsing.py` — `main.py` passes `dataset_name` and `parse_output` resolves the correct regex and extractor internally. No `if/else` branching in `main.py`.

**Other minor edits:**
- Module docstring updated: "For LogiQA and CodeQA, runs a full Chain-of-Thought pipeline..."
- `--shot_mode` help text generalized: "Prompting mode: zero-shot or few-shot CoT." (removed "for LogiQA").
- `breakpoint()` inside `generate_one()` left untouched (intentional dev artifact).

---

## `evaluate_trajectories.py` — full rewrite

The old script was a throwaway: hardcoded `RUN = "qwen_logiqa_zero_1"`, no CLI args, `breakpoint()` at the end.

**New design:**

### CLI
```
python evaluate_trajectories.py --run <run_id> --dataset <dataset_name>
```
Example: `python evaluate_trajectories.py --run qwen_codeqa_zero_100 --dataset codeqa`

The `--run` argument drives file path reconstruction using the same `rsplit("_", 1)` pattern the original script used.

### `load_trajectories(run: str) -> list[dict]`
Extracted into a reusable function. Same `json.JSONDecoder().raw_decode()` streaming loop as the original.

### `evaluate_logiqa(trajectories)`
- Compares `final_answer.upper()` against `ground_truth.upper()`.
- Tracks: total, answered (non-empty `final_answer`), missing, correct.
- Prints: total, answered/missing, correct, accuracy (correct / total).

### `evaluate_codeqa(trajectories)`
- Compares `final_answer.strip().lower()` against `ground_truth.strip().lower()` (case-insensitive exact-match).
- Tracks: total, covered (non-empty `final_answer`), correct.
- Prints: total, coverage (covered / total), correct, exact-match accuracy (correct / total).
- Coverage is reported separately because empty `final_answer` (model failed to emit "Final Answer:") is meaningfully different from a wrong answer.

### Dispatch
```python
if dataset in MCQ_DATASETS:      # {"logiqa"}
    evaluate_logiqa(trajectories)
elif dataset in OPEN_ENDED_DATASETS:  # {"codeqa"}
    evaluate_codeqa(trajectories)
else:
    # warn + fall back to evaluate_logiqa
```

---

## Verification

All changes verified by a 3-agent team (`codeqa-verify`), 5 tasks, all passing:

| Task | Result |
|---|---|
| Imports: all 5 modules import cleanly | PASS |
| `parse_output(dataset_name="codeqa")`: 4 test cases | PASS |
| `load_messages("codeqa")` zero-shot (3 msgs) and few-shot (7 msgs) | PASS |
| Few-shot format: role alternation, markers, code `compile()` | PASS |
| `evaluate_trajectories.py` CLI, logiqa/codeqa logic, main.py dispatch | PASS |

---

## Usage

```bash
# Zero-shot inference
python main.py --dataset codeqa --model qwen --sample_size 100 --shot_mode zero

# Few-shot inference (2 demonstrations)
python main.py --dataset codeqa --model qwen --sample_size 100 --shot_mode few

# Evaluation
python evaluate_trajectories.py --run qwen_codeqa_zero_100 --dataset codeqa
```

Output is written to `trajectories/qwen_codeqa_zero_100/qwen_codeqa_zero_trajectories_100.json` as newline-delimited JSON objects. KV caches are saved as `cache_<i>.pt` in the same directory.
