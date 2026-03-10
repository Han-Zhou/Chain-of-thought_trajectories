# i4 – LogiQA CoT Evaluation Pipeline

## Overview

This iteration adds a full LogiQA evaluation pipeline on top of the existing
`main.py` infrastructure. Two files were touched:

- **`evaluate_logiqa.py`** — new module, all pipeline logic lives here
- **`main.py`** — three small edits to wire the new module in

---

## New file: `evaluate_logiqa.py`

The file is divided into four sections that mirror the data flow.

### Section 1 — Prompt Formatting

Two public functions build the text string that gets tokenised and fed to the
model.

**`format_zero_shot(entry)`**
Prepends `SYSTEM_PROMPT` (the CoT instruction) directly before the test entry.
No demonstrations are shown; the model must follow the format from the
instruction alone.

**`format_few_shot(entry, n_shots=2)`**
Inserts `n_shots` fixed demonstrations before the test entry. Each
demonstration contains the full context/question/options block, the numbered
`Step N:` reasoning, and a `Final Answer: <letter>` line, teaching the model
the expected output format by example.

The two demonstrations in `FEW_SHOT_EXAMPLES` are hand-written and are not
drawn from any eval split, so they do not contaminate the benchmark.

Both functions call the shared helper `_render_entry()`, which formats a
normalised dataset entry (as returned by `load_logiqa()`) into a
context/question/options text block.

---

### Section 2 — Generation with `DynamicCache`

**`generate_one(model, tokenizer, prompt_text, ...)`** wraps `model.generate()`
and returns a `GenerationResult` dataclass containing the generated text, the
live KV cache, and the absolute token positions of the prompt and generated
tokens.

Three HuggingFace arguments drive the behaviour:

| Argument | Value | Why |
|---|---|---|
| `past_key_values` | `DynamicCache()` | Passes a pre-initialised cache into `generate()`. Transformers fills it in-place; the populated cache is returned via `outputs.past_key_values`, ready to be reused or inspected. |
| `return_dict_in_generate` | `True` | Makes `generate()` return a structured `GenerateDecoderOnlyOutput` instead of a bare tensor, giving access to `.sequences` and `.past_key_values`. |
| `do_sample` / `temperature` | `False` / `0.0` | Greedy decoding — deterministic and reproducible, which is standard practice for MCQ benchmarks. |

Token positions are derived from the sequence tensor shape:

```
prompt_positions    = [0, 1, ..., prompt_len - 1]
generated_positions = [prompt_len, ..., total_len - 1]
```

These are stored in the result dict so downstream analysis can map generated
tokens back to their absolute positions in the full sequence (e.g. for
attention or KV cache inspection).

---

### Section 3 — Parsing

**`parse_output(generated_text)`** takes the raw string returned by the model
(prompt is not included) and returns a `ParsedOutput` dataclass with three
fields: `cot_steps`, `final_answer_letter`, and `raw_cot_block`.

The parsing runs three steps:

1. **Split on `"final answer"`** (case-insensitive) to separate the CoT block
   from the answer line. If the model did not emit `Final Answer:`, the entire
   text is treated as CoT.

2. **Split the CoT block into steps** using `_STEP_MARKER_RE` (`Step N:`).
   If no markers are found, falls back to blank-line splitting, then
   single-line splitting as a last resort.

3. **Extract the letter** from the answer segment with `_FINAL_ANSWER_RE`,
   which tolerates lowercase letters, parentheses (`(B)`), brackets (`[c]`),
   and missing colons.

---

### Section 4 — Evaluation Loop

**`run_logiqa_eval(model_name, dataset, shot_mode, ...)`** is the drop-in
replacement for `generate_trajectories()` in `main.py` when
`--dataset logiqa` is active.

It loads the model once, then for each entry:
1. Calls `format_zero_shot` or `format_few_shot` depending on `shot_mode`.
2. Calls `generate_one` to run inference and capture the cache and positions.
3. Calls `parse_output` to extract the answer letter and CoT steps.
4. Appends a result dict to the output list.

It logs running accuracy and returns the full result list for `main.py` to
serialise.

Each result dict contains:

| Key | Description |
|---|---|
| `id` | Entry ID from the dataset |
| `question` | The question text |
| `ground_truth` | Correct answer letter (A–D) |
| `predicted` | Letter extracted by the parser |
| `correct` | Boolean — `predicted == ground_truth` |
| `cot_steps` | List of individual reasoning step strings |
| `raw_cot_block` | Full CoT text before the Final Answer line |
| `generated_text` | Raw model output (no prompt) |
| `prompt_token_positions` | Absolute positions `[0, prompt_len)` |
| `generated_token_positions` | Absolute positions `[prompt_len, total_len)` |

---

## Changes to `main.py`

Three edits, no existing behaviour altered.

**1. New import at the top**
```python
from evaluate_logiqa import run_logiqa_eval
```

**2. New CLI argument `--shot_mode`**
```
--shot_mode {zero,few}   default: zero
```
Only consumed by the LogiQA path; ignored for all other datasets.

**3. Early-return dispatch in `main()`**
```python
if args.dataset == "logiqa":
    results = run_logiqa_eval(
        model_name=model_name,
        dataset=dataset[:sample_size],
        shot_mode=args.shot_mode,
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return
```
When `--dataset logiqa` is passed, `main()` calls `run_logiqa_eval()` and
writes its structured output, then returns before reaching the original
`generate_trajectories()` path. All other datasets continue through the
original path unchanged.

---

## Usage

```bash
# Zero-shot CoT on 50 samples
python main.py --dataset logiqa --model llama --sample_size 50 --shot_mode zero

# Few-shot CoT (2 demonstrations) on 100 samples
python main.py --dataset logiqa --model llama --sample_size 100 --shot_mode few

# Different model
python main.py --dataset logiqa --model qwen --sample_size 50 --shot_mode few
```

Output is written to `<model>_logiqa_trajectories_<sample_size>.json` as a
JSON array of result dicts.
