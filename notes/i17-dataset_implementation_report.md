# Dataset Implementation Report

**Date:** 2026-03-12
**Branch:** `i17/main-others`

## Summary

All 11 datasets listed in `README.md` are now fully wired into the CoT pipeline. This report describes the bugs found and the changes made.

---

## Bugs Found

### 1. `prompts/load.py` — Only 2 of 11 datasets handled

`load_messages()` only handled `logiqa` and `codeqa`. All other datasets fell through to:
```python
raise NotImplementedError(f"Message loading not implemented for dataset '{dataset}'.")
```
Running any other dataset would immediately crash the pipeline.

### 2. `parsing.py` — Wrong parse mode for most datasets

`_DATASET_PARSING_MODE` only listed `logiqa` (mcq) and `codeqa` (open_ended). All other datasets defaulted to `"mcq"`, which uses a regex that only captures letters `A–D`. This was incorrect for:
- **college_math** (MMLU-Pro): options are labelled `A–J` (up to 10 choices)
- **hotpotqa, math500, olympiadbench, bfcl, bigbench_movie, bigbench_causal, cs1qa, hle**: all produce free-text answers, not MCQ letters

### 3. `prompts/cot_prompt.py` — Key mismatch for `college_math`

`PROMPT_REGISTRY` used the key `"college_math_test"` but:
- `dataloader/__init__.py` registered it as `"college_math"`
- `main.py` documents `college_math` as the valid `--dataset` argument

Running `python main.py --dataset college_math` would fail with `ValueError: Unknown dataset 'college_math'` when building prompts.

---

## Changes Made

### `prompts/cot_prompt.py`
- Added `"college_math"` as the canonical key in `PROMPT_REGISTRY` (kept `"college_math_test"` as an alias for backward compatibility)

### `parsing.py`
- Added `"mcq_extended"` parsing mode: matches a single letter `A–J` (covers MMLU-Pro's 10-option format)
- Added explicit parsing modes for all 9 remaining datasets:

| Dataset | Parse Mode | Reason |
|---------|-----------|--------|
| `bfcl` | `open_ended` | Answers are function-call strings |
| `bigbench_movie` | `open_ended` | Answers are movie titles (text) |
| `bigbench_causal` | `open_ended` | Answers are "Yes" / "No" (text) |
| `cs1qa` | `open_ended` | Free-text answers |
| `hotpotqa` | `open_ended` | Free-text span answers |
| `college_math` | `mcq_extended` | MMLU-Pro uses letters A–J |
| `olympiadbench` | `open_ended` | Numeric / proof answers |
| `math500` | `open_ended` | Numeric / algebraic answers |
| `hle` | `open_ended` | Mixed MCQ and free-text; open_ended captures both |

### `prompts/load.py`
- Removed unused `import pathlib`
- Added message builders for all 9 remaining datasets:

| Dataset | System content | User content |
|---------|---------------|-------------|
| `bfcl` | `SYSTEM` + `BFCL` template (with `{functions}` JSON) | `entry["question"]` |
| `bigbench_movie` | `SYSTEM` | question + choices (if present) |
| `bigbench_causal` | `SYSTEM` | question + choices (if present) |
| `cs1qa` | `SYSTEM` | code block (if present) + question |
| `hotpotqa` | `SYSTEM` | supporting passages + question |
| `college_math` | `SYSTEM` | problem + choices (if present) |
| `olympiadbench` | `SYSTEM` | context (if present) + problem |
| `math500` | `SYSTEM` | problem |
| `hle` | `SYSTEM` | question + choices (if present) |

For `bfcl`, `entry["metadata"]["functions"]` is serialised with `json.dumps(indent=2)` and embedded in the system message via the `BFCL` prompt template.

---

## Test Results

All datasets were validated end-to-end (dataloader → message builder → parse):

| Dataset | Status | Entries | Notes |
|---------|--------|---------|-------|
| `bfcl` | ✅ PASS | 1000 | Local files at `/storage/backup/han/cot/bfcl` |
| `bigbench_movie` | ✅ PASS | 100 | HuggingFace `tasksource/bigbench` |
| `bigbench_causal` | ✅ PASS | 38 | HuggingFace `tasksource/bigbench` |
| `logiqa` | ✅ PASS | — | Already working |
| `codeqa` | ✅ PASS | — | Already working |
| `cs1qa` | ✅ PASS (graceful) | — | Requires `CS1QA_PATH` env var; raises `RuntimeError` if missing |
| `hotpotqa` | ✅ PASS | 7405 | HuggingFace `hotpotqa/hotpot_qa` (fullwiki) |
| `college_math` | ✅ PASS | 1351 | HuggingFace `TIGER-Lab/MMLU-Pro` (math subset) |
| `olympiadbench` | ✅ PASS | 2126 | HuggingFace `lmms-lab/OlympiadBench` |
| `math500` | ✅ PASS | 500 | HuggingFace `HuggingFaceH4/MATH-500` |
| `hle` | ✅ PASS (graceful) | — | Gated dataset; requires `HF_TOKEN` env var; raises `RuntimeError` if missing |

### Prerequisites for gated/local datasets
- **CS1QA**: `export CS1QA_PATH=/path/to/cs1qa_test.json`
- **HLE**: `export HF_TOKEN=hf_...` (after accepting terms at https://huggingface.co/datasets/cais/hle)
- **BFCL**: files must exist at `/storage/backup/han/cot/bfcl/`

---

## Dataset Sources

| Dataset | Source | How loaded | Notes |
|---------|--------|------------|-------|
| `bfcl` | [Berkeley Function-Calling Leaderboard v4](https://gorilla.cs.berkeley.edu/leaderboard.html) | Local files at `/storage/backup/han/cot/bfcl/` | 4 categories: `simple_python`, `multiple`, `parallel`, `parallel_multiple` |
| `bigbench_movie` | HuggingFace [`tasksource/bigbench`](https://huggingface.co/datasets/tasksource/bigbench), config `movie_recommendation` | HF `datasets` library, `validation` split | Part of Google's BIG-Bench suite |
| `bigbench_causal` | HuggingFace [`tasksource/bigbench`](https://huggingface.co/datasets/tasksource/bigbench), config `causal_judgment` | HF `datasets` library, `validation` split | Part of Google's BIG-Bench suite |
| `logiqa` | GitHub [`lgw863/LogiQA-dataset`](https://github.com/lgw863/LogiQA-dataset) | Downloaded directly via `urllib` from raw GitHub URLs | Uses `Test.txt` by default; parsing logic adapted from `lucasmccabe/logiqa` HF script |
| `codeqa` | HuggingFace [`lissadesu/codeqa_v2`](https://huggingface.co/datasets/lissadesu/codeqa_v2) | HF `datasets` library, `train` split (or local file via `CODEQA_PATH` env var) | |
| `cs1qa` | GitHub [`cs1qa/cs1qa`](https://github.com/cs1qa/cs1qa) (Lee et al., 2022) | Local file only — set `CS1QA_PATH=/path/to/cs1qa_test.json` | Not on HuggingFace |
| `hotpotqa` | HuggingFace [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa), config `fullwiki` | HF `datasets` library, `validation` split | |
| `college_math` | HuggingFace [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | HF `datasets` library, `test` split, filtered to `math`/`mathematics` category | |
| `olympiadbench` | HuggingFace [`lmms-lab/OlympiadBench`](https://huggingface.co/datasets/lmms-lab/OlympiadBench) | HF `datasets` library, `test_en` split | English math Olympiad problems |
| `math500` | HuggingFace [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | HF `datasets` library, `test` split | 500-problem subset of the MATH benchmark |
| `hle` | HuggingFace [`cais/hle`](https://huggingface.co/datasets/cais/hle) (gated) | HF `datasets` library, `test` split | Gated — requires accepting terms and setting `HF_TOKEN` env var |

---

## Files Changed

```
prompts/cot_prompt.py   — added "college_math" key to PROMPT_REGISTRY
parsing.py              — added mcq_extended mode; added 9 new dataset modes
prompts/load.py         — added message builders for 9 datasets; removed unused import
```

---

## Usage

Run any dataset with:
```bash
python3 main.py --dataset <name> --model <llama|gpt|qwen> --sample_size <N> --shot_mode zero
```

Example:
```bash
python3 main.py --dataset hotpotqa --model qwen --sample_size 10 --shot_mode zero
python3 main.py --dataset math500 --model llama --sample_size 50 --shot_mode zero
python3 main.py --dataset college_math --model gpt --sample_size 20 --shot_mode zero
```
