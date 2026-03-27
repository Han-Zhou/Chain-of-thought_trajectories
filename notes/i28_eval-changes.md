# i28: Evaluation Harness — Implementation Notes

## What was built

A post-hoc evaluation system that reads saved trajectory JSONL/JSON files, extracts the model's final answer, compares it against ground truth, and reports accuracy/F1 metrics.

## Files created

### `eval/__init__.py`
Package init. Exports `evaluate_one`, `EVAL_REGISTRY`, `UNSUPPORTED`.

### `eval/extractors.py`
Three extraction functions operating on `generated_text`:

- **`extract_mcq_letter(text)`** — reuses the regex from `parsing.py` to pull a letter (A–D) after "Final Answer:".
- **`extract_text_answer(text)`** — returns the raw text after the last "Final Answer:" marker.
- **`extract_boxed_or_text(text)`** — for math: tries `\boxed{}` extraction on the answer segment, falls back to raw text. Strips surrounding `$...$`.

`_extract_boxed()` is copied from `lm-evaluation-harness/lm_eval/tasks/score/math/math_grader.py:564-612`.

### `eval/comparators.py`
Comparison functions:

- **`exact_match(pred, ref)`** — case-insensitive string equality.
- **`normalized_text_match(pred, ref)`** — strips punctuation, articles, extra whitespace, then compares.
- **`math_equal(pred, ref)`** — full math comparison: string normalization → numeric equality (`isclose`) → symbolic equality (`sympy`). Copied from `math_grader.py:91-654` with all helper functions. The `antlr4-python3-runtime==4.11` check is deferred to `symbolic_equal()` so simple numeric comparisons work without it.
- **`qa_f1_score(pred, ref)`** — token-level F1 after QA normalization. Copied from `lm-evaluation-harness/lm_eval/tasks/longbench/metrics.py:40-248`.

### `eval/registry.py`
Dataset dispatcher. Maps each supported dataset to `(extractor, comparator, metric_name)`:

| Dataset | Extractor | Comparator | Metric |
|---------|-----------|------------|--------|
| `logiqa` | `extract_mcq_letter` | `exact_match` | accuracy |
| `college_math` | `extract_boxed_or_text` | `math_equal` | accuracy |
| `bigbench_causal` | `extract_text_answer` | `exact_match` | accuracy |
| `bigbench_movie` | `extract_mcq_letter` | `exact_match` | accuracy |
| `hotpotqa` | `extract_text_answer` | `qa_f1_score` | f1 |
| `math500` | `extract_boxed_or_text` | `math_equal` | accuracy |
| `olympiadbench` | `extract_boxed_or_text` | `math_equal` | accuracy |

`UNSUPPORTED = {"bfcl", "codeqa", "cs1qa", "hle"}` — these are skipped with a warning.

`evaluate_one(trajectory, dataset_name)` handles OlympiadBench multi-answer (split on `"; "`).

### `evaluate_trajectories.py` (rewritten)
CLI entry point replacing the previous skeleton.

```
python evaluate_trajectories.py --trajectory_dir trajectories/<dir>
python evaluate_trajectories.py --trajectory_file <path> --dataset <name>
python evaluate_trajectories.py --trajectory_dir <dir> --output results.json
```

- Auto-detects dataset from directory name (3rd underscore-delimited segment).
- Reads concatenated JSON objects (the existing trajectory format).
- Prints per-entry results + aggregate summary to stdout.
- Writes detailed JSON to `<trajectory_dir>/eval_results.json` (or `--output`).

### `tests/test_eval.py`
39 unit tests covering all extractors, comparators, and the registry dispatcher.

## Files NOT modified

- `main.py` — no changes to the generation pipeline.
- `parsing.py` — unchanged; the eval extractors reuse its regex pattern independently.

## Dependencies

- `sympy` + `antlr4-python3-runtime==4.11` — required only for `math_equal()` (Math500, OlympiadBench). All other datasets work without these.

## What remains (from i28 notes)

Per the plan, the following datasets are not yet supported and need new code:
- **BFCL** — JSON/AST comparison of function calls
- **CodeQA / CS1QA** — free-form code QA; likely needs LLM-as-judge
- **HLE** — mixed answer types; needs routing + LLM-as-judge for open-ended
