# i28: Reusable Extraction & Evaluation Methods from lm-evaluation-harness

## Context

Our project generates CoT trajectories for 11 datasets. The pipeline (`main.py`) currently saves `final_answer` and `ground_truth` into trajectory JSON files but **never compares them** — there is no evaluation/grading step. `parsing.py` only handles MCQ letter extraction (`Final Answer: [A-D]`).

This note catalogues what extraction and evaluation methods already exist in the bundled `lm-evaluation-harness/` that we can reuse.

---

## 1. Answer Extraction (from CoT model output)

### 1a. MCQ letter extraction — `"Final Answer: B"`

**Already implemented** in our `parsing.py`. Regex: `final\s+answer\s*:?\s*[\(\[]?\s*([A-Da-d])\s*[\)\]]?`

Applies to: **LogiQA, College_Math (MMLU-Pro), BB Causal Judgement**

### 1b. `\boxed{}` extraction

Extracts the content of the **last** `\boxed{...}` in a response using brace-matching. Also handles `\fbox{}` and `\boxed <space>` variants.

**Best source to copy:** `lm_eval/tasks/score/math/math_grader.py:564` → `extract_answer()`
- Supports `\boxed{}` with correct nested-brace parsing
- Falls back to a configurable regex (default: `r"The final answer is (.+)$"`)
- Single self-contained function, no class dependencies

**Alternative sources** (simpler but less robust):
- `lm_eval/tasks/minerva_math/utils.py:99` → `last_boxed_only_string()` + `remove_boxed()`
- `lm_eval/tasks/hendrycks_math/utils.py:67` → same pattern, copy-pasted

Applies to: **Math500, OlympiadBench** (and College_Math if answers are open-form)

### 1c. BBH CoT regex — `"the answer is (.*)."`

Used by the BBH CoT fewshot tasks. Defined in `lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml:24`:
```yaml
filter_list:
  - name: "get-answer"
    filter:
      - function: "regex"
        regex_pattern: "(?<=the answer is )(.*)(?=.)"
      - function: "take_first"
```

Applies to: **BB Movie Recommendation, BB Causal Judgement** (if we prompt with BBH-style "So the answer is ...")

### 1d. `$...$` dollar-sign extraction

Extracts text between the first and last `$` delimiters. Used as a fallback when `\boxed{}` isn't present.

Source: `lm_eval/tasks/hendrycks_math/utils.py:20-24` and `lm_eval/tasks/aime/utils.py:10-14`

### 1e. `MultiChoiceRegexFilter`

General-purpose MCQ letter extractor in `lm_eval/filters/extraction.py:126`. Tries:
1. Primary regex match
2. Fuzzy match against full choice text
3. Bare letter match (`r":[\s]*(A|B|C|D)"`)

Applies to: any MCQ dataset as a robust fallback

---

## 2. Answer Comparison / Grading

### 2a. Exact match (string equality)

```python
predicted.strip().upper() == ground_truth.strip().upper()
```

Applies to: **LogiQA** (letter), **BB Causal Judgement** (Yes/No), **College_Math** (letter)

### 2b. `math_equal()` — numeric + symbolic equivalence

**Source:** `lm_eval/tasks/score/math/math_grader.py:378`

The most sophisticated math comparison. Handles:
- **String normalization** — strips units, `\text{}`, `\left`/`\right`, intervals, mixed numbers, commas
- **Numeric equality** — both sides parse to float, checked with `isclose(rel_tol=1e-4)`, including percentage variants (x, x/100, x*100)
- **Symbolic equality** — parses via `sympy.parse_expr` and `parse_latex`, then checks `simplify(a - b) == 0`
- **Tuple/interval/matrix comparison** — recursively compares elements
- **Timeout protection** — `signal.ITIMER_REAL` based, configurable

Dependencies: `sympy`, `antlr4-python3-runtime==4.11`

Applies to: **Math500, OlympiadBench**

### 2c. `is_equiv()` — lighter sympy equivalence

**Source:** `lm_eval/tasks/minerva_math/utils.py:159` (or `hendrycks_math/utils.py:36`)

Simpler than `math_equal()`: normalizes both strings with `strip_string()`, then compares via `parse_latex` + `sympy.simplify(diff) == 0`. No numeric tolerance, no tuple handling.

Two variants exist:
- `minerva_math` version: uses `parse_latex` + sympy simplification (heavier, more correct)
- `hendrycks_math` version: uses `strip_string()` for normalization then string equality (lighter, less correct)

### 2d. `math_verify` library

**Source:** used in `lm_eval/tasks/leaderboard/math/utils.py:73-90`

```python
from math_verify import parse, verify
verify(gold=parse(doc["solution"]), target=parse(candidates))
```

Third-party library that handles LaTeX parsing and verification. Used alongside the manual methods as a secondary check.

### 2e. `qa_f1_score` — token-level F1

**Source:** `lm_eval/tasks/longbench/metrics.py` → `qa_f1_score(prediction, ground_truth)`

Standard token-level F1 between predicted and gold answer strings. Tokenizes by whitespace, computes precision/recall/F1.

Applies to: **HotPotQA** (and potentially CodeQA, CS1QA as a baseline)

---

## 3. Coverage Matrix

| Dataset | Answer Type | Extraction | Comparison | Both in harness? |
|---------|-------------|------------|------------|-----------------|
| BFCL v1 | JSON function calls | **None** | **None** | No |
| BB Movie Rec | MCQ text (movie titles) | BBH regex / `MultiChoiceRegexFilter` | exact match on choice text | Partial |
| BB Causal Judgement | Yes/No | BBH regex | exact match | Yes |
| LogiQA | MCQ letter (A-D) | `parsing.py` (ours) | exact match | Yes (trivial) |
| CodeQA | Free-form text | **None** | **None** | No |
| CS1QA | Free-form text | **None** | **None** | No |
| HotPotQA | Free-form text | needs CoT-aware extraction | `qa_f1_score` | Partial |
| College_Math | MCQ letter (MMLU-Pro) | `parsing.py` (ours) | exact match | Yes (trivial) |
| OlympiadBench | Numeric/expression | `extract_answer()` (`\boxed{}`) | `math_equal()` | Yes |
| Math500 | Numeric/expression | `extract_answer()` (`\boxed{}`) | `math_equal()` / `math_verify` | Yes |
| HLE | Mixed (MCQ + open) | per-type routing needed | MCQ: exact match; open: needs LLM-judge | Partial |

---

## 4. What We Need to Build

### Already covered by harness code (copy-paste ready)
- `extract_answer()` for `\boxed{}` → from `score/math/math_grader.py:564`
- `math_equal()` for math grading → from `score/math/math_grader.py:378`
- `qa_f1_score` for HotPotQA → from `longbench/metrics.py`
- Exact-match for all MCQ datasets → trivial

### Gaps (need new code)
- **BFCL** — JSON/AST comparison of function calls (name + arguments)
- **CodeQA / CS1QA** — free-form code QA; likely needs LLM-as-judge or task-specific heuristics
- **HLE open-ended subset** — mixed answer types; needs routing logic + LLM-as-judge for non-MCQ
- **BB Movie Recommendation** — answer is a movie title string, not a letter; need fuzzy matching against choices
- **`parsing.py` extension** — currently only extracts MCQ letters; needs `\boxed{}` path for math datasets

### Integration point
`main.py:227` builds trajectory dicts with `final_answer` and `ground_truth` but never computes `is_correct`. Add a `evaluate(dataset_name, final_answer, ground_truth, entry)` dispatcher that routes to the appropriate comparison function.
