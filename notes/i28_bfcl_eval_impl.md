# BFCL Evaluation Harness Implementation

Reference repo: `../gorilla/berkeley-function-call-leaderboard/`

Scope: `simple_python`, `multiple`, `parallel`, `parallel_multiple` categories only.

## Repository Structure

```
berkeley-function-call-leaderboard/
└── bfcl_eval/
    ├── __main__.py                       # CLI entry point
    ├── _llm_response_generation.py       # Model response generation
    ├── constants/
    │   ├── enums.py                      # ModelStyle, Language, ReturnFormat
    │   ├── eval_config.py                # Paths and configs
    │   └── category_mapping.py           # Test category definitions
    ├── eval_checker/
    │   ├── eval_runner.py                # Main evaluation orchestrator
    │   ├── eval_runner_helper.py         # Accuracy calculation and scoring
    │   └── ast_eval/
    │       └── ast_checker.py            # Function call matching logic
    ├── model_handler/
    │   └── parser/                       # Output parsers (JSON, XML, Java, JS)
    └── data/
        ├── BFCL_v4_*.json               # Test prompts and functions
        └── possible_answer/              # Ground truth answers
```

## Relevant Categories

| Category | What it tests |
|---|---|
| `simple_python` | Single function call with correct name + params |
| `multiple` | Multiple function calls in sequence |
| `parallel` | Multiple function calls in parallel (unordered) |
| `parallel_multiple` | Mixed parallel and sequential calls |

All are single-turn, non-live, AST-evaluated.

## Evaluation Flow

```
For each model:
  For each category in {simple_python, multiple, parallel, parallel_multiple}:
    Load model responses + test prompts + ground truth
    ast_file_runner()
    accuracy = correct_count / total_count
    Save to score/{MODEL_NAME}/{category}/BFCL_v4_*_score.json
```

## Output Parsing (Python)

Model output → `decode_ast` → list of dicts with function names + params.

ReturnFormat.PYTHON: extracts `[{"func_name": {"param": value, ...}}, ...]`

## Matching and Scoring Logic

### Simple Function Checker (`simple_function_checker`)

Core checker used by all four categories. Per function call:

1. **Function name** — exact match (dots → underscores for OpenAI/Mistral/Google)
2. **Required params** — all must be present; missing required → FAIL
3. **Param type checking** — uses PYTHON_TYPE_MAPPING (string→str, integer→int, etc.); int→float auto-conversion allowed
4. **Param value validation:**
   - **Strings:** standardized comparison — removes spaces + punctuation (`,./\-_*^`), lowercased, single→double quotes
   - **Numerics:** exact match (int→float auto-conversion for Python)
   - **Lists:** standardize elements, check membership
   - **Dicts:** all keys present, no unexpected keys, validate each value; optional keys marked with `""`
   - **List of dicts:** validate count and each dict
5. **Variable detection** — if both possible_answer and value are string type → treated as variable reference, passes validation
6. **Optional params** — marked with `""` in ground truth; missing is OK, unexpected params → FAIL

### Category-Specific Checkers

**`simple_python`** — single call, uses `simple_function_checker` directly.

**`multiple`** (`multiple_function_checker`):
- Count must match exactly
- All calls for same function
- Validates first function (sequence implied)

**`parallel`** (`parallel_function_checker_no_order`):
- No ordering enforcement
- Greedy matching with backtracking against unmatched model outputs
- Each expected function must match some model output
- Any unmatched → FAIL with detailed error

**`parallel_multiple`** — uses `parallel_function_checker_no_order`, combining both parallel and sequential semantics.

## String Standardization (`standardize_string`)

```
Remove: spaces, comma, period, slash, dash, underscore, asterisk, caret
Convert to lowercase
Single quotes → double quotes
```

Purpose: handle formatting variations without penalizing models.

## Metric

Per-category:
```
accuracy = correct_count / total_count
```

Non-Live AST Acc (for these 4):
```
avg(simple_python, multiple, parallel, parallel_multiple)
```

## Score File Format

```json
[
  {"accuracy": 0.95, "correct_count": 19, "total_count": 20},
  {"id": "simple_python_1", "valid": false, "error": ["Wrong function name"], "error_type": "..."}
]
```

## Key Design Decisions

- **AST-based checking** — parses function calls into structured form, not string matching
- **Lenient string comparison** with extensive normalization
- **Optional param handling** via empty string markers in ground truth
- **Type auto-conversion:** int→float, tuple→list (JSON limitation)
- **Unordered parallel matching** with backtracking for `parallel` / `parallel_multiple`
