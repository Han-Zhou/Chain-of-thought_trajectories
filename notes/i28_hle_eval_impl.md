# HLE (Humanity's Last Exam) Evaluation Harness - Implementation Details

Source repo: `/storage/backup/han/backup_workspace/hle/`

HLE uses a **two-stage pipeline**: model prediction generation → LLM judge-based evaluation.

---

## Dependencies

```
numpy==1.26.4
openai==1.57.0
datasets==3.2.0
```

---

## Stage 1: Model Prediction Generation

**File:** `hle_eval/run_model_predictions.py`

### System Prompt (line 11)

```
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

### Pipeline

1. **Load dataset** from Hugging Face (`datasets.load_dataset(args.dataset, split="test")`)
2. **Format questions** via `format_message(question)` (lines 13-28):
   - Creates `[system_message, user_message]` pair
   - User message includes text + optional image URL (multimodal)
   - For o1 models: system role set to `"user"` (no system prompt support)
3. **Async batch inference** via `attempt_all(questions)` (lines 53-63):
   - Rate-limited by `asyncio.Semaphore(args.num_workers)`
   - Each call via `attempt_question()` (lines 31-50) → OpenAI API
   - Returns `(question_id, response_content, token_usage)` or `None` on failure
4. **Caching**: Skips already-answered questions (resumable runs)

### Output Format

```json
{
  "question_id": {
    "model": "model_name",
    "response": "Explanation: ...\nAnswer: ...\nConfidence: ...",
    "usage": {
      "prompt_tokens": int,
      "completion_tokens": int,
      "total_tokens": int
    }
  }
}
```

### CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | required | HuggingFace dataset ID |
| `--model` | required | Model endpoint name |
| `--max_completion_tokens` | None | Max tokens (recommended 8192+) |
| `--temperature` | 0.0 | Sampling temperature |
| `--num_workers` | 10 | Async concurrency limit |
| `--max_samples` | None | Limit to first N samples |

---

## Stage 2: Judge-Based Evaluation

**File:** `hle_eval/run_judge_results.py`

### Judge Prompt (lines 16-33)

Instructs the judge to:
1. Extract `extracted_final_answer` from the model response (or `"None"` if unclear)
2. Compare extracted answer to the provided `[correct_answer]`
3. Output `reasoning` explaining match/mismatch
4. Decide `correct: "yes"/"no"` — "yes" if match or within small margin for numerical problems
5. Extract `confidence` score (0-100%) from model response

### Structured Output (lines 35-40)

```python
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]
```

Uses OpenAI's `client.beta.chat.completions.parse()` for structured output.

### Answer Extraction

`extract_answer(question, correct_answer, response)` (lines 43-64):
- Calls judge with `max_completion_tokens=4096`
- Returns:
  ```python
  {
    "correct_answer": str,
    "model_answer": str,  # extracted_final_answer
    "reasoning": str,
    "correct": "yes" or "no",
    "confidence": int  # 0-100
  }
  ```

### Answer Comparison Logic

The comparison is **entirely delegated to the LLM judge**:
- Judge compares `extracted_final_answer` to `[correct_answer]`
- Matching: exact match OR "within a small margin of error for numerical problems"
- Correctness extraction: `"yes" in judge_response["correct"]` → boolean

### Metrics Computation

**`dump_metrics(predictions, n)`** (lines 129-155):

1. Collect correctness and confidence from all judged predictions:
   ```python
   correct_i = "yes" in judge_response["correct"]  # boolean
   confidence_i = judge_response["confidence"] / 100  # normalize to [0,1]
   ```

2. Compute metrics:

| Metric | Formula | Line |
|--------|---------|------|
| **Accuracy** | `100 * sum(correct) / n` | 148 |
| **95% CI** (Wald) | `1.96 * sqrt(accuracy * (100 - accuracy) / n)` | 150 |
| **Calibration Error** (L2) | `100 * calib_err(confidence, correct, p='2', beta=100)` | 151 |

**Note:** Accuracy denominator is total dataset size `n`, not just judged samples. This allows partial evaluation runs.

### Calibration Error

`calib_err(confidence, correct, p='2', beta=100)` (lines 97-127):

Source: [hendrycks/outlier-exposure](https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py)

Algorithm:
1. Sort predictions by confidence score
2. Partition into bins of size `beta` (default 100)
3. For each bin:
   - `mean_conf = mean(confidence_in_bin)`
   - `mean_correct = mean(correctness_in_bin)`
   - `diff = |mean_conf - mean_correct|`
   - `weighted = (bin_size / total) * diff^p`
4. Aggregate:
   - `p='2'` → `sqrt(sum(weighted))`
   - `p='1'` → `sum(weighted)`
   - `p='infty'` → `max(weighted)`

### Pipeline Orchestration (`main()`, lines 158-200)

1. Load dataset + model predictions
2. Load existing judged predictions if present (resumable)
3. Filter to unjudged predictions
4. Call `judge_all_responses()` async (Semaphore with `num_workers`)
5. Merge results, save to `judged_{input_filename}.json`
6. Call `dump_metrics()`

### CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | required | HuggingFace dataset ID |
| `--predictions` | required | Path to stage 1 predictions JSON |
| `--num_workers` | 100 | Async concurrency limit |
| `--judge` | `"o3-mini-2025-01-31"` | Judge model |

---

## Full Pipeline

```
Stage 1: run_model_predictions.py
  Load HF dataset → format questions (text+image) → async OpenAI API calls
  → save predictions JSON (resumable)

Stage 2: run_judge_results.py
  Load predictions → async judge calls (structured output) → extract answers
  → compare to ground truth → compute accuracy + CI + calibration error
  → save judged JSON + print metrics
```

### Example Usage

```bash
cd hle_eval

MODEL="gpt-4o-2024-11-20"
DATASET="cais/hle"

# Stage 1
python run_model_predictions.py \
  --dataset ${DATASET} --model ${MODEL} \
  --max_completion_tokens 8192 --num_workers 100

# Stage 2
python run_judge_results.py \
  --dataset ${DATASET} \
  --predictions hle_${MODEL}.json \
  --num_workers 100
```

Expected output:
```
*** Metrics ***
Accuracy: 3.07% +/- 0.65% | n = 2700
Calibration Error: 92.3
```

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| Judge max tokens | 4096 | run_judge_results.py:48 |
| Calibration bin size (beta) | 100 | run_judge_results.py:102 |
| Calibration p-norm | `'2'` (L2) | run_judge_results.py:151 |
| CI z-score | 1.96 (95%) | run_judge_results.py:150 |
| Default judge model | `o3-mini-2025-01-31` | run_judge_results.py:198 |
| Default temperature | 0.0 | run_model_predictions.py:110 |

---

## Data Format

### HF Dataset Fields
- `id`: str
- `question`: str
- `image`: str (URL or empty `""`)
- `answer`: str (ground truth)

### Judged Output
```json
{
  "question_id": {
    "model": "...",
    "response": "...",
    "usage": {...},
    "judge_response": {
      "correct_answer": "ground truth",
      "model_answer": "extracted answer",
      "reasoning": "why match/mismatch",
      "correct": "yes" or "no",
      "confidence": 92
    }
  }
}
```
