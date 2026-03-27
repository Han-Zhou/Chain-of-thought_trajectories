# CS1QA Evaluation Harness - Implementation Details

Source repo: `/storage/backup/han/backup_workspace/cs1qa/`

CS1QA evaluates 3 subtasks, each with its own pipeline and metrics.

---

## 1. Type Classification

**Files:**
- `src/type_classification/classify_type.py` — main eval pipeline
- `src/type_classification/utils.py` — metrics + data processing

### Question Type Labels (11 classes)

`"code_understanding"`, `"logical"`, `"error"`, `"usage"`, `"algorithm"`, `"task"`, `"comparison"`, `"reasoning"`, `"code_explain"`, `"variable"`, `"guiding"`

### Evaluation Pipeline

`evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev')` (classify_type.py:197)

1. **Load data**: `load_and_cache_examples()` reads JSONL, tokenizes question+code pairs
   - Long code sequences are split at newline boundaries via `_split_long_tokens()` — each chunk gets a separate feature with the same `example_id`
   - Returns TensorDataset: `(input_ids, input_mask, segment_ids, label_ids, example_ids)`

2. **Batch inference**: Sequential sampler, model outputs logits per chunk

3. **Aggregation by example** (classify_type.py:257-265):
   - Multiple chunks per example are aggregated via **majority voting**:
     ```python
     np.bincount(preds_label[example_ids == i]).argmax()
     ```
   - Labels aggregated the same way (all chunks share same label, so mode = that label)

4. **Metric computation**: `compute_metrics(task_name, preds, labels)` → dispatches to `acc_and_f1()`

### Metrics (utils.py:361-389)

| Metric | Formula |
|--------|---------|
| Accuracy | `(preds == labels).mean()` |
| F1 (Macro) | `f1_score(y_true=labels, y_pred=preds, average='macro')` |
| F1 (Weighted) | `f1_score(y_true=labels, y_pred=preds, average='weighted')` |
| Acc+F1 (Macro) | `(acc + f1_macro) / 2` |
| Acc+F1 (Weighted) | `(acc + f1_weighted) / 2` |

**Per-class metrics** from confusion matrix:
- Precision: `conf[i,i] / sum(conf[:,i])`
- Recall: `conf[i,i] / sum(conf[i,:])`
- F1: `2 * precision * recall / (precision + recall)`

### Models
RoBERTa, XLM-RoBERTa, Longformer (all via HuggingFace `*ForSequenceClassification`)

---

## 2. Span Prediction

**Files:**
- `src/span_prediction/select_span.py` — main eval pipeline
- `src/span_prediction/utils.py` — metrics, token-to-line conversion, top-k extraction

### Evaluation Pipeline

`evaluate(args, model, tokenizer, checkpoint=None, prefix="", mode='dev')` (select_span.py:267)

1. **Load data**: JSONL with `startLine`, `endLine` fields per example
   - Features include: `start_index`, `end_index` (ground truth token positions), `b_start_index`, `b_end_index` (code span boundaries), `line_starts`, `example_ids`

2. **Batch inference**: Model outputs `start_logits`, `end_logits`

3. **Top-K index extraction** via `get_top_k_indices(start_logits, end_logits, k)` (select_span.py:229-264):
   - Priority queue over sorted (start, end) logit pairs
   - Priority = `-(sorted_start[i] + sorted_end[j])`
   - Returns list of `(start_idx, end_idx, prob)` tuples

4. **Group by example** (select_span.py:398-413):
   - Chunks from same example grouped together for multi-chunk predictions

5. **Ground truth loading** via `file_to_spans_list()` (select_span.py:209-227):
   - Extracts all unique `(startLine, endLine)` tuples per question
   - Deduplicates identical questions within file

6. **Metric computation**: `compute_metrics()` (utils.py:642-752)

### Token-to-Line Conversion (utils.py:482-555)

`convert_token_index_to_lines()` maps predicted token indices back to source code line numbers by tracking newline tokens through the token sequence. Validates predictions fall within code span bounds.

Returns: `(start_line, end_line, curr_line, end_new_line, prob)` tuples

### Top-K Combination Extraction (utils.py:589-640)

`extract_top_k_combinations()` uses a priority queue to find the k-best combinations of predicted line ranges across multiple code chunks. Sums probabilities for ranking.

### F1 and EM Calculation (utils.py:557-587)

`calculate_F1_EM(selected_intervals, label)`:

```
label_set = set(range(label[0], label[1] + 1))
selection_set = union of all predicted interval line sets

TP = |label_set ∩ selection_set|
FP = |selection_set - label_set|
FN = |label_set - selection_set|

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
EM        = 1 if F1 == 1 else 0
```

### Full Metric Pipeline (utils.py:642-752)

`compute_metrics()`:
1. Convert token predictions → line ranges
2. Extract top-5 combinations
3. Merge adjacent/contiguous line ranges (utils.py:691-702)
4. Calculate F1/EM against **each** ground truth annotation (handles multiple valid answers)
5. Take best F1 across top-5 combinations
6. Optionally compute per-question-type breakdown

Returns: `{F1, EM, F1_type[], EM_type[]}`

---

## 3. Answer Retrieval

**Files:**
- `src/answer_retrieval/modules/encoder/model.py` — model, training, eval
- `src/answer_retrieval/get_bleu.py` — BLEU-1 evaluation

### Model Architecture

`BERTForTextSim` (model.py:62-113) extends `RobertaPreTrainedModel`:
- Backbone: `RobertaModel`
- Head: `Dropout → Linear(hidden_size, hidden_size)`
- Output: pooled embeddings (logits)

### Training Loss (model.py:256-270)

`get_loss(logits, n)`:
```
q_vecs = logits[:n]        # query embeddings
doc_vecs = logits[n:]      # document embeddings
sim = q_vecs @ doc_vecs.T  # similarity matrix
loss = CrossEntropyLoss(sim, [0, 1, ..., n-1])  # diagonal = positive pairs
```

### Evaluation Metrics

**Batch accuracy** (model.py:347-356):
```python
acc = (np.argmax(sim, axis=1) == np.arange(n)) * 1.0
```

**Ranking metrics** via `get_top1()` (model.py:449-480):

| Metric | Formula |
|--------|---------|
| Top-1 | 1 if `argmax(scores) == gt_index` else 0 |
| Top-5 | 1 if `gt_index in top_5_indices` else 0 |
| MRR | `1.0 / (1.0 + rank_of_gt)` |
| Upperbound | 1.0 if `gt_index != -1` else 0.0 (handles unanswerable) |

**Full evaluation** via `eval_model()` (model.py:482-512):
- Computes similarity: `np.dot(query_emb, candidate_embs.T)`
- Aggregates metrics: `np.mean(metric_list) * 100.0`
- Output format: `"Eval: ({top1:.04f}, {top5:.04f}, {mrr:.04f}) / {upperbound:.04f}"`

### BLEU-1 Evaluation (get_bleu.py:13-64)

Alternative text-overlap metric:

```python
bleu = sentence_bleu(
    reference=gt_answer.split(),
    candidate=pred_answer.split(),
    weights=(1, 0, 0, 0)  # unigram only
)
```

- Overall: mean BLEU across all predictions
- Per-type breakdown with ST (student) / TA grouping:
  - ST types (first 6): `code_understanding`, `logical`, `error`, `usage`, `algorithm`, `task`
  - TA types (last 3): `reasoning`, `code_explain`, `variable`
- Question types (9 total, subset of the 11 used in type classification — missing `comparison` and `guiding`)

---

## Data Format

JSONL files with fields:
- `labNo`, `taskNo` — lab/task identifiers
- `questioner` — `"student"` or `"TA"`
- `question` — question text
- `code` — code snippet at time of question
- `questionType` — ground truth label
- `startLine`, `endLine` — ground truth span (for span prediction)

Dataset splits: train, dev, test

---

## Summary Table

| Subtask | Primary Metrics | Aggregation | Key Detail |
|---------|----------------|-------------|------------|
| Type Classification | Acc, F1-macro, F1-weighted | Majority vote across code chunks | 11 question types |
| Span Prediction | F1, EM (line-level) | Top-5 combinations, best F1 across multiple valid annotations | Token→line mapping |
| Answer Retrieval | Top-1, Top-5, MRR | Dot-product similarity ranking | CrossEntropy training loss |
| Answer Retrieval (BLEU) | BLEU-1 | Mean, per-type (ST/TA split) | Unigram overlap only |
