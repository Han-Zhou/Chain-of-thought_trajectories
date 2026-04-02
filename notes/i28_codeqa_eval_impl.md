# CodeQA Evaluation Harness Implementation

Reference repo: `../codeqa/` (codeBERT baseline)

## Repository Structure

```
codeqa/
├── README.md
├── data_sample/
│   ├── java/      # .question, .answer, .code files
│   └── python/
└── codeBERT/
    └── code/
        ├── run.py          # Main training/evaluation script
        ├── model.py        # Seq2Seq model
        ├── bleu.py         # Standalone BLEU utility
        └── eval/
            ├── bleu/       # Google BLEU + NLTK BLEU
            ├── rouge/      # ROUGE-L
            └── meteor/     # METEOR (Java subprocess)
```

## Data Format

Each language (Java/Python) has paired files:
- `.question` — free-form questions about code
- `.answer` — reference answers
- `.code` — processed code snippets (`.code.original` for raw version)

Example:
- **Q:** "What does the code count?"
- **A:** "how many marks existed in string"
- **Code:** A method counting substring occurrences

## Metrics (7 total, all scaled 0–100)

| Metric | Description |
|---|---|
| EM | Binary exact match after normalization |
| BLEU-4 | Google's smoothed corpus BLEU with brevity penalty |
| ROUGE-L | LCS-based F-score (beta=1.2) |
| METEOR | Java subprocess (`meteor-1.5.jar`) |
| Precision | Token-level: `common_tokens / prediction_tokens` |
| Recall | Token-level: `common_tokens / gold_tokens` |
| F1 | Harmonic mean of token precision/recall |

## Evaluation Flow

```
predictions → normalize_answer() → {EM, token F1, BLEU, ROUGE-L, METEOR}
```

Orchestrator: `eval_accuracies()` in `run.py` (lines ~591–625).

```python
eval_accuracies(hypotheses, references)
    ├─→ corpus_bleu(hypotheses, references)
    ├─→ Rouge().compute_score(references, hypotheses)
    └─→ Meteor().compute_score(references, hypotheses)
```

## Answer Normalization

Simple lowercase + whitespace collapse. No punctuation stripping or stemming.

```python
def normalize_answer(s):
    return white_space_fix(lower(s))

def white_space_fix(text):
    return ' '.join(text.split())
```

## Answer Extraction (Beam Search)

```python
preds = model(source_ids=source_ids, source_mask=source_mask)
for pred in preds:
    t = pred[0].cpu().numpy()   # top beam
    if 0 in t:
        t = t[:t.index(0)]      # truncate at EOS
    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
```

- `--beam_size` default 10, evaluates top-1 only.

## Token-Level Matching

Uses `Counter` intersection — bag-of-words, order-insensitive.

```python
prediction_tokens = normalize_answer(prediction).split()
ground_truth_tokens = normalize_answer(ground_truth).split()
common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
num_same = sum(common.values())

precision = num_same / len(prediction_tokens) if prediction_tokens else 0
recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```

## Exact Match

```python
EM = (normalize_answer(prediction) == normalize_answer(ground_truth))
```

## Multi-Reference Handling

Picks the gold reference that maximizes F1:

```python
for gt in ground_truths:
    _EM, _prec, _rec, _f1 = eval_score(prediction, gt)
    if _f1 > f1:
        EM, precision, recall, f1 = _EM, _prec, _rec, _f1
```

## BLEU Details (google_bleu.py)

- Extracts unigrams through 4-grams
- Clipped counts to prevent over-matching
- Brevity penalty: `exp(1 - ref_len / hyp_len)` when hypothesis shorter than reference
- Smoothing: adds 1 to both numerator and denominator to avoid zero precision
- Reference length selection: picks reference closest in length to hypothesis

## ROUGE-L Details (rouge.py)

- Computes LCS for each reference
- Returns max precision and recall across references
- F-score: `(1 + 1.2^2) * P * R / (R + 1.2^2 * P)`

## METEOR Details (meteor.py)

- Java subprocess: `java -jar meteor-1.5.jar`
- Communicates via stdin/stdout
- Thread-safe with locking
- Provides sentence-level and corpus-level scores

## Training vs Testing Evaluation

**Training:** samples 1000 random dev examples per epoch; uses BLEU for checkpoint selection.

**Testing:** full test set; computes all 7 metrics; writes predictions and gold to files.

## Checkpoint Selection

Three checkpoints saved:
1. `checkpoint-last` — most recent epoch
2. `checkpoint-best-ppl` — lowest dev perplexity
3. `checkpoint-best-bleu` — highest dev BLEU (primary)

## Notable Design Choice: BLEU Implementation

Both NLTK BLEU and Google BLEU are included. **Google BLEU is used as primary.**

From `run.py` comments:
> CodeBERT smoothed BLEU-4 produces much higher scores than "A Transformer-based Approach for Source Code Summarization" paper. In CodeQA, we use the latter (Google BLEU) for consistency with the original CodeQA paper.

## Final Output Format

```
dev set: bleu = %.2f | rouge_l = %.2f | meteor = %.2f |
         EM = %.2f | Precision = %.2f | Recall = %.2f | F1 = %.2f
```

All scores returned as 0–100 range:

```python
return (EM.avg * 100, bleu * 100, rouge_l * 100, meteor * 100,
        precision.avg * 100, recall.avg * 100, f1.avg * 100)
```
