## Goal

This repo contains preliminary scripts for the paper "Statistical Testing of Large Language Model Reasoning". It concerns 2 parts:

1. Generate Chain-of-Thought trajectories for the following datasets:
   1. BFCL v1
   2. BigBench Movie Recommendation
   3. BigBench Causal Judgement
   4. LogiQA
   5. CodeQA
   6. CS1QA
   7. HotPotQA
   8. Math500

2. Evaluate confidence scores via the following methods:
   1. Answer token perplexity
   2. Answer token entropy
   3. Indirect logit
   4. Verbalized confidence


Using the following models:
```
llama3.1-8B (Instruct)
qwen3.5/27B
```

## Pipeline Overview

The pipeline runs end-to-end from a dataset name to scored trajectories:

`main.py` → `dataloader/` → `prompts/load.py` → `llm.py` → `parsing.py` → `eval/registry.py` → `confidence.py`

1. **Data loading** (`dataloader/`) — each dataset has a `BaseBenchmarkDataset` subclass that normalizes HuggingFace data into `{id, question, answer}` dicts. Entry point: `load_dataset(name)`.
2. **Prompt assembly** (`prompts/load.py`) — `load_messages()` looks up `(system_prompt, user_template, assistant_start)` from `PROMPT_REGISTRY`, fills entry-specific fields, and applies model/flag-specific overrides.
3. **Generation** (`llm.py`) — the `LLM` class wraps `AutoModelForCausalLM` and exposes `generate_one()` (chat-template-based) and `generate_raw()` (raw prompt string), returning a `GenerationResult` with text, token positions, and KV cache.
4. **Parsing** (`parsing.py`) — `parse_output()` splits on `Step N:` markers, finds `\boxed{answer}` via brace-matching, and records character offsets for answer boundaries.
5. **Evaluation** (`eval/registry.py`) — `evaluate_one()` dispatches to `(extractor, comparator)` tuples for standard datasets or to dedicated evaluators for BFCL / CodeQA / CS1QA.
6. **Confidence scoring** (`confidence.py`) — `compute_all_confidence_scores()` produces vanilla, dropout, and optional jackknife confidence scores, switching attention implementation between generation and confidence modes.

## Setup

```bash
conda activate cot
module load cuda/12.4
pip install -r requirements.txt
```

Place HuggingFace and OpenAI tokens in a local `.env` file (loaded via `python-dotenv`).

## Running the Pipeline

### Generate trajectories

```bash
python3 main.py --dataset math500 --model llama --type 2 --shot_mode zero --max_new_tokens 20000 --experimental_extra_box --confidence
```
some more official scripts can be viewed in `scripts/`


Common flags:

- `--dataset` — one of `bfcl`, `bigbench_movie`, `bigbench_causal`, `logiqa`, `codeqa`, `cs1qa`, `hotpotqa`, `math500`
- `--model` — `llama`, `qwen`
- `--type` — `1` (assistant prefill with `Step 1:`) (used by qwen) or `2` (free-form, `Let's think step-by-step.` appended to user) (used by llama)
- `--shot_mode` — `zero` (few-shot not working yet)
- `--temperature`, `--max_new_tokens`, `--nb_cot_samples`
- `--confidence` — enable confidence scoring
- `--experimental_extra_box` — two-phase generation; we generate content and then append "Therefore the answer is \boxed{"

### Evaluate existing trajectories

```bash
python3 evaluate_trajectories.py --trajectory_dir trajectories/<dir_name>
python3 evaluate_trajectories.py --trajectory_dir <hle_dir> \
    --judge_model o3-mini-2025-01-31 --num_workers 100
```

The dataset is auto-detected from the directory name's 3rd underscore-delimited segment.

### Slurm

Cluster jobs are launched via `sbatch` scripts under `scripts/`.

## Trajectory Output

Each sample writes `trajectories/<dir_name>/traj_N.json` containing the full record: question, ground truth, prompt, parsed CoT steps, final answer, generated text, token positions, confidence scores, and eval result.

Directory naming convention:

```
{tag}_{model}_{thinking|regular}_{dataset}_{shot}_{start}_{end}_type{N}_{conf|vanilla}_{jackknife_mode}_mod-{modification}
```


## Repository Layout

```
main.py                  # pipeline entry point
evaluate_trajectories.py # offline evaluation of saved trajectories
llm.py                   # LLM wrapper (HF AutoModelForCausalLM)
parsing.py               # CoT step + boxed answer extraction
confidence.py            # vanilla / dropout / jackknife confidence
dataloader/              # one BaseBenchmarkDataset subclass per dataset
prompts/                 # PROMPT_REGISTRY, system / user templates, few-shot
eval/                    # extractors, comparators, dataset-specific evaluators
utils/                   # enums, model utils, data structures
scripts/                 # Slurm launch scripts
tests/                   # pytest suite
trajectories/            # generated outputs
```

## Extending

**New dataset.**
1. Add a `BaseBenchmarkDataset` subclass under `dataloader/` and register it in `dataloader/__init__.py`.
2. Add a prompt template in `prompts/cot_prompt.py` and register it in `PROMPT_REGISTRY` (and `DATASET_ANSWER_FORMAT` if relevant).
3. Add the dataset name to `_SUPPORTED_MESSAGE_DATASETS` in `prompts/load.py`; handle any custom fields in `load_messages()`.
4. Register an evaluator: either an `(extractor, comparator, metric)` tuple in `eval/registry.py`'s `EVAL_REGISTRY`, or a dedicated evaluator dispatched from `evaluate_one()`.

**New model.**
1. Add an entry to `MODEL_DICT` in `utils/enum.py`.
2. If attention handling differs, update `_resolve_attn_implementation()` in `llm.py`.
3. If the model uses thinking tokens, add them to `THINKING_TOKENS` in `prompts/few_shot_prompt.py`.



