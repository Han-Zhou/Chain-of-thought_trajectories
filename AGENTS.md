# Repository Guidelines

## Project Structure & Module Organization

`main.py` is the primary entry point for generating chain-of-thought trajectories. Dataset adapters live in `dataloader/`, prompt builders in `prompts/`, evaluation and extraction logic in `eval/`, and shared model/data structures in `utils/`. Root-level utilities such as `evaluate_trajectories.py`, `confidence.py`, and `parsing.py` support evaluation and analysis. Add automated tests under `tests/`. Experiment launchers are grouped in `scripts/`, while `trajectories/`, `pickles/`, `debug_cache/`, `.slurm_logs/`, notebooks, and `figures_roc/` contain data or generated artifacts.

## Setup, Test, and Development Commands

- `python -m pip install -r requirements.txt` installs the pinned research environment; GPU packages require a compatible CUDA system.
- `python main.py --dataset logiqa --model llama --sample_size 4` runs a small local trajectory-generation smoke test.
- `python evaluate_trajectories.py --trajectory_dir trajectories/<run>` evaluates an existing run.
- `pytest -q` runs the complete test suite.
- `pytest tests/test_parsing.py -v` runs one focused module while debugging.
- `black . && isort .` formats Python imports and source. Review the diff because notebooks and archived research code may not need mechanical changes.

There is no separate build step. Cluster experiments use the SLURM wrappers in `scripts/`; copy an existing campaign script and adjust its dataset, model, resource request, and output tag.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and modules, `PascalCase` for classes, and uppercase names for constants. Prefer type hints for new public helpers and short docstrings for non-obvious parsing, confidence, or evaluation behavior. Keep dataset-specific behavior in its loader or evaluator rather than adding more branching to unrelated modules.

## Testing Guidelines

Tests use `pytest` and follow `tests/test_<feature>.py`; test functions should describe behavior, for example `test_parse_json_function_call_extracts_embedded_payload`. Add regression tests for parsing edge cases, evaluator scoring, prompt changes, and model-independent utilities. No coverage threshold is configured, so prioritize representative success, failure, and malformed-input cases.

## Commit & Pull Request Guidelines

Recent history uses short, lowercase, imperative or descriptive subjects (for example, `new figures` and `slurm helpers`). Make subjects more specific when possible and keep each commit scoped to one concern. Pull requests should explain the experiment or bug, list validation commands, identify affected datasets/models, and note generated artifacts. Include plots or metric summaries for evaluation changes; avoid committing caches, logs, model outputs, `.env`, API keys, or webhook URLs.
