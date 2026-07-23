# Greenfield Design Report

Starting from [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:57), the main architectural issue is that the project's variability is organized around flags and `if/elif` branches instead of stable interfaces. The code works, but `main.py` and `confidence.py` have both become control centers for too many concerns.

One terminology fix up front: the usual design goal is **high cohesion and low coupling**.

The good news is that the codebase already has the right raw ingredients:

- Registries in [dataloader/__init__.py](/storage/backup/han/backup_workspace/cot-eval/dataloader/__init__.py:23) and [eval/registry.py](/storage/backup/han/backup_workspace/cot-eval/eval/registry.py:25)
- A normalized dataset base in [dataloader/base.py](/storage/backup/han/backup_workspace/cot-eval/dataloader/base.py:11)
- Typed result objects in [utils/structures.py](/storage/backup/han/backup_workspace/cot-eval/utils/structures.py:8)

The right direction is to evolve those patterns rather than rewrite the project.

## Current Structural Problems

### 1. `main.py` is doing too much

`main.py` currently handles all of the following:

- CLI parsing and validation in [parse()](/storage/backup/han/backup_workspace/cot-eval/main.py:57)
- Runtime orchestration in [main()](/storage/backup/han/backup_workspace/cot-eval/main.py:1088)
- Generation flow in [generate_trajectories()](/storage/backup/han/backup_workspace/cot-eval/main.py:627)
- Pregenerated entry resolution in helpers like [_load_pregenerated_source_entries()](/storage/backup/han/backup_workspace/cot-eval/main.py:480)
- Model-specific generation branches such as [generate_qwen_outside_cot()](/storage/backup/han/backup_workspace/cot-eval/main.py:533) and [generate_continuation()](/storage/backup/han/backup_workspace/cot-eval/main.py:582)
- Parsing dispatch, confidence scoring, evaluation, persistence, and error recovery

This is low cohesion. The file is nominally an entrypoint, but it is also a workflow engine, an experiment router, a persistence layer, and a special-case host for dataset and model behavior.

### 2. Dataset-specific logic is scattered

Dataset behavior is spread across multiple modules:

- Dataset loading registry: [dataloader/__init__.py](/storage/backup/han/backup_workspace/cot-eval/dataloader/__init__.py:23)
- Prompt construction: [prompts/load.py](/storage/backup/han/backup_workspace/cot-eval/prompts/load.py:140)
- Evaluation dispatch: [eval/registry.py](/storage/backup/han/backup_workspace/cot-eval/eval/registry.py:53)
- Special cases in the runner: [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:480), [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:932), and [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:977)

That means a new benchmark is not a single unit of change. To add or modify one dataset, you often need to touch multiple files in unrelated layers. That is higher coupling than necessary.

### 3. Model-family behavior is also scattered

Qwen, GPT, and Llama behavior appears in several places:

- Model loading and generation behavior in [llm.py](/storage/backup/han/backup_workspace/cot-eval/llm.py:61)
- Prompt formatting in [prompts/load.py](/storage/backup/han/backup_workspace/cot-eval/prompts/load.py:40)
- Generation-time branches in [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:712)
- Parsing-time branches in [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:734)
- Confidence-time prompt rebuilding in [confidence.py](/storage/backup/han/backup_workspace/cot-eval/confidence.py:2151)

This means the project does not currently have a single model abstraction. `LLM` exists, but it is not the only place where model-specific invariants live.

### 4. Confidence scoring is a framework hidden inside one module

[compute_all_confidence_scores()](/storage/backup/han/backup_workspace/cot-eval/confidence.py:293) is effectively coordinating several different algorithms:

- Vanilla
- Coin-flip
- Jackknife default
- Jackknife true
- Bootstrap
- Fake bootstrap
- Fake bootstrap hook
- Hollow

Those are separate policies with different runtime behavior, but today they are mostly expressed as flags and branches inside a single orchestration function. That is low cohesion at the function level and high coupling to CLI/config flags.

### 5. Shared contracts are too implicit

The system passes around raw dicts for:

- Dataset entries
- Trajectories
- Evaluation payloads
- Prompt inputs

This shows up in modules like [dataloader/base.py](/storage/backup/han/backup_workspace/cot-eval/dataloader/base.py:33), [prompts/load.py](/storage/backup/han/backup_workspace/cot-eval/prompts/load.py:150), and [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:954).

That style is flexible early on, but once the system spans multiple datasets and experiment modes, the hidden key-level contracts increase coupling between modules.

## Design Patterns and Solutions

### 1. Make `main.py` the composition root only

`main.py` should wire objects together and start the run. It should not own the business logic.

Recommended split:

- `config.py`
  - `RunConfig`
  - `GenerationConfig`
  - `ConfidenceConfig`
  - `OutputConfig`
- `pipeline/runner.py`
  - `TrajectoryRunner`
- `pipeline/services/generation_service.py`
  - `GenerationService`
- `pipeline/services/confidence_service.py`
  - `ConfidenceService`
- `pipeline/services/evaluation_service.py`
  - `EvaluationService`
- `io/trajectory_store.py`
  - JSON and pickle persistence

Pattern: **service layer + functional core / imperative shell**

Why this improves cohesion:

- `main.py` becomes responsible only for wiring and invocation
- Each service owns one job
- Experimental logic stops leaking into the entrypoint

### 2. Introduce a `BenchmarkSpec` per dataset

Today the benchmark concern is fragmented. Instead, each benchmark should own its full lifecycle behavior.

Recommended interface:

```python
class BenchmarkSpec(Protocol):
    name: str

    def load_examples(self) -> Sequence[BenchmarkExample]: ...
    def build_messages(self, example: BenchmarkExample, request: PromptRequest) -> list[dict[str, str]]: ...
    def evaluate(self, trajectory: TrajectoryRecord) -> EvaluationResult | None: ...
    def resolve_pregenerated(self, example: BenchmarkExample, index: int, source: PregeneratedSource) -> BenchmarkExample: ...
```

Concrete implementations:

- `benchmarks/logiqa.py`
- `benchmarks/codeqa.py`
- `benchmarks/bfcl.py`
- `benchmarks/hotpotqa.py`
- `benchmarks/hle.py`

Pattern: **Strategy + Registry**

Why this improves cohesion:

- Each benchmark becomes one unit of behavior
- Adding a dataset no longer requires touching prompt code, eval registry, and runner branches separately
- Benchmark-specific quirks stay local

### 3. Introduce a `ModelAdapter`

The project needs a stable abstraction for model-family quirks beyond just `LLM.generate_one()`.

Recommended responsibilities:

- Chat template rendering
- Assistant prefill rules
- Thinking format handling
- Prompt normalization
- Stop-string policy
- Cache type selection
- Parsing normalization hooks

Recommended interface:

```python
class ModelAdapter(Protocol):
    model_name: str

    def render_messages(self, messages: list[dict[str, str]], *, continue_final: bool) -> str: ...
    def normalize_generated_text(self, text: str) -> str: ...
    def parse_generated_output(self, full_text: str, request: ParseRequest) -> ParsedOutput: ...
    def phase1_stop_strings(self, request: GenerationRequest) -> list[str] | None: ...
    def create_cache(self, config) -> object: ...
```

Concrete adapters:

- `models/llama_adapter.py`
- `models/qwen_adapter.py`
- `models/gpt_adapter.py`

Pattern: **Adapter / Bridge**

Why this improves cohesion:

- Qwen/GPT/Llama logic is centralized
- `main.py`, `prompts/load.py`, and `confidence.py` stop duplicating model assumptions
- New models become a localized extension instead of a cross-codebase patch

### 4. Split confidence methods into registered strategies

`confidence.py` is the clearest place where flag-driven branching should become object-driven behavior.

Recommended structure:

- `confidence/base.py`
  - `ConfidenceMethod`
- `confidence/methods/vanilla.py`
- `confidence/methods/coinflip.py`
- `confidence/methods/jackknife_mask.py`
- `confidence/methods/jackknife_reconstruct.py`
- `confidence/methods/bootstrap.py`
- `confidence/methods/fake_bootstrap.py`
- `confidence/methods/hollow.py`
- `confidence/engine.py`
  - Shared cache/tokenization utilities

Pattern: **Strategy + Template Method**

Why this improves cohesion:

- Each method owns one algorithm
- Shared logic stays in an engine/helper layer
- The orchestration layer becomes simpler and easier to test
- Experimental methods can be added without inflating one 400-line function

### 5. Create a single prompt rendering path

There is duplicated prompt building logic in:

- [llm.py](/storage/backup/han/backup_workspace/cot-eval/llm.py:145)
- [confidence.py](/storage/backup/han/backup_workspace/cot-eval/confidence.py:2151)

That is dangerous because the same conversation can be rendered differently in generation and in confidence scoring.

Recommended solution:

- Add `ConversationRenderer`
- Make both generation and confidence call the same renderer
- Keep all model-template normalization in the adapter layer

Pattern: **Builder / Single Source of Truth**

Why this improves cohesion:

- Rendering rules live in one place
- Confidence scoring stops depending on replicated generation logic
- Bugs from template drift become less likely

### 6. Replace raw dicts with typed DTOs

Recommended types:

```python
@dataclass
class BenchmarkExample:
    id: str
    question: str
    answer: Any
    context: str | None = None
    choices: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptRequest:
    few_shot: bool
    prompt_type: int
    thinking: bool
    experimental_extra_box: bool = False

@dataclass
class TrajectoryRecord:
    id: str
    prompt: str | None
    generated_text: str | None
    cot_steps: list[str] | None
    final_answer: str | None
    correct: bool | dict | None
    confidence: dict | None
```

Pattern: **DTO / Value Object**

Why this improves cohesion:

- Modules stop depending on undocumented dict keys
- Static reasoning becomes easier
- Refactors become safer

### 7. Move persistence behind repositories

Artifact writing is currently embedded in the generation loop in [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:897), [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:923), and [main.py](/storage/backup/han/backup_workspace/cot-eval/main.py:990).

Recommended abstractions:

- `TrajectoryRepository`
- `DebugCacheRepository`
- `PregeneratedTrajectoryRepository`

Pattern: **Repository / Ports-and-Adapters**

Why this improves cohesion:

- The runner no longer knows file naming and serialization details
- Storage policy can change independently
- Tests can mock persistence cleanly

### 8. Replace large argument lists with configuration objects

[generate_trajectories()](/storage/backup/han/backup_workspace/cot-eval/main.py:627) has too many parameters. [compute_all_confidence_scores()](/storage/backup/han/backup_workspace/cot-eval/confidence.py:293) also exposes a large experimental surface directly.

Recommended fix:

- `GenerationConfig`
- `ConfidenceConfig`
- `SamplingConfig`
- `ArtifactConfig`

Pattern: **Parameter Object**

Why this improves cohesion:

- Each function accepts coherent configuration bundles
- Related options are grouped
- Validation becomes centralized instead of spread across branches

## Proposed Greenfield Package Shape

```text
cot_eval/
  main.py
  config.py
  pipeline/
    runner.py
    contexts.py
    services/
      generation_service.py
      confidence_service.py
      evaluation_service.py
  benchmarks/
    base.py
    registry.py
    logiqa.py
    bfcl.py
    codeqa.py
    hotpotqa.py
    hle.py
  models/
    base.py
    registry.py
    llm_client.py
    llama_adapter.py
    qwen_adapter.py
    gpt_adapter.py
    rendering.py
  confidence/
    base.py
    engine.py
    methods/
      vanilla.py
      coinflip.py
      jackknife_mask.py
      jackknife_reconstruct.py
      bootstrap.py
      fake_bootstrap.py
      hollow.py
  parsing/
    cot_parser.py
  evaluation/
    registry.py
    comparators.py
    extractors.py
  io/
    trajectory_store.py
    debug_cache_store.py
    pregenerated_store.py
  domain/
    models.py
```

## Practical Migration Plan

This can be done incrementally without a risky rewrite.

### Phase 1: Stabilize interfaces

1. Extract `RunConfig` from `argparse`
2. Extract `TrajectoryRunner` from `main.py`
3. Move JSON/pickle writing into a store class

Expected benefit:

- `main.py` shrinks quickly
- No behavior change required yet

### Phase 2: Unify rendering and model behavior

1. Introduce `ModelAdapter`
2. Add `ConversationRenderer`
3. Make `llm.py` and `confidence.py` share the same rendering logic

Expected benefit:

- Removes duplicated model-specific invariants
- Reduces fragile prompt/token mismatch behavior

### Phase 3: Consolidate benchmark ownership

1. Introduce `BenchmarkSpec`
2. Migrate one benchmark first, ideally `logiqa` or `codeqa`
3. Remove dataset-specific branches from `prompts/load.py`, `eval/registry.py`, and `main.py`

Expected benefit:

- Each dataset becomes self-contained
- New benchmarks become easier to add

### Phase 4: Break apart confidence strategies

1. Keep shared cache/tokenization helpers in a common engine
2. Move each method into a separate class/module
3. Replace flag branches with strategy dispatch

Expected benefit:

- Easier testing
- Cleaner extension path for new confidence experiments

### Phase 5: Tighten domain contracts

1. Replace dicts with typed DTOs
2. Make services exchange typed domain objects
3. Limit raw dicts to I/O boundaries only

Expected benefit:

- Lower coupling between modules
- Fewer hidden assumptions

## What High Cohesion and Low Coupling Should Mean in This Project

If the design is healthy, then:

- `main.py` only wires dependencies and launches runs
- A benchmark module owns its own loading, prompting, and evaluation semantics
- A model adapter owns all model-family quirks
- The parser only parses
- Confidence methods only implement confidence algorithms
- Persistence code only reads and writes artifacts

That is the target separation.

## Highest-Impact Recommendations

If you only do a few things, do these first:

1. Turn `main.py` into a composition root
2. Introduce `BenchmarkSpec`
3. Introduce `ModelAdapter`
4. Refactor confidence methods into strategies
5. Unify prompt rendering into one shared path

Those five changes would remove most of the current cross-module coupling.

## Final Assessment

This codebase is not far from a cleaner architecture. It already has registry patterns, a dataset base abstraction, useful dataclasses, and a real test surface in `tests/`. The main problem is that variability is encoded procedurally in central files instead of being encapsulated behind interfaces.

The greenfield principle for this project should be:

> Put benchmark variation behind benchmark objects, model variation behind model adapters, confidence variation behind confidence strategies, and keep the runner dumb.

That is the cleanest path to high cohesion and low coupling here.
