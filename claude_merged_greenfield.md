# Merged Greenfield Plan — `cot-eval`

This document merges two independent design reviews of the project (`claude_greenfield.md` and `codex_greenfield.md`) into a single plan. The two reviews reached broadly the same target architecture from different angles; the merge keeps the strongest analysis from each and resolves the small places where they disagreed.

The design goal, restated: **high cohesion and low coupling**. The project already has most of the right raw ingredients — registries in `dataloader/__init__.py:23` and `eval/registry.py:25`, a normalized dataset base in `dataloader/base.py:11`, typed result objects in `utils/structures.py:8`. The problem is that variability is encoded *procedurally* in central files (`main.py`, `confidence.py`) instead of being encapsulated behind interfaces. The fix is to evolve the patterns that already exist, not to rewrite.

---

## Diagnosis

### D1. `main.py` is a god-module (~1200 lines)

It owns CLI parsing, cross-flag validation, runtime orchestration, pregenerated-entry resolution, three model-specific generation variants, parsing post-processing, confidence scoring wrap, evaluation dispatch, envelope construction, directory naming, persistence, and two parallel error-recovery paths. That is every layer of the pipeline stuffed into one file.

Key sites:
- `parse()` at `main.py:57-395` — CLI + validation + defaults
- `generate_trajectories()` at `main.py:627-1071` — the workflow engine
- `generate_qwen_outside_cot()` / `generate_continuation()` at `main.py:533-624` — model-family specializations
- `_load_pregenerated_source_entries()` at `main.py:480-516` — codeqa special case
- `main.py:1128-1152` — ad-hoc directory-name assembly

### D2. Flag explosion (root cause of most coupling)

Fourteen mutually-exclusive `experimental_*` booleans drive:

- O(n²) pairwise XOR validation at `main.py:332-395`
- A 25-parameter `generate_trajectories()` signature at `main.py:627`
- A 12-parameter `compute_all_confidence_scores()` signature at `confidence.py:293-313`
- A six-way nested ternary for the `confidence_method` label at `main.py:967`
- A *second* six-way ternary for the directory suffix `_jk_suffix` at `main.py:1128-1141`

These are really one variable — the confidence *method* — plus a couple of orthogonal modifiers (modification, extra_box, preserve_final_step). Treating them as independent booleans means every layer of the pipeline has to re-derive the same enum.

### D3. Dataset-specific logic is scattered across unrelated layers

Adding a benchmark today requires edits in:

- `dataloader/__init__.py:23` — the loader registry
- `prompts/load.py:150-183` — an `if/elif` chain per dataset
- `eval/registry.py:70-88` — five special-case `if` branches before the table lookup
- `main.py:480-516` — codeqa-specific pregenerated resolution
- `main.py:932-942` — `if dataset_name == "hle"` branch in the orchestrator
- `main.py:977-980` — `if dataset_name == "bfcl"` tacks extra fields onto the trajectory

A benchmark is not a single unit of change. This is high coupling that forces cross-layer edits for any benchmark-local decision.

### D4. Model-family logic is also scattered

Qwen / GPT / Llama quirks live in five places:

- `llm.py:61-72` — attention-implementation selection
- `llm.py:211-220` and `llm.py:311-318` — chat-template post-processing (duplicated across HF and vLLM backends)
- `prompts/load.py:40-47` — thinking-token resolution
- `main.py:712-735` — generation branching on model family
- `main.py:734-814` — parsing branching on model family
- `confidence.py:2151` — prompt re-rendering for confidence scoring

`LLM` exists but is not the only home for model-specific invariants. Adding a model or changing its chat template means edits across several files.

### D5. Confidence scoring is a framework hidden in one 2865-line file

`compute_all_confidence_scores()` at `confidence.py:293` coordinates eight distinct algorithms (vanilla, coinflip, coinflip-true, jackknife-default, jackknife-true, bootstrap, fake-bootstrap, hollow) through flag branches. The file also holds three near-duplicate dispatch functions (`dropout_answerlogits`, `dropout_indirectlogits`, `dropout_verbalconf`) that differ only in the target-logit extraction step — each takes ~15 parameters and repeats the same loop skeleton.

### D6. Prompt rendering is duplicated — and this is a silent correctness hazard

The same conversation is rendered in two places:

- `llm.py:145-220` for generation
- `confidence.py:2151` (`_build_prompt_text`) for confidence scoring

Because `apply_chat_template` can normalize whitespace, the confidence module has to round-trip through the tokenizer at `confidence.py:352-361` to realign step boundaries after generation. This is not just duplication — it is a real drift hazard where a change to generation-time rendering silently miscomputes confidence offsets.

### D7. Shared contracts are implicit dicts

Dataset entries, trajectories, eval payloads, and prompt inputs all flow as raw `dict[str, Any]` objects. Concretely:

- `evaluate_one` returns `{"score": float}` for most datasets but `{"score": {"em": bool, "f1": float}}` for hotpotqa, forcing an `isinstance(..., dict)` branch at `main.py:944-952`.
- Trajectory dicts have drifting schemas across three construction sites (see D8).

Dict-based contracts are fine early on; once the system spans multiple datasets and experiment modes, the hidden key-level contracts become a coupling surface that nothing enforces.

### D8. Trajectory envelope is constructed three times with schema drift

At `main.py:954-991` (success), `main.py:1000-1030` (sample-level failure), and `main.py:1039-1063` (entry-level failure), three independent dict literals build what is nominally the same JSON envelope. They have drifted:

- **Schema drift.** Success site has 20 keys including `debug_info`. Failure sites drop `debug_info` and add `error`. Adding a new field to the success site requires remembering to add `None` entries in both failure sites — nothing enforces this.
- **Clock drift.** Success and sample-failure use `time.time() - t_sample`; entry-failure uses `time.time() - t0`. `runtime_seconds` is not on a consistent clock.
- **Filename bug.** Entry-level failure writes `traj_{i}.json` unconditionally, ignoring `nb_cot_samples > 1`. During a multi-sample run, an entry-level exception at sample 0 writes `traj_5.json` instead of `traj_5_sample_0.json` — and subsequent sample writes use a different filename, so the error record is orphaned.
- **Dataset-specific injection happens only on success.** `if dataset_name == "bfcl": traj["func_description"] = ...` at `main.py:977-980` runs only on the success path, so BFCL failure records have a different shape than success records.

### D9. Lifecycles leaking into the caller

`main.py:830-892` does the attention-implementation dance manually:

```python
llm.switch_attn_implementation("confidence")
try:
    ...compute_all_confidence_scores(...)
finally:
    llm.switch_attn_implementation("cot")
```

The orchestrator should not know about attention-implementation modes. Same pattern applies to the `deepcopy` / crop logic for `gen_cache` at `confidence.py:400-407` and the manual `del gen.past_key_values; torch.cuda.empty_cache()` at `main.py:928-930`.

---

## Highest-leverage fix (Phase 0): `ConfidenceMode` discriminated union

Every other refactor gets easier if the flag bag collapses first. Introduce a closed hierarchy:

```python
# confidence/modes.py
class ConfidenceMode(Protocol):
    label: str                                 # replaces main.py:967 ternary
    dir_suffix: str                            # replaces main.py:1128-1141 ternary
    def run(self, ctx: ConfidenceContext) -> ConfidenceScores: ...

@dataclass(frozen=True)
class CoinFlip(ConfidenceMode):        nb_samples: int = 3
@dataclass(frozen=True)
class VanillaOnly(ConfidenceMode):     pass
@dataclass(frozen=True)
class Jackknife(ConfidenceMode):       variant: Literal["default", "true"]; grid: bool = False
@dataclass(frozen=True)
class Bootstrap(ConfidenceMode):       nb_samples: int = 100
@dataclass(frozen=True)
class FakeBootstrap(ConfidenceMode):   use_hook: bool = False; nb_samples: int = 100
@dataclass(frozen=True)
class CoinFlipTrue(ConfidenceMode):    nb_samples: int = 3
@dataclass(frozen=True)
class Hollow(ConfidenceMode):          mode: Literal[1, 2]
```

A single factory at the CLI boundary does all XOR validation:

```python
def build_confidence_mode(args) -> ConfidenceMode: ...  # one place for every mutual-exclusion check
```

Consequences across the codebase:
- `parse()` drops all N×N pairwise checks; the factory raises one descriptive error.
- `generate_trajectories(confidence_mode=CoinFlip(3), ...)` shrinks from ~25 params to <10.
- `main.py:967` becomes `mode.label`.
- `main.py:1128-1141` becomes `mode.dir_suffix`.
- `compute_all_confidence_scores` becomes a dispatcher: `return mode.run(ctx)`.
- Adding a new method = adding one class. No edits to CLI, orchestrator, or output naming.

This is explicitly Phase 0 — before service extraction or benchmark/model refactors — because it shrinks every downstream signature and removes the need to thread seven booleans through four layers.

---

## Core abstractions

### A1. `BenchmarkSpec` — one object per dataset

Consolidate benchmark variability behind a single protocol:

```python
class BenchmarkSpec(Protocol):
    name: str
    loader_cls: type[BaseBenchmarkDataset]

    def build_messages(self, example: BenchmarkExample, request: PromptRequest) -> list[dict[str, str]]: ...
    def evaluate(self, trajectory: TrajectoryRecord) -> EvalResult | None: ...
    def extra_trajectory_fields(self, example: BenchmarkExample) -> dict[str, Any]: ...
```

Concrete implementations live in `benchmarks/{logiqa,codeqa,bfcl,hotpotqa,hle,...}.py`. One `BENCHMARK_REGISTRY: dict[str, BenchmarkSpec]`.

This removes the `if dataset_name == "hle"` branches in the orchestrator, the `if/elif` chain in `prompts/load.py`, and the special cases in `eval/registry.py`. Adding a benchmark becomes one file with one registration.

**Design note (correction to one review):** pregenerated-entry resolution should **not** be a method on `BenchmarkSpec`. Only codeqa needs it, and it's an I/O concern, not a benchmark-semantics concern. Model it as a separate `PregeneratedSource` that a benchmark optionally composes:

```python
@dataclass
class BenchmarkSpec:
    pregenerated_source: PregeneratedSource | None = None
```

### A2. `ModelAdapter` — one object per model family

All Qwen / GPT / Llama quirks behind a single adapter:

```python
class ModelAdapter(Protocol):
    family: Literal["llama", "qwen", "gpt"]
    model_name: str

    def render_messages(self, messages, *, continue_final: bool) -> str: ...   # owns <think>/<|channel|> normalization
    def thinking_tokens(self) -> tuple[str, str]: ...                           # moves from prompts/load.py:40
    def generation_attn_impl(self) -> str: ...                                  # moves from llm.py:61
    def confidence_attn_impl(self) -> str: ...                                  # moves from llm.py:131
    def phase1_stop_strings(self, thinking: bool) -> list[str]: ...             # moves from utils.model_utils
    def create_cache(self, config) -> DynamicCache: ...                         # Qwen3_5DynamicCache branching
    def post_parse_normalize(self, full_generated_text: str, prompt_type: int) -> ParsedOutput: ...
```

Concrete: `models/{llama_adapter,qwen_adapter,gpt_adapter}.py`. Registry keyed by model name.

This collapses the model-family branching at `main.py:712`, `main.py:734-814`, `llm.py:211-220`, `llm.py:311-318`, `prompts/load.py:40-47`, and `confidence.py:2151` into one object per family. Downstream code uses `adapter.render_messages(...)` and stops calling `is_qwen_model_name()` / `is_llama_model_name()`.

### A3. `ConversationRenderer` — single source of prompt truth

Both `llm.py:145-220` and `confidence.py:2151` render conversations. They must not diverge. Extract one renderer that both paths call:

```python
class ConversationRenderer:
    def __init__(self, adapter: ModelAdapter, tokenizer): ...
    def render(self, messages, *, continue_final: bool) -> RenderedPrompt: ...

@dataclass
class RenderedPrompt:
    text: str
    token_ids: torch.Tensor
    prompt_len: int
    # boundary markers the confidence module needs to realign step offsets
    assistant_start_char: int
    assistant_start_token: int
```

Generation and confidence both consume `RenderedPrompt`. The awkward round-trip at `confidence.py:352-361` where the text is re-encoded and re-decoded to realign step boundaries disappears — there is no drift to correct, because both paths started from the same render.

This is the fix I initially undersold and that `codex_greenfield.md` correctly flagged as a silent correctness hazard.

### A4. Confidence methods as registered strategies

Split `confidence.py` into a package:

```
confidence/
  base.py              # ConfidenceMode ABC, ConfidenceContext, Template Method scaffold
  engine.py            # shared cache/tokenization utilities (crop_cache, early/late split)
  methods/
    vanilla.py
    coinflip.py
    coinflip_true.py
    jackknife.py       # both default and true variants
    bootstrap.py
    fake_bootstrap.py
    hollow.py
  probes/              # orthogonal to methods
    answer_logits.py   # replaces dropout_answerlogits
    indirect_ptrue.py  # replaces dropout_indirectlogits
    verbalized.py      # replaces dropout_verbalconf
  sampling/
    bootstrap.py       # _bootstrap_sample_steps etc.
    jackknife.py
    coinflip.py
  masking.py           # attention-mask weighting, forward hooks
  reconstruction.py    # _reconstruct_cot_text, _renumber_step, _strip_thinking
  debug.py             # all _debug_* helpers
```

Template Method in `base.py`:

```python
class ConfidenceMode(ABC):
    def run(self, ctx, probe: Probe) -> ProbeScores:
        samples = self._sample(ctx)                   # mode-specific
        inputs  = self._materialize(ctx, samples)     # tokens/mask/hook/recon-text
        with ctx.forward_scope(inputs) as out:
            return probe.extract(out)
```

This collapses the three near-duplicate dispatchers (`dropout_answerlogits`, `dropout_indirectlogits`, `dropout_verbalconf`) into one generic loop parameterized by a `Probe`. Method × Probe = matrix of 8×3 combinations, but implementation is 8 + 3 classes — not 24 branches.

### A5. Service layer — `TrajectoryRunner` as composition root

```
pipeline/
  runner.py            # TrajectoryRunner: iter + try/except + sink
  context.py           # PerSampleContext dataclass
  services/
    generation.py      # GenerationService (wraps GenerationStrategy)
    confidence.py      # ConfidenceService (owns attn_mode context, dispatches to ConfidenceMode)
    evaluation.py      # EvaluationService (dispatches to BenchmarkSpec.evaluate)
```

The runner becomes ~40 lines:

```python
for i, entry in self.iter(dataloader):
    try:
        ctx = self.build_context(i, entry)
        gen = self.generation.run(ctx)
        parsed = self.generation.parse(ctx, gen)
        conf = self.confidence.run(ctx, gen, parsed)    # no-op if mode is None
        eval_ = self.evaluation.run(ctx, parsed)
        self.sink.write(ctx, gen, parsed, conf, eval_)
    except Exception as e:
        self.sink.write_failure(ctx, e)
```

---

## Supporting infrastructure

### S1. `GenerationStrategy` — replace the `if/elif` at `main.py:702-724`

```python
class GenerationStrategy(Protocol):
    def generate(self, llm, messages, **opts) -> GenerationTriple: ...

class PlainGeneration: ...
class QwenOutsideCoTGeneration: ...       # from generate_qwen_outside_cot
class ExtraBoxContinuation: ...           # from generate_continuation
class PregeneratedReplay: ...             # from the from_pregenerated branch
```

Strategy is chosen once per run from `(model_family, thinking, extra_box, from_pregenerated)`. The orchestrator calls `strategy.generate(...)` without branching.

### S2. `TrajectorySink` and `TrajectoryRecord` — fix the triplication

Kill the three drifting envelope literals at `main.py:954-991`, `1000-1030`, `1039-1063`.

```python
@dataclass
class TrajectoryRecord:
    id: str
    question: str
    ground_truth: Any
    # all fields with None defaults — one schema for success and failure
    prompt: str | None = None
    cot_steps: list[str] | None = None
    final_answer: str | None = None
    raw_cot_block: str | None = None
    generated_text: str | None = None
    prompt_end_position: int | None = None
    generated_end_position: int | None = None
    answer_token_start_position: int | None = None
    confidence_score: dict | None = None
    confidence_method: str | None = None
    confidence_modification: str | None = None
    confidence_error: str | None = None
    confidence_timings: dict | None = None
    debug_info: dict | None = None
    runtime_seconds: float | None = None
    generation_seconds: float | None = None
    correct: bool | dict | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)    # bfcl func_description etc. — present on success AND failure

    @classmethod
    def success(cls, ...) -> "TrajectoryRecord": ...
    @classmethod
    def failure(cls, source_entry, error, elapsed) -> "TrajectoryRecord": ...

class TrajectorySink:
    def __init__(self, out_dir, multi_sample: bool): ...
    def write(self, index: int, sample_idx: int | None, record: TrajectoryRecord): ...
    def write_cot_cache(self, ...): ...
    def write_errors(self, errors: list[dict]): ...
    def _path(self, index, sample_idx) -> str: ...    # one place picks the filename
```

Fixes from D8 baked in:
- One schema — success and failure cannot diverge.
- `runtime_seconds` is on the same clock everywhere; both factory methods take `elapsed` as a parameter.
- Entry-level failures during multi-sample runs get `traj_{i}_sample_{s}.json`, not `traj_{i}.json`.
- `extra` dict is populated by `BenchmarkSpec.extra_trajectory_fields(example)` on *both* paths.

### S3. `EvalResult` — normalize the score contract

Replace the dict-or-float dispatch at `main.py:944-952`:

```python
@dataclass
class EvalResult:
    metric: str
    score: float                    # primary scalar for `correct`
    components: dict[str, float] = field(default_factory=dict)   # em/f1/etc.
    prediction: str | None = None
    ground_truth: str | None = None
    extraction_failed: bool = False
```

HotpotQA returns `EvalResult(metric="em+f1", score=em, components={"em": em, "f1": f1})`. Orchestrator reads `score` to set `correct`, writes `components` to the trajectory. `isinstance(..., dict)` branch disappears.

### S4. `attn_mode` context manager on `LLM`

Replace `main.py:830-892`:

```python
class LLM:
    @contextmanager
    def attn_mode(self, mode: Literal["cot", "confidence"]):
        prev = self._current_mode
        self._switch(mode)
        try:
            yield
        finally:
            self._switch(prev)

# caller:
with llm.attn_mode("confidence"):
    scores = confidence_service.run(ctx)
```

Same principle for `gen_cache` ownership in `confidence.py:400-407`:

```python
with owned_early_cache(gen_cache, split) as cache:
    ...
```

And for generation-cache cleanup in `main.py:928-930`:

```python
with gen.borrowed_cache():
    scores = ...
# cache freed automatically on exit
```

### S5. `PipelineConfig` — parameter object

Collapse 19 lines of kwargs at `main.py:627` and the 20-line argparse body into one frozen dataclass:

```python
@dataclass(frozen=True)
class PipelineConfig:
    model: ModelAdapter
    benchmark: BenchmarkSpec
    shot_mode: Literal["zero", "few"]
    prompt_type: Literal[1, 2]
    thinking: bool
    max_new_tokens: int
    temperature: float
    nb_cot_samples: int
    cot_sample_idx: int | None
    backend: GenerationBackend
    confidence_mode: ConfidenceMode | None           # replaces 7 booleans
    modification: Modification | None                # "no-cots" | "gibberish"
    extra_box: bool
    preserve_final_step: bool
    from_pregenerated: Path | None
    debug: bool
    debug_conf: bool

    def dir_name(self) -> str: ...                    # replaces main.py:1143-1151
```

### S6. Validation separated from parsing

`parse()` currently does argparse + validation + defaults + flag-to-object coercion. Split:

- `cli/parser.py::build_parser()` — argparse only.
- `cli/config.py::from_args(args) -> PipelineConfig` — validation, defaults, mutual-exclusion, coercion.

Validation becomes unit-testable without subprocess or argparse mocking.

### S7. `BenchmarkExample` and `PromptRequest` DTOs

Replace the raw dicts flowing between loader, prompt builder, and evaluator:

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
    extra_box: bool = False
```

Raw dicts are confined to the I/O boundary (loader reads HF, emits `BenchmarkExample`; sink writes `TrajectoryRecord` as JSON).

### S8. Small fixes that clean up loose ends

- **`HollowMode` enum** replaces `experimental_hollow: int` with magic 1/2 at `main.py:277-289`.
- **`PregeneratedSource` class** consolidates `_answer_lookup_value` + `_build_dataset_lookup` + `_merge_pregenerated_entry` + `_resolve_pregenerated_source_entry` (`main.py:425-516`). Codeqa gets a `CodeqaPregeneratedSource` subclass.
- **Timing via `Timer` context manager** replaces the 18-line manual projection at `main.py:872-889`. A `TimingRegistry` populated by scoped `with timer("vanilla_meanprob"):` blocks; `asdict(timings)` replaces the manual field-by-field copy.
- **Circular import** — `from utils.structures import ConfidenceTimings` inside `compute_all_confidence_scores` at `confidence.py:428` resolves naturally when the module is split per A4.

---

## Target architecture

```text
cot_eval/
  main.py                        # ~30 lines: parser → config → runner.run(config)
  cli/
    parser.py
    config.py                    # PipelineConfig + ConfidenceMode factory + validation

  pipeline/
    runner.py                    # TrajectoryRunner — composition root
    context.py                   # PerSampleContext
    services/
      generation.py
      confidence.py
      evaluation.py

  benchmarks/
    base.py                      # BenchmarkSpec protocol + BenchmarkExample DTO
    registry.py
    logiqa.py, codeqa.py, bfcl.py, hotpotqa.py, hle.py, ...

  models/
    base.py                      # ModelAdapter protocol
    registry.py
    llama_adapter.py, qwen_adapter.py, gpt_adapter.py
    llm.py                       # LLM + attn_mode context manager
    backends/
      hf.py, vllm.py             # GenerationBackend impls
    rendering.py                 # ConversationRenderer

  generation/
    strategy.py                  # GenerationStrategy + selector
    plain.py, qwen_outside.py, extra_box.py, pregenerated_replay.py

  parsing/
    strategy.py                  # ParsingStrategy + selector
    default.py, qwen_thinking.py, prompt_type2.py

  confidence/
    base.py, engine.py
    modes/, probes/, sampling/
    masking.py, reconstruction.py, debug.py

  evaluation/
    result.py                    # EvalResult DTO
    comparators.py, extractors.py

  io/
    trajectory_sink.py           # TrajectorySink + TrajectoryRecord
    debug_cache_store.py
    pregenerated_store.py        # PregeneratedSource base + subclasses

  domain/
    models.py                    # shared DTOs
```

Coupling after the refactor:
- `main.py` depends on `cli`, `pipeline.runner` only.
- `TrajectoryRunner` depends on abstractions: `BenchmarkSpec`, `ModelAdapter`, `ConfidenceMode`, `GenerationStrategy`, `EvalResult`, `TrajectorySink`. No concrete strategies.
- Adding a confidence method = one file under `confidence/modes/`. No edits to `cli`, `pipeline`, or `io`.
- Adding a benchmark = one file under `benchmarks/`. No edits to `prompts/`, `eval/`, or `main.py`.
- Adding a model = one file under `models/`. No edits anywhere else.

---

## Migration plan (6 phases, each independently shippable)

### Phase 0 — Collapse the flag explosion

1. Introduce `ConfidenceMode` discriminated union (§A4 types, before the package split).
2. Add `build_confidence_mode(args)` factory in CLI.
3. Thread `confidence_mode: ConfidenceMode` through `generate_trajectories` and `compute_all_confidence_scores`.
4. Replace the two ternaries at `main.py:967` and `main.py:1128-1141` with `mode.label` / `mode.dir_suffix`.

Output: trajectories bit-identical. Signatures shrink immediately. Every downstream refactor gets cheaper.

### Phase 1 — Stabilize contracts

1. `PipelineConfig` + `from_args` split (§S5, §S6).
2. `TrajectoryRecord` + `TrajectorySink` (§S2) — this phase also fixes the entry-failure filename bug from D8.
3. `EvalResult` DTO (§S3) — removes the `isinstance(..., dict)` branch.
4. `BenchmarkExample` + `PromptRequest` DTOs (§S7).

Output: `main.py` loses the three envelope blocks and the `isinstance` branch; schemas are enforced by dataclasses.

### Phase 2 — Unify model behavior and prompt rendering

1. `ModelAdapter` protocol (§A2), one adapter per family.
2. `ConversationRenderer` (§A3) — the key fix that `codex_greenfield.md` flagged.
3. `GenerationBackend` strategy (HF / vLLM) replaces the `if backend == "vllm"` dispatchers in `llm.py`.
4. `LLM.attn_mode` context manager (§S4).

Output: all model-family string-sniffing (`is_qwen_model_name()`, `is_llama_model_name()`) is gone. Prompt rendering lives in exactly one place.

### Phase 3 — Consolidate benchmark ownership

1. `BenchmarkSpec` protocol + `BenchmarkRegistry` (§A1).
2. Migrate one benchmark end-to-end as a template — suggest `logiqa` (most structured) or `math500` (simplest).
3. Port remaining benchmarks one at a time; delete their branches from `prompts/load.py`, `eval/registry.py`, and `main.py`.
4. `PregeneratedSource` consolidates codeqa's special case (§S8).

Output: `prompts/load.py` and `eval/registry.py` shrink to registry lookups; `main.py` loses all `if dataset_name == "..."` branches.

### Phase 4 — Extract services and runner

1. `TrajectoryRunner` as composition root (§A5).
2. `GenerationService`, `ConfidenceService`, `EvaluationService`.
3. `GenerationStrategy` replaces the `if/elif` at `main.py:702-724` (§S1).
4. `main.py` drops to ~30 lines.

Output: every layer of the pipeline owns one responsibility. Orchestration is testable in isolation.

### Phase 5 — Split `confidence.py` into a package

1. Extract shared utilities into `confidence/engine.py` first (cache crop, early/late split, tokenization).
2. Move sampling helpers into `confidence/sampling/`.
3. Port each forward-pass function to a `ConfidenceMode` class in `confidence/modes/`.
4. Collapse the three dispatchers (`dropout_answerlogits`, `dropout_indirectlogits`, `dropout_verbalconf`) into one `Probe`-parameterized loop in `base.py`.
5. Move debug helpers to `confidence/debug.py`.

This is last because it is the biggest edit, and Phase 0 + Phase 2 have already separated the *selection* of a method from its *implementation* — so this phase is pure code motion inside a settled interface.

Output: a 2865-line file becomes ~15 files averaging <200 lines, each with one responsibility. New confidence methods become drop-in additions.

---

## Highest-impact recommendations

If budget forces a subset, do these five in order:

1. **Phase 0** — `ConfidenceMode` discriminated union. Removes flag explosion everywhere.
2. `ConversationRenderer` — fixes the silent prompt-rendering drift between generation and confidence (the one thing most likely to be a latent correctness bug).
3. `ModelAdapter` — single home for model-family quirks; stops string-sniffing.
4. `BenchmarkSpec` — each dataset becomes one unit of change.
5. `TrajectoryRecord` + `TrajectorySink` — fixes the multi-sample filename bug and ends schema drift.

These five changes remove the great majority of current cross-module coupling. The confidence-package split (Phase 5) is valuable but can safely wait until the interfaces around it have settled.

---

## Final assessment

This codebase is not far from a cleaner architecture. It already has registries, a dataset base, useful dataclasses, and a real test surface. The problem is that variability is encoded procedurally in central files instead of being encapsulated behind interfaces.

The greenfield principle, stated once:

> Put benchmark variation behind benchmark objects, model variation behind model adapters, confidence variation behind confidence strategies, and keep the runner dumb. Collapse flag bags into discriminated unions at the CLI boundary so the pipeline sees one object, not seven booleans.

That is the cleanest path to high cohesion and low coupling here.
