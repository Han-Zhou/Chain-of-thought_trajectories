# Design Review: `cot-eval` — toward high cohesion / low coupling

I read `main.py:1-1204`, `llm.py`, `confidence.py` (structure), `eval/registry.py`, `prompts/load.py`, `dataloader/__init__.py`, and `utils/structures.py`. The pipeline is correct and feature-rich, but `main.py` has become a *god-module* — ~1200 lines mixing CLI, validation, orchestration, model-specific generation strategies, parsing branching, dataset-specific munging, pregenerated-I/O, envelope construction, directory naming, and two almost-duplicate error paths. `confidence.py` is 2865 lines with ~30 top-level functions. Most coupling leaks through **feature flags** (14 `experimental_*` booleans) that are mutually exclusive but encoded as independent fields, forcing O(n²) cross-validation at the edges and nested ternaries at the output.

The fixes below are not stylistic — each one removes a real coupling edge.

---

## 1. The flag-explosion problem (root cause of most coupling)

Evidence — `main.py:332-395` cross-validates:

```
vanilla_only  ⊕ experimental_jackknife ⊕ experimental_grid
              ⊕ experimental_bootstrap ⊕ experimental_fake_bootstrap
              ⊕ experimental_fake_bootstrap_hook ⊕ experimental_coinflip_true
              ⊕ experimental_hollow ⊕ experimental_modification
              ⊕ experimental_preserve_final_step ⊕ experimental_extra_box
```

These are really *one* variable — the confidence method — plus a couple of orthogonal modifiers. The same bag of booleans is then:

- passed into `generate_trajectories()` (25+ params at `main.py:627`)
- re-passed into `compute_all_confidence_scores()` (12 flags at `confidence.py:293-313`)
- rematerialized as a label via a six-way nested ternary at `main.py:967`:
  ```python
  "confidence_method": ("vanilla_only" if vanilla_only else "bootstrap" if ...
  ```
- rematerialized a second time to pick `_jk_suffix` at `main.py:1128-1141` for the output directory name.

### Fix — Discriminated Union + Strategy

Introduce a closed set of *confidence modes* as a sealed hierarchy:

```python
# confidence/modes.py
class ConfidenceMode(Protocol):
    label: str                                 # for filenames / trajectory field
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

And a small factory at the CLI boundary that converts the flag bag into *one* object:

```python
def build_confidence_mode(args) -> ConfidenceMode: ...  # one place for all XOR checks
```

Consequences:
- `parse()` drops all N×N pairwise checks; the factory raises a single descriptive error.
- `generate_trajectories(..., confidence_mode=CoinFlip(3), preserve_final_step=False, ...)` shrinks from ~25 params to <10.
- `main.py:967` becomes `mode.label`.
- `main.py:1128-1141` becomes `mode.dir_suffix`.
- `compute_all_confidence_scores` becomes a dispatcher: `return mode.run(ctx)`.
- New methods register themselves — no flag-table editing.

This is the single highest-leverage refactor. Everything else gets easier afterward.

---

## 2. `generate_trajectories()` is doing five jobs

At `main.py:627-1071` the function holds:

1. Per-entry prompt assembly (delegates to `load_messages`)
2. Three generation variants (plain / Qwen-outside-CoT / extra-box two-phase)
3. Model-specific parsing post-processing (Qwen `</think>` stripping, type-2 `<think>` block, llama cleanup)
4. Confidence scoring wrap + attention-impl toggling
5. Evaluation + trajectory envelope writing + two error-path envelopes

### Fix — Pipeline composed of Strategy collaborators

Extract a small object graph and drive it from a single tight loop:

```
TrajectoryPipeline
├── PromptBuilder            # load_messages is already close
├── GenerationStrategy       # plain | qwen_outside_cot | extra_box | pregenerated_replay
├── OutputParser             # model/prompt-type aware
├── ConfidenceRunner         # owns attn-impl context mgr (see §4)
├── Evaluator                # from EvalRegistry
├── TrajectorySink           # writes traj_{i}[_sample_{s}].json, cot_{i}.pkl, errors.json
└── ProgressReporter         # tqdm | tqdm_discord
```

The driver becomes ~40 lines:

```python
for i, entry in pipeline.iter(dataloader):
    try:
        ctx = pipeline.build_context(i, entry)
        gen = pipeline.generate(ctx)
        parsed = pipeline.parse(ctx, gen)
        conf = pipeline.score_confidence(ctx, gen, parsed)   # no-op if mode is None
        eval_ = pipeline.evaluate(ctx, parsed)
        pipeline.sink.write(ctx, gen, parsed, conf, eval_)
    except Exception as e:
        pipeline.sink.write_failure(ctx, e)
```

### 2a. Generation strategy — replace the `if/elif` chain at `main.py:702-724`

```python
class GenerationStrategy(Protocol):
    def generate(self, llm, messages, **opts) -> GenerationTriple: ...

class PlainGeneration(GenerationStrategy): ...
class QwenOutsideCoTGeneration(GenerationStrategy): ...   # from generate_qwen_outside_cot
class ExtraBoxContinuation(GenerationStrategy): ...       # from generate_continuation
class PregeneratedReplay(GenerationStrategy): ...         # from the from_pregenerated branch

def select_generation_strategy(model_name, thinking, prompt_type,
                               extra_box, from_pregenerated) -> GenerationStrategy:
    if from_pregenerated: return PregeneratedReplay(...)
    if is_qwen(model_name) and thinking: return QwenOutsideCoTGeneration(prompt_type, extra_box)
    if extra_box: return ExtraBoxContinuation()
    return PlainGeneration()
```

Same for parsing: a `ParsingStrategy` that returns a `ParsedOutput` with offsets already relative to `full_generated_text`. Today that logic lives awkwardly inline at `main.py:734-814` and writes back onto `parsed` via `setattr`.

### 2b. Trajectory envelope — extract a builder, kill the duplicates

`main.py:954-991`, `1000-1030`, and `1039-1063` build three versions of the same envelope. Replace with:

```python
@dataclass
class TrajectoryEnvelope:
    index: int
    sample_idx: int | None
    question: str
    trajectory: Trajectory          # dataclass, already well-structured

class TrajectorySink:
    def write(self, env: TrajectoryEnvelope): ...
    def write_failure(self, index, sample_idx, source_entry, error, elapsed): ...
    def write_cot_cache(self, ...): ...
    def write_errors(self, errors: list[dict]): ...
```

That removes ~100 lines of duplication and the three different ways the envelope gets `sample_idx` tacked on conditionally.

---

## 3. Dataset-specific branches are leaking everywhere

Evidence:
- `main.py:932-942` — `if dataset_name == "hle": ... else: evaluate_one(...)` and then ad-hoc dict-score unpacking for hotpotqa.
- `main.py:977-980` — `if dataset_name == "bfcl": traj["func_description"] = ...` appended post-hoc.
- `eval/registry.py:70-88` — five special-case `if` branches in front of the registry table.
- `main.py:480-516` — codeqa special case in pregenerated source loading.
- `prompts/load.py:150-183` — per-dataset `if` chain inside `load_messages()`.

These are four different sites all mutating behavior on the same string, which is the textbook symptom of missing polymorphism.

### Fix — uniform per-dataset Registry/Strategy objects

One entry per dataset, carrying *everything* that is dataset-specific:

```python
@dataclass
class DatasetProfile:
    name: str
    loader: type[BaseBenchmarkDataset]
    prompt_builder: PromptBuilder         # replaces the if-chain in prompts/load.py
    evaluator: Evaluator                  # replaces evaluate_one dispatch
    extra_traj_fields: Callable[[Entry, Trajectory], dict] = lambda *_: {}
    pregenerated_source: PregeneratedSource | None = None
    requires_judge: bool = False          # replaces the hle special-case
```

Then there is exactly one registry — `DATASET_REGISTRY: dict[str, DatasetProfile]` — and the orchestrator asks the profile, not the string. Adding a dataset means adding one object; no edits to `main.py`, `prompts/load.py`, or `eval/registry.py`.

### 3a. Eval score normalization

Today `evaluate_one` returns either `{"score": float}` or `{"score": {"em":bool,"f1":float}}`. Downstream, `main.py:944-952` has to `isinstance(eval_result["score"], dict)` to decide what `correct` means. Normalize the contract:

```python
@dataclass
class EvalResult:
    metric: str
    score: float                    # primary score used for `correct`
    components: dict[str, float] = field(default_factory=dict)  # em/f1/etc.
    prediction: str | None = None
    ground_truth: str | None = None
    extraction_failed: bool = False
```

HotpotQA returns `EvalResult(metric="em+f1", score=em, components={"em":em,"f1":f1})`. The orchestrator no longer branches on type.

---

## 4. Lifecycles leaking into the caller

`main.py:830-892`:

```python
llm.switch_attn_implementation("confidence")
try:
    ...compute_all_confidence_scores(...)...
finally:
    llm.switch_attn_implementation("cot")
```

The orchestrator should not know about attention-implementation modes.

### Fix — Context manager on `LLM`

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
    scores = confidence_runner.run(ctx)
```

Bonus: the mode string stops being a magic literal at the call site. Same principle applies to the `deepcopy`/crop dance for `gen_cache` in `confidence.py:400-407` — wrap in `with owned_early_cache(gen_cache, split) as cache:`.

---

## 5. `LLM` class mixes backend dispatch with backend implementation

`llm.py:145-466` has four near-identical method bodies: `_generate_one_hf`, `_generate_one_vllm`, `_generate_raw_hf`, `_generate_raw_vllm`, each preceded by a thin `if self.backend == "vllm": ... else: ...` dispatcher. Chat-template normalization (Qwen `<think>` stripping, GPT `<|channel|>analysis` swap) is duplicated in both backends (`llm.py:211-220` and `311-318`).

### Fix — Strategy + Template Method

```python
class GenerationBackend(Protocol):
    def generate(self, input_ids, opts) -> BackendOutput: ...

class HFBackend(GenerationBackend): ...
class VLLMBackend(GenerationBackend): ...

class LLM:
    def __init__(self, model_name, thinking, backend: GenerationBackend):
        self.backend = backend
        self.chat_template = ChatTemplateAdapter(model_name, thinking)  # dedup the two normalizers

    def generate_one(self, messages, **opts):
        prompt_text = self.chat_template.format(messages, opts.has_assistant_prefill)
        tokens = self.backend.tokenize(prompt_text)
        out = self.backend.generate(tokens, opts)
        return self._to_generation_result(prompt_text, out)
```

This also solves the "confidence-requires-HF" asymmetry cleanly: `confidence_runner` just requires `backend: HFBackend` in its type signature, not a runtime string check at `confidence.py:410-415`.

---

## 6. `confidence.py` is a 2865-line flat file with eight related algorithms

The top-level namespace holds nine independent forward passes (`_jackknife_true_forward`, `_coinflip_true_forward`, `_hollow_forward`, `_bootstrap_forward`, `_fake_bootstrap_forward`, `_fake_bootstrap_hook_forward`, `dropout_forward`, `dropout_late_forward`) plus three dispatch functions (`dropout_answerlogits`, `dropout_indirectlogits`, `dropout_verbalconf`) that each take the same 15 parameters, plus ~9 debug helpers. The three dispatchers are near-duplicates that differ only in the target-logit extraction step.

### Fix — package split + Template Method for the forward loop

```
confidence/
├── context.py           # ConfidenceContext dataclass, early/late split, cache prep
├── modes/
│   ├── vanilla.py       # existing vanilla_* paths
│   ├── coinflip.py
│   ├── coinflip_true.py
│   ├── jackknife.py
│   ├── bootstrap.py
│   ├── fake_bootstrap.py
│   ├── hollow.py
│   └── base.py          # ConfidenceMode ABC + shared forward scaffold
├── probes/
│   ├── answer_logits.py
│   ├── indirect_ptrue.py
│   └── verbalized.py    # each exposes .extract(model_output) -> tensor
├── sampling/
│   ├── bootstrap.py
│   ├── fake_bootstrap.py
│   ├── jackknife.py     # _jackknife_sample_steps etc.
│   └── coinflip.py
├── masking.py           # crop_cache, attention-mask weighting, hook installers
├── reconstruction.py    # _reconstruct_cot_text, _renumber_step, _strip_thinking
└── debug.py             # all the _debug_* helpers
```

The Template Method lives in `modes/base.py`:

```python
class ConfidenceMode(ABC):
    def run(self, ctx, probe: Probe) -> ProbeScores:
        samples = self._sample(ctx)             # method-specific
        inputs  = self._materialize(ctx, samples)  # tokens/mask/hook/recon-text
        with ctx.forward_scope(inputs) as out:
            return probe.extract(out)
```

Then `answer_logits`, `indirect_ptrue`, and `verbconf` become three `Probe` implementations, and the three 200-line dispatcher functions collapse into one generic loop with a probe parameter. The existing triplication across `dropout_answerlogits` / `dropout_indirectlogits` / `dropout_verbalconf` disappears.

---

## 7. I/O and orchestration are interleaved

Throughout `generate_trajectories()` there are file writes to `trajectories/`, `trajectories_cached_cots/`, `debug_cache/`, and `errors.json` — each with its own path-construction logic that branches on `nb_cot_samples > 1`. Sample-idx-aware naming is a formatting concern and should not be visible to the orchestrator.

### Fix — single `TrajectoryStore`

```python
class TrajectoryStore:
    def __init__(self, out_dir, cot_dir=None, debug_dir="debug_cache"):
        ...
    def traj_path(self, index, sample_idx) -> str: ...    # one place picks the filename
    def cot_path(self, index, sample_idx) -> str: ...
    def write_trajectory(self, envelope): ...
    def write_cot_cache(self, index, sample_idx, cot_data): ...
    def write_errors(self, errors): ...
    def write_debug_cache(self, entries_to_save, prompt_type): ...
```

Orchestrator says `store.write_trajectory(env)` and nothing else. This is where most of the duplicate `if nb_cot_samples > 1` branching lives today (`main.py:919-924`, `982-991`, `1022-1030`).

---

## 8. Configuration object instead of 25 kwargs

`generate_trajectories` signature spans 19 lines at `main.py:627`. Every callsite edit propagates. Collapse into a single frozen dataclass:

```python
@dataclass(frozen=True)
class PipelineConfig:
    model_name: str
    dataset: str
    shot_mode: Literal["zero", "few"]
    prompt_type: Literal[1, 2]
    thinking: bool
    max_new_tokens: int
    temperature: float
    nb_cot_samples: int
    cot_sample_idx: int | None
    backend: Literal["hf", "vllm"]
    confidence_mode: ConfidenceMode | None           # replaces 7 booleans
    modification: Modification | None                # "no-cots" | "gibberish"
    extra_box: bool
    preserve_final_step: bool
    from_pregenerated: Path | None
    debug: bool
    debug_conf: bool
```

And a `PipelineOutputs` for `out_dir`, `cot_dir`, `debug_dir`, `tag`.

This also kills the five-line `dir_name = f"..."` at `main.py:1143-1151` — it becomes `config.dir_name()`, and the `if experimental_hollow` / `if nb_cot_samples > 1` / `if cot_sample_idx` / `if tag` additions each live with the field they describe.

---

## 9. Validation separated from parsing

`parse()` at `main.py:57-395` does argparse *and* cross-flag validation *and* default-filling. Split:

- `cli.py::build_parser()` — argparse only
- `config.py::from_args(args) -> PipelineConfig` — validation, defaults, mutual-exclusion, coercion to `ConfidenceMode`/`Modification` objects

Benefits: you can unit-test validation with no subprocess, no argparse mocking, and `main()` stops being the integration-test boundary.

---

## 10. Smaller but still worth doing

- **`experimental_hollow: int` with magic values 1/2** (`main.py:277-289`) → `HollowMode.KEEP_LAST | HollowMode.REPLACE_LAST`.
- **`experimental_jackknife` is a nullable tri-state string** (`None`, `"default"`, `"true"`) doubling as a boolean — becomes `Jackknife(variant=...)` in the mode object above.
- **`_answer_lookup_value` + `_build_dataset_lookup` + `_merge_pregenerated_entry` + `_resolve_pregenerated_source_entry`** (`main.py:425-516`) form a coherent `PregeneratedSource` class. Move them together and give codeqa its own subclass.
- **`MODEL_DICT[args.model]`** (`main.py:1114`) — wrap in a `ModelProfile` that knows its family flags (`is_qwen`, `is_llama`), thinking-token pair, attention-impl preferences. Then `is_qwen_model_name(model_name)` checks at `main.py:712, 734` stop being string-sniffing and become `profile.family == "qwen"`.
- **Timing instrumentation is manually threaded** (see `conf_timings.vanilla_meanprob_total_seconds = ...` at `main.py:872-889` and `confidence.py`). Replace with a `Timer` context manager that writes into a `TimingRegistry`, then `asdict(timings)` replaces the 18-line manual projection at `main.py:872-889`.
- **`del gen.past_key_values; gen.past_key_values = None; torch.cuda.empty_cache()`** (`main.py:928-930`) is a cleanup step that belongs inside a `with gen.borrowed_cache():` context manager on `GenerationResult`.
- **Circular-import workaround** — `from utils.structures import ConfidenceTimings` *inside* `compute_all_confidence_scores` at `confidence.py:428` is a smell. With the module split in §6 this disappears naturally.

---

## Target architecture (one page)

```
cli/
  parser.py                      # argparse only
  config.py                      # PipelineConfig, ConfidenceMode factory, validation

pipeline/
  driver.py                      # TrajectoryPipeline, iter + try/except loop
  context.py                     # PerSampleContext dataclass

generation/
  strategy.py                    # GenerationStrategy Protocol + selector
  plain.py, qwen_outside.py, extra_box.py, pregenerated_replay.py

llm/
  llm.py                         # LLM + attn_mode context manager
  backends/{hf.py, vllm.py}      # GenerationBackend impls
  chat_template.py               # prompt-text normalizers deduped

parsing/
  strategy.py                    # ParsingStrategy Protocol + selector
  default.py, qwen_thinking.py, prompt_type2.py

confidence/                      # see §6
  context.py, modes/, probes/, sampling/, masking.py, reconstruction.py

datasets/
  registry.py                    # DatasetProfile objects (loader+prompt+eval+traj-fields)
  profiles/{bfcl,hotpotqa,math500,...}.py

evaluation/
  result.py                      # EvalResult dataclass
  evaluators/{bfcl,hle,codeqa,cs1qa,standard}.py

io/
  store.py                       # TrajectorySink, path construction, envelope builder
  progress.py                    # ProgressReporter (tqdm | tqdm_discord)
  envelope.py                    # Trajectory, TrajectoryEnvelope dataclasses

main.py                          # ~30 lines: parser → config → pipeline.run(config) → summary
```

Coupling after the refactor:
- `main.py` depends on `cli`, `pipeline.driver` only.
- `pipeline.driver` depends on abstractions (`GenerationStrategy`, `ParsingStrategy`, `ConfidenceMode`, `Evaluator`, `TrajectorySink`) — no concrete strategy.
- Adding a new confidence method: one file under `confidence/modes/`, register in the factory. No edits to `cli`, `pipeline`, or `io`.
- Adding a new dataset: one `DatasetProfile` object. No edits to `prompts/load.py` or `eval/registry.py`.

---

## Roadmap (order matters — each step unblocks the next)

1. **Introduce `PipelineConfig` + `ConfidenceMode`** (§1, §8). Pure refactor, no behavior change. Removes the N×N validation and the two ternary suffixes. Highest ROI.
2. **Extract `TrajectorySink` and `TrajectoryEnvelope`** (§2b, §7). Removes ~100 lines of duplication and lets the error paths share one writer.
3. **Extract `GenerationStrategy` and `ParsingStrategy`** (§2a). `generate_trajectories` drops to its loop skeleton.
4. **`LLM.attn_mode` context manager + `GenerationBackend` split** (§4, §5).
5. **`EvalResult` normalization + `DatasetProfile` registry** (§3). Touches more files but removes the last string-sniffing branches.
6. **Split `confidence.py` into a package with probes/modes/sampling** (§6). Biggest file, but safest to do last because §1 has already separated the mode selection from the algorithms.

Each step is independently shippable and keeps trajectories bit-identical (the output envelope schema is unchanged except in §3a, which is additive).
