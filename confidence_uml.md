# Confidence Sampler UML

UML diagrams for the design patterns proposed for `confidence_pseudocode.py`.

## Class diagram

```mermaid
classDiagram
    direction LR

    %% ===================== Value objects / DTOs =====================
    class SampleContext {
        +str question
        +str reference_cot
        +str reference_answer
        +KVCache question_cache
        +Backend backend
        +SamplingConfig cfg
    }

    class SamplingConfig {
        +int nb_samples
        +float temperature
        +int max_new_tokens
        +bool include_question_cache
    }

    class Candidate {
        +str cot
        +str answer
        +KVCache cot_with_answer_cache
        +TokenProbs cot_with_answer_prob
        +bool answer_came_from_generation
    }

    class ScoredCompletion {
        +str prefix_with_answer
        +KVCache cache
        +TokenProbs answer_token_probs
    }

    class Confidence {
        +TokenProbs answer_token_probs
        +TokenProbs indirect_probs
        +TokenProbs verbal_probs
    }

    class Sample {
        +str cot
        +str answer
        +Confidence confidence
    }

    class Timings {
        +float generation_time
        +float confidence_time
    }

    class SamplerResult {
        +str method_name
        +list~Sample~ samples
        +Timings timings
    }

    Sample --> Confidence
    SamplerResult --> Sample : 0..*
    SamplerResult --> Timings
    SampleContext --> SamplingConfig
    SampleContext --> Backend

    %% ===================== Backend Adapter =====================
    class Backend {
        <<interface>>
        +generate(prefix, nb, temperature, cache) GenerationBatch
        +forward(prefix, cache) ForwardResult
        +supports_kv_cache() bool
    }

    class APIBackend {
        -ApiClient client
        -str model_id
        +generate(...) GenerationBatch
        +forward(...) ForwardResult
        +supports_kv_cache() bool
    }

    class LocalBackend {
        -CausalLM model
        -Tokenizer tokenizer
        -ModelAdapter adapter
        +generate(...) GenerationBatch
        +forward(...) ForwardResult
        +supports_kv_cache() bool
    }

    class GenerationBatch {
        +list~str~ texts
        +list~TokenProbs~ probs
        +list~KVCache~ caches
        +KVCache shared_prefix_cache
    }

    class ForwardResult {
        +TokenProbs probs
        +KVCache cache
    }

    Backend <|.. APIBackend
    Backend <|.. LocalBackend
    Backend ..> GenerationBatch : returns
    Backend ..> ForwardResult : returns

    %% ===================== Sampler Strategy / Template Method =====================
    class SamplerMethod {
        <<abstract>>
        +str name
        +propose(ctx) list~Candidate~
        +validate(cand, ctx) bool
        +to_scored_completion(cand, ctx) ScoredCompletion
    }

    class VanillaSampler {
        +propose(ctx) list~Candidate~
        +to_scored_completion(cand, ctx) ScoredCompletion
    }

    class RejectionSampler {
        -int nb
        -float temperature
        +propose(ctx) list~Candidate~
        +validate(cand, ctx) bool
        +to_scored_completion(cand, ctx) ScoredCompletion
    }

    class LawyerSampler {
        -int nb
        -float temperature
        +propose(ctx) list~Candidate~
        +to_scored_completion(cand, ctx) ScoredCompletion
    }

    class StepBootstrapSampler {
        -int nb
        +propose(ctx) list~Candidate~
        +to_scored_completion(cand, ctx) ScoredCompletion
    }

    SamplerMethod <|-- VanillaSampler
    SamplerMethod <|-- RejectionSampler
    SamplerMethod <|-- LawyerSampler
    SamplerMethod <|-- StepBootstrapSampler

    SamplerMethod ..> SampleContext : reads
    SamplerMethod ..> Backend : delegates I/O
    SamplerMethod ..> Candidate : produces
    SamplerMethod ..> ScoredCompletion : builds

    %% ===================== Confidence engine + Builder =====================
    class ConfidenceEngine {
        -Backend backend
        -AnswerSuffix suffix
        +score(question, sc) Confidence
    }

    class AnswerSuffix {
        <<builder>>
        +str boxed_open
        +str boxed_close
        +str indirect_probe
        +str verbal_probe
        +format_boxed(answer) str
        +indirect_continuation() str
        +verbal_continuation() str
    }

    ConfidenceEngine --> Backend
    ConfidenceEngine --> AnswerSuffix
    ConfidenceEngine ..> ScoredCompletion : reads
    ConfidenceEngine ..> Confidence : produces

    %% ===================== Step splitter (used by StepBootstrap) =====================
    class StepResampler {
        +split_steps(cot) list~str~
        +resample(steps, rng) list~str~
        +rebuild(steps) str
    }

    StepBootstrapSampler ..> StepResampler

    %% ===================== Orchestrator =====================
    class SamplerRunner {
        -ConfidenceEngine engine
        +run(method, ctx) SamplerResult
    }

    SamplerRunner --> ConfidenceEngine
    SamplerRunner ..> SamplerMethod : drives
    SamplerRunner ..> SamplerResult : returns
    SamplerRunner ..> Sample : assembles
    SamplerRunner ..> Timings : measures

    %% ===================== Registry (entry point) =====================
    class SamplerRegistry {
        +register(name, factory)
        +build(name, cfg) SamplerMethod
        +names() list~str~
    }

    SamplerRegistry ..> SamplerMethod : creates
```

## Sequence diagram — one sampler invocation

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant R as SamplerRunner
    participant M as SamplerMethod
    participant B as Backend
    participant E as ConfidenceEngine
    participant S as AnswerSuffix

    Caller->>R: run(method, ctx)
    R->>R: t0 = now()
    R->>M: propose(ctx)
    M->>B: generate(prefix, nb, temperature, ctx.question_cache)
    B-->>M: GenerationBatch
    M-->>R: list~Candidate~
    R->>R: t1 = now()  (generation_time = t1 - t0)

    loop for each Candidate
        R->>M: validate(cand, ctx)
        alt rejected (e.g. answer != reference)
            M-->>R: false
        else accepted
            M-->>R: true
            R->>M: to_scored_completion(cand, ctx)
            alt cand has cache (vanilla / rejection)
                M-->>R: ScoredCompletion(cache=cand.cache, atp=cand.probs)
            else string-only candidate (lawyer / stepbootstrap)
                M->>S: format_boxed(reference_answer)
                S-->>M: "\nThe answer is \\boxed{...}."
                M->>B: forward(question + cot + suffix, ctx.question_cache)
                B-->>M: ForwardResult
                M-->>R: ScoredCompletion(cache, atp)
            end

            R->>E: score(question, sc)
            E->>S: indirect_continuation()
            E->>B: forward(prefix_with_answer + indirect, sc.cache)
            B-->>E: ForwardResult
            E->>S: verbal_continuation()
            E->>B: forward(prefix_with_answer + verbal, sc.cache)
            B-->>E: ForwardResult
            E-->>R: Confidence
            R->>R: samples.append(Sample(...))
        end
    end

    R->>R: t2 = now()  (confidence_time = t2 - t1)
    R-->>Caller: SamplerResult(samples, Timings(t1-t0, t2-t1))
```

## Pattern map

| Pattern | Where it lives in the diagram |
|---|---|
| Template Method | `SamplerMethod` defines the shape; `SamplerRunner.run` is the invariant skeleton; subclasses fill `propose` / `validate` / `to_scored_completion` |
| Strategy + Registry | The four `*Sampler` classes are interchangeable; `SamplerRegistry` is the lookup |
| Adapter | `Backend` interface with `APIBackend` / `LocalBackend` realizations hides `use_api` |
| Unified probe scoring | `ConfidenceEngine.score` consumes a `ScoredCompletion` regardless of how it was produced |
| Builder | `AnswerSuffix` is the single source of truth for `\boxed{...}` and the indirect/verbal probe strings |
| Filter pipeline | `validate(cand, ctx)` on `SamplerMethod`; rejection and lawyer use it, vanilla and stepbootstrap default it to `true` |
| Parameter Object | `SampleContext` and `SamplingConfig` replace the long argument lists |
| Value objects / DTOs | `Candidate`, `ScoredCompletion`, `Confidence`, `Sample`, `Timings`, `SamplerResult` replace the raw dicts in the pseudocode |

The structural insight the diagram makes visible: every arrow that used to be a `use_api` branch in the pseudocode is now an arrow into `Backend`, and every arrow that used to be `confidence_from_answerprob` vs `confidence_from_cot_string` is now an arrow into `ConfidenceEngine.score` via a `ScoredCompletion`. Those are the two seams that collapse the duplication.
