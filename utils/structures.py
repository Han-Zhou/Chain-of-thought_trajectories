import torch
from dataclasses import dataclass, field

from utils.cache_compat import DynamicCache, Qwen3_5DynamicCache



@dataclass
class GenerationResult:
    """Everything produced by a single generate_one() call."""
    prompt_text: str
    generated_text: str                  # only the newly generated tokens (no prompt echo)
    prompt_end_position: int             # prompt_len = last prompt position + 1
    generated_end_position: int          # last generated position + 1
    past_key_values: DynamicCache | Qwen3_5DynamicCache | None       # the fully populated KV cache (None for vLLM backend)
    prompt_tail_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.long))
    scores: tuple | None = field(default=None)
    generated_ids: torch.Tensor | None = field(default=None)

@dataclass
class ParsedOutput:
    """Structured view of a single model generation."""
    cot_steps: list           # individual reasoning steps as strings
    final_answer: str         # MCQ or other strings
    raw_cot_block: str        # the full CoT text before the \boxed{} answer
    answer_fullstring_start: int | None  # char offset in generated_text where "\boxed{" begins (None if not found)
    answer_start: int | None # char offset of the first character inside \boxed{...} braces
    answer_sentence_start: int | None  # char offset where the sentence containing \boxed{} begins (after CoT steps)


@dataclass
class ConfidenceScores:
    """All confidence scores computed for a single generation."""
    answer_probabilities: list[dict[str, float]]
    answer_entropy: list[dict[str, float]]
    indirect_ptrue1_probabilities: list[dict[str, float]]
    indirect_ptrue2_probabilities: list[dict[str, float]]
    indirect_ptrue3_probabilities: list[dict[str, float]]
    verbconf_probabilities: list[float]
    verbconf_distribution: list[float] | None = None   # softmaxed probs for scores 0-100 (length 101)
    verbconf_top_score: int | None = None               # score with highest probability
    verbconf_top_prob: float | None = None              # probability of that score
    step_masks: list[list[int]] | None = None           # binary list per sample: 1 = kept, 0 = masked


@dataclass
class ConfidenceTimings:
    """Granular timing for confidence computation components."""
    # Vanilla per-metric timing (includes forward pass + post-processing)
    vanilla_meanprob_total_seconds: float = 0.0
    vanilla_meanprob_forward_seconds: float = 0.0

    vanilla_meanent_total_seconds: float = 0.0
    vanilla_meanent_forward_seconds: float = 0.0

    vanilla_indirect1_total_seconds: float = 0.0
    vanilla_indirect1_forward_seconds: float = 0.0

    vanilla_verbconf_total_seconds: float = 0.0
    vanilla_verbconf_forward_seconds: float = 0.0
    vanilla_verbconf_joint_probs_seconds: float = 0.0

    # Bootstrap-specific timing (when experimental_bootstrap=True)
    bootstrap_meanprob_total_seconds: float = 0.0
    bootstrap_meanprob_forward_seconds: float = 0.0

    bootstrap_indirect1_total_seconds: float = 0.0
    bootstrap_indirect1_forward_seconds: float = 0.0

    bootstrap_verbconf_total_seconds: float = 0.0
    bootstrap_verbconf_forward_seconds: float = 0.0
    bootstrap_verbconf_joint_probs_seconds: float = 0.0


@dataclass
class AllConfidenceData:
    """All confidence data for a single question, for both non-dropout and dropout versions."""
    vanilla_confidences: ConfidenceScores
    dropout_confidences: ConfidenceScores
    jackknife_confidences: ConfidenceScores | None = None
    bootstrap_confidences: ConfidenceScores | None = None
    fake_bootstrap_confidences: ConfidenceScores | None = None
    fake_bootstrap_hook_confidences: ConfidenceScores | None = None
    coinflip_true_confidences: ConfidenceScores | None = None
    hollow_confidences: ConfidenceScores | None = None
    debug_info: dict = field(default_factory=dict)
    timings: ConfidenceTimings = field(default_factory=ConfidenceTimings)
