import torch
from dataclasses import dataclass, field
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache



@dataclass
class GenerationResult:
    """Everything produced by a single generate_one() call."""
    prompt_text: str
    generated_text: str                  # only the newly generated tokens (no prompt echo)
    prompt_end_position: int             # prompt_len = last prompt position + 1
    generated_end_position: int          # last generated position + 1
    past_key_values: DynamicCache | Qwen3_5DynamicCache       # the fully populated KV cache
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


@dataclass
class ConfidenceScores:
    """All confidence scores computed for a single generation."""
    answer_probabilities: list[float]
    answer_entropy: list[float]
    indirect_ptrue1_probabilities: list[float]
    indirect_ptrue2_probabilities: list[float]
    verbconf_probabilities: list[float]


@dataclass
class AllConfidenceData:
    """All confidence data for a single question, for both non-dropout and dropout versions."""
    vanilla_confidences: ConfidenceScores
    dropout_confidences: ConfidenceScores
    debug_info: dict = field(default_factory=dict)

