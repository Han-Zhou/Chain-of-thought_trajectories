from dataclasses import dataclass
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache



@dataclass
class GenerationResult:
    """Everything produced by a single generate_one() call."""
    prompt_text: str
    generated_text: str                  # only the newly generated tokens (no prompt echo)
    prompt_token_positions: int          # prompt_len = last prompt position + 1
    generated_token_positions: int       # last generated position + 1
    past_key_values: DynamicCache | Qwen3_5DynamicCache       # the fully populated KV cache

@dataclass
class ParsedOutput:
    """Structured view of a single model generation."""
    cot_steps: list           # individual reasoning steps as strings
    final_answer: str         # MCQ or other strings
    raw_cot_block: str        # the full CoT text before the Final Answer line
    answer_char_start: int | None  # char offset in generated_text where "final answer" begins (None if not found)
