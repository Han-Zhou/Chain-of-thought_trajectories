from dataclasses import dataclass
from transformers import DynamicCache


@dataclass
class GenerationResult:
    """Everything produced by a single generate_one() call."""
    prompt_text: str
    generated_text: str                  # only the newly generated tokens (no prompt echo)
    prompt_token_positions: list         # absolute positions [0, prompt_len)
    generated_token_positions: list      # absolute positions [prompt_len, total_len)
    past_key_values: DynamicCache        # the fully populated KV cache

@dataclass
class ParsedOutput:
    """Structured view of a single model generation."""
    cot_steps: list           # individual reasoning steps as strings
    final_answer_letter: str  # "A", "B", "C", "D", or "" if not detected
    raw_cot_block: str        # the full CoT text before the Final Answer line
