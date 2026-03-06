"""
evaluate_logiqa.py

LogiQA evaluation pipeline: prompt formatting, DynamicCache-based generation,
and structured output parsing.

Designed to slot into main.py via run_logiqa_eval(), which is the drop-in
replacement for generate_trajectories() when --dataset logiqa is selected.

Sections:
  1. Prompt Formatting   – zero-shot and few-shot CoT builders
  2. Generation          – model.generate() with DynamicCache + position tracking
  3. Parsing             – CoT step extraction and final-answer detection
  4. Evaluation Loop     – orchestrates sections 1-3 across the dataset
"""

import re
import json
import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)

logger = logging.getLogger(__name__)

LETTERS = "ABCD"

# =============================================================================
# Section 1 – Prompt Formatting
# =============================================================================

# The system instruction that heads every prompt (zero-shot and few-shot).
# We keep it identical to the shared SYSTEM constant in prompts/cot_prompt.py
# but extend it with the explicit letter-only constraint for easier parsing.
SYSTEM_PROMPT = (
    "You are an expert logical reasoning assistant. "
    "For each problem, think carefully and reason step-by-step before answering. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "After all steps, write your final answer on a new line in the form:\n"
    "Final Answer: <letter>   (letter must be A, B, C, or D)"
)

# Two fixed few-shot demonstrations.
# These illustrate the expected CoT format without coming from the eval split,
# so they do not contaminate the benchmark.
FEW_SHOT_EXAMPLES = [
    {
        "context": (
            "All members of the chess club are also members of the math club. "
            "No member of the math club is a member of the debate club."
        ),
        "question": "Which of the following must be true?",
        "choices": [
            "A. All chess club members are in the debate club.",
            "B. No chess club member is in the debate club.",
            "C. Some math club members are in the chess club.",
            "D. Some debate club members are in the math club.",
        ],
        "answer": "B",
        "cot": (
            "Step 1: The first premise says chess club ⊆ math club.\n"
            "Step 2: The second premise says math club ∩ debate club = ∅.\n"
            "Step 3: Because chess members are a subset of math members, and no "
            "math member is in the debate club, no chess member can be either.\n"
            "Step 4: This directly matches option B.\n"
            "Step 5: Option A contradicts step 3. C and D are not entailed."
        ),
    },
    {
        "context": (
            "A company reported falling profits in Q1. "
            "Whenever profits fall, the board considers budget cuts. "
            "The board considered budget cuts last quarter."
        ),
        "question": "Which conclusion is best supported by the statements above?",
        "choices": [
            "A. The company will recover in Q2.",
            "B. Budget cuts were implemented last quarter.",
            "C. The board acted consistently with its stated policy.",
            "D. Falling profits always lead to employee layoffs.",
        ],
        "answer": "C",
        "cot": (
            "Step 1: Policy: falling profits → board considers budget cuts.\n"
            "Step 2: Profits fell in Q1, so the policy triggers.\n"
            "Step 3: The passage confirms the board did consider cuts — matching the policy.\n"
            "Step 4: This is exactly what option C states.\n"
            "Step 5: A is speculative, B says 'implemented' (beyond what was stated), "
            "D introduces layoffs which are never mentioned."
        ),
    },
]


def _render_entry(entry: dict, include_answer: bool = False) -> str:
    """Render one normalised LogiQA entry as a text block.

    Args:
        entry:          A normalised dict from load_logiqa() or a FEW_SHOT_EXAMPLES item.
        include_answer: If True, append the CoT steps and 'Final Answer:' line.
    """
    choices_text = "\n".join(entry["choices"])
    block = (
        f"Context:\n{entry['context']}\n\n"
        f"Question:\n{entry['question']}\n\n"
        f"Options:\n{choices_text}"
    )
    if include_answer:
        block += f"\n\n{entry['cot']}\n\nFinal Answer: {entry['answer']}"
    return block


def format_zero_shot(entry: dict) -> str:
    """Build a zero-shot Chain-of-Thought prompt for one LogiQA entry.

    The system instruction is prepended verbatim; no demonstrations are shown.
    The model must discover the CoT format purely from the instruction.

    Returns:
        A single string ready to be tokenised.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "--- Problem ---\n"
        f"{_render_entry(entry, include_answer=False)}\n\n"
        "Now reason step-by-step, then give your Final Answer."
    )


def format_few_shot(entry: dict, n_shots: int = 2) -> str:
    """Build a few-shot Chain-of-Thought prompt using fixed demonstrations.

    Each demonstration shows the full CoT and the correct 'Final Answer:' line,
    teaching the model the exact output format before it sees the real question.

    Args:
        entry:   The actual test entry to answer.
        n_shots: How many demonstrations to include (capped at len(FEW_SHOT_EXAMPLES)).

    Returns:
        A single string ready to be tokenised.
    """
    shots = FEW_SHOT_EXAMPLES[:n_shots]
    parts = [SYSTEM_PROMPT, ""]

    for i, ex in enumerate(shots, 1):
        parts.append(f"--- Example {i} ---")
        parts.append(_render_entry(ex, include_answer=True))
        parts.append("")  # blank line between examples

    parts.append("--- Your Turn ---")
    parts.append(_render_entry(entry, include_answer=False))
    parts.append("\nNow reason step-by-step, then give your Final Answer.")

    return "\n".join(parts)


# =============================================================================
# Section 2 – Generation with DynamicCache
# =============================================================================

@dataclass
class GenerationResult:
    """Everything produced by a single generate_one() call."""
    prompt_text: str
    generated_text: str                  # only the newly generated tokens (no prompt echo)
    prompt_token_positions: list         # absolute positions [0, prompt_len)
    generated_token_positions: list      # absolute positions [prompt_len, total_len)
    past_key_values: DynamicCache        # the fully populated KV cache


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> GenerationResult:
    """Run one forward pass through model.generate() with a DynamicCache.

    Key HuggingFace arguments and why they are used here:

    past_key_values=DynamicCache()
        Passes a pre-initialised DynamicCache object into generate().
        Transformers fills it in-place during the forward passes; the
        populated cache is returned via outputs.past_key_values.
        This avoids re-computing attention keys/values on the prompt for any
        subsequent continuation (useful if you want to branch or re-use).

    return_dict_in_generate=True
        Makes generate() return a GenerateDecoderOnlyOutput (a structured
        object) instead of a raw tensor.  This gives access to:
          .sequences        – full token IDs (prompt + generated)
          .past_key_values  – the updated DynamicCache
          .scores           – per-step logits (if output_scores=True)

    do_sample / temperature
        temperature=0.0 → greedy decoding (deterministic, good for evals).
        temperature>0.0 → multinomial sampling.

    Args:
        model:          A loaded AutoModelForCausalLM.
        tokenizer:      The matching tokenizer.
        prompt_text:    The fully formatted prompt string.
        max_new_tokens: Maximum tokens to generate beyond the prompt.
        temperature:    Sampling temperature; 0.0 = greedy.

    Returns:
        A GenerationResult dataclass with text, positions, and the live cache.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len: int = inputs["input_ids"].shape[1]

    # Initialise an empty cache; generate() will populate it in-place.
    cache = DynamicCache()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,          # hand the cache object to generate()
            return_dict_in_generate=True,   # structured output, not a bare tensor
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # outputs.sequences shape: (1, prompt_len + n_new_tokens)
    total_len: int = outputs.sequences.shape[1]

    # Absolute positions of prompt tokens vs. generated tokens in the sequence.
    # These are the indices into the full sequence tensor, not relative offsets.
    prompt_positions = list(range(0, prompt_len))
    generated_positions = list(range(prompt_len, total_len))

    # Decode only the generated portion to avoid echoing the prompt back.
    generated_ids = outputs.sequences[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        prompt_text=prompt_text,
        generated_text=generated_text,
        prompt_token_positions=prompt_positions,
        generated_token_positions=generated_positions,
        past_key_values=outputs.past_key_values,  # the updated DynamicCache
    )


# =============================================================================
# Section 3 – Parsing
# =============================================================================

@dataclass
class ParsedOutput:
    """Structured view of a single model generation."""
    cot_steps: list          # individual reasoning steps as strings
    final_answer_letter: str  # "A", "B", "C", "D", or "" if not detected
    raw_cot_block: str        # the full CoT text before the Final Answer line


# Matches: "Final Answer: B" / "final answer: (C)" / "Final Answer: [d]" etc.
_FINAL_ANSWER_RE = re.compile(
    r"final\s+answer\s*:?\s*[\(\[]?\s*([A-Da-d])\s*[\)\]]?",
    re.IGNORECASE,
)

# Matches "Step 1:", "Step 2:", ... as step delimiters inside the CoT block.
_STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)


def parse_output(generated_text: str) -> ParsedOutput:
    """Split a raw generation into structured CoT steps and the final answer.

    Strategy:
      1. Find "Final Answer:" to split the text into a CoT block and an
         answer segment.  Case- and punctuation-insensitive.
      2. Within the CoT block, split on "Step N:" markers to get individual
         steps.  If no markers are found, fall back to blank-line splitting,
         then single-line splitting.
      3. Extract the letter (A-D) from the answer segment with a lenient regex
         that handles parentheses, brackets, and lowercase letters.

    Args:
        generated_text: The raw string returned by the model (no prompt).

    Returns:
        A ParsedOutput dataclass.
    """
    text = generated_text.strip()

    # --- 1. Split CoT from the final answer ---
    split_pos = text.lower().find("final answer")
    if split_pos != -1:
        cot_block = text[:split_pos].strip()
        answer_segment = text[split_pos:]
    else:
        # Model did not emit "Final Answer:"; treat everything as CoT.
        cot_block = text
        answer_segment = ""

    # --- 2. Extract the answer letter ---
    final_letter = ""
    if answer_segment:
        m = _FINAL_ANSWER_RE.search(answer_segment)
        if m:
            final_letter = m.group(1).upper()

    # --- 3. Split the CoT block into individual steps ---
    # re.split with a capturing group keeps the delimiter in the list:
    # ["preamble", "Step 1:", "body1", "Step 2:", "body2", ...]
    parts = _STEP_MARKER_RE.split(cot_block)

    steps: list = []
    if len(parts) > 1:
        preamble = parts[0].strip()
        if preamble:
            steps.append(preamble)
        # Parts alternate: marker, body, marker, body, ...
        for i in range(1, len(parts) - 1, 2):
            marker = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            steps.append(f"{marker} {body}".strip())
    else:
        # No "Step N:" markers — fall back to blank-line splitting.
        steps = [s.strip() for s in re.split(r"\n{2,}", cot_block) if s.strip()]
        if len(steps) <= 1:
            # Last resort: split by single newline.
            steps = [s.strip() for s in cot_block.splitlines() if s.strip()]

    return ParsedOutput(
        cot_steps=steps,
        final_answer_letter=final_letter,
        raw_cot_block=cot_block,
    )


# =============================================================================
# Section 4 – Evaluation Loop
# =============================================================================

def run_logiqa_eval(
    model_name: str,
    dataset: list,
    shot_mode: str = "zero",
    n_shots: int = 2,
    max_new_tokens: int = 512,
) -> list:
    """Full LogiQA evaluation pipeline.

    This is the drop-in replacement for generate_trajectories() in main.py
    when --dataset logiqa is active.  It returns a serialisable list of result
    dicts that main.py can write directly to its output JSON file.

    Args:
        model_name:     HuggingFace model ID string (from MODEL_DICT).
        dataset:        Normalised entries produced by load_logiqa().
        shot_mode:      "zero" → zero-shot CoT, "few" → few-shot CoT.
        n_shots:        Number of demonstrations (only used when shot_mode="few").
        max_new_tokens: Token budget for generation.

    Returns:
        List of result dicts, one per example, with keys:
            id, question, ground_truth, predicted, correct,
            cot_steps, raw_cot_block, generated_text,
            prompt_token_positions, generated_token_positions.
    """
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    results = []
    n_correct = 0

    for i, entry in enumerate(dataset):
        logger.info("Evaluating %d / %d  (id=%s)", i + 1, len(dataset), entry["id"])

        # Build the prompt for this entry.
        if shot_mode == "few":
            prompt_text = format_few_shot(entry, n_shots=n_shots)
        else:
            prompt_text = format_zero_shot(entry)

        # Generate with DynamicCache and capture token positions.
        gen: GenerationResult = generate_one(
            model, tokenizer, prompt_text, max_new_tokens=max_new_tokens
        )

        # Parse CoT steps and extract the final answer letter.
        parsed: ParsedOutput = parse_output(gen.generated_text)

        correct = parsed.final_answer_letter == entry["answer"]
        n_correct += int(correct)

        results.append({
            "id":                        entry["id"],
            "question":                  entry["question"],
            "ground_truth":              entry["answer"],
            "predicted":                 parsed.final_answer_letter,
            "correct":                   correct,
            "cot_steps":                 parsed.cot_steps,
            "raw_cot_block":             parsed.raw_cot_block,
            "generated_text":            gen.generated_text,
            # Absolute token positions for downstream analysis of the KV cache.
            "prompt_token_positions":    gen.prompt_token_positions,
            "generated_token_positions": gen.generated_token_positions,
        })

    accuracy = n_correct / len(dataset) if dataset else 0.0
    logger.info(
        "Final accuracy: %d / %d = %.2f%%",
        n_correct, len(dataset), accuracy * 100,
    )
    return results
