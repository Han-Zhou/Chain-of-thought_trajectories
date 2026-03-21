import re

from utils.structures import ParsedOutput



# =============================================================================
# Section 3 – Parsing (LogiQA)
# =============================================================================

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
    """
    text = generated_text.strip()

    split_pos = text.lower().find("final answer:")
    if split_pos != -1:
        cot_block = text[:split_pos].strip()
        answer_segment = text[split_pos:]
    else:
        cot_block = text
        answer_segment = ""

    final_answer = ""
    answer_start = None
    if answer_segment:
        m = _FINAL_ANSWER_RE.search(answer_segment)
        if m:
            final_answer = m.group(1).upper()
            answer_start = split_pos + m.start(1)

    parts = _STEP_MARKER_RE.split(cot_block)

    steps: list = []
    if len(parts) > 1:
        preamble = parts[0].strip()
        if preamble:
            steps.append(preamble)
        for i in range(1, len(parts) - 1, 2):
            marker = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            steps.append(f"{marker} {body}".strip())
    else:
        steps = [s.strip() for s in re.split(r"\n{2,}", cot_block) if s.strip()]
        if len(steps) <= 1:
            steps = [s.strip() for s in cot_block.splitlines() if s.strip()]

    return ParsedOutput(
        cot_steps=steps,
        final_answer=final_answer,
        raw_cot_block=cot_block,
        answer_fullstring_start=split_pos if split_pos != -1 else None,
        answer_start=answer_start,
    )
