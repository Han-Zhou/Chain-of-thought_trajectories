import re

from utils.structures import ParsedOutput



# =============================================================================
# Section 3 – Parsing
# =============================================================================

# Maps a regex pattern to a post-processing function that extracts the
# final answer string from the first capture group.
_PARSING_REGEX: dict[str, tuple[re.Pattern, callable]] = {
    # MCQ datasets: capture exactly one letter (A-D), return it uppercased.
    "mcq": (
        re.compile(r"final\s+answer\s*:?\s*[\(\[]?\s*([A-Da-d])\s*[\)\]]?", re.IGNORECASE),
        lambda m: m.group(1).upper(),
    ),
    # Extended MCQ datasets (e.g. MMLU-Pro with options A-J): capture one letter A-J.
    "mcq_extended": (
        re.compile(r"final\s+answer\s*:?\s*[\(\[]?\s*([A-Ja-j])\s*[\)\]]?", re.IGNORECASE),
        lambda m: m.group(1).upper(),
    ),
    # Open-ended datasets: capture all remaining text after "Final Answer:".
    "open_ended": (
        re.compile(r"final\s+answer\s*:\s*([\s\S]+)", re.IGNORECASE),
        lambda m: m.group(1).strip(),
    ),
}

# Maps dataset names to a key in _PARSING_REGEX. Unlisted datasets default to "mcq".
_DATASET_PARSING_MODE: dict[str, str] = {
    "logiqa":               "mcq",
    "college_math_test":    "mcq_extended",
    "codeqa":           "open_ended",
    "bfcl":             "open_ended",
    "bigbench_movie":   "open_ended",
    "bigbench_causal":  "open_ended",
    "cs1qa":            "open_ended",
    "hotpotqa":         "open_ended",
    "olympiadbench":    "open_ended",
    "math500":          "open_ended",
    "hle":              "open_ended",
}

# Matches "Step 1:", "Step 2:", ... as step delimiters inside the CoT block.
_STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)


def parse_output(generated_text: str, dataset_name: str = "logiqa") -> ParsedOutput:
    """Split a raw generation into structured CoT steps and the final answer.

    Strategy:
      1. Find "Final Answer:" to split the text into a CoT block and an
         answer segment.  Case- and punctuation-insensitive.
      2. Within the CoT block, split on "Step N:" markers to get individual
         steps.  If no markers are found, fall back to blank-line splitting,
         then single-line splitting.
      3. Extract the final answer from the answer segment using the regex and
         post-processor registered for the given dataset_name in _PARSING_REGEX.
         MCQ datasets yield a single uppercase letter; open-ended datasets yield
         the full text after "Final Answer:".
    """
    text = generated_text.strip()

    split_pos = text.lower().find("final answer")
    if split_pos != -1:
        cot_block = text[:split_pos].strip()
        answer_segment = text[split_pos:]
    else:
        cot_block = text
        answer_segment = ""

    mode = _DATASET_PARSING_MODE.get(dataset_name, "mcq")
    pattern, extract = _PARSING_REGEX[mode]

    final_answer = ""
    if answer_segment:
        m = pattern.search(answer_segment)
        if m:
            final_answer = extract(m)

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
        answer_char_start=split_pos if split_pos != -1 else None,
    )
