import re

from utils.structures import ParsedOutput



# Matches "Step 1:", "Step 2:", ... as step delimiters inside the CoT block.
_STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)


def _extract_boxed_with_positions(text: str):
    """Find the last \\boxed{...} in *text* and return (content, boxed_start, content_start, content_end).

    Returns (None, None, None, None) if no valid \\boxed{} is found.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None, None, None, None

    # Walk forward to find the opening brace
    i = idx + len("\\boxed")
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None, None, None, None

    brace_start = i
    num_open = 0
    last_close = -1
    while i < len(text):
        if text[i] == "{":
            num_open += 1
        elif text[i] == "}":
            num_open -= 1
            if num_open == 0:
                content = text[brace_start + 1 : i]
                return content, idx, brace_start + 1, i
            last_close = i
        i += 1

    # Braces never balanced — fall back to content up to the last '}' seen.
    if last_close > brace_start:
        content = text[brace_start + 1 : last_close]
        return content, idx, brace_start + 1, last_close

    return None, None, None, None


def parse_output(generated_text: str) -> ParsedOutput:
    """Split a raw generation into structured CoT steps and the final answer.

    Strategy:
      1. Find the last \\boxed{...} to split the text into a CoT block and an
         answer.  Uses brace-matching to handle nested braces.
      2. Within the CoT block, split on "Step N:" markers to get individual
         steps.  If no markers are found, fall back to blank-line splitting,
         then single-line splitting.
      3. Extract the content inside \\boxed{...} as the final answer.
    """
    text = generated_text.strip()

    content, boxed_start, content_start, content_end = _extract_boxed_with_positions(text)

    if content is not None:
        cot_block = text[:boxed_start].strip()
        final_answer = content
        answer_fullstring_start = boxed_start
        answer_start = content_start

        # Find the start of the answer sentence containing \boxed{}.
        # This is the text after the last CoT step — transition phrases like
        # "Therefore the final answer is\nThe answer is \boxed{...}".
        # Strategy: find the end of the last "Step N:" body in the original
        # text, then scan forward to the first non-blank line start.
        last_step_match = None
        for m in _STEP_MARKER_RE.finditer(text[:boxed_start]):
            last_step_match = m
        if last_step_match is not None:
            # The last step body runs from end-of-marker to either a double
            # newline (paragraph break) or the next Step marker (whichever
            # comes first).  After that is the answer sentence.
            body_start = last_step_match.end()
            remaining = text[body_start:boxed_start]
            double_nl = remaining.find("\n\n")
            if double_nl >= 0:
                # Skip past the blank line to reach the answer sentence
                answer_sentence_start = body_start + double_nl + 1
                while answer_sentence_start < boxed_start and text[answer_sentence_start] in "\n\r":
                    answer_sentence_start += 1
            else:
                # No paragraph break — answer sentence is on the last line
                last_nl = text.rfind("\n", last_step_match.start(), boxed_start)
                if last_nl >= last_step_match.end():
                    answer_sentence_start = last_nl + 1
                else:
                    answer_sentence_start = boxed_start
        else:
            # No step markers — use the last paragraph/line break before \boxed
            double_nl = text.rfind("\n\n", 0, boxed_start)
            if double_nl >= 0:
                answer_sentence_start = double_nl + 2
            else:
                last_nl = text.rfind("\n", 0, boxed_start)
                answer_sentence_start = last_nl + 1 if last_nl >= 0 else 0
    else:
        cot_block = text
        final_answer = ""
        answer_fullstring_start = None
        answer_start = None
        answer_sentence_start = None

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
        answer_fullstring_start=answer_fullstring_start,
        answer_start=answer_start,
        answer_sentence_start=answer_sentence_start,
    )
