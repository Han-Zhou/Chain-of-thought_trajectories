import re

from utils.structures import ParsedOutput



# Matches "Step 1:", "Step 2:", ... as step delimiters inside the CoT block.
_STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^\s*<think>\s*", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"\s*</think>\s*$", re.IGNORECASE)
_THINK_CLOSE_TAG_RE = re.compile(r"</think>", re.IGNORECASE)
_ANSWER_MARKER_RE = re.compile(
    r"(?im)^\s*(?:"
    r"answer"
    r"|final answer(?:\s+seems\s+to\s+be|\s+is)?"
    r"|the final answer(?:\s+seems\s+to\s+be|\s+is)?"
    r"|the answer(?:\s+seems\s+to\s+be|\s+is)?"
    r"|therefore the answer(?:\s+is)?"
    r"|therefore the final answer(?:\s+is)?"
    r")\s*:?"
)
_MCQ_STANDALONE_LETTER_RE = re.compile(r"^(?:\(?[A-Ea-e]\)?|\[[A-Ea-e]\])$")


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


def _build_cot_block_for_steps(cot_block: str, extra_box_suffix: str | None = None) -> tuple[str, bool]:
    """Return a cleaned view of the CoT block for step extraction only.

    The returned string may differ from raw_cot_block because we strip any
    extra answer suffix text and outer <think> wrappers so those tokens do not
    leak into cot_steps.
    """
    cot_block_for_steps = cot_block

    if extra_box_suffix is not None:
        # The suffix without the \boxed{ part is what appears in cot_block.
        suffix_text = extra_box_suffix.replace("\\boxed{", "").strip()
        suffix_idx = cot_block_for_steps.rfind(suffix_text)
        if suffix_idx >= 0:
            cot_block_for_steps = cot_block_for_steps[:suffix_idx].strip()

    cot_block_for_steps = cot_block_for_steps.strip()
    had_think_wrapper = False

    if _THINK_OPEN_RE.match(cot_block_for_steps):
        cot_block_for_steps = _THINK_OPEN_RE.sub("", cot_block_for_steps, count=1)
        had_think_wrapper = True
        close_matches = list(_THINK_CLOSE_TAG_RE.finditer(cot_block_for_steps))
        if close_matches:
            cot_block_for_steps = cot_block_for_steps[:close_matches[-1].start()]
    elif _THINK_CLOSE_RE.search(cot_block_for_steps):
        cot_block_for_steps = _THINK_CLOSE_RE.sub("", cot_block_for_steps, count=1)
        had_think_wrapper = True

    return cot_block_for_steps.strip(), had_think_wrapper


def _cleanup_cot_step(step: str) -> str:
    """Strip leftover think wrapper tokens from step boundaries."""
    step = _THINK_OPEN_RE.sub("", step.strip(), count=1)
    step = _THINK_CLOSE_RE.sub("", step, count=1)
    return step.strip()


def _answer_aliases(final_answer: str) -> set[str]:
    """Return normalized answer strings that may appear before \\boxed{}."""
    answer = final_answer.strip()
    if not answer:
        return set()

    aliases = {answer}
    unwrapped = answer.strip("()[]{}").strip()
    if unwrapped:
        aliases.add(unwrapped)

    if len(unwrapped) == 1 and unwrapped.isalpha():
        aliases.update({
            unwrapped.upper(),
            unwrapped.lower(),
            f"({unwrapped.upper()})",
            f"({unwrapped.lower()})",
            f"[{unwrapped.upper()}]",
            f"[{unwrapped.lower()}]",
        })

    return {alias for alias in aliases if alias}


def _find_standalone_answer_line_start(cot_block: str, final_answer: str) -> int | None:
    """Find a final line that is only the answer itself, e.g. a bare 'C'."""
    aliases = _answer_aliases(final_answer)
    if not aliases:
        return None

    trimmed = cot_block.rstrip()
    if not trimmed:
        return None

    line_start = trimmed.rfind("\n") + 1 if "\n" in trimmed else 0
    last_line = trimmed[line_start:].strip()

    if last_line not in aliases and _MCQ_STANDALONE_LETTER_RE.fullmatch(last_line) is None:
        return None

    return line_start


def _find_answer_tail_start(
    cot_block: str,
    final_answer: str,
    extra_box_suffix: str | None = None,
) -> int | None:
    """Find where answer-like text begins inside the pre-\\boxed CoT block."""
    candidates: list[int] = []
    suffix_idx = None

    if extra_box_suffix is not None:
        suffix_text = extra_box_suffix.replace("\\boxed{", "").strip()
        suffix_idx = cot_block.rfind(suffix_text)
        if suffix_idx >= 0:
            candidates.append(suffix_idx)

    last_step_match = None
    for match in _STEP_MARKER_RE.finditer(cot_block):
        last_step_match = match
    search_start = last_step_match.end() if last_step_match is not None else 0

    for match in _ANSWER_MARKER_RE.finditer(cot_block):
        if match.start() < search_start:
            continue
        if _STEP_MARKER_RE.search(cot_block, match.end()) is None:
            candidates.append(match.start())

    standalone_idx = _find_standalone_answer_line_start(cot_block, final_answer)
    if standalone_idx is not None and standalone_idx >= search_start:
        candidates.append(standalone_idx)

    if suffix_idx is not None and suffix_idx >= 0:
        standalone_before_suffix = _find_standalone_answer_line_start(
            cot_block[:suffix_idx],
            final_answer,
        )
        if standalone_before_suffix is not None and standalone_before_suffix >= search_start:
            candidates.append(standalone_before_suffix)

    return min(candidates) if candidates else None


def _find_generic_answer_sentence_start(text: str, boxed_start: int) -> int:
    """Find the generic answer-sentence boundary before \\boxed{}."""
    last_step_match = None
    for match in _STEP_MARKER_RE.finditer(text[:boxed_start]):
        last_step_match = match

    if last_step_match is not None:
        body_start = last_step_match.end()
        remaining = text[body_start:boxed_start]
        double_nl = remaining.find("\n\n")
        if double_nl >= 0:
            answer_sentence_start = body_start + double_nl + 1
            while answer_sentence_start < boxed_start and text[answer_sentence_start] in "\n\r":
                answer_sentence_start += 1
        else:
            last_nl = text.rfind("\n", last_step_match.start(), boxed_start)
            if last_nl >= last_step_match.end():
                answer_sentence_start = last_nl + 1
            else:
                answer_sentence_start = boxed_start
    else:
        double_nl = text.rfind("\n\n", 0, boxed_start)
        if double_nl >= 0:
            answer_sentence_start = double_nl + 2
        else:
            last_nl = text.rfind("\n", 0, boxed_start)
            answer_sentence_start = last_nl + 1 if last_nl >= 0 else 0

    return answer_sentence_start


def parse_output(
    generated_text: str,
    extra_box_suffix: str | None = None,
    llama_answer_cleanup: bool = False,
) -> ParsedOutput:
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
        cot_prefix = text[:boxed_start]
        cot_block = cot_prefix.strip()
        final_answer = content
        answer_fullstring_start = boxed_start
        answer_start = content_start
        if llama_answer_cleanup:
            answer_sentence_start = _find_answer_tail_start(
                cot_prefix,
                final_answer,
                extra_box_suffix=extra_box_suffix,
            )
        else:
            answer_sentence_start = _find_generic_answer_sentence_start(text, boxed_start)
    else:
        cot_block = text
        final_answer = ""
        answer_fullstring_start = None
        answer_start = None
        answer_sentence_start = None

    cot_block_for_steps = cot_block
    if answer_sentence_start is not None and (llama_answer_cleanup or _STEP_MARKER_RE.search(cot_block)):
        cot_block_for_steps = text[:answer_sentence_start].strip()

    cot_block_for_steps, had_think_wrapper = _build_cot_block_for_steps(
        cot_block_for_steps,
        extra_box_suffix=extra_box_suffix,
    )
    parts = _STEP_MARKER_RE.split(cot_block_for_steps)

    steps: list = []
    if len(parts) > 1:
        preamble = parts[0].strip()
        if preamble and not had_think_wrapper:
            steps.append(preamble)
        for i in range(1, len(parts) - 1, 2):
            marker = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            step = (marker + body).strip()
            steps.append(step)
    else:
        steps = [s.strip() for s in re.split(r"\n{2,}", cot_block_for_steps) if s.strip()]
        if len(steps) <= 1:
            steps = [s.strip() for s in cot_block_for_steps.splitlines() if s.strip()]

    steps = [_cleanup_cot_step(step) for step in steps]
    steps = [step for step in steps if step]

    return ParsedOutput(
        cot_steps=steps,
        final_answer=final_answer,
        raw_cot_block=cot_block,
        answer_fullstring_start=answer_fullstring_start,
        answer_start=answer_start,
        answer_sentence_start=answer_sentence_start,
    )
