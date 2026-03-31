"""Answer extraction functions for CoT trajectory evaluation.

Each extractor operates on the full `generated_text` string from a trajectory
and returns the extracted answer (str) or None if extraction fails.
"""

import re


# ---------------------------------------------------------------------------
# Copied from lm-evaluation-harness
# lm_eval/tasks/score/math/math_grader.py:564-612  (Apache-2.0 / MIT)
# ---------------------------------------------------------------------------
def _extract_boxed(string: str) -> str | None:
    """Extract content from the last \\boxed{...} or \\fbox{...} expression."""
    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    retval = string[idx : right_brace_idx + 1]

    left = "\\boxed{"
    try:
        assert retval[: len(left)] == left
        assert retval[-1] == "}"
        return retval[len(left) : -1]
    except (AssertionError, AssertionError):
        return None


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_mcq_letter(generated_text: str) -> str | None:
    """Extract a multiple-choice letter (A-D) from \\boxed{} in text.

    Returns the uppercase letter or None.
    """
    content = _extract_boxed(generated_text)
    if content is None:
        return None
    content = content.strip()
    # Single letter A-F
    if len(content) == 1 and content.upper() in "ABCDEF":
        return content.upper()
    # Handle cases like \boxed{(A)} or \boxed{[B]}
    m = re.match(r"[\(\[]?\s*([A-Fa-f])\s*[\)\]]?$", content)
    if m:
        return m.group(1).upper()
    return None


def extract_text_answer(generated_text: str) -> str | None:
    """Extract the text content from the last \\boxed{} in text.

    Returns stripped text or None if no \\boxed{} is found.
    """
    content = _extract_boxed(generated_text)
    if content is None:
        return None
    return content.strip() if content.strip() else None


def extract_boxed_or_text(generated_text: str) -> str | None:
    """For math datasets: extract content from \\boxed{} expression.

    Returns the content inside \\boxed{...} or None.
    """
    content = _extract_boxed(generated_text)
    if content is None:
        return None
    # Strip surrounding dollar signs if present (e.g., model outputs \boxed{$42$})
    cleaned = content.strip()
    if cleaned.startswith("$") and cleaned.endswith("$"):
        cleaned = cleaned[1:-1].strip()
    return cleaned if cleaned else None
