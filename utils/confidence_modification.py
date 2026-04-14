import hashlib
import random
import re
import string
from dataclasses import dataclass


_STEP_MARKER_RE = re.compile(r"^(\s*Step\s+\d+\s*:)(.*)$", re.IGNORECASE | re.DOTALL)
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass
class ConfidenceModificationResult:
    mode: str | None
    modified_content: str
    modified_steps: list[str]
    step_mappings: list[dict[str, str]]


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _match_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template.islower():
        return replacement.lower()
    if template.istitle():
        return replacement.capitalize()

    chars = []
    for src_char, repl_char in zip(template, replacement):
        chars.append(repl_char.upper() if src_char.isupper() else repl_char.lower())
    return "".join(chars)


def _random_letters(length: int, template: str, rng: random.Random) -> str:
    replacement = "".join(rng.choice(string.ascii_lowercase) for _ in range(length))
    return _match_case(template, replacement)


def _gibberish_text(text: str, rng: random.Random) -> str:
    return _ALPHA_WORD_RE.sub(
        lambda match: _random_letters(len(match.group(0)), match.group(0), rng),
        text,
    )


def _gibberish_step(step: str, rng: random.Random) -> str:
    match = _STEP_MARKER_RE.match(step)
    if match:
        marker, body = match.groups()
        return f"{marker}{_gibberish_text(body, rng)}"
    return _gibberish_text(step, rng)


def _replace_steps_in_content(
    normalized_content: str,
    cot_steps: list[str],
    modified_steps: list[str],
) -> str:
    pieces = []
    cursor = 0

    for original_step, modified_step in zip(cot_steps, modified_steps):
        step_start = normalized_content.find(original_step, cursor)
        if step_start < 0:
            raise ValueError(f"Unable to locate CoT step in normalized content: {original_step!r}")
        pieces.append(normalized_content[cursor:step_start])
        pieces.append(modified_step)
        cursor = step_start + len(original_step)

    pieces.append(normalized_content[cursor:])
    return "".join(pieces)


def _apply_no_cots(
    normalized_content: str,
    cot_steps: list[str],
) -> ConfidenceModificationResult:
    del normalized_content
    return ConfidenceModificationResult(
        mode="no-cots",
        modified_content="",
        modified_steps=[],
        step_mappings=[
            {"original_step": step, "modified_step": ""}
            for step in cot_steps
        ],
    )


def _apply_gibberish(
    normalized_content: str,
    cot_steps: list[str],
) -> ConfidenceModificationResult:
    rng = random.Random(_seed_from_text(normalized_content))
    modified_steps = [_gibberish_step(step, rng) for step in cot_steps]
    modified_content = _replace_steps_in_content(normalized_content, cot_steps, modified_steps)
    return ConfidenceModificationResult(
        mode="gibberish",
        modified_content=modified_content,
        modified_steps=modified_steps,
        step_mappings=[
            {"original_step": original_step, "modified_step": modified_step}
            for original_step, modified_step in zip(cot_steps, modified_steps)
        ],
    )


_MODIFICATION_HANDLERS = {
    "no-cots": _apply_no_cots,
    "gibberish": _apply_gibberish,
}


def apply_confidence_modification(
    normalized_content: str,
    cot_steps: list[str],
    mode: str | None,
) -> ConfidenceModificationResult:
    if mode is None:
        return ConfidenceModificationResult(
            mode=None,
            modified_content=normalized_content,
            modified_steps=list(cot_steps),
            step_mappings=[
                {"original_step": step, "modified_step": step}
                for step in cot_steps
            ],
        )

    return _MODIFICATION_HANDLERS[mode](normalized_content, cot_steps)
