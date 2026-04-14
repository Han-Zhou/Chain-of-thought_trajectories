from utils.enum import LETTERS


_PHASE1_LLAMA_ANSWER_STOP_STRINGS = [
    "\nAnswer:",
    "\nFinal Answer",
    "\nFinal answer",
    "\nThe final answer",
    "\nThe answer",
    "\nTherefore the answer",
    "\nTherefore the final answer",
]


def is_llama_model_name(model_name: str) -> bool:
    return "llama" in model_name.lower()


def is_qwen_model_name(model_name: str) -> bool:
    return "qwen" in model_name.lower()


def phase1_stop_strings_for_model(model_name: str, thinking: bool) -> list[str] | None:
    """Return phase-1 stop strings for experimental_extra_box."""
    if is_llama_model_name(model_name):
        stop_strings = list(_PHASE1_LLAMA_ANSWER_STOP_STRINGS)
        stop_strings.extend(f"\n\n{letter}\n" for letter in LETTERS)
        stop_strings.extend(f"\n\n({letter})\n" for letter in LETTERS)
        return stop_strings

    if thinking and is_qwen_model_name(model_name):
        return ["</think>"]

    return None


def strip_trailing_stop_strings(text: str, stop_strings: list[str] | None) -> str:
    """Remove any matched stop strings from the end of text."""
    if not stop_strings:
        return text

    cleaned = text

    while True:
        matched = None
        for marker in sorted(stop_strings, key=len, reverse=True):
            if marker and cleaned.endswith(marker):
                matched = marker
                break
            compact_marker = marker.rstrip()
            if compact_marker and cleaned.endswith(compact_marker):
                matched = compact_marker
                break

        if matched is None:
            return cleaned

        cleaned = cleaned[: -len(matched)].rstrip()


def clean_phase1_generated_text(model_name: str, text: str, stop_strings: list[str] | None) -> str:
    """Apply llama-specific phase-1 cleanup without affecting other models."""
    if not is_llama_model_name(model_name):
        return text

    return strip_trailing_stop_strings(text, stop_strings)
