"""
load_prompt.py

Utility for loading CoT prompts by dataset name.

Usage:
    from prompts.load_prompt import load_prompt

    system_prompt, user_template = load_prompt("hotpotqa")
    user_message = user_template.format(context=ctx, question=q)
"""

from prompts.cot_prompt import PROMPT_REGISTRY

# Canonical dataset names supported by the registry.
SUPPORTED_DATASETS = list(PROMPT_REGISTRY.keys())


def load_prompt(dataset: str) -> tuple[str, str, str]:
    """Return (system_prompt, user_prompt_template) for *dataset*.

    The user_prompt_template is a str.format()-compatible string.
    Placeholder names vary per dataset; see cot_prompt.py for details:
        - All datasets: {question}
        - bfcl:         {functions}
        - logiqa:       {context}, {options}
        - codeqa:       {code}
        - hotpotqa:     {context}

    Args:
        dataset: Case-insensitive dataset name, e.g. "hotpotqa", "math500".

    Returns:
        A (system_prompt, user_prompt_template) tuple.

    Raises:
        ValueError: If the dataset name is not recognised.
    """
    key = dataset.strip().lower()
    if key not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )
    return PROMPT_REGISTRY[key]
