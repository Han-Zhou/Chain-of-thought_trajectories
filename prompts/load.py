
import pathlib
import json
from prompts.cot_prompt import PROMPT_REGISTRY
from prompts.few_shot_prompt import FEW_SHOT_PROMPT_REGISTRY

# Canonical dataset names supported by the registry.
SUPPORTED_DATASETS = list(PROMPT_REGISTRY.keys())


def load_prompt_from_registry(dataset: str) -> tuple[str, str, str]:
    key = dataset.strip().lower()
    if key not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )
    return PROMPT_REGISTRY[key]


def load_few_shot_prompt_from_registry(dataset: str) -> dict[str, str]:
    key = dataset.strip().lower()
    if key not in FEW_SHOT_PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}' for few-shot prompt. "
            f"Supported datasets: {list(FEW_SHOT_PROMPT_REGISTRY.keys())}"
        )
    return FEW_SHOT_PROMPT_REGISTRY[key]




def load_messages(dataset: str, few_shot: bool, entry: dict) -> list[dict[str, str]]:
    if dataset == "logiqa":
        # For LogiQA, we want to use the system prompt as the reasoning prompt and the original prompt as the specific prompt.
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)

        choices_text = "\n".join(entry["choices"])
        user_prompt = (
            f"Context:\n{entry['context']}\n\n"
            f"Question:\n{entry['question']}\n\n"
            f"Options:\n{choices_text}"
        )

        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start}
        ]

        if few_shot:
            few_shot_messages = load_few_shot_prompt_from_registry(dataset)
            messages = messages[0:1] + few_shot_messages + messages[1:]

        # breakpoint()

        return messages

    else:
        raise NotImplementedError(f"Message loading not implemented for dataset '{dataset}'.")

