
import pathlib
import json
from prompts.cot_prompt import PROMPT_REGISTRY

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

# Few-shot demonstrations loaded from prompts/few_shot_examples/logiqa.json.
_FEW_SHOT_PATH = (
    pathlib.Path(__file__).parent / "prompts" / "few_shot_examples" / "logiqa.json"
)
with _FEW_SHOT_PATH.open() as _f:
    FEW_SHOT_EXAMPLES: list = json.load(_f)


def load_few_shot(dataset: str) -> str:
    raise NotImplementedError("Few-shot loading not implemented for any dataset yet.")


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

        if few_shot:
            user_prompt = load_few_shot(dataset) + "\n\n" + user_prompt

        return [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start}
        ]

    else:
        raise NotImplementedError(f"Message loading not implemented for dataset '{dataset}'.")

