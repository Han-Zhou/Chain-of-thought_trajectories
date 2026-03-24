
import pathlib
import json
from prompts.cot_prompt import PROMPT_REGISTRY
from prompts.few_shot_prompt import FEW_SHOT_PROMPT_REGISTRY, THINKING_TOKENS

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




def load_messages(dataset: str, few_shot: bool, entry: dict, model_name: str, thinking: bool, prompt_type: int = 1) -> list[dict[str, str]]:
    if dataset == "logiqa":
        # For LogiQA, we want to use the system prompt as the reasoning prompt and the original prompt as the specific prompt.
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)

        choices_text = "\n".join(entry["choices"])
        user_prompt = (
            f"Context:\n{entry['context']}\n\n"
            f"Question:\n{entry['question']}\n\n"
            f"Options:\n{choices_text}"
        )

        thinking_token_open, thinking_token_close = THINKING_TOKENS["none"]
        if thinking:
            try:
                thinking_token_open, thinking_token_close = THINKING_TOKENS[model_name]
            except KeyError:
                raise ValueError(f'"{model_name}" does not support thinking yet')

        if prompt_type == 1:
            # Type 1: assistant prefill with thinking tokens + "Step 1:"
            assistant_start = assistant_start.format(
                thinking_token_open=thinking_token_open,
            )

            messages = [
                {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_start}
            ]
        else:
            # Type 2: "Let's think step-by-step." appended to user prompt, no assistant prefill
            user_prompt += "\n\nLet's think step-by-step."

            messages = [
                {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
                {"role": "user", "content": user_prompt},
            ]

        if few_shot:
            few_shot_messages_raw = load_few_shot_prompt_from_registry(dataset)

            use_thinking_field = thinking and "gpt" in model_name.lower()

            few_shot_messages = []
            for message in few_shot_messages_raw:
                if message["role"] == "assistant":
                    if prompt_type == 2:
                        # Type 2: steps outside thinking blocks — strip thinking tokens
                        content = message["content"].format(
                            thinking_token_open="",
                            thinking_token_close="",
                        )
                    else:
                        content = message["content"].format(
                            thinking_token_open=thinking_token_open,
                            thinking_token_close=thinking_token_close,
                        )
                    if use_thinking_field:
                        sep = "\n\\boxed{"
                        idx = content.find(sep)
                        if idx != -1:
                            few_shot_messages.append({
                                "role": "assistant",
                                "thinking": content[:idx].strip(),
                                "content": content[idx + 1:].strip(),
                            })
                        else:
                            few_shot_messages.append({"role": "assistant", "content": content})
                    else:
                        few_shot_messages.append({"role": "assistant", "content": content})
                else:
                    few_shot_messages.append(message)

            messages = messages[0:1] + few_shot_messages + messages[1:]

            # breakpoint()

        return messages

    else:
        raise NotImplementedError(f"Message loading not implemented for dataset '{dataset}'.")

