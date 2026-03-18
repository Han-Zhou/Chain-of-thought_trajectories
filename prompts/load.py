
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




def load_messages(dataset: str, few_shot: bool, entry: dict, model_name: str, thinking: bool) -> list[dict[str, str]]:
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

        assistant_start = assistant_start.format(
            thinking_token_open=thinking_token_open,
        )

        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start}
        ]

        if few_shot:
            few_shot_messages_raw = load_few_shot_prompt_from_registry(dataset)

            use_thinking_field = thinking and "gpt" in model_name.lower()

            few_shot_messages = []
            for message in few_shot_messages_raw:
                if message["role"] == "assistant":
                    content = message["content"].format(
                        thinking_token_open=thinking_token_open,
                        thinking_token_close=thinking_token_close,
                    )
                    if use_thinking_field:
                        sep = "\nFinal Answer:"
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

    elif dataset == "codeqa":
        # For CodeQA, SYSTEM is the generic CoT instruction (system message),
        # and CODEQA template is filled in as the user message.
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)

        user_prompt = (
            f"Code snippet:\n```python\n{entry['context']}\n```\n\n"
            f"Question:\n{entry['question']}"
        )

        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start}
        ]

        if few_shot:
            few_shot_messages = load_few_shot_prompt_from_registry(dataset)
            messages = messages[0:1] + few_shot_messages + messages[1:]

        return messages

    elif dataset == "bfcl":
        system_prompt, bfcl_template, assistant_start = load_prompt_from_registry(dataset)
        functions = entry.get("metadata", {}).get("functions", [])
        functions_json = json.dumps(functions, indent=2)
        system_content = system_prompt + "\n\n" + bfcl_template.format(functions=functions_json)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset in ("bigbench_movie", "bigbench_causal"):
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        user_prompt = entry["question"]
        if entry.get("choices"):
            choices_text = "\n".join(entry["choices"])
            user_prompt += f"\n\nChoices:\n{choices_text}"
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset == "cs1qa":
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        if entry.get("context"):
            user_prompt = (
                f"Code:\n```python\n{entry['context']}\n```\n\n"
                f"Question:\n{entry['question']}"
            )
        else:
            user_prompt = f"Question:\n{entry['question']}"
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset == "hotpotqa":
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        context = entry.get("context") or ""
        user_prompt = (
            f"Supporting passages:\n{context}\n\n"
            f"Question:\n{entry['question']}"
        )
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset == "college_math_test":
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        user_prompt = f"Problem:\n{entry['question']}"
        if entry.get("choices"):
            choices_text = "\n".join(entry["choices"])
            user_prompt += f"\n\nChoices:\n{choices_text}"
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset == "olympiadbench":
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        if entry.get("context"):
            user_prompt = f"Context:\n{entry['context']}\n\nProblem:\n{entry['question']}"
        else:
            user_prompt = f"Problem:\n{entry['question']}"
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset == "math500":
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        user_prompt = f"Problem:\n{entry['question']}"
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    elif dataset == "hle":
        system_reasoning_prompt, system_specific_prompt, assistant_start = load_prompt_from_registry(dataset)
        user_prompt = entry["question"]
        if entry.get("choices"):
            choices_text = "\n".join(entry["choices"])
            user_prompt += f"\n\nChoices:\n{choices_text}"
        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
        return messages

    else:
        raise NotImplementedError(f"Message loading not implemented for dataset '{dataset}'.")

