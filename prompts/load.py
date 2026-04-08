
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


def _resolve_thinking_tokens(model_name: str, thinking: bool) -> tuple[str, str]:
    thinking_token_open, thinking_token_close = THINKING_TOKENS["none"]
    if thinking:
        try:
            thinking_token_open, thinking_token_close = THINKING_TOKENS[model_name]
        except KeyError:
            raise ValueError(f'"{model_name}" does not support thinking yet')
    return thinking_token_open, thinking_token_close


def _materialize_few_shot_messages(
    dataset: str,
    model_name: str,
    thinking: bool,
    prompt_type: int,
) -> list[dict[str, str]]:
    few_shot_messages_raw = load_few_shot_prompt_from_registry(dataset)
    thinking_token_open, thinking_token_close = _resolve_thinking_tokens(model_name, thinking)
    use_thinking_field = thinking and "gpt" in model_name.lower()

    few_shot_messages = []
    for message in few_shot_messages_raw:
        if message["role"] == "assistant":
            if prompt_type == 2:
                content = (message["content"]
                    .replace("{thinking_token_open}", "")
                    .replace("{thinking_token_close}", ""))
            else:
                content = (message["content"]
                    .replace("{thinking_token_open}", thinking_token_open)
                    .replace("{thinking_token_close}", thinking_token_close))
            # if use_thinking_field:
            #     sep = "\\boxed{"
            #     idx = content.find(sep)
            #     if idx != -1:
            #         few_shot_messages.append({
            #             "role": "assistant",
            #             "thinking": content[:idx].strip(),
            #             "content": content[idx:].strip(),
            #         })
            #     else:
            #         few_shot_messages.append({"role": "assistant", "content": content})
            # else:
            few_shot_messages.append({"role": "assistant", "content": content})
        else:
            few_shot_messages.append(message)
    
    return few_shot_messages



def _build_messages(
    system_prompt: str,
    user_prompt: str,
    assistant_start: str,
    model_name: str,
    thinking: bool,
    prompt_type: int,
    few_shot: bool,
    few_shot_key: str,
) -> list[dict[str, str]]:
    """Common message assembly: thinking tokens, prompt_type branching, few-shot insertion."""
    thinking_token_open, _ = _resolve_thinking_tokens(model_name, thinking)

    if prompt_type == 1:
        assistant_start = assistant_start.format(thinking_token_open=thinking_token_open)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start},
        ]
    else:
        user_prompt += "\n\nLet's think step-by-step."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    if few_shot:
        few_shot_messages = _materialize_few_shot_messages(few_shot_key, model_name, thinking, prompt_type)
        messages = messages[0:1] + few_shot_messages + messages[1:]

    return messages


_SUPPORTED_MESSAGE_DATASETS = {
    "logiqa", "hle",
    "cs1qa_type", "cs1qa_span", "cs1qa_answer",
    "college_math", "bigbench_causal", "bigbench_movie",
    "hotpotqa", "math500", "olympiadbench",
    "bigbench_movie_nobox",
}


def load_messages(dataset: str, few_shot: bool, entry: dict, model_name: str, thinking: bool, prompt_type: int = 1, experimental_extra_box: bool = False) -> list[dict[str, str]]:
    if dataset not in _SUPPORTED_MESSAGE_DATASETS:
        raise NotImplementedError(f"Message loading not implemented for dataset '{dataset}'.")

    # college_math uses a different registry key than its dataset name
    registry_key = "college_math_test" if dataset == "college_math" else dataset
    if experimental_extra_box:
        registry_key = "bigbench_movie_nobox"
    prompt_parts = load_prompt_from_registry(registry_key)

    if dataset == "logiqa":
        # Registry returns (reasoning_prompt, specific_prompt, assistant_start) for logiqa
        reasoning_prompt, specific_prompt, assistant_start = prompt_parts
        system_prompt = reasoning_prompt + "\n\n" + specific_prompt
        choices_text = "\n".join(entry["choices"])
        user_prompt = (
            f"Context:\n{entry['context']}\n\n"
            f"Question:\n{entry['question']}\n\n"
            f"Options:\n{choices_text}"
        )

    elif dataset in ("cs1qa_type", "cs1qa_span", "cs1qa_answer"):
        system_prompt, user_template, assistant_start = prompt_parts
        code = entry.get("context") or entry.get("code") or ""
        user_prompt = user_template.format(question=entry["question"], code=code)

    else:
        # Default path: covers hle, college_math, hotpotqa, bigbench_causal,
        # bigbench_movie, math500, olympiadbench, and any future simple datasets.
        system_prompt, user_template, assistant_start = prompt_parts
        user_prompt = user_template.format(question=entry["question"])

        if dataset in ("college_math", "hotpotqa") and entry.get("context"):
            user_prompt = f"Context:\n{entry['context']}\n\n{user_prompt}"

        if dataset == "hle" and entry.get("choices"):
            choices_text = "\n".join(entry["choices"])
            user_prompt += f"\n\nAnswer Choices:\n{choices_text}"

    return _build_messages(
        system_prompt, user_prompt, assistant_start,
        model_name, thinking, prompt_type, few_shot, registry_key,
    )

