#!/usr/bin/env python3
"""Score the reasoning tokens in stored StepBootstrap prompts.

The input prompts are already fully rendered chat-template strings. This script
therefore tokenizes each complete prompt directly and computes teacher-forced
causal log likelihoods for only the numbered reasoning-step tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.enum import MODEL_DICT


LOGGER = logging.getLogger(__name__)
STEP_MARKER_RE = re.compile(r"(?im)^[ \t]*Step\s+(\d+)\s*:")
ANSWER_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:therefore[ \t]+)?(?:the[ \t]+)?"
    r"(?:final[ \t]+)?answer\b"
)


@dataclass(frozen=True)
class StepSpan:
    """A numbered reasoning step's character span in a complete prompt."""

    display_number: int
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-step and whole-CoT mean token log likelihoods for "
            "the complete prompts stored in a StepBootstrap JSON file."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="StepBootstrap JSON file.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model alias from utils/enum.py or a Hugging Face model identifier.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of full prompts per forward pass (default: 1).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model loading dtype (default: bfloat16 on CUDA, float32 on CPU).",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Optional Transformers attention implementation override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N bootstrap prompts.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def resolve_model_name(model: str) -> str:
    """Resolve a repository model alias, leaving full model IDs unchanged."""

    return MODEL_DICT.get(model, model)


def load_stepbootstrap_data(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Load complete answer-logit prompts and their bootstrap metadata."""

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    containers: list[Any] = [data]
    if isinstance(data, dict):
        trajectory = data.get("trajectory")
        if isinstance(trajectory, dict):
            containers.append(trajectory)
            debug_info = trajectory.get("debug_info")
            if isinstance(debug_info, dict):
                containers.append(debug_info)
        debug_info = data.get("debug_info")
        if isinstance(debug_info, dict):
            containers.append(debug_info)

    prompt_container = None
    sample_container = None
    for container in containers:
        if not isinstance(container, dict):
            continue
        if prompt_container is None and "bootstrap_prompts" in container:
            prompt_container = container["bootstrap_prompts"]
        if sample_container is None and "bootstrap_samples" in container:
            sample_container = container["bootstrap_samples"]

    if not isinstance(prompt_container, dict):
        raise ValueError(
            "Could not find bootstrap_prompts in the input JSON. Expected "
            "trajectory.debug_info.bootstrap_prompts or a direct equivalent."
        )
    prompts = prompt_container.get("answer_logits")
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("bootstrap_prompts.answer_logits must be a list of strings.")
    if not isinstance(sample_container, list) or not all(
        isinstance(item, dict) for item in sample_container
    ):
        raise ValueError("bootstrap_samples must be a list of objects.")
    if len(prompts) != len(sample_container):
        raise ValueError(
            "StepBootstrap prompt/sample count mismatch: "
            f"{len(prompts)} prompts versus {len(sample_container)} samples."
        )
    return prompts, sample_container


def _find_answer_start(prompt: str, search_start: int) -> int:
    """Find the answer-sentence boundary following the reasoning chain."""

    answer_match = ANSWER_LINE_RE.search(prompt, search_start)
    boxed_at = prompt.find(r"\boxed", search_start)
    candidates: list[int] = []
    if answer_match is not None:
        candidates.append(answer_match.start())
    if boxed_at >= 0:
        line_start = prompt.rfind("\n", search_start, boxed_at) + 1
        candidates.append(max(line_start, search_start))
    if not candidates:
        raise ValueError("Could not find a final-answer boundary after the reasoning steps.")
    return min(candidates)


def find_reasoning_step_spans(prompt: str, expected_step_count: int) -> list[StepSpan]:
    """Find the final consecutive Step 1..N chain in a complete rendered prompt."""

    if expected_step_count < 1:
        raise ValueError("A bootstrap sample must contain at least one kept step.")

    markers = list(STEP_MARKER_RE.finditer(prompt))
    expected_numbers = list(range(1, expected_step_count + 1))
    candidates: list[tuple[list[re.Match[str]], int]] = []

    for start_idx, marker in enumerate(markers):
        candidate = markers[start_idx : start_idx + expected_step_count]
        if len(candidate) != expected_step_count:
            continue
        numbers = [int(item.group(1)) for item in candidate]
        if numbers != expected_numbers:
            continue
        try:
            answer_start = _find_answer_start(prompt, candidate[-1].end())
        except ValueError:
            continue
        if answer_start > candidate[-1].end():
            candidates.append((candidate, answer_start))

    if not candidates:
        raise ValueError(
            f"Could not locate a final consecutive Step 1..{expected_step_count} "
            "chain followed by an answer boundary."
        )

    selected, answer_start = candidates[-1]
    spans: list[StepSpan] = []
    for idx, marker in enumerate(selected):
        end = selected[idx + 1].start() if idx + 1 < len(selected) else answer_start
        while end > marker.start() and prompt[end - 1].isspace():
            end -= 1
        spans.append(
            StepSpan(
                display_number=int(marker.group(1)),
                start=marker.start(),
                end=end,
            )
        )
    return spans


def map_tokens_to_steps(
    offsets: Sequence[Sequence[int]], spans: Sequence[StepSpan]
) -> list[list[int]]:
    """Assign tokens to the step span with which they have greatest overlap."""

    step_token_indices: list[list[int]] = [[] for _ in spans]
    for token_idx, raw_offset in enumerate(offsets):
        token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
        if token_end <= token_start:
            continue
        overlaps = [
            max(0, min(token_end, span.end) - max(token_start, span.start))
            for span in spans
        ]
        best_overlap = max(overlaps, default=0)
        if best_overlap > 0:
            step_token_indices[overlaps.index(best_overlap)].append(token_idx)

    empty_steps = [idx + 1 for idx, indices in enumerate(step_token_indices) if not indices]
    if empty_steps:
        raise ValueError(f"No tokenizer tokens mapped to reasoning step(s) {empty_steps}.")
    if step_token_indices[0][0] == 0:
        raise ValueError("The first input token cannot be scored causally.")
    return step_token_indices


def selected_token_log_likelihoods(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_indices: Sequence[int],
) -> torch.Tensor:
    """Return log p(input_ids[t] | input_ids[:t]) for selected token positions."""

    if logits.ndim != 2 or input_ids.ndim != 1:
        raise ValueError("Expected logits [sequence, vocabulary] and input_ids [sequence].")
    indices = torch.as_tensor(token_indices, device=logits.device, dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError("At least one token index is required.")
    if torch.any(indices <= 0) or torch.any(indices >= input_ids.shape[0]):
        raise ValueError("Token indices must be in the causal range [1, sequence_length).")

    selected_logits = logits.index_select(0, indices - 1).float()
    targets = input_ids.to(logits.device).index_select(0, indices)
    target_logits = selected_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return target_logits - torch.logsumexp(selected_logits, dim=-1)


def summarize_step_log_likelihoods(
    token_log_likelihoods: torch.Tensor,
    step_token_counts: Sequence[int],
) -> tuple[list[float], float]:
    """Aggregate flat token scores into per-step and token-weighted CoT means."""

    if sum(step_token_counts) != token_log_likelihoods.numel():
        raise ValueError("Step token counts do not match the number of token scores.")
    if not step_token_counts or any(count < 1 for count in step_token_counts):
        raise ValueError("Every reasoning step must contain at least one token.")

    means: list[float] = []
    offset = 0
    for count in step_token_counts:
        means.append(token_log_likelihoods[offset : offset + count].mean().item())
        offset += count
    return means, token_log_likelihoods.mean().item()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _model_input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _context_limit(tokenizer: Any, model: torch.nn.Module) -> int | None:
    limits: list[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = getattr(model.config, "max_position_embeddings", None)
    for value in (tokenizer_limit, model_limit):
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            limits.append(value)
    return min(limits) if limits else None


def _numbered_source_step_indices(
    sample: dict[str, Any], sample_idx: int
) -> list[Any] | None:
    """Return source indices for numbered kept steps, excluding any preamble."""

    kept_steps = sample.get("kept_steps")
    if not isinstance(kept_steps, list) or not all(
        isinstance(step, str) for step in kept_steps
    ):
        raise ValueError(f"bootstrap_samples[{sample_idx}].kept_steps must be a list of strings.")
    numbered_positions = [
        idx for idx, step in enumerate(kept_steps) if STEP_MARKER_RE.match(step)
    ]
    if not numbered_positions:
        raise ValueError(
            f"bootstrap_samples[{sample_idx}] contains no numbered kept steps."
        )

    step_counts = sample.get("step_counts")
    if isinstance(step_counts, list) and all(
        isinstance(count, int) and count >= 0 for count in step_counts
    ):
        if sum(step_counts) != len(numbered_positions):
            raise ValueError(
                f"bootstrap_samples[{sample_idx}] metadata mismatch: step_counts "
                f"sum to {sum(step_counts)}, but {len(numbered_positions)} numbered "
                "kept steps were found."
            )

    kept_indices = sample.get("kept_step_indices")
    if not isinstance(kept_indices, list):
        return None
    if len(kept_indices) != len(kept_steps):
        raise ValueError(
            f"bootstrap_samples[{sample_idx}] has different kept_steps and "
            "kept_step_indices lengths."
        )
    return [kept_indices[idx] for idx in numbered_positions]


def score_prompts(
    prompts: Sequence[str],
    samples: Sequence[dict[str, Any]],
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score complete prompts in batches and return JSON-serializable records."""

    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results: list[dict[str, Any]] = []
    context_limit = _context_limit(tokenizer, model)
    device = _model_input_device(model)

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[batch_start : batch_start + batch_size])
        batch_samples = samples[batch_start : batch_start + batch_size]
        numbered_source_indices = [
            _numbered_source_step_indices(sample, batch_start + local_idx)
            for local_idx, sample in enumerate(batch_samples)
        ]
        expected_counts = [
            len(source_indices)
            if source_indices is not None
            else sum(1 for step in sample["kept_steps"] if STEP_MARKER_RE.match(step))
            for source_indices, sample in zip(numbered_source_indices, batch_samples)
        ]
        batch_spans = [
            find_reasoning_step_spans(prompt, expected_count)
            for prompt, expected_count in zip(batch_prompts, expected_counts)
        ]

        encoded = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(dim=1).tolist()
        if context_limit is not None and max(lengths) > context_limit:
            raise ValueError(
                f"Prompt length {max(lengths)} exceeds model context limit {context_limit}; "
                "the complete StepBootstrap prompt will not be truncated."
            )

        encoded_on_device = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded_on_device, use_cache=False)

        for local_idx, (prompt, sample, spans, sequence_length) in enumerate(
            zip(batch_prompts, batch_samples, batch_spans, lengths)
        ):
            global_idx = batch_start + local_idx
            sample_offsets = offsets[local_idx, :sequence_length].tolist()
            step_token_indices = map_tokens_to_steps(sample_offsets, spans)
            flat_indices = [idx for indices in step_token_indices for idx in indices]
            token_lls = selected_token_log_likelihoods(
                output.logits[local_idx, :sequence_length],
                encoded_on_device["input_ids"][local_idx, :sequence_length],
                flat_indices,
            )
            step_counts = [len(indices) for indices in step_token_indices]
            step_means, all_cot_mean = summarize_step_log_likelihoods(
                token_lls, step_counts
            )

            source_indices = numbered_source_indices[local_idx]
            per_step = []
            for step_idx, (span, token_count, mean_ll) in enumerate(
                zip(spans, step_counts, step_means)
            ):
                record: dict[str, Any] = {
                    "step_idx": step_idx,
                    "display_number": span.display_number,
                    "token_count": token_count,
                    "mean_log_likelihood": mean_ll,
                    "text": prompt[span.start : span.end],
                }
                if source_indices is not None:
                    record["source_step_index"] = source_indices[step_idx]
                per_step.append(record)

            results.append(
                {
                    "sample_idx": sample.get("sample_idx", global_idx),
                    "prompt_token_count": int(sequence_length),
                    "cot_token_count": sum(step_counts),
                    "per_step": per_step,
                    "all_cot": {
                        "token_count": sum(step_counts),
                        "mean_log_likelihood": all_cot_mean,
                    },
                    "kept_step_indices": sample.get("kept_step_indices"),
                    "step_counts": sample.get("step_counts"),
                }
            )
            LOGGER.info(
                "Scored sample %s: %d prompt tokens, %d CoT tokens, mean LL %.6f",
                results[-1]["sample_idx"],
                sequence_length,
                sum(step_counts),
                all_cot_mean,
            )

        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must not overwrite the input StepBootstrap JSON file.")

    prompts, samples = load_stepbootstrap_data(args.input)
    if args.limit is not None:
        prompts = prompts[: args.limit]
        samples = samples[: args.limit]

    model_name = resolve_model_name(args.model)
    dtype = _resolve_dtype(args.dtype)
    LOGGER.info("Loading %s with dtype=%s", model_name, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    scored_samples = score_prompts(
        prompts=prompts,
        samples=samples,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
    )
    result = {
        "source_file": str(args.input.resolve()),
        "model": model_name,
        "dtype": str(dtype).replace("torch.", ""),
        "num_samples": len(scored_samples),
        "samples": scored_samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary_output.replace(args.output)
    LOGGER.info("Wrote %d samples to %s", len(scored_samples), args.output)


if __name__ == "__main__":
    main()
