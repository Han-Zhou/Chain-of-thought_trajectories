"""
main_debug.py

Debug entry point for the CoT evaluation pipeline.

Loads pre-generated trajectories from a pickle file (e.g.,
llama_8b_31_bigbench_movie_recommendation.pkl) and runs confidence scoring
on them. The pickle file contains entries with system_prompt, fs_prompt,
question, answer, and sampled_cots.
"""

import argparse
import os
import logging
import json
import time
import pickle

import torch
from tqdm.contrib.discord import tqdm as tqdm_discord
from tqdm import tqdm
from dotenv import load_dotenv

from utils.enum import MODEL_DICT
from dataclasses import asdict
from utils.structures import ParsedOutput, AllConfidenceData
from parsing import parse_output
from confidence_debug import compute_all_confidence_scores
from llm import LLM
from eval.registry import evaluate_one

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_pickle_data(pickle_path: str) -> list[dict]:
    """Load pre-generated trajectories from a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def construct_messages_from_entry(entry: dict, generated_text: str, few_shot: bool = True) -> list[dict]:
    """Construct chat messages from a pickle entry.

    Args:
        entry: Dict with system_prompt, fs_prompt, question, answer, sampled_cots
        generated_text: The full generated CoT text (from sampled_cots)
        few_shot: Whether to include few-shot examples in the messages

    Returns:
        List of message dicts in chat format
    """
    messages = []

    # System message
    messages.append({"role": "system", "content": entry["system_prompt"]})

    # Few-shot examples
    if few_shot and entry.get("fs_prompt"):
        for fs in entry["fs_prompt"]:
            messages.append({"role": "user", "content": fs["question"]})
            messages.append({"role": "assistant", "content": fs["cot_with_answer"]})

    # Actual question
    messages.append({"role": "user", "content": entry["question"]})

    # Assistant prefill with the generated text
    messages.append({"role": "assistant", "content": generated_text})

    return messages






def parse():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Path to the pickle file containing pre-generated trajectories."
    )
    args.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "gpt", "qwen", "qwen-fp8", "qwen-gptq"],
        help="Model to use for confidence scoring."
    )
    args.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process from the dataset. Defaults to all."
    )
    args.add_argument(
        "--sample_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Slice [start, end) of the dataset to process. Mutually exclusive with --sample_size."
    )
    args.add_argument(
        "--confidence",
        action="store_true",
        default=False,
        help="Whether or not to evaluate confidence on the COTs."
    )
    args.add_argument(
        "--discord",
        action="store_true",
        default=False,
        help="Send tqdm progress to Discord via TQDM_DISCORD_TOKEN and TQDM_DISCORD_CHANNEL_ID env vars."
    )
    args.add_argument(
        "--debug_conf",
        action="store_true",
        default=False,
        help="Save detailed confidence debug info to debug_conf.json in the output directory."
    )
    args.add_argument(
        "--experimental_jackknife",
        action="store_true",
        default=False,
        help=(
            "Use jackknife step masking instead of coin-flip dropout. "
            "Keeps ceil(log(k)) of k CoT steps per sample (uniform random without replacement) "
            "rather than flipping an independent coin per step."
        ),
    )
    args.add_argument(
        "--nb_dropout_samples",
        type=int,
        default=3,
        help="Number of dropout samples for confidence scoring."
    )
    args.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional prefix for the output directory name."
    )
    args.add_argument(
        "--shot_mode",
        type=str,
        default="zero",
        choices=["zero", "few"],
        help="Prompting mode: zero-shot (no few-shot examples) or few-shot."
    )
    args.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name for evaluation (e.g., 'bigbench_movie'). If not provided, will try to extract from pickle."
    )
    parsed = args.parse_args()
    if parsed.sample_size is not None and parsed.sample_range is not None:
        args.error("--sample_size and --sample_range are mutually exclusive.")
    return parsed





def reconstruct_cot_from_partial(partial_cots: list[str], cot_answer: str) -> str:
    """Reconstruct CoT text from partial_cots with Step prefixes.

    partial_cots[0] is the preamble (e.g., "Answer: Let's think step by step.")
    partial_cots[i] - partial_cots[i-1] gives step i's content (for i >= 1)

    Returns formatted text like:
    Answer: Let's think step by step.
    Step 1: <content>
    Step 2: <content>
    ...
    Therefore the final answer is \\boxed{<answer>}.
    """
    # Keep the answer as-is (besides whitespace trimming)
    clean_answer = cot_answer.strip()

    # NOTE: previously we stripped surrounding parentheses like "(C)" -> "C".
    # This is intentionally disabled.
    # if clean_answer.startswith("(") and clean_answer.endswith(")"):
    #     clean_answer = clean_answer[1:-1]

    if not partial_cots:
        return f"Therefore the final answer is \\boxed{{{clean_answer}}}."

    parts = []

    # First element is the preamble (e.g., "Answer: Let's think step by step.")
    preamble = partial_cots[0].strip()
    parts.append(preamble)

    # Subsequent elements are actual CoT steps
    for i in range(1, len(partial_cots)):
        prev = partial_cots[i - 1]
        step_content = partial_cots[i][len(prev):].strip()
        if step_content:
            parts.append(f"Step {i}: {step_content}")

    # Join parts and add final answer
    cot_text = "\n".join(parts)
    cot_text += f"\nTherefore the final answer is \\boxed{{{clean_answer}}}."
    return cot_text


def score_trajectories(model_name, pickle_data, out_dir, confidence=False, debug_conf=False, experimental_jackknife=False, discord=False, nb_dropout_samples=3, shot_mode="zero", dataset_name=None):
    """Score pre-generated trajectories from pickle data using confidence metrics."""
    llm = LLM(model_name, thinking=False)
    tokenizer = llm.tokenizer
    errors = []
    few_shot = (shot_mode == "few")

    progress = tqdm_discord if discord else tqdm
    for i, entry in enumerate(progress(pickle_data, desc="Scoring", unit="sample")):
        logger.info(f"Scoring {i}/{len(pickle_data)}")
        t0 = time.time()

        try:
            # Get the CoT text from sampled_cots
            if not entry.get("sampled_cots"):
                raise ValueError("No sampled_cots in entry")

            # Take first 10 sampled_cots that have valid answers (not just parentheses)
            sampled_cots_with_answers = [
                cot for cot in entry["sampled_cots"]
                if cot.get("cot_answer") and cot["cot_answer"].strip("() \t\n")
            ][:10]

            if not sampled_cots_with_answers:
                raise ValueError("No sampled_cots with answers in entry")

            for sample_idx, cot_entry in enumerate(sampled_cots_with_answers):
                partial_cots = cot_entry.get("partial_cots", [])
                cot_answer = cot_entry.get("cot_answer", "")
                full_generated_text = reconstruct_cot_from_partial(partial_cots, cot_answer)

                # Construct messages from entry
                messages = construct_messages_from_entry(entry, full_generated_text, few_shot=few_shot)

                # Parse the output
                parsed: ParsedOutput = parse_output(full_generated_text)

                if parsed.answer_fullstring_start is not None:
                    generated_prefix = full_generated_text[:parsed.answer_fullstring_start]
                    prefix_ids = tokenizer(generated_prefix, add_special_tokens=False)["input_ids"]
                    answer_token_start_position = len(prefix_ids)
                else:
                    answer_token_start_position = None

                confidence_score = None
                debug_info = None
                if confidence and parsed.final_answer:
                    llm.switch_attn_implementation("confidence")
                    confidence_data: AllConfidenceData = compute_all_confidence_scores(
                        llm,
                        messages,
                        full_generated_text,
                        parsed,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=False,
                        assistant_prefill="",
                        debug_conf=debug_conf,
                        gen_cache=None,
                        experimental_jackknife=experimental_jackknife,
                    )
                    llm.switch_attn_implementation("cot")
                    debug_info = confidence_data.debug_info if debug_conf else None
                    confidence_score = asdict(confidence_data)
                    confidence_score.pop("debug_info", None)

                torch.cuda.empty_cache()

                # Evaluate correctness
                if dataset_name:
                    eval_result = evaluate_one({"generated_text": full_generated_text, "ground_truth": entry["answer"]}, dataset_name)
                    correct = eval_result["score"] > 0 if eval_result else None
                else:
                    correct = None

                traj = {
                    "question":                     entry["question"],
                    "ground_truth":                 entry["answer"],
                    "sample_idx":                   sample_idx,
                    "cot_steps":                    parsed.cot_steps,
                    "final_answer":                 parsed.final_answer,
                    "raw_cot_block":                parsed.raw_cot_block,
                    "generated_text":               full_generated_text,
                    "answer_token_start_position":  answer_token_start_position,
                    "confidence_score":             confidence_score,
                    "confidence_method":            "coin_flip+jackknife" if experimental_jackknife else "coin_flip",
                    "debug_info":                   debug_info if debug_conf else None,
                    "runtime_seconds":              time.time() - t0,
                    "correct":                      correct,
                }

                traj_file = os.path.join(out_dir, f"traj_{i}_sample_{sample_idx}.json")
                with open(traj_file, "w") as f:
                    json.dump({"index": i, "sample_idx": sample_idx, "question": entry["question"], "trajectory": traj}, f, indent=2)

        except Exception as e:
            logger.error(f"Error processing entry {i}: {e}", exc_info=True)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                logger.warning("Cleared CUDA cache after OOM error")
            errors.append({"index": i, "error": str(e)})
            traj = {
                "question":                     entry.get("question", ""),
                "ground_truth":                 entry.get("answer", ""),
                "sample_idx":                   None,
                "error":                        str(e),
                "cot_steps":                    None,
                "final_answer":                 None,
                "raw_cot_block":                None,
                "generated_text":               None,
                "answer_token_start_position":  None,
                "confidence_score":             None,
                "confidence_method":            None,
                "runtime_seconds":              time.time() - t0,
                "correct":                      None,
            }
            traj_file = os.path.join(out_dir, f"traj_{i}_error.json")
            with open(traj_file, "w") as f:
                json.dump({"index": i, "trajectory": traj}, f, indent=2)

    return errors


def main(args):
    t_start = time.time()

    pickle_data = load_pickle_data(args.pickle_path)
    logger.info(f"Loaded {len(pickle_data)} entries from '{args.pickle_path}'.")
    logger.info("Sample entry keys: %s", list(pickle_data[0].keys()))

    model_name = MODEL_DICT[args.model]

    if args.sample_range is not None:
        start = max(args.sample_range[0], 0)
        end = min(args.sample_range[1], len(pickle_data))
        pickle_data = pickle_data[start:end]
    elif args.sample_size is not None:
        start = 0
        end = min(args.sample_size, len(pickle_data))
        pickle_data = pickle_data[start:end]
    else:
        start = 0
        end = len(pickle_data)

    logger.info(f"Processing [{start}:{end}] ({len(pickle_data)} entries)")

    # Determine dataset name for evaluation
    dataset_name = args.dataset
    if not dataset_name and pickle_data:
        # Try to extract from pickle data
        raw_name = pickle_data[0].get("dataset_name", "")
        # Convert to lowercase and map to registry name (e.g., "Bigbench_Movie_Recommendation" -> "bigbench_movie")
        dataset_name = raw_name.lower().replace("bigbench_", "bigbench_").replace("_recommendation", "")
        logger.info(f"Extracted dataset name from pickle: {dataset_name}")

    pickle_basename = os.path.splitext(os.path.basename(args.pickle_path))[0]
    dir_name = f"{args.model}_{pickle_basename}_{start}_{end}_{'conf' if args.confidence else 'vanilla'}_{'jackknife' if args.experimental_jackknife else 'coinflip'}"
    if args.tag:
        dir_name = f"{args.tag}_{dir_name}"
    out_dir = f"trajectories/{dir_name}"
    os.makedirs(out_dir, exist_ok=True)

    errors = score_trajectories(
        model_name, pickle_data, out_dir,
        confidence=args.confidence,
        debug_conf=args.debug_conf,
        experimental_jackknife=args.experimental_jackknife,
        discord=args.discord,
        nb_dropout_samples=args.nb_dropout_samples,
        shot_mode=args.shot_mode,
        dataset_name=dataset_name,
    )

    if errors:
        errors_file = os.path.join(out_dir, "errors.json")
        with open(errors_file, "w") as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"{len(errors)} errors encountered. Saved to {errors_file}")

    total_time = time.time() - t_start
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    args = parse()
    main(args)
