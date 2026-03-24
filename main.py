"""
main.py

Entry point for the CoT evaluation pipeline.

Supports multiple datasets via --dataset. For LogiQA, runs a full
Chain-of-Thought pipeline with DynamicCache-based generation and structured
output parsing. All other datasets use the generic generate_trajectories() path.

LogiQA pipeline sections (inlined below):
  1. Prompt Formatting   – zero-shot and few-shot CoT builders
  2. Generation          – model.generate() with DynamicCache + position tracking
  3. Parsing             – CoT step extraction and final-answer detection
  4. Evaluation Loop     – orchestrates sections 1-3 across the dataset
"""

import re
import argparse
import os
import logging
import json
import time
import pickle

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
# from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from utils.enum import MODEL_DICT, LETTERS
from dataclasses import asdict
from utils.structures import GenerationResult, ParsedOutput, AllConfidenceData, ConfidenceScores
from prompts.load import load_messages
from prompts.cot_prompt import SYSTEM as _SYSTEM_BASE
from dataloader import load_dataset, make_dataloader
from parsing import parse_output
# from utils.confidence_prev import compute_confidence_metrics
from confidence import compute_all_confidence_scores
from llm import LLM

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
#






def parse():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Dataset to process. One of: bfcl, bigbench_movie, "
            "bigbench_causal, logiqa, codeqa, cs1qa, hotpotqa, "
            "college_math, olympiadbench, math500, hle."
        )
    )
    args.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "gpt", "qwen"],
        help="Model to use for inference."
    )
    args.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to process from the dataset. Defaults to all."
    )
    args.add_argument(
        "--shot_mode",
        type=str,
        default="zero",
        choices=["zero", "few"],
        help="Prompting mode for LogiQA: zero-shot or few-shot CoT."
    )
    args.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate for each prompt."
    )
    args.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Whether to use thinking.",
    )

    args.add_argument(
        "--type",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "Prompt type. "
            "1 (default): assistant prefill with thinking tokens and 'Step 1:'. "
            "2: 'Let's think step-by-step.' in user prompt, no assistant prefill."
        ),
    )
    args.add_argument(
        "--confidence",
        action="store_true",
        default=False,
        help="Whether or not to evaluate confidence on the COTs."
    )
    args.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Skip generation and load from saved pickle files in debug_cache/."
    )
    args.add_argument(
        "--debug_conf",
        action="store_true",
        default=False,
        help="Save detailed confidence debug info to debug_conf.json in the output directory."
    )
    return args.parse_args()





def generate_trajectories(model_name, dataloader, max_new_tokens, dataset_name=None, shot_mode="zero", thinking=False, confidence=None, debug=False, prompt_type=1, debug_conf=False):

    debug_dir = "debug_cache"

    if debug:
        # Load pre-saved generation results instead of running the model
        cache_file = os.path.join(debug_dir, f"gen_parsed_type{prompt_type}.pkl")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Debug cache not found at {cache_file}. Run without --debug first to generate it.")
        logger.info(f"Debug mode: loading cached generation results from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_entries = pickle.load(f)

        # Only need the tokenizer for answer_token_start_position computation
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_for_confidence = None
    else:
        llm = LLM(model_name, thinking)
        tokenizer = llm.tokenizer
        cached_entries = None
        os.makedirs(debug_dir, exist_ok=True)
        entries_to_save = []

    trajectories = []

    for i, entry in enumerate(dataloader):
        logger.info(f"Generating {i}/{len(dataloader)}")
        t0 = time.time()

        if debug:
            gen, parsed, assistant_prefill, full_generated_text, messages = cached_entries[i]
        else:
            messages = load_messages(dataset_name, few_shot=(shot_mode=="few"), entry=entry, model_name=model_name, thinking=thinking, prompt_type=prompt_type)
            has_assistant_prefill = messages[-1]["role"] == "assistant"
            gen: GenerationResult = llm.generate_one(messages, max_new_tokens=max_new_tokens, output_scores=bool(confidence), has_assistant_prefill=has_assistant_prefill)
            assistant_prefill = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
            full_generated_text = assistant_prefill + gen.generated_text

            # For type 2 the model generates freely and may wrap reasoning in
            # <think>...</think>.  We parse only the text *after* the think block
            # so that CoT steps are extracted from the visible output.
            # We then shift the character offsets back so they are relative to
            # full_generated_text (which confidence.py indexes into).
            if prompt_type == 2:
                think_match = re.search(r"</think>\s*", full_generated_text)
                think_block_len = think_match.end() if think_match else 0
                text_for_parsing = full_generated_text[think_block_len:]
            else:
                think_block_len = 0
                text_for_parsing = full_generated_text
            parsed: ParsedOutput = parse_output(text_for_parsing)
            # Shift character offsets back to full_generated_text coordinates
            if think_block_len > 0:
                if parsed.answer_fullstring_start is not None:
                    parsed.answer_fullstring_start += think_block_len
                if parsed.answer_start is not None:
                    parsed.answer_start += think_block_len
            entries_to_save.append((gen, parsed, assistant_prefill, full_generated_text, messages))

        if parsed.answer_fullstring_start is not None:
            # answer_fullstring_start is an offset into full_generated_text (prefill + new tokens).
            # Token positions only count newly generated tokens, so strip the prefill chars first.
            generated_prefix = full_generated_text[len(assistant_prefill):parsed.answer_fullstring_start]
            prefix_ids = tokenizer(generated_prefix, add_special_tokens=False)["input_ids"]
            answer_token_start_position = gen.prompt_end_position + len(prefix_ids)
        else:
            answer_token_start_position = None

        if confidence and parsed.final_answer:
            if debug and llm_for_confidence is None:
                llm_for_confidence = LLM(model_name, thinking)
            active_llm = llm_for_confidence if debug else llm
            confidence_score: AllConfidenceData = compute_all_confidence_scores(
                active_llm,
                messages,
                gen.generated_text,
                parsed,
                nb_dropout_samples=3,
                use_fullstring=False,   # whether to apply dropout to the entire "\boxed{...}" string or just the answer tokens
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
            )
            debug_info = confidence_score.debug_info if debug_conf else None
            confidence_score = asdict(confidence_score)
            confidence_score.pop("debug_info", None)
        else:
            confidence_score = None
            debug_info = None

        #  - id — the unique identifier from the dataset entry (e.g., a BFCL question ID)
        #   - question — the raw question text from the dataset
        #   - ground_truth — the expected correct answer from the dataset, used for evaluation
        #   - cot_steps — the parsed chain-of-thought steps (e.g., "Step 1: ...", "Step 2: ...") extracted from the model's output by parse_output()
        #   - raw_cot_block — the full unparsed reasoning block as a single string (everything before "\boxed{}")
        #   - final_answer — the model's extracted final answer (the content inside "\boxed{...}")
        #   - generated_text — the complete raw output including the assistant prefill + all generated tokens, before any parsing
        #   - prompt_end_position — the number of tokens in the prompt (i.e., the index where generation starts). Useful for indexing into scores/cache
        #   - generated_end_position — the total sequence length (prompt + generated). So generated_end_position - prompt_end_position = number of new
        #   tokens
        #   - answer_token_start_position — the absolute token position where the "\boxed{...}" content begins. This is the boundary between CoT and answer tokens
        #   — critical for computing confidence only over the answer portion
        #   - confidence_metric — which confidence method was used (e.g., the string passed via --confidence), or None if confidence wasn't computed
        #   - confidence_score — the computed confidence value for the answer tokens using the specified metric, or None
        #   - past_key_values — the KV cache from generation (a DynamicCache object). This gets popped out and saved as a separate .pt file at write time
        #   (main.py:298) so you can resume/branch generation later without recomputing the prompt
        trajectories.append({
            "id":                           entry["id"],
            "question":                     entry["question"],
            "ground_truth":                 entry["answer"],
            "cot_steps":                    parsed.cot_steps,
            "final_answer":                 parsed.final_answer,
            "raw_cot_block":                parsed.raw_cot_block,
            "generated_text":               full_generated_text,
            "prompt_end_position":           gen.prompt_end_position,
            "generated_end_position":       gen.generated_end_position,
            "answer_token_start_position":  answer_token_start_position,
            # "confidence_metric": confidence if confidence else None,
            "confidence_score": confidence_score,
            "debug_info": debug_info,
            "runtime_seconds":              time.time() - t0,
            "past_key_values":              gen.past_key_values,
        })

    if not debug:
        cache_file = os.path.join(debug_dir, f"gen_parsed_type{prompt_type}.pkl")
        logger.info(f"Saving debug cache to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(entries_to_save, f)

    return trajectories


def main(args):
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} entries from '{args.dataset}'.")
    logger.info("Sample entry:")
    logger.info(json.dumps(dataset[0], indent=2, default=str))

    model_name = MODEL_DICT[args.model]
    sample_size = len(dataset) if args.sample_size is None else min(args.sample_size, len(dataset))
    dataloader = make_dataloader(dataset, n=sample_size)

    trajectories = generate_trajectories(
        model_name, dataloader, args.max_new_tokens,
        dataset_name=args.dataset,
        shot_mode=args.shot_mode,
        thinking=args.thinking,
        confidence=args.confidence,
        debug=args.debug,
        prompt_type=args.type,
        debug_conf=args.debug_conf,
    )


    out_dir = f"trajectories/{args.model}_{'thinking' if args.thinking else 'regular'}_{args.dataset}_{args.shot_mode}_{sample_size}_type{args.type}_{'conf' if args.confidence else 'vanilla'}"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{args.model}_{'thinking' if args.thinking else 'regular'}_{args.dataset}_{args.shot_mode}_trajectories_{sample_size}.json")
    all_debug_info = []
    with open(out_file, "w") as f:
        for i, traj in enumerate(trajectories):
            cache = traj.pop("past_key_values")
            debug_info = traj.pop("debug_info", None)
            if debug_info is not None:
                all_debug_info.append({"index": i, "id": traj["id"], **debug_info})
            # torch.save(cache, os.path.join(out_dir, f"cache_{i}.pt"))

            traj["prompt_end_position"] = traj.pop("prompt_end_position")
            traj["generated_end_position"] = traj.pop("generated_end_position")

            entry = {
                "index": i,
                "question": traj["question"],
                "trajectory": traj,
            }
            json.dump(entry, f, indent=2)
            f.write("\n")

    if all_debug_info:
        debug_file = os.path.join(out_dir, "debug_conf.json")
        with open(debug_file, "w") as f:
            json.dump(all_debug_info, f, indent=2)
        logger.info(f"Saved confidence debug info to {debug_file}")


if __name__ == "__main__":
    args = parse()
    main(args)
