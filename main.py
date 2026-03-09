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

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from utils.enum import MODEL_DICT
from utils.structures import GenerationResult, ParsedOutput
from prompts.load import load_messages
from prompts.cot_prompt import SYSTEM as _SYSTEM_BASE
from load_datasets import load

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Section 1 – Prompt Formatting (LogiQA)
# =============================================================================

LETTERS = "ABCD"




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
        default=10,
        help="Number of samples to process from the dataset."
    )
    args.add_argument(
        "--shot_mode",
        type=str,
        default="zero",
        choices=["zero", "few"],
        help="Prompting mode for LogiQA: zero-shot or few-shot CoT."
    )
    return args.parse_args()



def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> GenerationResult:
    """Run one forward pass through model.generate() with a DynamicCache.

    Key HuggingFace arguments and why they are used here:

    past_key_values=DynamicCache()
        Passes a pre-initialised DynamicCache object into generate().
        Transformers fills it in-place during the forward passes; the
        populated cache is returned via outputs.past_key_values.
        This avoids re-computing attention keys/values on the prompt for any
        subsequent continuation (useful if you want to branch or re-use).

    return_dict_in_generate=True
        Makes generate() return a GenerateDecoderOnlyOutput (a structured
        object) instead of a raw tensor.  This gives access to:
          .sequences        – full token IDs (prompt + generated)
          .past_key_values  – the updated DynamicCache
          .scores           – per-step logits (if output_scores=True)

    do_sample / temperature
        temperature=0.0 → greedy decoding (deterministic, good for evals).
        temperature>0.0 → multinomial sampling.
    """
    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=False,     # MUST be False if you provide the assistant role
        continue_final_message=True      # MUST be True for Assistant Prefilling
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len: int = inputs["input_ids"].shape[1]

    cache = DynamicCache()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    total_len: int = outputs.sequences.shape[1]
    prompt_positions = list(range(0, prompt_len))
    generated_positions = list(range(prompt_len, total_len))

    generated_ids = outputs.sequences[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        prompt_text=prompt_text,
        generated_text=generated_text,
        prompt_token_positions=prompt_positions,
        generated_token_positions=generated_positions,
        past_key_values=outputs.past_key_values,
    )


# =============================================================================
# Section 3 – Parsing (LogiQA)
# =============================================================================

# Matches: "Final Answer: B" / "final answer: (C)" / "Final Answer: [d]" etc.
_FINAL_ANSWER_RE = re.compile(
    r"final\s+answer\s*:?\s*[\(\[]?\s*([A-Da-d])\s*[\)\]]?",
    re.IGNORECASE,
)

# Matches "Step 1:", "Step 2:", ... as step delimiters inside the CoT block.
_STEP_MARKER_RE = re.compile(r"(Step\s+\d+\s*:)", re.IGNORECASE)


def parse_output(generated_text: str) -> ParsedOutput:
    """Split a raw generation into structured CoT steps and the final answer.

    Strategy:
      1. Find "Final Answer:" to split the text into a CoT block and an
         answer segment.  Case- and punctuation-insensitive.
      2. Within the CoT block, split on "Step N:" markers to get individual
         steps.  If no markers are found, fall back to blank-line splitting,
         then single-line splitting.
      3. Extract the letter (A-D) from the answer segment with a lenient regex
         that handles parentheses, brackets, and lowercase letters.
    """
    text = generated_text.strip()

    split_pos = text.lower().find("final answer")
    if split_pos != -1:
        cot_block = text[:split_pos].strip()
        answer_segment = text[split_pos:]
    else:
        cot_block = text
        answer_segment = ""

    final_letter = ""
    if answer_segment:
        m = _FINAL_ANSWER_RE.search(answer_segment)
        if m:
            final_letter = m.group(1).upper()

    parts = _STEP_MARKER_RE.split(cot_block)

    steps: list = []
    if len(parts) > 1:
        preamble = parts[0].strip()
        if preamble:
            steps.append(preamble)
        for i in range(1, len(parts) - 1, 2):
            marker = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            steps.append(f"{marker} {body}".strip())
    else:
        steps = [s.strip() for s in re.split(r"\n{2,}", cot_block) if s.strip()]
        if len(steps) <= 1:
            steps = [s.strip() for s in cot_block.splitlines() if s.strip()]

    return ParsedOutput(
        cot_steps=steps,
        final_answer_letter=final_letter,
        raw_cot_block=cot_block,
    )


# =============================================================================
# CLI
# =============================================================================


def generate_trajectories(model, dataset, dataset_name=None, shot_mode="zero", n_shots=2):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto",
        # attn_implementation="flash_attention_2"
        torch_dtype="auto",
        # attn_implementation="flash_attention_2"
    )

    trajectories = []


    for i, entry in enumerate(dataset):
        logger.info(f"Generating {i}/{len(dataset)}")

        messages = load_messages(dataset_name, few_shot=(shot_mode=="few"), entry=entry)

        if dataset_name == "logiqa":
            gen: GenerationResult = generate_one(model, tokenizer, messages)
            parsed: ParsedOutput = parse_output(gen.generated_text)
            trajectories.append({
                "id":                        entry["id"],
                "question":                  entry["question"],
                "ground_truth":              entry["answer"],
                "cot_steps":                 parsed.cot_steps,
                "raw_cot_block":             parsed.raw_cot_block,
                "generated_text":            gen.generated_text,
                "prompt_token_positions":    gen.prompt_token_positions,
                "generated_token_positions": gen.generated_token_positions,
            })
        else:
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=False,     # MUST be False if you provide the assistant role
                continue_final_message=True      # MUST be True for Assistant Prefilling
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_length=4096
            )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            trajectories.append(output_text)

    return trajectories


def main(args):
    dataset = load(args.dataset)
    logger.info(f"Loaded {len(dataset)} entries from '{args.dataset}'.")
    logger.info("Sample entry:")
    logger.info(json.dumps(dataset[0], indent=2, default=str))

    model_name = MODEL_DICT[args.model]
    sample_size = min(args.sample_size, len(dataset))

    trajectories = generate_trajectories(
        model_name, dataset[:sample_size],
        dataset_name=args.dataset,
        shot_mode=args.shot_mode,
    )


    out_path = f"{args.model}_{args.dataset}_trajectories_{sample_size}.json"
    with open(out_path, "w") as f:
        for i in range(sample_size):
            json.dump({
                "index": i,
                "question": dataset[i]["question"],
                "trajectory": trajectories[i]
            }, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    args = parse()
    main(args)
