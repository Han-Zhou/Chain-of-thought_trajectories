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
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from utils.enum import MODEL_DICT
from utils.structures import GenerationResult, ParsedOutput
from prompts.load import load_messages
from prompts.cot_prompt import SYSTEM as _SYSTEM_BASE
from dataloader import load_dataset, make_dataloader
from parsing import parse_output

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



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

    return args.parse_args()



def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    thinking: bool,
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

    model_name = model.config._name_or_path

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,     # MUST be False if you provide the assistant role
        continue_final_message=True,      # MUST be True for Assistant Prefilling
        # add_generation_prompt=True,
        # continue_final_message=False,
        # chat_template_kwargs={"enable_thinking": True},
        # enable_thinking=True,
    )

    # Post processing
    # For qwen: strip the empty think block that Qwen injects
    if "qwen" in model_name.lower():
        prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)

    # For gpt: GPT-OSS templates default to <|channel|>final for the assistant prefill.
    # if we are using thinking mode, Switch the last occurrence to <|channel|>analysis so the model reasons before answering.
    if "gpt" in model_name.lower() and thinking:
        target = "<|channel|>final"
        last_idx = prompt_text.rfind(target)
        if last_idx != -1:
            prompt_text = prompt_text[:last_idx] + "<|channel|>analysis" + prompt_text[last_idx + len(target):]

    print(prompt_text)

    # breakpoint()
        
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    # input_ids = inputs["input_ids"][0]
    # tokens = tokenizer.tokenize(prompt_text)

    # print("Last 20 tokens with input_ids:")
    # for token, token_id in zip(tokens[-20:], input_ids[-20:].tolist()):
    #     print(f"  {token_id}: {token}")

    # breakpoint()


    



    prompt_len: int = inputs["input_ids"].shape[1]

    # Qwen3.5 has a custom DynamicCache implementation that is required for correct attention behavior.  
    if hasattr(model.config, "layer_types") and "linear_attention" in model.config.layer_types:
        cache = Qwen3_5DynamicCache(config=model.config)
    else:
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
    # prompt_positions - 1 is the end position of the prompt text
    prompt_positions = prompt_len
    # generated_positions - 1 is the end position of the generated text
    generated_positions = total_len

    generated_ids = outputs.sequences[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        prompt_text=prompt_text,
        generated_text=generated_text,
        prompt_token_positions=prompt_positions,
        generated_token_positions=generated_positions,
        past_key_values=outputs.past_key_values,
    )



def generate_trajectories(model_name, dataloader, max_new_tokens, dataset_name=None, shot_mode="zero", thinking=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name in MODEL_DICT["qwen"]: 
        attn_implementation = "sdpa" 
    elif model_name in MODEL_DICT["gpt"] or model_name in MODEL_DICT["llama"]:
        attn_implementation = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation=attn_implementation
    )



    trajectories = []

    for i, entry in enumerate(dataloader):
        logger.info(f"Generating {i}/{len(dataloader)}")

        messages = load_messages(dataset_name, few_shot=(shot_mode=="few"), entry=entry, model_name=model_name, thinking=thinking)

        gen: GenerationResult = generate_one(model, tokenizer, messages, max_new_tokens=max_new_tokens, thinking=thinking)

        assistant_prefill = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
        full_generated_text = assistant_prefill + gen.generated_text
        parsed: ParsedOutput = parse_output(full_generated_text)

        if parsed.answer_char_start is not None:
            # answer_char_start is an offset into full_generated_text (prefill + new tokens).
            # Token positions only count newly generated tokens, so strip the prefill chars first.
            generated_prefix = full_generated_text[len(assistant_prefill):parsed.answer_char_start]
            prefix_ids = tokenizer(generated_prefix, add_special_tokens=False)["input_ids"]
            answer_token_start_position = gen.prompt_token_positions + len(prefix_ids)
        else:
            answer_token_start_position = None

        trajectories.append({
            "id":                           entry["id"],
            "question":                     entry["question"],
            "ground_truth":                 entry["answer"],
            "cot_steps":                    parsed.cot_steps,
            "raw_cot_block":                parsed.raw_cot_block,
            "final_answer":                 parsed.final_answer,
            "generated_text":               full_generated_text,
            "prompt_token_positions":       gen.prompt_token_positions,
            "generated_token_positions":    gen.generated_token_positions,
            "answer_token_start_position":  answer_token_start_position,
            "past_key_values":              gen.past_key_values,
        })

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
    )


    out_dir = f"trajectories/{args.model}_{'thinking' if args.thinking else 'regular'}_{args.dataset}_{args.shot_mode}_{sample_size}"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{args.model}_{'thinking' if args.thinking else 'regular'}_{args.dataset}_{args.shot_mode}_trajectories_{sample_size}.json")
    with open(out_file, "w") as f:
        for i, traj in enumerate(trajectories):
            cache = traj.pop("past_key_values")
            torch.save(cache, os.path.join(out_dir, f"cache_{i}.pt"))

            traj["prompt_token_positions"] = traj.pop("prompt_token_positions")
            traj["generated_token_positions"] = traj.pop("generated_token_positions")

            entry = {
                "index": i,
                "question": traj["question"],
                "trajectory": traj,
            }
            json.dump(entry, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    args = parse()
    main(args)
