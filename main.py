import argparse
import os
import logging
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.enum import MODEL_DICT


from prompts.load_prompt import load_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


from load_datasets import load


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
    return args.parse_args()



def generate_trajectories(model: str, dataset: list, prompt: tuple[str, str, str]):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto",
        # attn_implementation="flash_attention_2"
    )

    trajectories = []

    for i, entry in enumerate(dataset):
        logger.info(f"Generating {i}/{len(dataset)}")
        # system_prompt, user_template = prompt
        # system_reasoning_prompt: prompt for COTs
        # system_specific_prompt: system prompt specific to the dataset (as close as possible to the original prompt in their repo)
        system_reasoning_prompt, system_specific_prompt, assistant_start = prompt
        # user_prompt = user_template.format(question=entry["question"], functions=entry["metadata"]["functions"])
        system_specific_prompt = system_specific_prompt.format(functions=entry["metadata"]["functions"])

        user_prompt = entry["question"]


        # input_text = system_reasoning_prompt + "\n\n" + system_specific_prompt + "\n\n" + user_prompt

        messages = [
            {"role": "system", "content": system_reasoning_prompt + "\n\n" + system_specific_prompt},
            # {"role": "system", "content": system_specific_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_start}
        ]

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
        # breakpoint()
        trajectories.append(output_text)
    
    return trajectories



def main(args):
    dataset = load(args.dataset)
    logger.info(f"Loaded {len(dataset)} entries from '{args.dataset}'.")
    logger.info("Sample entry:")
    logger.info(json.dumps(dataset[0], indent=2, default=str))

    model = MODEL_DICT[args.model]
    sample_size = min(args.sample_size, len(dataset))

    prompt: tuple[str, str, str] = load_prompt(args.dataset)

    trajectories = generate_trajectories(model, dataset[:sample_size], prompt)


    with open(f"/shared_work/han/cot/trajectories/{args.model}_{args.dataset}_trajectories_{sample_size}.json", "w") as f:
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

