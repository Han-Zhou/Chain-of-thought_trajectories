import argparse
import os
import logging
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.enum import MODEL_DICT

from prompts.load_prompt import load_prompt
from evaluate_logiqa import run_logiqa_eval

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
    args.add_argument(
        "--shot_mode",
        type=str,
        default="zero",
        choices=["zero", "few"],
        help="Prompting mode for LogiQA: zero-shot or few-shot CoT."
    )
    return args.parse_args()



def generate_trajectories(model, dataset, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto"
    )

    trajectories = []

    for i, entry in enumerate(dataset):
        logger.info(f"Generating {i}/{len(dataset)}")
        system_prompt, user_template = prompt
        user_prompt = user_template.format(question=entry["question"], functions=entry["metadata"]["functions"])

        input_text = system_prompt + "\n\n" + user_prompt
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=2048)
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
    out_path = f"{args.model}_{args.dataset}_trajectories_{sample_size}.json"

    if args.dataset == "logiqa":
        # Full LogiQA pipeline: CoT prompting + DynamicCache + structured parsing.
        results = run_logiqa_eval(
            model_name=model_name,
            dataset=dataset[:sample_size],
            shot_mode=args.shot_mode,
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {out_path}")
        return

    # Generic path for all other datasets.
    prompt = load_prompt(args.dataset)
    trajectories = generate_trajectories(model_name, dataset[:sample_size], prompt)

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

