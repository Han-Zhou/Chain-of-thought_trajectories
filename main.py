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

import copy

import torch
from tqdm.contrib.discord import tqdm as tqdm_discord
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
# from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

from utils.enum import MODEL_DICT
from utils.model_utils import (
    clean_phase1_generated_text,
    is_llama_model_name,
    is_qwen_model_name,
    phase1_stop_strings_for_model,
)
from dataclasses import asdict
from utils.structures import GenerationResult, ParsedOutput, AllConfidenceData, ConfidenceScores
from prompts.load import load_messages
from prompts.cot_prompt import SYSTEM as _SYSTEM_BASE
from dataloader import load_dataset, make_dataloader
from parsing import parse_output
# from utils.confidence_prev import compute_confidence_metrics
from confidence import compute_all_confidence_scores, ConfidenceAlignmentError
from llm import LLM
from eval.registry import evaluate_one

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset to process. One of: bfcl, bigbench_movie, "
            "bigbench_causal, logiqa, codeqa, cs1qa, hotpotqa, "
            "college_math, olympiadbench, math500, hle."
        )
    )
    args.add_argument(
        "--pickle_path",
        type=str,
        default=None,
        help="Path to a pickle file containing pre-generated entries. --dataset is still required for prompt/eval routing."
    )
    args.add_argument(
        "--model",
        type=str,
        default="llama",
        choices=["llama", "gpt", "qwen", "qwen-fp8", "qwen-gptq"],
        help="Model to use for inference."
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation. 0.0 = greedy decoding.",
    )
    args.add_argument(
        "--nb_cot_samples",
        type=int,
        default=1,
        help="Number of CoT samples per datapoint. Requires temperature > 0 for diversity.",
    )
    args.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Whether to use thinking.",
    )
    args.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Backend for model inference. vLLM is faster but incompatible with --confidence.",
    )
    args.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Maximum vLLM context length. Uses the model default when omitted.",
    )
    args.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory available to the vLLM engine.",
    )
    args.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs used for vLLM tensor parallelism.",
    )
    args.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=None,
        help="Maximum number of sequences scheduled concurrently by vLLM.",
    )
    args.add_argument(
        "--vllm_language_model_only",
        action="store_true",
        default=False,
        help="Load only the language model for hybrid text/multimodal models.",
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
        "--discord",
        action="store_true",
        default=False,
        help="Send tqdm progress to Discord via TQDM_DISCORD_TOKEN and TQDM_DISCORD_CHANNEL_ID env vars."
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
    args.add_argument(
        "--experimental_jackknife",
        nargs="?",
        const="default",
        default=None,
        choices=["default", "true"],
        help=(
            "Jackknife step masking mode. "
            "Omit for coin-flip only. "
            "'--experimental_jackknife' or '--experimental_jackknife default': "
            "current mask-based jackknife. "
            "'--experimental_jackknife true': prompt-reconstruction jackknife "
            "(fresh forward passes, no cache reuse or attention-mask edits)."
        ),
    )
    args.add_argument(
        "--nb_dropout_samples",
        type=int,
        default=None,
        help="Number of dropout samples for confidence scoring. Defaults to 3. Must not be set with --vanilla_only."
    )
    args.add_argument(
        "--vanilla_only",
        action="store_true",
        default=False,
        help=(
            "Compute only vanilla confidence scores (no coin-flip or jackknife dropout). "
            "Mutually exclusive with --experimental_jackknife, --experimental_grid, "
            "and --nb_dropout_samples."
        ),
    )
    args.add_argument(
        "--experimental_grid",
        action="store_true",
        default=False,
        help="Enable grid-based jackknife sampling (requires --experimental_jackknife)."
    )
    args.add_argument(
        "--experimental_preserve_final_step",
        action="store_true",
        default=False,
        help=(
            "Always keep the final CoT step during jackknife/dropout masking. "
            "Masking pool is computed from n-1 steps instead of n."
        ),
    )
    args.add_argument(
        "--experimental_bootstrap",
        action="store_true",
        default=False,
        help=(
            "Bootstrap confidence scoring: sample n-1 steps with replacement, "
            "always keep final step. Defaults to 100 samples (override with --nb_dropout_samples). "
            "Mutually exclusive with --experimental_jackknife, --experimental_grid, --vanilla_only."
        ),
    )
    args.add_argument(
        "--experimental_fake_bootstrap",
        action="store_true",
        default=False,
        help=(
            "Fake bootstrap confidence scoring: sample n-1 steps with replacement, "
            "use attention mask weighting (weight k for steps sampled k times, 0 otherwise). "
            "Mutually exclusive with --experimental_jackknife, --experimental_grid, --vanilla_only, --experimental_bootstrap."
        ),
    )
    args.add_argument(
        "--experimental_fake_bootstrap_hook",
        action="store_true",
        default=False,
        help=(
            "Fake bootstrap with forward hooks: sample n-1 steps with replacement, "
            "use forward hooks to directly scale attention weights (multiply by k for steps sampled k times). "
            "Mutually exclusive with --experimental_jackknife, --experimental_grid, --vanilla_only, --experimental_bootstrap, --experimental_fake_bootstrap."
        ),
    )
    args.add_argument(
        "--experimental_coinflip_true",
        action="store_true",
        default=False,
        help=(
            "Token-level coin-flip dropout: reconstruct prompts from independently "
            "Bernoulli-sampled CoT steps (p=0.5) and run fresh forward passes. "
            "Mutually exclusive with --experimental_jackknife, --experimental_bootstrap, "
            "--vanilla_only."
        ),
    )
    args.add_argument(
        "--experimental_extra_box",
        action="store_true",
        default=False,
        help=(
            "Two-phase generation: generate without \\boxed instruction, then "
            "append '\\nTherefore the answer is \\boxed{' and generate the answer "
            "with stop string '}'."
        ),
    )
    args.add_argument(
        "--experimental_modification",
        type=str,
        default=None,
        help=(
            "Confidence-only prompt modification mode. Supported values: "
            "'no-cots' and 'gibberish'."
        ),
    )
    args.add_argument(
        "--experimental_hollow",
        type=int,
        default=None,
        choices=[1, 2],
        help=(
            "Hollow confidence run: drop every step except the last at the "
            "token level (prompt reconstruction, no attention-mask edits). "
            "Mode 1: keep last step as-is. "
            "Mode 2: replace last step with "
            "'Considering the question, {final_answer} seems to be the most appropriate answer.'. "
            "The retained step is always renumbered as Step 1."
        ),
    )
    args.add_argument(
        "--cot_sample_idx",
        type=int,
        default=None,
        help=(
            "Run only this sample index per datapoint instead of iterating "
            "0..nb_cot_samples-1. Useful for parallelising multi-sample runs."
        ),
    )
    args.add_argument(
        "--save_cots",
        action="store_true",
        default=False,
        help=(
            "Save complete generation trajectory and all inputs needed for "
            "compute_all_confidence_scores as a per-sample pickle file "
            "(cot_{i}.pkl). Allows deferred confidence scoring without re-running generation."
        ),
    )
    args.add_argument(
        "--from_pregenerated",
        type=str,
        default=None,
        help=(
            "Directory of pregenerated trajectory JSONs. Uses traj_{i}_sample_0.json "
            "(or traj_{i}.json) for each datapoint. Skips generation and runs only "
            "confidence scoring. Requires --confidence."
        ),
    )
    args.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional prefix for the output directory name."
    )
    parsed = args.parse_args()
    if parsed.sample_size is not None and parsed.sample_range is not None:
        args.error("--sample_size and --sample_range are mutually exclusive.")
    if parsed.dataset is None and parsed.pickle_path is None:
        args.error("--dataset is required when not using --pickle_path.")
    if parsed.pickle_path is not None and parsed.dataset is None:
        args.error("--dataset is required when using --pickle_path (needed for evaluation).")
    if parsed.vllm_max_model_len is not None and parsed.vllm_max_model_len < 1:
        args.error("--vllm_max_model_len must be >= 1.")
    if not 0 < parsed.vllm_gpu_memory_utilization <= 1:
        args.error("--vllm_gpu_memory_utilization must be in (0, 1].")
    if parsed.vllm_tensor_parallel_size < 1:
        args.error("--vllm_tensor_parallel_size must be >= 1.")
    if parsed.vllm_max_num_seqs is not None and parsed.vllm_max_num_seqs < 1:
        args.error("--vllm_max_num_seqs must be >= 1.")
    if parsed.vanilla_only:
        if parsed.experimental_jackknife is not None:
            args.error("--vanilla_only is mutually exclusive with --experimental_jackknife.")
        if parsed.experimental_grid:
            args.error("--vanilla_only is mutually exclusive with --experimental_grid.")
        if parsed.nb_dropout_samples is not None:
            args.error("--vanilla_only is mutually exclusive with --nb_dropout_samples.")
        parsed.nb_dropout_samples = 0
    elif parsed.nb_dropout_samples is None:
        parsed.nb_dropout_samples = 3
    if parsed.experimental_bootstrap:
        if parsed.experimental_jackknife is not None:
            args.error("--experimental_bootstrap is mutually exclusive with --experimental_jackknife.")
        if parsed.experimental_grid:
            args.error("--experimental_bootstrap is mutually exclusive with --experimental_grid.")
        if parsed.vanilla_only:
            args.error("--experimental_bootstrap is mutually exclusive with --vanilla_only.")
        if parsed.experimental_fake_bootstrap:
            args.error("--experimental_bootstrap is mutually exclusive with --experimental_fake_bootstrap.")
        if parsed.nb_dropout_samples == 3:
            parsed.nb_dropout_samples = 100
    if parsed.experimental_fake_bootstrap:
        if parsed.experimental_jackknife is not None:
            args.error("--experimental_fake_bootstrap is mutually exclusive with --experimental_jackknife.")
        if parsed.experimental_grid:
            args.error("--experimental_fake_bootstrap is mutually exclusive with --experimental_grid.")
        if parsed.vanilla_only:
            args.error("--experimental_fake_bootstrap is mutually exclusive with --vanilla_only.")
        if parsed.nb_dropout_samples == 3:
            parsed.nb_dropout_samples = 100
    if parsed.experimental_fake_bootstrap_hook:
        if parsed.experimental_jackknife is not None:
            args.error("--experimental_fake_bootstrap_hook is mutually exclusive with --experimental_jackknife.")
        if parsed.experimental_grid:
            args.error("--experimental_fake_bootstrap_hook is mutually exclusive with --experimental_grid.")
        if parsed.vanilla_only:
            args.error("--experimental_fake_bootstrap_hook is mutually exclusive with --vanilla_only.")
        if parsed.experimental_fake_bootstrap:
            args.error("--experimental_fake_bootstrap_hook is mutually exclusive with --experimental_fake_bootstrap.")
        if parsed.nb_dropout_samples == 3:
            parsed.nb_dropout_samples = 100
    if parsed.experimental_coinflip_true:
        if parsed.experimental_jackknife is not None:
            args.error("--experimental_coinflip_true is mutually exclusive with --experimental_jackknife.")
        if parsed.experimental_bootstrap:
            args.error("--experimental_coinflip_true is mutually exclusive with --experimental_bootstrap.")
        if parsed.experimental_fake_bootstrap:
            args.error("--experimental_coinflip_true is mutually exclusive with --experimental_fake_bootstrap.")
        if parsed.experimental_fake_bootstrap_hook:
            args.error("--experimental_coinflip_true is mutually exclusive with --experimental_fake_bootstrap_hook.")
        if parsed.vanilla_only:
            args.error("--experimental_coinflip_true is mutually exclusive with --vanilla_only.")
    if parsed.from_pregenerated is not None:
        if not parsed.confidence:
            args.error("--from_pregenerated requires --confidence.")
        if parsed.debug:
            args.error("--from_pregenerated and --debug are mutually exclusive.")
    if parsed.backend == "vllm" and parsed.confidence:
        logger.warning(
            "vLLM backend is incompatible with confidence scoring. "
            "Falling back to HuggingFace backend."
        )
        parsed.backend = "hf"
    return parsed





def _available_pregenerated_indices(pregen_dir: str) -> set[int]:
    indices: set[int] = set()
    for fname in os.listdir(pregen_dir):
        m = re.match(r"traj_(\d+)(?:_sample_0)?\.json$", fname)
        if m:
            indices.add(int(m.group(1)))
    return indices


def _load_pregenerated_envelope(pregen_dir: str, index: int) -> dict:
    for fname in (f"traj_{index}_sample_0.json", f"traj_{index}.json"):
        path = os.path.join(pregen_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No pregenerated trajectory found for index {index} in {pregen_dir}"
    )


def _load_pregenerated_traj(pregen_dir: str, index: int) -> dict:
    return _load_pregenerated_envelope(pregen_dir, index)["trajectory"]


def _answer_lookup_value(answer) -> str:
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, sort_keys=True, default=str)
    return str(answer)


def _build_dataset_lookup(dataset_entries) -> dict[str, dict]:
    by_id: dict[str, list[dict]] = {}
    by_qa: dict[tuple[str, str], list[dict]] = {}

    for entry in dataset_entries:
        entry_id = entry.get("id")
        if entry_id is not None:
            by_id.setdefault(str(entry_id), []).append(entry)

        key = (entry.get("question"), _answer_lookup_value(entry.get("answer")))
        by_qa.setdefault(key, []).append(entry)

    return {"by_id": by_id, "by_qa": by_qa}


def _merge_pregenerated_entry(dataset_entry: dict, traj_saved: dict, dataset_lookup: dict | None = None) -> dict:
    """Use pregenerated identity as source of truth, resolve prompt fields by identity."""
    resolved_entry = dataset_entry

    if dataset_lookup is not None:
        traj_id = traj_saved.get("id")
        if traj_id is not None:
            matches = dataset_lookup["by_id"].get(str(traj_id), [])
            if len(matches) == 1:
                resolved_entry = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous dataset lookup for pregenerated id={traj_id}")

        if resolved_entry is dataset_entry:
            key = (traj_saved["question"], _answer_lookup_value(traj_saved["ground_truth"]))
            matches = dataset_lookup["by_qa"].get(key, [])
            if len(matches) == 1:
                resolved_entry = matches[0]
            elif len(matches) > 1:
                raise ValueError(
                    "Ambiguous dataset lookup for pregenerated question/ground_truth pair"
                )

    merged = copy.deepcopy(resolved_entry)
    merged["id"] = traj_saved["id"]
    merged["question"] = traj_saved["question"]
    merged["answer"] = traj_saved["ground_truth"]
    return merged


def _codeqa_pregenerated_pickle_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickles", "codeqa.pkl")


def _load_pregenerated_source_entries(dataset_name: str | None, dataloader) -> tuple[list[dict] | None, dict | None]:
    dataset_key = (dataset_name or "").strip().lower()
    if dataset_key == "codeqa":
        pickle_path = _codeqa_pregenerated_pickle_path()
        source_entries = load_pickle_data(pickle_path)
        logger.info(
            "Using %d entries from %s as pregenerated source data for codeqa (live dataloader has %d rows)",
            len(source_entries),
            pickle_path,
            len(dataloader),
        )
        return source_entries, None

    return None, _build_dataset_lookup(dataloader.dataset)


def _resolve_pregenerated_source_entry(
    dataset_name: str | None,
    entry: dict,
    index: int,
    traj_saved: dict,
    dataset_lookup: dict | None = None,
    pregenerated_source_entries: list[dict] | None = None,
) -> dict:
    dataset_key = (dataset_name or "").strip().lower()
    if dataset_key == "codeqa":
        if pregenerated_source_entries is None:
            raise ValueError(
                "codeqa from_pregenerated requires pickle-backed source entries"
            )
        if index >= len(pregenerated_source_entries):
            raise IndexError(
                f"Pregenerated codeqa index {index} exceeds pickle size {len(pregenerated_source_entries)}"
            )
        return _merge_pregenerated_entry(pregenerated_source_entries[index], traj_saved)

    return _merge_pregenerated_entry(entry, traj_saved, dataset_lookup=dataset_lookup)


def _extract_boxed_answer(generated_text):
    depth = 1
    answer_text = generated_text
    for i, char in enumerate(generated_text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer_text = generated_text[:i + 1]
                break
    return answer_text


def generate_qwen_outside_cot(llm, messages, max_new_tokens, temperature=0.0, prompt_type=1, experimental_extra_box=False):
    has_assistant_prefill = messages[-1]["role"] == "assistant"
    assistant_prefill = messages[-1]["content"] if has_assistant_prefill else ""

    gen1 = llm.generate_one(
        messages, max_new_tokens=max_new_tokens, temperature=temperature,
        has_assistant_prefill=has_assistant_prefill,
        stop_strings=["</think>"],
    )

    # Strip trailing </think> — stop_strings includes it in generated_text
    if gen1.generated_text.rstrip().endswith("</think>"):
        gen1.generated_text = gen1.generated_text.rstrip()
        gen1.generated_text = gen1.generated_text[:-len("</think>")]

    if prompt_type == 1:
        cot_scaffold = "\n</think>\nLet's think step-by-step.\nStep 1: "
    else:
        cot_scaffold = "\n</think>\n"

    phase2_prompt = gen1.prompt_text + gen1.generated_text + cot_scaffold

    if experimental_extra_box:
        gen2 = llm.generate_raw(phase2_prompt, max_new_tokens=max_new_tokens, temperature=temperature, stop_strings=[])
        # Strip trailing special tokens (e.g. <|im_end|>) before appending suffix
        for special_tok in llm.tokenizer.all_special_tokens:
            if gen2.generated_text.endswith(special_tok):
                gen2.generated_text = gen2.generated_text[:-len(special_tok)]
                break
        suffix = "\nTherefore the answer is \\boxed{"
        phase3_prompt = phase2_prompt + gen2.generated_text + suffix
        gen3 = llm.generate_raw(phase3_prompt, max_new_tokens=200, temperature=temperature, stop_strings=[])
        answer_text = _extract_boxed_answer(gen3.generated_text)
        combined_new = gen1.generated_text + cot_scaffold + gen2.generated_text + suffix + answer_text
        full_generated_text = assistant_prefill + combined_new
        final_gen = gen3
        final_gen.generated_text = combined_new
    else:
        gen2 = llm.generate_raw(phase2_prompt, max_new_tokens=max_new_tokens, temperature=temperature, stop_strings=[])
        combined_new = gen1.generated_text + cot_scaffold + gen2.generated_text
        full_generated_text = assistant_prefill + combined_new
        final_gen = gen2
        final_gen.generated_text = combined_new

    final_gen.prompt_text = gen1.prompt_text
    final_gen.prompt_end_position = gen1.prompt_end_position
    return final_gen, assistant_prefill, full_generated_text


def generate_continuation(llm, messages, max_new_tokens, temperature=0.0, suffix="\nTherefore the answer is \\boxed{", stop_strings=None):
    """Two-phase generation: free generation, then forced continuation with suffix.

    Returns (gen, assistant_prefill, full_generated_text) matching the
    contract of the normal generation path so downstream code is unchanged.
    """
    if stop_strings is None:
        stop_strings = ["}"]

    has_assistant_prefill = messages[-1]["role"] == "assistant"

    phase1_stop = phase1_stop_strings_for_model(llm.model_name, llm.thinking)
    gen1 = llm.generate_one(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        has_assistant_prefill=has_assistant_prefill,
        stop_strings=phase1_stop,
    )
    assistant_prefill = messages[-1]["content"] if has_assistant_prefill else ""
    gen1.generated_text = clean_phase1_generated_text(llm.model_name, gen1.generated_text, phase1_stop)
    phase1_text = assistant_prefill + gen1.generated_text

    # Strip end-of-text tokens before appending the suffix
    for special_tok in llm.tokenizer.all_special_tokens:
        if phase1_text.endswith(special_tok):
            phase1_text = phase1_text[:-len(special_tok)]
            gen1.generated_text = gen1.generated_text[:-len(special_tok)]
            break
    gen1.generated_text = clean_phase1_generated_text(llm.model_name, gen1.generated_text, phase1_stop)
    phase1_text = assistant_prefill + gen1.generated_text

    # Phase 2: generate without stop string, then trim to matching brace
    # (which would reformat the completed think block and break continue_final_message)
    phase2_prompt = gen1.prompt_text + gen1.generated_text + suffix
    gen2 = llm.generate_raw(phase2_prompt, max_new_tokens=200, temperature=temperature, stop_strings=[])

    # Find matching closing brace to handle answers containing "}"
    answer_text = _extract_boxed_answer(gen2.generated_text)

    full_generated_text = phase1_text + suffix + answer_text
    gen2.generated_text = gen1.generated_text + suffix + answer_text
    return gen2, assistant_prefill, full_generated_text


def generate_trajectories(model_name, dataloader, max_new_tokens, out_dir, dataset_name=None, shot_mode="zero", thinking=False, confidence=None, debug=False, prompt_type=1, debug_conf=False, experimental_jackknife=None, discord=False, nb_dropout_samples=3, experimental_extra_box=False, experimental_modification=None, experimental_grid=False, tag=None, cot_dir=None, temperature=0.0, nb_cot_samples=1, vanilla_only=False, from_pregenerated=None, experimental_preserve_final_step=False, experimental_bootstrap=False, experimental_coinflip_true=False, experimental_hollow=None, cot_sample_idx=None, experimental_fake_bootstrap=False, experimental_fake_bootstrap_hook=False, backend="hf", vllm_max_model_len=None, vllm_gpu_memory_utilization=0.92, vllm_tensor_parallel_size=1, vllm_max_num_seqs=None, vllm_language_model_only=False):

    debug_dir = "debug_cache"

    llm = LLM(
        model_name,
        thinking,
        backend=backend,
        vllm_max_model_len=vllm_max_model_len,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_max_num_seqs=vllm_max_num_seqs,
        vllm_language_model_only=vllm_language_model_only,
    )
    tokenizer = llm.tokenizer
    is_llama = is_llama_model_name(model_name)

    if debug:
        # Load pre-saved generation results instead of running the model
        cache_file = os.path.join(debug_dir, f"gen_parsed_type{prompt_type}.pkl")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Debug cache not found at {cache_file}. Run without --debug first to generate it.")
        logger.info(f"Debug mode: loading cached generation results from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_entries = pickle.load(f)
    else:
        cached_entries = None
        os.makedirs(debug_dir, exist_ok=True)
        entries_to_save = []

    errors = []

    if from_pregenerated:
        _pregen_indices = _available_pregenerated_indices(from_pregenerated)
        logger.info(f"Pregenerated dir has {len(_pregen_indices)} indices available")
        pregenerated_source_entries, dataset_lookup = _load_pregenerated_source_entries(dataset_name, dataloader)
    else:
        _pregen_indices = None
        pregenerated_source_entries = None
        dataset_lookup = None

    progress = tqdm_discord if discord else tqdm
    desc = f"Generating [{tag}]" if tag else "Generating"
    total = min(len(_pregen_indices), len(dataloader)) if _pregen_indices is not None else len(dataloader)
    for i, entry in enumerate(progress(dataloader, desc=desc, unit="sample", total=total)):
        if _pregen_indices is not None and i not in _pregen_indices:
            continue
        logger.info(f"Generating {i}/{len(dataloader)}")
        t0 = time.time()

        try:
            source_entry = entry
            traj_saved = None
            if debug:
                gen, parsed, assistant_prefill, full_generated_text, messages = cached_entries[i]
                nb_samples_for_entry = 1
            elif from_pregenerated:
                traj_saved = _load_pregenerated_traj(from_pregenerated, i)
                source_entry = _resolve_pregenerated_source_entry(
                    dataset_name,
                    entry,
                    i,
                    traj_saved,
                    dataset_lookup=dataset_lookup,
                    pregenerated_source_entries=pregenerated_source_entries,
                )
                messages = load_messages(dataset_name, few_shot=(shot_mode=="few"), entry=source_entry, model_name=model_name, thinking=thinking, prompt_type=prompt_type, experimental_extra_box=experimental_extra_box)
                nb_samples_for_entry = 1
            else:
                messages = load_messages(dataset_name, few_shot=(shot_mode=="few"), entry=source_entry, model_name=model_name, thinking=thinking, prompt_type=prompt_type, experimental_extra_box=experimental_extra_box)
                nb_samples_for_entry = nb_cot_samples

            sample_indices = [cot_sample_idx] if cot_sample_idx is not None else range(nb_samples_for_entry)
            for sample_idx in sample_indices:
              try:
                t_sample = time.time()

                # Track generation time separately
                t_generation_start = time.time()

                if debug:
                    gen, parsed, assistant_prefill, full_generated_text, messages = cached_entries[i]
                    generation_time = 0.0  # Debug mode loads from cache, no generation
                elif not debug:
                    if from_pregenerated:
                        full_generated_text = traj_saved["generated_text"]
                        assistant_prefill = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
                        gen = GenerationResult(
                            prompt_text=traj_saved.get("prompt", ""),
                            generated_text=full_generated_text[len(assistant_prefill):],
                            prompt_end_position=traj_saved["prompt_end_position"],
                            generated_end_position=traj_saved["generated_end_position"],
                            past_key_values=None,
                        )
                    elif is_qwen_model_name(model_name) and thinking:
                        gen, assistant_prefill, full_generated_text = generate_qwen_outside_cot(
                            llm, messages, max_new_tokens, temperature=temperature,
                            prompt_type=prompt_type, experimental_extra_box=experimental_extra_box,
                        )
                    elif experimental_extra_box:
                        gen, assistant_prefill, full_generated_text = generate_continuation(llm, messages, max_new_tokens, temperature=temperature)
                    else:
                        has_assistant_prefill = messages[-1]["role"] == "assistant"

                        gen: GenerationResult = llm.generate_one(messages, max_new_tokens=max_new_tokens, temperature=temperature, output_scores=False, has_assistant_prefill=has_assistant_prefill)
                        assistant_prefill = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
                        full_generated_text = assistant_prefill + gen.generated_text

                    generation_time = time.time() - t_generation_start

                    # For Qwen, CoT steps are outside </think>. Even without
                    # --thinking the chat template emits the <think> opener in
                    # the prefill, so a generated trajectory often contains a
                    # lone </think> mid-stream. Strip everything up to and
                    # including that closer so parse_output doesn't pick up
                    # the model's scratchwork "Step N:" lines from inside.
                    if is_qwen_model_name(model_name) and "</think>" in full_generated_text:
                        think_end = full_generated_text.find("</think>")
                        post_think_start = think_end + len("</think>")
                        post_think_content = full_generated_text[post_think_start:]
                        extra_box_suffix = "\nTherefore the answer is \\boxed{" if experimental_extra_box else None
                        parsed: ParsedOutput = parse_output(
                            post_think_content,
                            extra_box_suffix=extra_box_suffix,
                            llama_answer_cleanup=False,
                        )
                        for attr in ("answer_fullstring_start", "answer_start", "answer_sentence_start"):
                            val = getattr(parsed, attr)
                            if val is not None:
                                setattr(parsed, attr, post_think_start + val)
                    # For type 2 the model generates freely and may wrap reasoning in
                    # <think>...</think>.  CoT steps are extracted from *inside* the
                    # think block, while the answer is extracted from the text *after*
                    # </think>.  We then shift all character offsets back so they are
                    # relative to full_generated_text (which confidence.py indexes into).
                    elif prompt_type == 2:
                        think_match = re.search(r"<think>(.*?)</think>(.*)", full_generated_text, re.DOTALL)
                        if think_match:
                            # Parse CoT steps from inside <think>...</think>
                            think_content = think_match.group(1).strip()
                            parsed_cot: ParsedOutput = parse_output(
                                think_content,
                                extra_box_suffix=None,
                                llama_answer_cleanup=is_llama,
                            )

                            # Parse answer from text after </think>
                            post_think_content = think_match.group(2).strip()
                            post_think_group_start = think_match.start(2)  # where group(2) starts in full_generated_text

                            # Adjust for whitespace stripping
                            # think_match.group(2) includes leading whitespace, but we stripped it
                            # So we need to find where the stripped content actually starts
                            post_think_actual_start = full_generated_text.find(post_think_content, post_think_group_start)
                            if post_think_actual_start == -1:
                                # Fallback if we can't find it (shouldn't happen)
                                post_think_actual_start = post_think_group_start

                            extra_box_suffix = "\nTherefore the answer is \\boxed{" if experimental_extra_box else None
                            parsed_answer: ParsedOutput = parse_output(
                                post_think_content,
                                extra_box_suffix=extra_box_suffix,
                                llama_answer_cleanup=is_llama,
                            )

                            # Merge: take CoT steps from think block, answer from post-think
                            parsed = ParsedOutput(
                                cot_steps=parsed_cot.cot_steps,
                                final_answer=parsed_answer.final_answer,
                                raw_cot_block=parsed_cot.raw_cot_block,
                                answer_fullstring_start=None,
                                answer_start=None,
                                answer_sentence_start=None,
                            )

                            # Shift answer offsets to full_generated_text coordinates
                            if parsed_answer.answer_fullstring_start is not None:
                                parsed.answer_fullstring_start = post_think_actual_start + parsed_answer.answer_fullstring_start
                            if parsed_answer.answer_start is not None:
                                parsed.answer_start = post_think_actual_start + parsed_answer.answer_start
                            if parsed_answer.answer_sentence_start is not None:
                                parsed.answer_sentence_start = post_think_actual_start + parsed_answer.answer_sentence_start
                        else:
                            # Fallback: no think block found, parse everything normally
                            extra_box_suffix = "\nTherefore the answer is \\boxed{" if experimental_extra_box else None
                            parsed: ParsedOutput = parse_output(
                                full_generated_text,
                                extra_box_suffix=extra_box_suffix,
                                llama_answer_cleanup=is_llama,
                            )
                    else:
                        extra_box_suffix = "\nTherefore the answer is \\boxed{" if experimental_extra_box else None
                        parsed: ParsedOutput = parse_output(
                            full_generated_text,
                            extra_box_suffix=extra_box_suffix,
                            llama_answer_cleanup=is_llama,
                        )
                    if not from_pregenerated:
                        entries_to_save.append((gen, parsed, assistant_prefill, full_generated_text, messages))

                if parsed.answer_fullstring_start is not None:
                    # answer_fullstring_start is an offset into full_generated_text (prefill + new tokens).
                    # Token positions only count newly generated tokens, so strip the prefill chars first.
                    generated_prefix = full_generated_text[len(assistant_prefill):parsed.answer_fullstring_start]
                    prefix_ids = tokenizer(generated_prefix, add_special_tokens=False)["input_ids"]
                    answer_token_start_position = gen.prompt_end_position + len(prefix_ids)
                else:
                    answer_token_start_position = None

                confidence_error = None
                confidence_timings = None
                if confidence and parsed.final_answer:
                    llm.switch_attn_implementation("confidence")
                    try:
                        try:
                            confidence_score_obj: AllConfidenceData = compute_all_confidence_scores(
                                llm,
                                messages,
                                gen.generated_text,
                                parsed,
                                nb_dropout_samples=nb_dropout_samples,
                                use_fullstring=False,   # whether to apply dropout to the entire "\boxed{...}" string or just the answer tokens
                                assistant_prefill=assistant_prefill,
                                debug_conf=debug_conf,
                                gen_cache=gen.past_key_values,
                                experimental_jackknife=experimental_jackknife,
                                experimental_modification=experimental_modification,
                                experimental_grid=experimental_grid,
                                vanilla_only=vanilla_only,
                                preserve_final_step=experimental_preserve_final_step,
                                experimental_bootstrap=experimental_bootstrap,
                                experimental_coinflip_true=experimental_coinflip_true,
                                experimental_hollow=experimental_hollow,
                                experimental_fake_bootstrap=experimental_fake_bootstrap,
                                experimental_fake_bootstrap_hook=experimental_fake_bootstrap_hook,
                            )
                        except ConfidenceAlignmentError as exc:
                            confidence_error = str(exc)
                            logger.warning(
                                "Confidence scoring skipped for entry %s due to step-alignment failure: %s",
                                source_entry["id"],
                                exc,
                            )
                            confidence_score = None
                            debug_info = {"confidence_error": confidence_error} if debug_conf else None
                        else:
                            debug_info = confidence_score_obj.debug_info if debug_conf else None
                            if debug_info is not None and confidence_error is not None:
                                debug_info = copy.deepcopy(debug_info)
                                debug_info["confidence_error"] = confidence_error
                            confidence_score = asdict(confidence_score_obj)
                            confidence_score.pop("debug_info", None)

                            # Extract granular timing from confidence_score_obj.timings
                            confidence_timings = {
                                "vanilla_meanprob_total_seconds": confidence_score_obj.timings.vanilla_meanprob_total_seconds,
                                "vanilla_meanprob_forward_seconds": confidence_score_obj.timings.vanilla_meanprob_forward_seconds,
                                "vanilla_meanent_total_seconds": confidence_score_obj.timings.vanilla_meanent_total_seconds,
                                "vanilla_meanent_forward_seconds": confidence_score_obj.timings.vanilla_meanent_forward_seconds,
                                "vanilla_indirect1_total_seconds": confidence_score_obj.timings.vanilla_indirect1_total_seconds,
                                "vanilla_indirect1_forward_seconds": confidence_score_obj.timings.vanilla_indirect1_forward_seconds,
                                "vanilla_verbconf_total_seconds": confidence_score_obj.timings.vanilla_verbconf_total_seconds,
                                "vanilla_verbconf_forward_seconds": confidence_score_obj.timings.vanilla_verbconf_forward_seconds,
                                "vanilla_verbconf_joint_probs_seconds": confidence_score_obj.timings.vanilla_verbconf_joint_probs_seconds,
                                "bootstrap_meanprob_total_seconds": confidence_score_obj.timings.bootstrap_meanprob_total_seconds,
                                "bootstrap_meanprob_forward_seconds": confidence_score_obj.timings.bootstrap_meanprob_forward_seconds,
                                "bootstrap_indirect1_total_seconds": confidence_score_obj.timings.bootstrap_indirect1_total_seconds,
                                "bootstrap_indirect1_forward_seconds": confidence_score_obj.timings.bootstrap_indirect1_forward_seconds,
                                "bootstrap_verbconf_total_seconds": confidence_score_obj.timings.bootstrap_verbconf_total_seconds,
                                "bootstrap_verbconf_forward_seconds": confidence_score_obj.timings.bootstrap_verbconf_forward_seconds,
                                "bootstrap_verbconf_joint_probs_seconds": confidence_score_obj.timings.bootstrap_verbconf_joint_probs_seconds,
                            }
                    finally:
                        # Always reset to "cot" mode, even if compute_all_confidence_scores throws
                        llm.switch_attn_implementation("cot")
                else:
                    confidence_score = None
                    debug_info = None

                # Save generation trajectory and confidence inputs for deferred scoring
                if cot_dir:
                    cot_data = {
                        # Complete generation trajectory
                        "messages": messages,
                        "prompt_text": gen.prompt_text,
                        "generated_text": gen.generated_text,
                        "full_generated_text": full_generated_text,
                        "assistant_prefill": assistant_prefill,
                        "prompt_end_position": gen.prompt_end_position,
                        "generated_end_position": gen.generated_end_position,
                        "parsed_output": parsed,
                        "entry": {"id": source_entry["id"], "question": source_entry["question"], "answer": source_entry["answer"]},
                        # All inputs for compute_all_confidence_scores (except llm)
                        "confidence_info": {
                            "generated_text": gen.generated_text,
                            "parsed_output": parsed,
                            "assistant_prefill": assistant_prefill,
                            "gen_cache": gen.past_key_values,
                            "messages": messages,
                        },
                    }
                    if nb_cot_samples > 1:
                        cot_file = os.path.join(cot_dir, f"cot_{i}_sample_{sample_idx}.pkl")
                    else:
                        cot_file = os.path.join(cot_dir, f"cot_{i}.pkl")
                    with open(cot_file, "wb") as f:
                        pickle.dump(cot_data, f)

                # Free KV cache — only needed for confidence scoring above.
                # Prevents N caches from accumulating in the trajectories list.
                del gen.past_key_values
                gen.past_key_values = None
                torch.cuda.empty_cache()

                if dataset_name == "hle":
                    eval_result = None
                    correct = None
                else:
                    eval_result = evaluate_one({
                        "generated_text": full_generated_text,
                        "ground_truth": source_entry["answer"],
                        "prediction": parsed.final_answer,
                        "func_description": (source_entry.get("metadata") or {}).get("functions", []),
                        "test_category": (source_entry.get("metadata") or {}).get("bfcl_type", "simple_python"),
                    }, dataset_name)

                    if eval_result:
                        # For hotpotqa, score is a dict with 'em' and 'f1'
                        if isinstance(eval_result["score"], dict):
                            correct = eval_result["score"]
                        else:
                            # For other datasets, score is a float/bool
                            correct = eval_result["score"] == 1.0
                    else:
                        correct = None

                traj = {
                    "id":                           source_entry["id"],
                    "question":                     source_entry["question"],
                    "ground_truth":                 source_entry["answer"],
                    "prompt":                       gen.prompt_text,
                    "cot_steps":                    parsed.cot_steps,
                    "final_answer":                 parsed.final_answer,
                    "raw_cot_block":                parsed.raw_cot_block,
                    "generated_text":               full_generated_text,
                    "prompt_end_position":           gen.prompt_end_position,
                    "generated_end_position":       gen.generated_end_position,
                    "answer_token_start_position":  answer_token_start_position,
                    "confidence_score": confidence_score,
                    "confidence_method": ("vanilla_only" if vanilla_only else "bootstrap" if experimental_bootstrap else "fake_bootstrap" if experimental_fake_bootstrap else "fake_bootstrap_hook" if experimental_fake_bootstrap_hook else "coinflip_true" if experimental_coinflip_true else {None: "coin_flip", "default": f"jackknife{'_grid' if experimental_grid else ''}_default", "true": f"coin_flip+jackknife{'_grid' if experimental_grid else ''}_true"}[experimental_jackknife]) if confidence_score else None,
                    "confidence_modification": experimental_modification if confidence_score else None,
                    "confidence_error":            confidence_error,
                    "confidence_timings":          confidence_timings,
                    "debug_info":                   debug_info if debug_conf else None,
                    "runtime_seconds":              time.time() - t_sample,
                    "generation_seconds":           generation_time,
                    "correct":                      correct,
                }

                if dataset_name == "bfcl":
                    meta = source_entry.get("metadata") or {}
                    traj["func_description"] = meta.get("functions", [])
                    traj["test_category"] = meta.get("bfcl_type", "simple_python")

                if nb_cot_samples > 1:
                    traj["sample_idx"] = sample_idx
                    traj_file = os.path.join(out_dir, f"traj_{i}_sample_{sample_idx}.json")
                    envelope = {"index": i, "sample_idx": sample_idx, "question": source_entry["question"], "trajectory": traj}
                else:
                    traj_file = os.path.join(out_dir, f"traj_{i}.json")
                    envelope = {"index": i, "question": source_entry["question"], "trajectory": traj}

                with open(traj_file, "w") as f:
                    json.dump(envelope, f, indent=2)

              except Exception as e:
                logger.error(f"Error processing entry {i} sample {sample_idx} (id={source_entry['id']}): {e}", exc_info=True)
                # Recover GPU memory after OOM so subsequent iterations can proceed
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    logger.warning("Cleared CUDA cache after OOM error")
                errors.append({"index": i, "sample_idx": sample_idx, "id": source_entry["id"], "error": str(e)})
                traj = {
                    "id":                           source_entry["id"],
                    "question":                     source_entry["question"],
                    "ground_truth":                 source_entry["answer"],
                    "prompt":                       None,
                    "error":                        str(e),
                    "cot_steps":                    None,
                    "final_answer":                 None,
                    "raw_cot_block":                None,
                    "generated_text":               None,
                    "prompt_end_position":           None,
                    "generated_end_position":       None,
                    "answer_token_start_position":  None,
                    "confidence_score":             None,
                    "confidence_method":           None,
                    "confidence_modification":     None,
                    "confidence_error":            None,
                    "confidence_timings":          None,
                    "runtime_seconds":              time.time() - t_sample,
                    "generation_seconds":           None,
                    "correct":                      None,
                }
                if nb_cot_samples > 1:
                    traj["sample_idx"] = sample_idx
                    traj_file = os.path.join(out_dir, f"traj_{i}_sample_{sample_idx}.json")
                    envelope = {"index": i, "sample_idx": sample_idx, "question": source_entry["question"], "trajectory": traj}
                else:
                    traj_file = os.path.join(out_dir, f"traj_{i}.json")
                    envelope = {"index": i, "question": source_entry["question"], "trajectory": traj}
                with open(traj_file, "w") as f:
                    json.dump(envelope, f, indent=2)

        except Exception as e:
            # Entry-level error (e.g., load_messages failed before sample loop)
            logger.error(f"Error processing entry {i} (id={source_entry['id']}): {e}", exc_info=True)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                logger.warning("Cleared CUDA cache after OOM error")
            errors.append({"index": i, "id": source_entry["id"], "error": str(e)})
            traj = {
                "id":                           source_entry["id"],
                "question":                     source_entry["question"],
                "ground_truth":                 source_entry["answer"],
                "prompt":                       None,
                "error":                        str(e),
                "cot_steps":                    None,
                "final_answer":                 None,
                "raw_cot_block":                None,
                "generated_text":               None,
                "prompt_end_position":           None,
                "generated_end_position":       None,
                "answer_token_start_position":  None,
                "confidence_score":             None,
                "confidence_method":           None,
                "confidence_modification":     None,
                "confidence_error":            None,
                "confidence_timings":          None,
                "runtime_seconds":              time.time() - t0,
                "generation_seconds":           None,
                "correct":                      None,
            }
            traj_file = os.path.join(out_dir, f"traj_{i}.json")
            with open(traj_file, "w") as f:
                json.dump({"index": i, "question": source_entry["question"], "trajectory": traj}, f, indent=2)

    if not debug:
        cache_file = os.path.join(debug_dir, f"gen_parsed_type{prompt_type}.pkl")
        logger.info(f"Saving debug cache to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(entries_to_save, f)

    return errors


def load_pickle_data(pickle_path: str) -> list[dict]:
    """Load pre-generated entries from a pickle file.

    Ensures each entry has 'id', 'question', and 'answer' keys
    so it can be consumed by generate_trajectories().
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    for i, entry in enumerate(data):
        if "id" not in entry:
            entry["id"] = str(i)
    return data


def main(args):
    t_start = time.time()

    # Validate experimental_grid requires experimental_jackknife
    if args.experimental_grid and args.experimental_jackknife is None:
        raise ValueError("--experimental_grid requires --experimental_jackknife to be set")

    if args.nb_cot_samples < 1:
        raise ValueError("--nb_cot_samples must be >= 1")
    if args.nb_cot_samples > 1 and args.temperature == 0.0:
        logger.warning(
            "nb_cot_samples=%d with temperature=0.0 (greedy decoding): "
            "all samples will be identical. Consider setting --temperature > 0.",
            args.nb_cot_samples,
        )

    if args.pickle_path:
        dataset = load_pickle_data(args.pickle_path)
        logger.info(f"Loaded {len(dataset)} entries from pickle '{args.pickle_path}'.")
    else:
        dataset = load_dataset(args.dataset)
        logger.info(f"Loaded {len(dataset)} entries from '{args.dataset}'.")

    logger.info("Sample entry:")
    logger.info(json.dumps(dataset[0], indent=2, default=str))

    model_name = MODEL_DICT[args.model]
    if args.sample_range is not None:
        start = max(args.sample_range[0], 0)
        end = min(args.sample_range[1], len(dataset))
        dataset = dataset[start:end]
        sample_size = len(dataset)
    else:
        start = 0
        sample_size = len(dataset) if args.sample_size is None else min(args.sample_size, len(dataset))
        end = start + sample_size
    dataloader = make_dataloader(dataset, n=sample_size)

    logger.info(f"generating {args.dataset} [{start}:{end}]")

    if args.vanilla_only:
        _jk_suffix = "vanillaonly"
    elif args.experimental_bootstrap:
        _jk_suffix = "bootstrap"
    elif args.experimental_fake_bootstrap:
        _jk_suffix = "fake_bootstrap"
    elif args.experimental_fake_bootstrap_hook:
        _jk_suffix = "fake_bootstrap_hook"
    elif args.experimental_coinflip_true:
        _jk_suffix = "coinflip_true"
    elif args.experimental_grid:
        _jk_suffix = {None: "coinflip", "default": "jackknife_grid_default", "true": "jackknife_grid_true"}[args.experimental_jackknife]
    else:
        _jk_suffix = {None: "coinflip", "default": "jackknife_default", "true": "jackknife_true"}[args.experimental_jackknife]
    mod_suffix = args.experimental_modification or "none"
    dir_name = f"{args.model}_{'thinking' if args.thinking else 'regular'}_{args.dataset}_{args.shot_mode}_{start}_{end}_type{args.type}_{'conf' if args.confidence else 'vanilla'}_{_jk_suffix}_mod-{mod_suffix}"
    if args.experimental_hollow is not None:
        dir_name += f"_hollow{args.experimental_hollow}"
    if args.nb_cot_samples > 1:
        dir_name += f"_samples{args.nb_cot_samples}"
    if args.cot_sample_idx is not None:
        dir_name += f"_sidx{args.cot_sample_idx}"
    if args.tag:
        dir_name = f"{args.tag}_{dir_name}"
    out_dir = f"trajectories/{dir_name}"
    os.makedirs(out_dir, exist_ok=True)

    cot_dir = None
    if args.save_cots:
        cot_dir = f"trajectories_cached_cots/{dir_name}"
        os.makedirs(cot_dir, exist_ok=True)

    errors = generate_trajectories(
        model_name, dataloader, args.max_new_tokens,
        out_dir=out_dir,
        dataset_name=args.dataset,
        shot_mode=args.shot_mode,
        thinking=args.thinking,
        confidence=args.confidence,
        debug=args.debug,
        prompt_type=args.type,
        debug_conf=args.debug_conf,
        experimental_jackknife=args.experimental_jackknife,
        discord=args.discord,
        nb_dropout_samples=args.nb_dropout_samples,
        experimental_extra_box=args.experimental_extra_box,
        experimental_modification=args.experimental_modification,
        experimental_grid=args.experimental_grid,
        tag=args.tag,
        cot_dir=cot_dir,
        temperature=args.temperature,
        nb_cot_samples=args.nb_cot_samples,
        vanilla_only=args.vanilla_only,
        from_pregenerated=args.from_pregenerated,
        experimental_preserve_final_step=args.experimental_preserve_final_step,
        experimental_bootstrap=args.experimental_bootstrap,
        experimental_coinflip_true=args.experimental_coinflip_true,
        experimental_hollow=args.experimental_hollow,
        cot_sample_idx=args.cot_sample_idx,
        experimental_fake_bootstrap=args.experimental_fake_bootstrap,
        experimental_fake_bootstrap_hook=args.experimental_fake_bootstrap_hook,
        backend=args.backend,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_language_model_only=args.vllm_language_model_only,
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
