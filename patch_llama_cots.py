"""
patch_llama_cots.py

Recompute vanilla confidence scores for previously generated Llama CoT
trajectories and write the patched per-sample traj_*.json files into a new
output directory (the source directory is left untouched).

The original scores were computed when the vanilla forward in confidence.py
reused a cached CoT-prefix KV cache and forwarded only the suffix tokens.
That scheme silently produced wrong logits whenever BPE tokenization diverged
across the CoT/suffix boundary. `dropout_forward` now runs a fresh
full-sequence forward for vanilla instead, so this script re-runs the
vanilla-only path against the saved generations and writes fresh scores.

No CoT generation happens — only the confidence forward passes.
"""

import argparse
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict

import torch
from tqdm import tqdm
from tqdm.contrib.discord import tqdm as tqdm_discord

from confidence import ConfidenceAlignmentError, compute_all_confidence_scores
from dataloader import load_dataset
from llm import LLM
from parsing import parse_output
from prompts.load import load_messages
from utils.enum import MODEL_DICT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_TRAJ_RE = re.compile(r"^traj_(\d+)(?:_sample_(\d+))?\.json$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("trajectory_dir", type=str, help="Directory containing traj_*.json files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write patched traj_*.json files into. Created if missing. "
                        "Must differ from trajectory_dir (no in-place mutation).")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name used during generation")
    p.add_argument("--model", type=str, default="llama", choices=list(MODEL_DICT.keys()))
    p.add_argument("--shot_mode", type=str, default="zero", choices=["zero", "few"])
    p.add_argument("--type", type=int, default=2, choices=[1, 2])
    p.add_argument("--experimental_extra_box", action="store_true", default=False)
    p.add_argument("--discord", action="store_true", default=False,
                   help="Send tqdm progress to Discord via TQDM_DISCORD_TOKEN and TQDM_DISCORD_CHANNEL_ID env vars.")
    return p.parse_args()


def _atomic_write_json(path: str, data: dict) -> None:
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".patch_", suffix=".json", dir=dir_)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _reparse_for_llama(full_generated_text: str, experimental_extra_box: bool):
    """Parse a saved Llama generation. Llama has no <think>...</think> block,
    so parsing is direct — no offset adjustment needed."""
    extra_box_suffix = "\nTherefore the answer is \\boxed{" if experimental_extra_box else None
    return parse_output(full_generated_text, extra_box_suffix=extra_box_suffix)


def main():
    args = parse_args()
    if not os.path.isdir(args.trajectory_dir):
        raise SystemExit(f"Not a directory: {args.trajectory_dir}")
    if os.path.abspath(args.output_dir) == os.path.abspath(args.trajectory_dir):
        raise SystemExit("output_dir must differ from trajectory_dir (no in-place mutation).")
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = MODEL_DICT[args.model]
    llm = LLM(model_name, thinking=False)

    dataset = load_dataset(args.dataset)
    dataset_by_id = {str(e["id"]): e for e in dataset}

    files = sorted(f for f in os.listdir(args.trajectory_dir) if _TRAJ_RE.match(f))
    logger.info("Found %d trajectory files to patch in %s", len(files), args.trajectory_dir)

    progress = tqdm_discord if args.discord else tqdm
    n_patched = n_skipped = n_failed = 0
    for fname in progress(files, desc="Patching", unit="traj"):
        path = os.path.join(args.trajectory_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        with open(path) as f:
            envelope = json.load(f)
        traj = envelope["trajectory"]

        if traj.get("error") or not traj.get("final_answer") or not traj.get("generated_text"):
            n_skipped += 1
            continue

        entry = dataset_by_id.get(str(traj["id"]))
        if entry is None:
            logger.warning("id %s missing from dataset, skipping %s", traj["id"], fname)
            n_skipped += 1
            continue

        try:
            messages = load_messages(
                args.dataset,
                few_shot=(args.shot_mode == "few"),
                entry=entry,
                model_name=model_name,
                thinking=False,
                prompt_type=args.type,
                experimental_extra_box=args.experimental_extra_box,
            )
            assistant_prefill = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""

            full_generated_text = traj["generated_text"]
            if not full_generated_text.startswith(assistant_prefill):
                logger.warning("prefill mismatch in %s; using full text as generated_text", fname)
                generated_text = full_generated_text
                assistant_prefill_for_conf = ""
            else:
                generated_text = full_generated_text[len(assistant_prefill):]
                assistant_prefill_for_conf = assistant_prefill

            parsed = _reparse_for_llama(full_generated_text, args.experimental_extra_box)
            if not parsed.final_answer:
                logger.warning("could not reparse final_answer for %s, skipping", fname)
                n_skipped += 1
                continue

            llm.switch_attn_implementation("confidence")
            try:
                scores_obj = compute_all_confidence_scores(
                    llm,
                    messages,
                    generated_text,
                    parsed,
                    nb_dropout_samples=0,
                    use_fullstring=False,
                    assistant_prefill=assistant_prefill_for_conf,
                    debug_conf=True,
                    gen_cache=None,
                    vanilla_only=True,
                )
            finally:
                llm.switch_attn_implementation("cot")

            scores_dict = asdict(scores_obj)
            debug_info = scores_dict.pop("debug_info", None)
            traj["confidence_score"] = scores_dict
            traj["confidence_method"] = "vanilla_only"
            traj["confidence_error"] = None
            traj["confidence_modification"] = None
            traj["debug_info"] = debug_info

            _atomic_write_json(out_path, envelope)
            n_patched += 1

        except ConfidenceAlignmentError as exc:
            logger.warning("alignment error for %s: %s", fname, exc)
            n_failed += 1
        except Exception as exc:
            logger.error("error patching %s: %s", fname, exc, exc_info=True)
            n_failed += 1
        finally:
            torch.cuda.empty_cache()

    logger.info("Done. patched=%d skipped=%d failed=%d", n_patched, n_skipped, n_failed)


if __name__ == "__main__":
    main()
