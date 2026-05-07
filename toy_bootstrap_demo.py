"""
toy_bootstrap_demo.py

For the first datapoint of bigbench_movie:
  1. Build the prompt exactly like patch_qwen_cots.py.
  2. Run a FRESH vanilla forward (full-sequence tokenize + one forward pass,
     no KV-cache reuse) — matching how bootstrap forward works.
  3. Run compute_all_confidence_scores with experimental_bootstrap=True so
     bootstrap re-runs the same probes on 3 resampled CoTs.

Writes a JSON file containing:
  - vanilla_prompt (the literal answer_logits prompt fed to the fresh forward)
  - vanilla_confidences
  - bootstrap_prompts (3)
  - bootstrap_confidences (3)

Run:
  python3 toy_bootstrap_demo.py \
      trajectories/1_full_trajectories/MERGED-cots-430-qwen-bm-8/traj_0_sample_0.json \
      --out toy_bootstrap_demo.json
"""

import argparse
import json
from dataclasses import asdict

import torch

from confidence import (
    ANSWER_TOKENS,
    _build_prompt_text,
    _compute_verbconf_joint_probs,
    compute_all_confidence_scores,
    get_token_ids,
)
from dataloader import load_dataset
from llm import LLM
from parsing import parse_output
from prompts.load import load_messages
from utils.enum import MODEL_DICT
from utils.text_utils import find_token_indices_from_end


def _reparse_for_qwen(full_generated_text: str, experimental_extra_box: bool):
    extra_box_suffix = "\nTherefore the answer is \\boxed{" if experimental_extra_box else None
    if "</think>" in full_generated_text:
        think_end = full_generated_text.find("</think>")
        post_think_start = think_end + len("</think>")
        post_think_content = full_generated_text[post_think_start:]
        parsed = parse_output(post_think_content,
                              extra_box_suffix=extra_box_suffix,
                              llama_answer_cleanup=False)
        for attr in ("answer_fullstring_start", "answer_start", "answer_sentence_start"):
            v = getattr(parsed, attr)
            if v is not None:
                setattr(parsed, attr, post_think_start + v)
        return parsed
    return parse_output(full_generated_text, extra_box_suffix=extra_box_suffix,
                        llama_answer_cleanup=False)


def _per_sample_bootstrap_dicts(bs_scores) -> list[dict]:
    sd = asdict(bs_scores)
    n = len(sd.get("answer_probabilities", []))
    out = []
    for i in range(n):
        out.append({
            "answer_probabilities": sd["answer_probabilities"][i],
            "answer_entropy": sd["answer_entropy"][i],
            "indirect_ptrue1_probabilities": sd["indirect_ptrue1_probabilities"][i],
            "indirect_ptrue2_probabilities": sd["indirect_ptrue2_probabilities"][i],
            "indirect_ptrue3_probabilities": sd["indirect_ptrue3_probabilities"][i],
            "verbconf_probabilities": sd["verbconf_probabilities"][i],
            "verbconf_top_score": sd["verbconf_top_score"][i],
            "verbconf_top_prob": sd["verbconf_top_prob"][i],
            "step_counts": sd["step_masks"][i] if sd.get("step_masks") else None,
        })
    return out


def fresh_vanilla_forward(llm, messages, cot_text: str, final_answer: str) -> dict:
    """Mirror _bootstrap_forward's per-sample loop with the original (unmodified)
    CoT. Builds one full-sequence prompt per probe, tokenizes it from scratch,
    runs a single forward pass with NO past_key_values — exactly the path
    bootstrap uses, but with kept_indices = all steps."""
    device = next(llm.model.parameters()).device

    ans_suffix = f"\nThe answer is \\boxed{{{final_answer}}}."
    ptrue1_suffix = f"\nThe answer is \\boxed{{{final_answer}}}.\nTrue/False:"
    ptrue2_suffix = f"\nIs the answer \\boxed{{{final_answer}}}?"
    verbconf_suffix = (
        f"\nThe answer is \\boxed{{{final_answer}}}."
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )

    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
    negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])
    yes_only_id = llm.tokenizer.encode(' Yes', add_special_tokens=False)
    no_only_id = llm.tokenizer.encode(' No', add_special_tokens=False)

    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.tensor([int(s) / 100 for s in verbconf_strings], dtype=torch.float64)

    base = cot_text.strip()

    # --- answer_logits ---
    ans_content = (base + ans_suffix).strip()
    ans_prompt = _build_prompt_text(llm, messages, ans_content)
    ans_tokens = llm.tokenizer(ans_prompt, return_tensors="pt")["input_ids"].to(device)

    boxed_string = f"\\boxed{{{final_answer}}}"
    with torch.no_grad():
        ans_output = llm.model.forward(input_ids=ans_tokens)

    ans_token_ids = ans_tokens[0]
    boxed_start, boxed_end = find_token_indices_from_end(
        llm.tokenizer, ans_token_ids, boxed_string, llm.model_name)
    a_start, a_end = find_token_indices_from_end(
        llm.tokenizer, ans_token_ids[boxed_start:boxed_end],
        final_answer, llm.model_name)
    a_start += boxed_start
    a_end += boxed_start

    answer_toks = ans_token_ids[a_start:a_end]
    token_strings = [llm.tokenizer.decode([t.item()]) for t in answer_toks]
    v_logits = ans_output.logits[0, a_start - 1:a_end - 1, :]
    v_probs = v_logits.double().softmax(dim=-1)
    answer_probs = [{token_strings[i]: v_probs[i, answer_toks[i]].item()} for i in range(len(answer_toks))]

    v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
    entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).cpu().numpy().tolist()
    answer_entropy = [{token_strings[i]: entropy_values[i]} for i in range(len(answer_toks))]

    # --- ptrue1 ---
    ptrue1_content = (base + ptrue1_suffix).strip()
    ptrue1_prompt = _build_prompt_text(llm, messages, ptrue1_content)
    ptrue1_tokens = llm.tokenizer(ptrue1_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        ptrue1_output = llm.model.forward(input_ids=ptrue1_tokens)
    p1_pos = ptrue1_output.logits[0, -1, positive_true_ids].sum()
    p1_neg = ptrue1_output.logits[0, -1, negative_false_ids].sum()
    p1_sm = torch.softmax(torch.stack([p1_pos.double(), p1_neg.double()]), dim=0)
    ptrue1 = {"True": p1_sm[0].item(), "False": p1_sm[1].item()}

    # --- ptrue2 / ptrue3 ---
    ptrue2_content = (base + ptrue2_suffix).strip()
    ptrue2_prompt = _build_prompt_text(llm, messages, ptrue2_content)
    ptrue2_tokens = llm.tokenizer(ptrue2_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        ptrue2_output = llm.model.forward(input_ids=ptrue2_tokens)
    p2_pos = ptrue2_output.logits[0, -1, positive_yes_ids].sum()
    p2_neg = ptrue2_output.logits[0, -1, negative_no_ids].sum()
    p2_sm = torch.softmax(torch.stack([p2_pos.double(), p2_neg.double()]), dim=0)
    ptrue2 = {"Yes": p2_sm[0].item(), "No": p2_sm[1].item()}

    p3_pos = ptrue2_output.logits[0, -1, yes_only_id[0]]
    p3_neg = ptrue2_output.logits[0, -1, no_only_id[0]]
    p3_sm = torch.softmax(torch.stack([p3_pos.double(), p3_neg.double()]), dim=0)
    ptrue3 = {"Yes": p3_sm[0].item(), "No": p3_sm[1].item()}

    # --- verbconf ---
    verbconf_content = (base + verbconf_suffix).strip()
    verbconf_prompt = _build_prompt_text(llm, messages, verbconf_content)
    verbconf_tokens = llm.tokenizer(verbconf_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        verbconf_output = llm.model.forward(input_ids=verbconf_tokens)
    vc_probs = _compute_verbconf_joint_probs(llm, verbconf_output, token_seqs, device)
    vc_probs_1d = vc_probs[0] if vc_probs.dim() > 1 else vc_probs
    verbconf = (score_values * vc_probs_1d).sum().item()
    top_idx = vc_probs_1d.argmax().item()

    return {
        "prompt": ans_prompt,
        "confidences": {
            "answer_probabilities": answer_probs,
            "answer_entropy": answer_entropy,
            "indirect_ptrue1_probabilities": ptrue1,
            "indirect_ptrue2_probabilities": ptrue2,
            "indirect_ptrue3_probabilities": ptrue3,
            "verbconf_probabilities": verbconf,
            "verbconf_top_score": top_idx,
            "verbconf_top_prob": vc_probs_1d[top_idx].item(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_path", type=str)
    ap.add_argument("--dataset", default="bigbench_movie")
    ap.add_argument("--model", default="qwen", choices=list(MODEL_DICT.keys()))
    ap.add_argument("--thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--shot_mode", default="zero", choices=["zero", "few"])
    ap.add_argument("--type", type=int, default=1, choices=[1, 2])
    ap.add_argument("--experimental_extra_box", action="store_true", default=True)
    ap.add_argument("--nb_bootstrap", type=int, default=3)
    ap.add_argument("--out", default="toy_bootstrap_demo.json")
    args = ap.parse_args()

    with open(args.traj_path) as f:
        envelope = json.load(f)
    traj = envelope["trajectory"]
    if traj.get("error") or not traj.get("final_answer"):
        raise SystemExit(f"Trajectory has no usable final_answer: {args.traj_path}")

    model_name = MODEL_DICT[args.model]

    dataset = load_dataset(args.dataset)
    entry = next((e for e in dataset if str(e["id"]) == str(traj["id"])), None)
    if entry is None:
        raise SystemExit(f"id {traj['id']} not found in dataset {args.dataset}")

    messages = load_messages(
        args.dataset,
        few_shot=(args.shot_mode == "few"),
        entry=entry,
        model_name=model_name,
        thinking=args.thinking,
        prompt_type=args.type,
        experimental_extra_box=args.experimental_extra_box,
    )
    assistant_prefill = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""

    full_generated_text = traj["generated_text"]
    if full_generated_text.startswith(assistant_prefill):
        generated_text = full_generated_text[len(assistant_prefill):]
        prefill_for_conf = assistant_prefill
    else:
        generated_text = full_generated_text
        prefill_for_conf = ""

    parsed = _reparse_for_qwen(full_generated_text, args.experimental_extra_box)
    if not parsed.final_answer:
        raise SystemExit("Could not reparse final_answer from generation.")

    llm = LLM(model_name, args.thinking)
    llm.switch_attn_implementation("confidence")

    try:
        # --- Bootstrap pass via compute_all_confidence_scores ---
        bootstrap_out = compute_all_confidence_scores(
            llm,
            messages,
            generated_text,
            parsed,
            nb_dropout_samples=args.nb_bootstrap,
            use_fullstring=False,
            assistant_prefill=prefill_for_conf,
            debug_conf=True,
            gen_cache=None,
            vanilla_only=True,
            experimental_bootstrap=True,
        )
        bootstrap_prompts = bootstrap_out.debug_info["bootstrap_prompts"]["answer_logits"]
        bootstrap_samples_dbg = bootstrap_out.debug_info.get("bootstrap_samples", [])
        bootstrap_per_sample = _per_sample_bootstrap_dicts(bootstrap_out.bootstrap_confidences)
        for i, s in enumerate(bootstrap_per_sample):
            s["kept_step_indices"] = (bootstrap_samples_dbg[i]["kept_step_indices"]
                                      if i < len(bootstrap_samples_dbg) else None)

        # The tokenizer-normalized CoT text that bootstrap used. Use the same
        # string so vanilla and bootstrap are over identical prompt content.
        cot_text_for_vanilla = bootstrap_out.debug_info["modified_assistant_content"]
        torch.cuda.empty_cache()

        # --- Fresh vanilla forward (same path as bootstrap, kept_indices=all) ---
        vanilla = fresh_vanilla_forward(llm, messages, cot_text_for_vanilla, parsed.final_answer)
    finally:
        llm.switch_attn_implementation("cot")
        torch.cuda.empty_cache()

    output = {
        "meta": {
            "traj_path": args.traj_path,
            "id": traj["id"],
            "dataset": args.dataset,
            "model": model_name,
            "final_answer": parsed.final_answer,
            "nb_cot_steps": len(parsed.cot_steps),
            "nb_bootstrap_samples": len(bootstrap_prompts),
        },
        "vanilla_prompt": vanilla["prompt"],
        "vanilla_confidences": vanilla["confidences"],
        "bootstrap_prompts": bootstrap_prompts,
        "bootstrap_confidences": bootstrap_per_sample,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {args.out}  (fresh vanilla + {len(bootstrap_prompts)} bootstrap samples)")


if __name__ == "__main__":
    main()
