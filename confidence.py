import re
import json
import numpy as np
import torch
import copy
from collections import defaultdict
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from llm import LLM
from utils.text_utils import find_token_indices_from_end
from utils.structures import ParsedOutput, ConfidenceScores, AllConfidenceData


ANSWER_TOKENS = {
    ' Yes': [' Yes', ' yes', ' YES', ' Yeah', ' yeah', ' Yep', ' yep'],
    ' No': [' No', ' no', ' NO', ' Nah', ' nah', ' Nope', ' nope'],
    ' True': [' True'],
    ' False': [' False'],
    'VERBCONF': [str(i) for i in range(0, 101)]
}


def get_token_ids(tokenizer, tokens):
    labels = []
    for token in tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            labels.append(token_ids[0])
    return labels


def compute_all_confidence_scores(
    llm: LLM,
    messages: list[dict],
    generated_text: str,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int = 10,
    use_fullstring: bool = False,
    assistant_prefill: str = "",
    debug_conf: bool = False,
) -> AllConfidenceData:
    debug_info = {}

    vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy, dbg = \
        dropout_answerlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf)
    if dbg:
        debug_info["answer_logits"] = dbg

    vanilla_ptrue1, vanilla_ptrue2, dropout_ptrue1, dropout_ptrue2, dbg = \
        dropout_indirectlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf)
    if dbg:
        debug_info["indirect_logits"] = dbg

    vanilla_verbconf, dropout_verbconf, dbg = \
        dropout_verbalconf(llm, messages, generated_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf)
    if dbg:
        debug_info["verbconf"] = dbg

    return AllConfidenceData(
        vanilla_confidences=ConfidenceScores(
            answer_probabilities=vanilla_answer_probs,
            answer_entropy=vanilla_answer_entropy,
            indirect_ptrue1_probabilities=vanilla_ptrue1,
            indirect_ptrue2_probabilities=vanilla_ptrue2,
            verbconf_probabilities=vanilla_verbconf,
        ),
        dropout_confidences=ConfidenceScores(
            answer_probabilities=dropout_answer_probs,
            answer_entropy=dropout_answer_entropy,
            indirect_ptrue1_probabilities=dropout_ptrue1,
            indirect_ptrue2_probabilities=dropout_ptrue2,
            verbconf_probabilities=dropout_verbconf,
        ),
        debug_info=debug_info,
    )


def dropout_answerlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False):
    """Logit-based confidence on the answer tokens themselves."""
    late_tokens, vanilla_out, dropout_out, ans_start, ans_end = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text="", nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)

    answer_tokens = late_tokens[0, ans_start:ans_end]
    nb_answer_tokens = len(answer_tokens)

    # --- Vanilla ---
    # Logit at position t-1 predicts token at position t
    v_logits = vanilla_out.logits[0, ans_start - 1:ans_end - 1, :]
    v_probs = v_logits.softmax(dim=-1)

    vanilla_answer_probs = [v_probs[t, answer_tokens[t]].item() for t in range(nb_answer_tokens)]

    v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
    vanilla_answer_entropy = (-(v_probs * v_logprobs).sum(dim=-1)).float().cpu().numpy().tolist()

    # --- Dropout ---
    if dropout_out is not None:
        d_probs = dropout_out.logits[:, ans_start - 1:ans_end - 1, :].softmax(dim=-1)

        dropout_answer_probs = [[d_probs[s, t, answer_tokens[t]].item()
                                  for t in range(nb_answer_tokens)]
                                 for s in range(d_probs.shape[0])]

        d_logprobs = torch.nan_to_num(d_probs.log(), neginf=-99)
        dropout_answer_entropy = (-(d_probs * d_logprobs).sum(dim=-1)).float().cpu().numpy().tolist()
    else:
        dropout_answer_probs = []
        dropout_answer_entropy = []

    # --- Debug ---
    dbg = _debug_answer_logit_tokens(llm, late_tokens, ans_start, ans_end, v_probs, vanilla_answer_probs) if debug_conf else None

    return vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy, dbg


def dropout_indirectlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False):
    """P(True) and P(Yes) probing after the answer."""
    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
    negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])

    # ptrue1: "True/False:"
    _, vanilla1, dropout1, _, _ = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text="\nTrue/False:",
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)

    # ptrue2: "Is the answer <X> correct?"
    _, vanilla2, dropout2, _, _ = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=f"\nIs the answer {parsed_output.final_answer} correct?",
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)

    # Vanilla ptrue1
    v_pos1 = vanilla1.logits[0, -1, positive_true_ids].sum()
    v_neg1 = vanilla1.logits[0, -1, negative_false_ids].sum()
    vanilla_ptrue1 = torch.softmax(
        torch.stack([v_pos1.float(), v_neg1.float()]), dim=0
    )[0].item()

    # Vanilla ptrue2
    v_pos2 = vanilla2.logits[0, -1, positive_yes_ids].sum()
    v_neg2 = vanilla2.logits[0, -1, negative_no_ids].sum()
    vanilla_ptrue2 = torch.softmax(
        torch.stack([v_pos2.float(), v_neg2.float()]), dim=0
    )[0].item()

    # Dropout
    if dropout1 is not None:
        d_pos1 = dropout1.logits[:, -1, positive_true_ids].sum(-1)
        d_neg1 = dropout1.logits[:, -1, negative_false_ids].sum(-1)
        dropout_ptrue1 = torch.softmax(
            torch.stack([d_pos1.float(), d_neg1.float()], dim=-1), dim=-1
        )[:, 0].cpu().numpy().tolist()

        d_pos2 = dropout2.logits[:, -1, positive_yes_ids].sum(-1)
        d_neg2 = dropout2.logits[:, -1, negative_no_ids].sum(-1)
        dropout_ptrue2 = torch.softmax(
            torch.stack([d_pos2.float(), d_neg2.float()], dim=-1), dim=-1
        )[:, 0].cpu().numpy().tolist()
    else:
        dropout_ptrue1 = []
        dropout_ptrue2 = []

    # --- Debug ---
    if debug_conf:
        dbg = {
            "ptrue1_true_false": _debug_indirect_logit_tokens(llm, vanilla1, positive_true_ids, negative_false_ids, "True/False"),
            "ptrue2_yes_no": _debug_indirect_logit_tokens(llm, vanilla2, positive_yes_ids, negative_no_ids, "Yes/No"),
        }
    else:
        dbg = None

    return [vanilla_ptrue1], [vanilla_ptrue2], dropout_ptrue1, dropout_ptrue2, dbg


def _compute_verbconf_joint_probs(llm, model_output, token_seqs, device):
    """Compute joint probabilities for multi-token number sequences.

    For a number like 42 tokenized as [tok_4, tok_2], computes:
        P(42) = P(tok_4) * P(tok_2 | tok_4)
    Each prefix gets its own cloned KV cache so conditioning is exact.

    Args:
        model_output: forward-pass output with .logits and .past_key_values
        token_seqs: list of token-id lists, one per number (0-100)
        device: torch device

    Returns:
        probs: [batch, len(token_seqs)] normalized joint probabilities
    """

    batch_size = model_output.logits.shape[0]
    joint_logprobs = torch.zeros(batch_size, len(token_seqs))

    # --- depth 0: first-token log-probs from existing logits ---
    logprobs_0 = model_output.logits[:, -1, :].float().log_softmax(-1).cpu()
    for i, seq in enumerate(token_seqs):
        joint_logprobs[:, i] = logprobs_0[:, seq[0]]

    max_depth = max(len(seq) for seq in token_seqs)
    if max_depth <= 1:
        return joint_logprobs.softmax(-1)

    # --- depth >= 1: one forward pass per unique prefix token ---
    prefix_cache = {(): model_output.past_key_values}

    for d in range(1, max_depth):
        groups = defaultdict(list)
        for i, seq in enumerate(token_seqs):
            if len(seq) > d:
                prefix = tuple(seq[:d])
                groups[prefix].append((i, seq[d]))

        if not groups:
            break

        new_prefix_cache = {}
        for prefix, entries in groups.items():
            parent_prefix = prefix[:-1]
            feed_token = prefix[-1]

            parent_kv = prefix_cache[parent_prefix]
            kv = copy.deepcopy(parent_kv)

            tok_input = torch.full((batch_size, 1), feed_token, device=device)

            with torch.no_grad():
                out = llm.model.forward(input_ids=tok_input, past_key_values=kv)

            logprobs_d = out.logits[:, -1, :].float().log_softmax(-1).cpu()

            for seq_idx, next_tok in entries:
                joint_logprobs[:, seq_idx] += logprobs_d[:, next_tok]

            new_prefix_cache[prefix] = out.past_key_values

        prefix_cache = new_prefix_cache

    return joint_logprobs.softmax(-1)


def dropout_verbalconf(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False):
    """Verbalized confidence (0-100 score)."""
    suffix = (
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )

    _, vanilla_out, dropout_out, _, _ = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)

    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.FloatTensor([int(s) / 100 for s in verbconf_strings])
    device = next(llm.model.parameters()).device

    # Vanilla
    v_probs = _compute_verbconf_joint_probs(llm, vanilla_out, token_seqs, device)
    vanilla_verbconf = (score_values * v_probs).sum().item()

    # Dropout
    if dropout_out is not None:
        d_probs = _compute_verbconf_joint_probs(llm, dropout_out, token_seqs, device)
        dropout_verbconf = (score_values * d_probs).sum(-1).numpy().tolist()
    else:
        dropout_verbconf = []

    # --- Debug ---
    dbg = _debug_verbconf_tokens(llm, token_seqs, v_probs, vanilla_verbconf) if debug_conf else None

    return [vanilla_verbconf], dropout_verbconf, dbg


def _tokenize_for_confidence(llm, messages, full_assistant_content):
    """Tokenize a conversation with the full assistant response.

    Matches the tokenization approach in llm.generate_one() for consistency,
    including model-specific post-processing.
    """
    conf_messages = copy.deepcopy(messages)
    if conf_messages[-1]["role"] == "assistant":
        conf_messages[-1]["content"] = full_assistant_content
    else:
        conf_messages.append({"role": "assistant", "content": full_assistant_content})

    prompt_text = llm.tokenizer.apply_chat_template(
        conf_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Apply same post-processing as generate_one()
    if "qwen" in llm.model_name.lower():
        prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)

    tokens = llm.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    return tokens


def dropout_forward(
    llm: LLM,
    messages: list[dict],
    generated_text: str,
    parsed_output: ParsedOutput,
    suffix_text: str = "",
    nb_dropout_samples: int = 10,
    use_fullstring: bool = False,
    threshold: float = 0.5,
    assistant_prefill: str = "",
):
    """Core forward pass for dropout experiment.

    Builds the full prompt+response+suffix as text, tokenizes from scratch,
    splits into early (prompt+CoT) and late (answer+suffix), then runs
    a vanilla and dropout forward pass on the late portion.
    """
    device = next(llm.model.parameters()).device

    # Build full assistant content with suffix
    full_content = (assistant_prefill + generated_text + suffix_text).strip()

    # Tokenize the full conversation from scratch
    tokens = _tokenize_for_confidence(llm, messages, full_content)  # [1, seq_len]

    # Find the answer region ("Final Answer: ...") in the full token sequence
    full_text = (assistant_prefill + generated_text).strip()
    fullstring_text = full_text[parsed_output.answer_fullstring_start:]
    fs_start, _ = find_token_indices_from_end(llm.tokenizer, tokens[0], fullstring_text)

    # Early/late split: one token before the answer fullstring as context
    early_late_split = fs_start - 1

    early_tokens = tokens[:, :early_late_split].to(device)
    late_tokens = tokens[:, early_late_split:].to(device)

    # Find answer positions within late tokens
    answer_start_late, answer_end_late = find_token_indices_from_end(
        llm.tokenizer, late_tokens[0], parsed_output.final_answer)

    if use_fullstring:
        modify_start_late = 1
        modify_end_late = len(late_tokens[0])
    else:
        modify_start_late = answer_start_late
        modify_end_late = answer_end_late

    # -- Early forward pass ---------------------------------------------------
    with torch.no_grad():
        empty_cache = Qwen3_5DynamicCache(llm.model.config) if "qwen" in llm.model_name.lower() else DynamicCache()
        early_output = llm.model(input_ids=early_tokens, past_key_values=empty_cache)

    early_cache = early_output.past_key_values
    nb_early_tokens = early_tokens.shape[1]

    # -- Vanilla forward ------------------------------------------------------
    with torch.no_grad():
        vanilla_output = llm.model.forward(
            input_ids=late_tokens,
            past_key_values=copy.deepcopy(early_cache),
            output_hidden_states=True,
        )

    # -- Dropout forward ------------------------------------------------------
    dropout_output = None
    if nb_dropout_samples > 0 and len(parsed_output.cot_steps) > 1:
        dropout_output = dropout_late_forward(
            llm,
            parsed_output.cot_steps,
            early_tokens[0],
            copy.deepcopy(early_cache),
            late_tokens,
            modify_start_late,
            modify_end_late,
            nb_early_tokens,
            nb_dropout_samples,
            threshold,
        )

    return late_tokens, vanilla_output, dropout_output, answer_start_late, answer_end_late


def dropout_late_forward(
    llm,
    cot_steps,
    early_token_ids,      # 1D tensor: all tokens in the early portion (prompt + CoT steps)
    early_cache,
    late_tokens,          # [1, late_len] tensor: tokens to run forward on (answer region + suffix)
    modify_start_late,
    modify_end_late,
    nb_early_tokens,
    nb_dropout_samples,
    threshold,
):
    """Batched forward with per-sample dropout attention masks.

    For each sample, a random subset of reasoning steps is masked so that
    the answer-region tokens cannot attend to those steps.
    """
    steps = cot_steps[1:]          # always keep the first step
    nb_late = late_tokens.shape[1]
    device = late_tokens.device

    # Base causal mask: late tokens attend to all early + causal late
    late_mask = torch.cat([
        torch.ones(nb_late, nb_early_tokens),
        torch.ones(nb_late, nb_late).tril(),
    ], dim=1)
    late_mask[late_mask == 0] = -10000.
    late_mask[late_mask == 1] = 0.
    late_mask = late_mask.repeat(nb_dropout_samples, 1, 1).unsqueeze(1)

    # Randomly select which steps to keep per sample
    is_step_selected = (np.random.random((nb_dropout_samples, len(steps))) <= threshold)

    # Walk backwards through steps, finding each one's token range and masking.
    # Positions in early_token_ids directly correspond to early cache columns.
    remaining_ids = early_token_ids.clone()
    for step_id, step in reversed(list(enumerate(steps))):
        try:
            step_start, step_end = find_token_indices_from_end(
                llm.tokenizer, remaining_ids, step)
        except ValueError:
            break

        for i in range(nb_dropout_samples):
            late_mask[i, 0,
                      modify_start_late - 1:modify_end_late - 1,
                      step_start:step_end] = 0. if is_step_selected[i, step_id] else -10000.

        remaining_ids = remaining_ids[:step_start + 1]

    # Expand for the dropout batch
    late_tokens_batch = late_tokens.expand(nb_dropout_samples, -1)
    early_cache.reorder_cache(torch.tensor([0] * nb_dropout_samples))
    late_mask = late_mask.to(device=device, dtype=next(llm.model.parameters()).dtype)

    with torch.no_grad():
        late_output = llm.model.forward(
            input_ids=late_tokens_batch,
            attention_mask=late_mask,
            past_key_values=early_cache,
            output_hidden_states=True,
        )
        late_output['input_ids'] = late_tokens_batch

    return late_output


# ---------------------------------------------------------------------------
# Debug helpers — return dicts for JSON serialization
# ---------------------------------------------------------------------------

def _debug_early_late_split(llm, early_tokens, late_tokens, answer_start_late, answer_end_late, parsed_output, suffix_text=""):
    """Return debug info about the early/late token split and answer region."""
    return {
        "early_token_count": early_tokens.shape[1],
        "late_token_count": late_tokens.shape[1],
        "total_token_count": early_tokens.shape[1] + late_tokens.shape[1],
        "early_tail_30_tokens": llm.tokenizer.decode(early_tokens[0][-30:]),
        "late_tokens_full": llm.tokenizer.decode(late_tokens[0]),
        "answer_start_late": answer_start_late,
        "answer_end_late": answer_end_late,
        "expected_answer": parsed_output.final_answer,
        "decoded_answer_tokens": llm.tokenizer.decode(late_tokens[0, answer_start_late:answer_end_late]),
        "suffix_text": suffix_text,
        "late_tail_after_answer": llm.tokenizer.decode(late_tokens[0, answer_end_late:]) if suffix_text else None,
    }


def _debug_answer_logit_tokens(llm, late_tokens, ans_start, ans_end, v_probs, vanilla_answer_probs):
    """Return debug info about per-token answer probabilities."""
    answer_tokens = late_tokens[0, ans_start:ans_end]
    per_token = []
    for t in range(len(answer_tokens)):
        tok_id = answer_tokens[t].item()
        tok_str = llm.tokenizer.decode([tok_id])
        context_tok = llm.tokenizer.decode([late_tokens[0, ans_start - 1 + t].item()])
        top5_probs, top5_ids = v_probs[t].topk(5)
        top5 = [{"token": llm.tokenizer.decode([tid.item()]), "prob": round(p, 6)}
                for tid, p in zip(top5_ids, top5_probs.tolist())]
        per_token.append({
            "position": t,
            "context_token": context_tok,
            "predicted_token": tok_str,
            "token_id": tok_id,
            "prob": round(vanilla_answer_probs[t], 6),
            "top5": top5,
        })
    return {
        "ans_start": ans_start,
        "ans_end": ans_end,
        "full_answer_text": llm.tokenizer.decode(answer_tokens),
        "per_token": per_token,
    }


def _debug_indirect_logit_tokens(llm, vanilla_out, positive_ids, negative_ids, label="True/False"):
    """Return debug info about indirect P(True) probing tokens."""
    logits = vanilla_out.logits[0, -1, :]
    probs = logits.float().softmax(-1)
    pos_entries = [{"token": llm.tokenizer.decode([tid]), "id": tid,
                    "logit": round(logits[tid].item(), 4), "prob": round(probs[tid].item(), 6)}
                   for tid in positive_ids]
    neg_entries = [{"token": llm.tokenizer.decode([tid]), "id": tid,
                    "logit": round(logits[tid].item(), 4), "prob": round(probs[tid].item(), 6)}
                   for tid in negative_ids]
    pos_sum = logits[positive_ids].sum()
    neg_sum = logits[negative_ids].sum()
    normalized = torch.softmax(torch.stack([pos_sum.float(), neg_sum.float()]), dim=0)
    return {
        "label": label,
        "positive_tokens": pos_entries,
        "negative_tokens": neg_entries,
        "p_positive": round(normalized[0].item(), 6),
        "p_negative": round(normalized[1].item(), 6),
    }


def _debug_verbconf_tokens(llm, token_seqs, v_probs, vanilla_verbconf):
    """Return debug info about verbalized confidence token predictions."""
    probs_1d = v_probs[0] if v_probs.dim() > 1 else v_probs
    top_vals, top_idxs = probs_1d.topk(min(10, len(probs_1d)))
    top_numbers = []
    for val, idx in zip(top_vals, top_idxs):
        num = idx.item()
        tok_ids = token_seqs[num]
        tok_str = llm.tokenizer.decode(tok_ids)
        top_numbers.append({
            "number": num,
            "token_ids": tok_ids,
            "token_str": tok_str,
            "prob": round(val.item(), 6),
        })
    return {
        "weighted_score": round(vanilla_verbconf, 6),
        "top10_numbers": top_numbers,
    }


def _debug_masked_text(llm, early_token_ids, late_mask, modify_start_late, nb_early_tokens, is_step_selected, steps, nb_dropout_samples):
    """Return debug info about which text is masked/kept per dropout sample."""
    samples = []
    for i in range(nb_dropout_samples):
        mask_row = late_mask[i, 0, modify_start_late - 1, :nb_early_tokens]
        masked_positions = (mask_row == -10000.).nonzero(as_tuple=True)[0]
        kept_positions = (mask_row == 0.).nonzero(as_tuple=True)[0]
        samples.append({
            "sample_idx": i,
            "kept_text": llm.tokenizer.decode(early_token_ids[kept_positions]),
            "masked_text": llm.tokenizer.decode(early_token_ids[masked_positions]),
            "dropped_steps": [steps[j] for j in range(len(steps)) if not is_step_selected[i, j]],
        })
    return samples
