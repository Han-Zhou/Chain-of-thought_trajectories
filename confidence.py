import re
import json
import logging
import math
import time
import numpy as np
import torch
import copy
from collections import defaultdict
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from llm import LLM
from utils.text_utils import find_token_indices_from_end, find_token_overlap
from utils.structures import ParsedOutput, ConfidenceScores, AllConfidenceData

logger = logging.getLogger(__name__)


def _jackknife_nb_keep(k: int) -> int:
    """Number of optional CoT steps to keep under jackknife dropout."""
    return math.ceil(math.sqrt(k)) if k >= 1 else 0


def crop_cache(cache, max_length):
    """Crop a KV cache in-place to max_length tokens.

    Works for both DynamicCache (.crop method) and Qwen3_5DynamicCache
    (which stores KV in key_cache/value_cache with shape
    [batch, heads, seq_len, head_dim] on attention layers only).
    """
    if hasattr(cache, 'crop'):
        cache.crop(max_length)
    else:
        # Qwen3_5DynamicCache: crop only attention layers (non-None KV entries)
        for idx in range(len(cache.key_cache)):
            if cache.key_cache[idx] is not None and cache.key_cache[idx].dim() == 4:
                cache.key_cache[idx] = cache.key_cache[idx][:, :, :max_length, :]
                cache.value_cache[idx] = cache.value_cache[idx][:, :, :max_length, :]


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
    gen_cache=None,
    experimental_jackknife: bool = False,
) -> AllConfidenceData:
    debug_info = {}

    if experimental_jackknife:
        k = len(parsed_output.cot_steps) - 1  # first step is always kept
        nb_keep = _jackknife_nb_keep(k)
        debug_info["jackknife_k"] = k
        debug_info["jackknife_nb_mask"] = k - nb_keep

    # Truncate generated_text to only include CoT steps (before the answer sentence)
    full_text = (assistant_prefill + generated_text).strip()
    if parsed_output.answer_sentence_start is not None:
        # Truncate at the answer sentence boundary (excludes transition text like "Therefore...")
        cot_end_offset = parsed_output.answer_sentence_start - len(assistant_prefill)
        if cot_end_offset > 0:
            cot_only_text = generated_text[:cot_end_offset].strip()
        else:
            # Answer is in the prefill, use empty string
            cot_only_text = ""
    else:
        # No answer found, use full text
        cot_only_text = generated_text

    logger.info("compute_all_confidence_scores: truncated generated_text from %d to %d chars",
                len(generated_text), len(cot_only_text))

    # Compute base tokenization (no suffix) once, shared by all methods
    base_content = (assistant_prefill + cot_only_text).strip()
    base_tokens = _tokenize_for_confidence(llm, messages, base_content)

    # Early/late split is now simply at the end of base_tokens (CoT only)
    early_late_split = base_tokens.shape[1]

    if gen_cache is not None:
        precomputed_early_cache = copy.deepcopy(gen_cache)
        crop_cache(precomputed_early_cache, early_late_split)
    else:
        # No generation cache available — one forward pass to build early_cache
        device = next(llm.model.parameters()).device
        early_tokens = base_tokens[:, :early_late_split].to(device)
        empty_cache = Qwen3_5DynamicCache(llm.model.config) if "qwen" in llm.model_name.lower() else DynamicCache()
        with torch.no_grad():
            early_output = llm.model(input_ids=early_tokens, past_key_values=empty_cache)
        precomputed_early_cache = early_output.past_key_values

    # --- Coin-flip dropout (always computed) ---
    vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy, dbg, dropout_step_masks = \
        dropout_answerlogits(llm, messages, cot_only_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                             gen_cache=gen_cache, base_tokens=base_tokens,
                             precomputed_early_cache=precomputed_early_cache)
    if dbg:
        debug_info["answer_logits"] = dbg

    vanilla_ptrue1, vanilla_ptrue2, dropout_ptrue1, dropout_ptrue2, dbg, _ = \
        dropout_indirectlogits(llm, messages, cot_only_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                               gen_cache=gen_cache, base_tokens=base_tokens,
                               precomputed_early_cache=precomputed_early_cache)
    if dbg:
        debug_info["indirect_logits"] = dbg

    (vanilla_verbconf, dropout_verbconf,
     v_verbconf_dist, v_verbconf_top_score, v_verbconf_top_prob,
     d_verbconf_dist, d_verbconf_top_scores, d_verbconf_top_probs,
     dbg, _) = \
        dropout_verbalconf(llm, messages, cot_only_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                           gen_cache=gen_cache, base_tokens=base_tokens,
                           precomputed_early_cache=precomputed_early_cache,
                           consume_early_cache=not experimental_jackknife)
    if dbg:
        debug_info["verbconf"] = dbg

    # --- Jackknife dropout (only when flag is set) ---
    jackknife_scores = None
    jk_step_masks = None
    if experimental_jackknife:
        _, _, jk_answer_probs, jk_answer_entropy, dbg, jk_step_masks = \
            dropout_answerlogits(llm, messages, cot_only_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                                 gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                                 precomputed_early_cache=precomputed_early_cache)
        if dbg:
            debug_info["jackknife_answer_logits"] = dbg

        _, _, jk_ptrue1, jk_ptrue2, dbg, _ = \
            dropout_indirectlogits(llm, messages, cot_only_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                                   gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                                   precomputed_early_cache=precomputed_early_cache)
        if dbg:
            debug_info["jackknife_indirect_logits"] = dbg

        (_, jk_verbconf,
         _, _, _,
         jk_verbconf_dist, jk_verbconf_top_scores, jk_verbconf_top_probs,
         dbg, _) = \
            dropout_verbalconf(llm, messages, cot_only_text, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                               gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                               precomputed_early_cache=precomputed_early_cache,
                               consume_early_cache=True)
        if dbg:
            debug_info["jackknife_verbconf"] = dbg

        jackknife_scores = ConfidenceScores(
            answer_probabilities=jk_answer_probs,
            answer_entropy=jk_answer_entropy,
            indirect_ptrue1_probabilities=jk_ptrue1,
            indirect_ptrue2_probabilities=jk_ptrue2,
            verbconf_probabilities=jk_verbconf,
            verbconf_distribution=jk_verbconf_dist,
            verbconf_top_score=jk_verbconf_top_scores,
            verbconf_top_prob=jk_verbconf_top_probs,
            step_masks=jk_step_masks,
        )

    return AllConfidenceData(
        vanilla_confidences=ConfidenceScores(
            answer_probabilities=vanilla_answer_probs,
            answer_entropy=vanilla_answer_entropy,
            indirect_ptrue1_probabilities=vanilla_ptrue1,
            indirect_ptrue2_probabilities=vanilla_ptrue2,
            verbconf_probabilities=vanilla_verbconf,
            verbconf_distribution=v_verbconf_dist,
            verbconf_top_score=v_verbconf_top_score,
            verbconf_top_prob=v_verbconf_top_prob,
        ),
        dropout_confidences=ConfidenceScores(
            answer_probabilities=dropout_answer_probs,
            answer_entropy=dropout_answer_entropy,
            indirect_ptrue1_probabilities=dropout_ptrue1,
            indirect_ptrue2_probabilities=dropout_ptrue2,
            verbconf_probabilities=dropout_verbconf,
            verbconf_distribution=d_verbconf_dist,
            verbconf_top_score=d_verbconf_top_scores,
            verbconf_top_prob=d_verbconf_top_probs,
            step_masks=dropout_step_masks,
        ),
        jackknife_confidences=jackknife_scores,
        debug_info=debug_info,
    )


def dropout_answerlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False, gen_cache=None, base_tokens=None, use_jackknife=False, precomputed_early_cache=None, consume_early_cache=False):
    """Logit-based confidence on the answer tokens themselves."""
    suffix_text = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
    logger.info("dropout_answerlogits: generated_text='%s', suffix='%s'", generated_text[:100], suffix_text)

    late_tokens, vanilla_out, dropout_out, ans_start, ans_end, dropout_step_masks = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=suffix_text, nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        consume_early_cache=consume_early_cache)

    answer_tokens = late_tokens[0, ans_start:ans_end]
    nb_answer_tokens = len(answer_tokens)
    token_strings = [llm.tokenizer.decode([answer_tokens[t].item()]) for t in range(nb_answer_tokens)]

    # --- Vanilla ---
    # Logit at position t-1 predicts token at position t
    v_logits = vanilla_out.logits[0, ans_start - 1:ans_end - 1, :]
    v_probs = v_logits.softmax(dim=-1)

    vanilla_answer_probs = [{token_strings[t]: v_probs[t, answer_tokens[t]].item()} for t in range(nb_answer_tokens)]

    v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
    entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).float().cpu().numpy().tolist()
    vanilla_answer_entropy = [{token_strings[t]: entropy_values[t]} for t in range(nb_answer_tokens)]

    # --- Dropout ---
    if dropout_out is not None:
        d_probs = dropout_out.logits[:, ans_start - 1:ans_end - 1, :].softmax(dim=-1)

        dropout_answer_probs = [[{token_strings[t]: d_probs[s, t, answer_tokens[t]].item()}
                                  for t in range(nb_answer_tokens)]
                                 for s in range(d_probs.shape[0])]

        d_logprobs = torch.nan_to_num(d_probs.log(), neginf=-99)
        d_entropy_values = (-(d_probs * d_logprobs).sum(dim=-1)).float().cpu().numpy().tolist()
        dropout_answer_entropy = [[{token_strings[t]: d_entropy_values[s][t]} for t in range(nb_answer_tokens)]
                                   for s in range(len(d_entropy_values))]
    else:
        dropout_answer_probs = []
        dropout_answer_entropy = []

    # --- Debug ---
    dbg = _debug_answer_logit_tokens(llm, late_tokens, ans_start, ans_end, v_probs, vanilla_answer_probs) if debug_conf else None

    return vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy, dbg, dropout_step_masks


def dropout_indirectlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False, gen_cache=None, base_tokens=None, use_jackknife=False, precomputed_early_cache=None, consume_early_cache=False):
    """P(True) and P(Yes) probing after the answer."""
    logger.info("dropout_indirectlogits: generated_text='%s'", generated_text[:100])

    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
    negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])

    # ptrue1: "True/False:"
    ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
    logger.info("dropout_indirectlogits ptrue1: suffix='%s'", ptrue1_suffix)
    _, vanilla1, dropout1, _, _, dropout_step_masks = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=ptrue1_suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache)

    # ptrue2: "Is the answer \boxed{<X>}?"
    ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"
    logger.info("dropout_indirectlogits ptrue2: suffix='%s'", ptrue2_suffix)
    _, vanilla2, dropout2, _, _, _ = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=ptrue2_suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        consume_early_cache=consume_early_cache)

    # Vanilla ptrue1
    v_pos1 = vanilla1.logits[0, -1, positive_true_ids].sum()
    v_neg1 = vanilla1.logits[0, -1, negative_false_ids].sum()
    v_softmax1 = torch.softmax(torch.stack([v_pos1.float(), v_neg1.float()]), dim=0)
    vanilla_ptrue1 = {"True": v_softmax1[0].item(), "False": v_softmax1[1].item()}

    # Vanilla ptrue2
    v_pos2 = vanilla2.logits[0, -1, positive_yes_ids].sum()
    v_neg2 = vanilla2.logits[0, -1, negative_no_ids].sum()
    v_softmax2 = torch.softmax(torch.stack([v_pos2.float(), v_neg2.float()]), dim=0)
    vanilla_ptrue2 = {"Yes": v_softmax2[0].item(), "No": v_softmax2[1].item()}

    # Dropout
    if dropout1 is not None:
        d_pos1 = dropout1.logits[:, -1, positive_true_ids].sum(-1)
        d_neg1 = dropout1.logits[:, -1, negative_false_ids].sum(-1)
        d_softmax1 = torch.softmax(torch.stack([d_pos1.float(), d_neg1.float()], dim=-1), dim=-1)
        dropout_ptrue1 = [{"True": d_softmax1[s, 0].item(), "False": d_softmax1[s, 1].item()}
                           for s in range(d_softmax1.shape[0])]

        d_pos2 = dropout2.logits[:, -1, positive_yes_ids].sum(-1)
        d_neg2 = dropout2.logits[:, -1, negative_no_ids].sum(-1)
        d_softmax2 = torch.softmax(torch.stack([d_pos2.float(), d_neg2.float()], dim=-1), dim=-1)
        dropout_ptrue2 = [{"Yes": d_softmax2[s, 0].item(), "No": d_softmax2[s, 1].item()}
                           for s in range(d_softmax2.shape[0])]
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

    return [vanilla_ptrue1], [vanilla_ptrue2], dropout_ptrue1, dropout_ptrue2, dbg, dropout_step_masks


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
            _t0 = time.perf_counter()
            kv = copy.deepcopy(parent_kv)
            logger.info("deepcopy(parent_kv) in _compute_verbconf_joint_probs took %.4fs", time.perf_counter() - _t0)

            tok_input = torch.full((batch_size, 1), feed_token, device=device)

            with torch.no_grad():
                out = llm.model.forward(input_ids=tok_input, past_key_values=kv)

            logprobs_d = out.logits[:, -1, :].float().log_softmax(-1).cpu()

            for seq_idx, next_tok in entries:
                joint_logprobs[:, seq_idx] += logprobs_d[:, next_tok]

            new_prefix_cache[prefix] = out.past_key_values

        prefix_cache = new_prefix_cache

    return joint_logprobs.softmax(-1)


def dropout_verbalconf(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False, gen_cache=None, base_tokens=None, use_jackknife=False, precomputed_early_cache=None, consume_early_cache=False):
    """Verbalized confidence (0-100 score)."""
    suffix = (
        f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )
    logger.info("dropout_verbalconf: generated_text='%s', suffix='%s'", generated_text[:100], suffix)

    _, vanilla_out, dropout_out, _, _, dropout_step_masks = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        consume_early_cache=consume_early_cache)

    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.FloatTensor([int(s) / 100 for s in verbconf_strings])
    device = next(llm.model.parameters()).device

    # Vanilla
    v_probs = _compute_verbconf_joint_probs(llm, vanilla_out, token_seqs, device)
    v_probs_1d = v_probs[0] if v_probs.dim() > 1 else v_probs
    vanilla_verbconf = (score_values * v_probs_1d).sum().item()
    vanilla_distribution = v_probs_1d.tolist()
    vanilla_top_idx = v_probs_1d.argmax().item()
    vanilla_top_score = vanilla_top_idx          # score 0-100
    vanilla_top_prob = v_probs_1d[vanilla_top_idx].item()

    # Dropout
    if dropout_out is not None:
        d_probs = _compute_verbconf_joint_probs(llm, dropout_out, token_seqs, device)
        dropout_verbconf = (score_values * d_probs).sum(-1).numpy().tolist()
        dropout_distribution = d_probs.tolist()
        dropout_top_idxs = d_probs.argmax(dim=-1)
        dropout_top_scores = dropout_top_idxs.tolist()
        dropout_top_probs = [d_probs[s, idx].item() for s, idx in enumerate(dropout_top_idxs)]
    else:
        dropout_verbconf = []
        dropout_distribution = []
        dropout_top_scores = []
        dropout_top_probs = []

    # --- Debug ---
    dbg = _debug_verbconf_tokens(llm, token_seqs, v_probs, vanilla_verbconf) if debug_conf else None

    return (
        [vanilla_verbconf], dropout_verbconf,
        vanilla_distribution, vanilla_top_score, vanilla_top_prob,
        dropout_distribution, dropout_top_scores, dropout_top_probs,
        dbg,
        dropout_step_masks,
    )


def _extract_thinking_and_content(content: str, model_name: str) -> tuple[str, str]:
    """Extract thinking and content from GPT model responses with channel tags.

    For GPT models, responses contain channel-tag structure:
    <|channel|>analysis<|message|>...thinking...<|end|><|start|>assistant<|channel|>final<|message|>...answer...<|return|>

    Returns (thinking, content) tuple. If no channel tags found, returns ("", content).
    """
    if "gpt" not in model_name.lower() or "<|channel|>" not in content:
        return "", content

    # Extract thinking: between <|channel|>analysis<|message|> and <|end|>
    thinking_match = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", content, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    # Extract final content: between <|channel|>final<|message|> and <|return|>
    content_match = re.search(r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", content, re.DOTALL)
    final_content = content_match.group(1).strip() if content_match else ""

    # If we couldn't extract, fall back to original content
    if not final_content:
        return "", content

    return thinking, final_content


def _tokenize_for_confidence(llm, messages, full_assistant_content):
    """Tokenize a conversation with the full assistant response.

    Matches the tokenization approach in llm.generate_one() for consistency,
    including model-specific post-processing.
    """
    _t0 = time.perf_counter()
    conf_messages = copy.deepcopy(messages)
    logger.info("deepcopy(messages) in _tokenize_for_confidence took %.4fs", time.perf_counter() - _t0)

    # Extract thinking and content for GPT models with channel tags
    thinking, content = _extract_thinking_and_content(full_assistant_content, llm.model_name)

    if conf_messages[-1]["role"] == "assistant":
        if thinking:
            conf_messages[-1]["thinking"] = thinking
            conf_messages[-1]["content"] = content
        else:
            conf_messages[-1]["content"] = full_assistant_content
    else:
        msg = {"role": "assistant", "content": content if thinking else full_assistant_content}
        if thinking:
            msg["thinking"] = thinking
        conf_messages.append(msg)

    # breakpoint()

    prompt_text = llm.tokenizer.apply_chat_template(
        conf_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # breakpoint()

    # Apply same post-processing as generate_one()
    if "qwen" in llm.model_name.lower():
        prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)

    # breakpoint()

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
    gen_cache=None,
    base_tokens=None,
    use_jackknife: bool = False,
    precomputed_early_cache=None,
    consume_early_cache: bool = False,
):
    """Core forward pass for dropout experiment.

    When gen_cache (the KV cache from generation) is provided, skips the
    expensive early forward pass by cropping the generation cache.  For
    suffix calls the overlap between the base and suffix tokenizations
    determines how much of the cache is reusable for the vanilla forward.

    When consume_early_cache is True, the last consumer of early_cache
    will use it directly instead of deepcopying — the caller asserts that
    early_cache (or precomputed_early_cache) will not be needed again.
    """
    device = next(llm.model.parameters()).device

    # -- Tokenize ---------------------------------------------------------------
    if base_tokens is None:
        base_content = (assistant_prefill + generated_text).strip()
        base_tokens = _tokenize_for_confidence(llm, messages, base_content)

    if suffix_text:
        full_content = (assistant_prefill + generated_text + suffix_text).strip()
        tokens = _tokenize_for_confidence(llm, messages, full_content)
        logger.info("dropout_forward: full_content (with suffix) = '%s'", full_content)
    else:
        tokens = base_tokens
        full_content = (assistant_prefill + generated_text).strip()
        logger.info("dropout_forward: full_content (no suffix) = '%s'", full_content)

    # -- Locate answer region ---------------------------------------------------
    # Since generated_text now ends before \boxed{}, the split is at the end of base_tokens
    early_late_split = base_tokens.shape[1]

    early_tokens = tokens[:, :early_late_split].to(device)
    late_tokens = tokens[:, early_late_split:].to(device)

    # Find \boxed{answer} in late tokens first, then find answer within that region
    boxed_string = f"\\boxed{{{parsed_output.final_answer}}}"
    boxed_start, boxed_end = find_token_indices_from_end(
        llm.tokenizer, late_tokens[0], boxed_string, llm.model_name)

    # Then find the answer within the boxed region
    answer_start_late, answer_end_late = find_token_indices_from_end(
        llm.tokenizer, late_tokens[0, boxed_start:boxed_end],
        parsed_output.final_answer, llm.model_name)

    # Adjust offsets to be relative to late_tokens
    answer_start_late += boxed_start
    answer_end_late += boxed_start

    if use_fullstring:
        modify_start_late = boxed_start
        modify_end_late = boxed_end
    else:
        modify_start_late = answer_start_late
        modify_end_late = answer_end_late

    nb_early_tokens = early_tokens.shape[1]

    # -- Determine ownership & whether dropout will run -------------------------
    # early_cache is "owned" (can be consumed destructively by the last user)
    # when it was freshly created in this call, or the caller explicitly allows it.
    _owns_early_cache = (precomputed_early_cache is None) or consume_early_cache
    will_run_dropout = nb_dropout_samples > 0 and len(parsed_output.cot_steps) > 1

    # -- Build early_cache (reuse gen_cache or recompute) -----------------------
    if precomputed_early_cache is not None:
        early_cache = precomputed_early_cache
        gen_cache_len = gen_cache.get_seq_length() if gen_cache is not None else early_cache.get_seq_length()
    elif gen_cache is not None:
        _t0 = time.perf_counter()
        early_cache = copy.deepcopy(gen_cache)
        logger.info("deepcopy(gen_cache) for early_cache took %.4fs", time.perf_counter() - _t0)
        gen_cache_len = early_cache.get_seq_length()
        crop_cache(early_cache, early_late_split)
    else:
        with torch.no_grad():
            empty_cache = Qwen3_5DynamicCache(llm.model.config) if "qwen" in llm.model_name.lower() else DynamicCache()
            early_output = llm.model(input_ids=early_tokens, past_key_values=empty_cache)
        early_cache = early_output.past_key_values

    # -- Vanilla forward --------------------------------------------------------
    if suffix_text and gen_cache is not None:
        overlap_len = min(find_token_overlap(base_tokens[0], tokens[0]), gen_cache_len)
        nb_discarded = gen_cache_len - overlap_len
        logger.info(
            "KV cache reuse: overlap=%d, discarded=%d (of %d)",
            overlap_len, nb_discarded, gen_cache_len,
        )

        if overlap_len > early_late_split:
            # Reuse more of the cache via overlap — forward only the tail
            _t0 = time.perf_counter()
            vanilla_cache = copy.deepcopy(gen_cache)
            logger.info("deepcopy(gen_cache) for vanilla_cache (suffix+overlap) took %.4fs", time.perf_counter() - _t0)
            crop_cache(vanilla_cache, overlap_len)
            remaining_tokens = tokens[:, overlap_len:].to(device)
            logger.info(
                "KV cache overlap boundary: cache_tail='%s' | remaining_head='%s'",
                llm.tokenizer.decode(tokens[0, overlap_len-3:overlap_len].tolist()),
                llm.tokenizer.decode(remaining_tokens[0, :3].tolist()),
            )

            with torch.no_grad():
                vanilla_output = llm.model.forward(
                    input_ids=remaining_tokens,
                    past_key_values=vanilla_cache,
                    output_hidden_states=True,
                )
        else:
            # Pathological case: overlap doesn't extend past split
            with torch.no_grad():
                if _owns_early_cache and not will_run_dropout:
                    logger.info("consuming early_cache directly for vanilla (suffix, no overlap)")
                    vanilla_output = llm.model.forward(
                        input_ids=late_tokens,
                        past_key_values=early_cache,
                        output_hidden_states=True,
                    )
                else:
                    _t0 = time.perf_counter()
                    _early_cache_copy = copy.deepcopy(early_cache)
                    logger.info("deepcopy(early_cache) for vanilla (suffix, no overlap) took %.4fs", time.perf_counter() - _t0)
                    vanilla_output = llm.model.forward(
                        input_ids=late_tokens,
                        past_key_values=_early_cache_copy,
                        output_hidden_states=True,
                    )
    else:
        if gen_cache is not None:
            nb_discarded = gen_cache_len - early_late_split
            logger.info(
                "KV cache reuse (no suffix): reused=%d, discarded=%d (of %d)",
                early_late_split, nb_discarded, gen_cache_len,
            )
            logger.info(
                "KV cache overlap boundary (no suffix): cache_tail='%s' | remaining_head='%s'",
                llm.tokenizer.decode(tokens[0, early_late_split-3:early_late_split].tolist()),
                llm.tokenizer.decode(late_tokens[0, :3].tolist()),
            )
        with torch.no_grad():
            if _owns_early_cache and not will_run_dropout:
                logger.info("consuming early_cache directly for vanilla (no suffix)")
                vanilla_output = llm.model.forward(
                    input_ids=late_tokens,
                    past_key_values=early_cache,
                    output_hidden_states=True,
                )
            else:
                _t0 = time.perf_counter()
                _early_cache_copy = copy.deepcopy(early_cache)
                logger.info("deepcopy(early_cache) for vanilla (no suffix) took %.4fs", time.perf_counter() - _t0)
                vanilla_output = llm.model.forward(
                    input_ids=late_tokens,
                    past_key_values=_early_cache_copy,
                    output_hidden_states=True,
                )

    # -- Dropout forward --------------------------------------------------------
    dropout_output = None
    dropout_step_masks = None
    if will_run_dropout:
        if _owns_early_cache:
            logger.info("consuming early_cache directly for dropout_late_forward")
            _dropout_cache = early_cache
        else:
            _t0 = time.perf_counter()
            _dropout_cache = copy.deepcopy(early_cache)
            logger.info("deepcopy(early_cache) for dropout_late_forward took %.4fs", time.perf_counter() - _t0)
        dropout_output, dropout_step_masks = dropout_late_forward(
            llm,
            parsed_output.cot_steps,
            early_tokens[0],
            _dropout_cache,
            late_tokens,
            modify_start_late,
            modify_end_late,
            nb_early_tokens,
            nb_dropout_samples,
            threshold,
            use_jackknife=use_jackknife,
        )

    return late_tokens, vanilla_output, dropout_output, answer_start_late, answer_end_late, dropout_step_masks


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
    use_jackknife=False,
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
    if use_jackknife:
        # Jackknife: keep ceil(log(k)) steps, mask the rest
        k = len(steps)
        nb_keep = _jackknife_nb_keep(k)
        nb_mask = k - nb_keep
        is_step_selected = np.ones((nb_dropout_samples, k), dtype=bool)
        for i in range(nb_dropout_samples):
            masked_indices = np.random.choice(k, size=nb_mask, replace=False)
            is_step_selected[i, masked_indices] = False
    else:
        # Coin-flip: each step independently kept with probability = threshold
        is_step_selected = (np.random.random((nb_dropout_samples, len(steps))) <= threshold)

    # Walk backwards through steps, finding each one's token range and masking.
    # Positions in early_token_ids directly correspond to early cache columns.
    remaining_ids = early_token_ids.clone()
    for step_id, step in reversed(list(enumerate(steps))):
        try:
            step_start, step_end = find_token_indices_from_end(
                llm.tokenizer, remaining_ids, step, llm.model_name)
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

    # Convert is_step_selected to binary masks (1 = kept, 0 = masked)
    step_masks = is_step_selected.astype(int).tolist()

    return late_output, step_masks


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
            "prob": round(list(vanilla_answer_probs[t].values())[0], 6),
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
