import re
import json
import logging
import math
import time
import numpy as np
import torch
import torch.nn as nn
import copy
from collections import defaultdict
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward, repeat_kv
from llm import LLM
from utils.cache_compat import DynamicCache, Qwen3_5DynamicCache
from utils.confidence_modification import apply_confidence_modification
from utils.text_utils import find_token_indices_from_end, find_token_overlap
from parsing import parse_output
from utils.structures import ParsedOutput, ConfidenceScores, AllConfidenceData

logger = logging.getLogger(__name__)


class ConfidenceAlignmentError(ValueError):
    """Raised when sampled CoT-step masking cannot be aligned to token spans."""


def _jackknife_nb_keep(k: int) -> int:
    """Number of optional CoT steps to keep under jackknife dropout."""
    # return math.ceil(math.sqrt(k)) if k >= 1 else 0
    return k // 2 if k >= 1 else 0


def _dropout_eligible_steps(cot_steps: list[str]) -> tuple[list[str], bool]:
    """Return the CoT steps that are eligible for dropout masking.

    If cot_steps[0] is an actual numbered reasoning step, it should be treated
    like every other reasoning step. Otherwise, cot_steps[0] is a preamble
    (for example "Let's think step-by-step.") and must stay always visible.
    """
    first_is_reasoning = bool(
        cot_steps and re.match(r'^\s*step\s', cot_steps[0], re.IGNORECASE)
    )
    return (cot_steps if first_is_reasoning else cot_steps[1:]), first_is_reasoning


def _jackknife_sample_steps(cot_steps: list[str], nb_dropout_samples: int, experimental_grid: bool = False, preserve_final_step: bool = False):
    """Sample jackknife keep/drop masks shared by both 'default' and 'true' modes.

    If the first step starts with "Step" (case/whitespace insensitive), it's treated as
    a reasoning step and can be masked. Otherwise, it's treated as a preamble and always kept.
    From the eligible steps, keeps exactly ceil(sqrt(k)) per sample (or k directly in grid mode).

    When preserve_final_step is True, the last eligible step is always kept and the
    masking pool is computed from n-1 steps instead of n.

    When experimental_grid is True, performs nb_dropout_samples runs of a 10-step grid:
    - For each of nb_dropout_samples runs:
      - Iteration i (1-indexed, i=1 to 10): k_i = floor(i * n / 10), keep k_i steps (no sqrt)
      - Iteration 10: k_10 = n, keep all n steps
    - Total samples returned: nb_dropout_samples * 10

    Returns:
        kept_indices: list[list[int]] — per-sample list of original cot_steps indices that are kept
        step_masks: list[list[int]] — per-sample binary masks over the optional steps (1=kept, 0=dropped)
    """
    steps, first_is_reasoning = _dropout_eligible_steps(cot_steps)
    n = len(steps)
    maskable = n - 1 if (preserve_final_step and n > 1) else n

    # Grid mode: nb_dropout_samples runs of 10-step grid each
    if experimental_grid:
        kept_indices = []
        step_masks = []

        for run in range(nb_dropout_samples):
            for i in range(1, 11):
                if i < 10:
                    k_i = math.floor(i * maskable / 10)
                    nb_keep = k_i
                else:
                    nb_keep = maskable

                selected_indices = np.random.choice(maskable, size=nb_keep, replace=False)
                is_step_selected = np.zeros(n, dtype=bool)
                is_step_selected[selected_indices] = True
                if preserve_final_step and n > 1:
                    is_step_selected[n - 1] = True

                # Build kept_indices accounting for preamble
                if first_is_reasoning:
                    sample_kept = [j for j in range(n) if is_step_selected[j]]
                else:
                    sample_kept = [0] + [1 + j for j in range(n) if is_step_selected[j]]

                kept_indices.append(sample_kept)
                step_masks.append(is_step_selected.astype(int).tolist())

        return kept_indices, step_masks

    # Fall through to existing logic for non-grid mode
    k = maskable
    nb_keep = _jackknife_nb_keep(k)
    nb_mask = k - nb_keep

    is_step_selected = np.ones((nb_dropout_samples, n), dtype=bool)
    if preserve_final_step and n > 1:
        is_step_selected[:, n - 1] = True
    for i in range(nb_dropout_samples):
        masked_indices = np.random.choice(k, size=nb_mask, replace=False)
        is_step_selected[i, masked_indices] = False

    kept_indices = []
    for i in range(nb_dropout_samples):
        if first_is_reasoning:
            sample_kept = [j for j in range(n) if is_step_selected[i, j]]
        else:
            sample_kept = [0] + [1 + j for j in range(n) if is_step_selected[i, j]]
        kept_indices.append(sample_kept)

    step_masks = is_step_selected.astype(int).tolist()
    return kept_indices, step_masks


def _bootstrap_sample_steps(cot_steps: list[str], nb_samples: int = 100):
    """Sample bootstrap replicates of CoT steps (with replacement).

    Always keeps the final eligible step. Samples (n-1) steps with replacement
    from the first (n-1) eligible steps, sorts them, appends the final step.
    Total steps per sample = n (same length as original eligible set).

    Returns:
        kept_indices: list[list[int]] — per-sample list of original cot_steps
            indices (may contain duplicates)
        step_counts: list[list[int]] — per-sample count of how many times each
            eligible step was sampled (analogous to binary step_masks but ≥ 0)
    """
    steps, first_is_reasoning = _dropout_eligible_steps(cot_steps)
    n = len(steps)
    if n == 0:
        return [], []
    if n == 1:
        counts = [1]
        if first_is_reasoning:
            kept = [0]
        else:
            kept = [0, 1]
        return [kept for _ in range(nb_samples)], [counts for _ in range(nb_samples)]

    pool_size = n - 1

    kept_indices = []
    step_counts = []
    for _ in range(nb_samples):
        sampled = np.sort(np.random.choice(pool_size, size=pool_size, replace=True))
        sampled = np.append(sampled, n - 1)

        counts = np.zeros(n, dtype=int)
        for idx in sampled:
            counts[idx] += 1

        if first_is_reasoning:
            sample_kept = sampled.tolist()
        else:
            sample_kept = [0] + [int(1 + j) for j in sampled]

        kept_indices.append(sample_kept)
        step_counts.append(counts.tolist())

    return kept_indices, step_counts


def _fake_bootstrap_sample_steps(cot_steps: list[str], nb_samples: int = 100):
    """Sample fake bootstrap replicates: bootstrap sampling with attention mask weighting.

    Like _bootstrap_sample_steps, samples (n-1) steps with replacement and always
    keeps the final step. Returns step counts (how many times each step was sampled)
    for use with weighted attention masks.

    Returns:
        step_counts: list[list[int]] — per-sample count of how many times each
            eligible step was sampled (0 = masked, k>0 = weighted by k)
    """
    steps, first_is_reasoning = _dropout_eligible_steps(cot_steps)
    n = len(steps)
    if n <= 1:
        return []

    pool_size = n - 1

    step_counts = []
    for _ in range(nb_samples):
        sampled = np.random.choice(pool_size, size=pool_size, replace=True)
        counts = np.zeros(n, dtype=int)
        for idx in sampled:
            counts[idx] += 1
        counts[n - 1] += 1  # Always keep final step

        step_counts.append(counts.tolist())

    return step_counts


def _coinflip_sample_steps(cot_steps: list[str], nb_dropout_samples: int, threshold: float = 0.5, preserve_final_step: bool = False):
    """Sample coin-flip keep/drop masks for token-level dropout.

    Each eligible step is independently kept with probability `threshold`.
    If all steps are dropped in a sample, that sample is redrawn.

    Returns:
        kept_indices: list[list[int]] — per-sample list of original cot_steps indices that are kept
        step_masks: list[list[int]] — per-sample binary masks over the optional steps (1=kept, 0=dropped)
    """
    steps, first_is_reasoning = _dropout_eligible_steps(cot_steps)
    n = len(steps)
    maskable = n - 1 if (preserve_final_step and n > 1) else n

    kept_indices = []
    step_masks = []

    for _ in range(nb_dropout_samples):
        while True:
            selected = np.random.random(maskable) <= threshold
            if selected.any():
                break

        is_step_selected = np.zeros(n, dtype=bool)
        is_step_selected[:maskable] = selected
        if preserve_final_step and n > 1:
            is_step_selected[n - 1] = True

        if first_is_reasoning:
            sample_kept = [j for j in range(n) if is_step_selected[j]]
        else:
            sample_kept = [0] + [1 + j for j in range(n) if is_step_selected[j]]

        kept_indices.append(sample_kept)
        step_masks.append(is_step_selected.astype(int).tolist())

    return kept_indices, step_masks


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


def _empty_sampled_confidence_scores() -> ConfidenceScores:
    return ConfidenceScores(
        answer_probabilities=[],
        answer_entropy=[],
        indirect_ptrue1_probabilities=[],
        indirect_ptrue2_probabilities=[],
        indirect_ptrue3_probabilities=[],
        verbconf_probabilities=[],
        verbconf_distribution=[],
        verbconf_top_score=[],
        verbconf_top_prob=[],
        step_masks=None,
    )


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
    experimental_jackknife: str | None = None,
    experimental_modification: str | None = None,
    experimental_grid: bool = False,
    vanilla_only: bool = False,
    preserve_final_step: bool = False,
    experimental_bootstrap: bool = False,
    experimental_coinflip_true: bool = False,
    experimental_hollow: int | None = None,
    experimental_fake_bootstrap: bool = False,
    experimental_fake_bootstrap_hook: bool = False,
) -> AllConfidenceData:
    debug_info = {}
    timings: dict[str, float] = {}
    _t_total_start = time.perf_counter()
    _t_setup_start = time.perf_counter()

    # Truncate generated_text to only include CoT steps (before the answer sentence)
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

    # Strip <think>...</think> blocks from the trajectory once, up front. All
    # downstream tokenization, re-parsing, modification, dropout, and
    # reconstruction-based passes then operate on the thinking-free text.
    pre_strip_cot = cot_only_text
    pre_strip_prefill = assistant_prefill
    cot_only_text = _strip_thinking(cot_only_text)
    if assistant_prefill:
        assistant_prefill = _strip_thinking(assistant_prefill)
    # gen_cache was built against the unstripped token sequence; invalidate it
    # if the strip actually changed the text so we don't reuse a stale KV cache.
    if cot_only_text != pre_strip_cot or assistant_prefill != pre_strip_prefill:
        gen_cache = None

    logger.info("compute_all_confidence_scores: truncated generated_text from %d to %d chars",
                len(generated_text), len(cot_only_text))

    # Re-parse cot_steps from the tokenizer-normalized text so that step
    # strings exactly match the token stream.  apply_chat_template may
    # normalise whitespace (e.g. collapse multiple spaces), causing the
    # original parsed steps to diverge from what the tokenizer produces.
    base_content = (assistant_prefill + cot_only_text).strip()
    normalized_content = llm.tokenizer.decode(
        llm.tokenizer.encode(base_content, add_special_tokens=False),
        skip_special_tokens=False,
    )
    reparsed = parse_output(normalized_content)
    if reparsed.cot_steps:
        parsed_output = copy.copy(parsed_output)
        parsed_output.cot_steps = reparsed.cot_steps
        logger.info("Re-parsed %d cot_steps from tokenizer-decoded text", len(reparsed.cot_steps))

    modification = apply_confidence_modification(
        normalized_content,
        parsed_output.cot_steps,
        experimental_modification,
    )
    parsed_output = copy.copy(parsed_output)
    parsed_output.cot_steps = modification.modified_steps
    cot_only_text = modification.modified_content
    assistant_prefill = ""
    normalized_content = cot_only_text
    can_run_step_dropout = len(parsed_output.cot_steps) > 1

    # Modification changes the token sequence, so the original gen_cache
    # (aligned to the unmodified tokens) is no longer valid.
    if modification.mode is not None:
        gen_cache = None

    if debug_conf:
        debug_info["experimental_modification"] = experimental_modification
        debug_info["modified_assistant_content"] = cot_only_text
        debug_info["modified_step_mappings"] = modification.step_mappings

    # Compute jackknife debug values AFTER all step list modifications
    if experimental_jackknife is not None:
        eligible_steps, _ = _dropout_eligible_steps(parsed_output.cot_steps)
        k = len(eligible_steps)
        nb_keep = _jackknife_nb_keep(k)
        debug_info["jackknife_k"] = k
        debug_info["jackknife_nb_mask"] = k - nb_keep

    # Compute base tokenization (no suffix) once, shared by all methods
    base_content = cot_only_text.strip()
    base_tokens = _tokenize_for_confidence(llm, messages, base_content)

    # Early/late split is now simply at the end of base_tokens (CoT only)
    early_late_split = base_tokens.shape[1]

    if gen_cache is not None:
        if vanilla_only:
            # Take ownership of gen_cache directly — no deepcopy. Crop in place.
            precomputed_early_cache = gen_cache
            crop_cache(precomputed_early_cache, early_late_split)
        else:
            precomputed_early_cache = copy.deepcopy(gen_cache)
            crop_cache(precomputed_early_cache, early_late_split)
    else:
        # No generation cache available — one forward pass to build early_cache
        if llm.backend == "vllm":
            raise ValueError(
                "Confidence scoring requires KV cache from generation, "
                "which is not available with vLLM backend. "
                "Use --backend hf for confidence scoring."
            )
        device = next(llm.model.parameters()).device
        early_tokens = base_tokens[:, :early_late_split].to(device)
        empty_cache = (
            Qwen3_5DynamicCache(config=llm.model.config)
            if "qwen" in llm.model_name.lower()
            else DynamicCache(config=llm.model.config)
        )
        with torch.no_grad():
            early_output = llm.model(input_ids=early_tokens, past_key_values=empty_cache)
        precomputed_early_cache = early_output.past_key_values

    # --- Coin-flip dropout (skipped when mode is "default" or "true") ---
    timings["setup"] = time.perf_counter() - _t_setup_start
    _t_vanilla_start = time.perf_counter()

    # Initialize confidence timings
    from utils.structures import ConfidenceTimings
    conf_timings = ConfidenceTimings()

    _coinflip_samples = 0 if experimental_jackknife in ["default", "true"] or experimental_coinflip_true or experimental_bootstrap or not can_run_step_dropout else nb_dropout_samples
    vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy, dbg, dropout_step_masks, masked_steps_dbg, meanprob_timings = \
        dropout_answerlogits(llm, messages, cot_only_text, parsed_output, _coinflip_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                             gen_cache=gen_cache, base_tokens=base_tokens,
                             precomputed_early_cache=precomputed_early_cache,
                             vanilla_only=vanilla_only,
                             consume_early_cache=vanilla_only,
                             recrop_early_cache_len=early_late_split if vanilla_only else None,
                             preserve_final_step=preserve_final_step)

    # Collect meanprob and meanent timing (same function call)
    conf_timings.vanilla_meanprob_total_seconds = meanprob_timings["total_seconds"]
    conf_timings.vanilla_meanprob_forward_seconds = meanprob_timings["forward_seconds"]
    conf_timings.vanilla_meanent_total_seconds = meanprob_timings["total_seconds"]
    conf_timings.vanilla_meanent_forward_seconds = meanprob_timings["forward_seconds"]

    if dbg:
        debug_info["answer_logits"] = dbg
    if masked_steps_dbg:
        debug_info["masked_steps"] = masked_steps_dbg

    vanilla_ptrue1, vanilla_ptrue2, vanilla_ptrue3, dropout_ptrue1, dropout_ptrue2, dropout_ptrue3, dbg, _, indirect_timings = \
        dropout_indirectlogits(llm, messages, cot_only_text, parsed_output, _coinflip_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                               gen_cache=gen_cache, base_tokens=base_tokens,
                               precomputed_early_cache=precomputed_early_cache,
                               vanilla_only=vanilla_only,
                               consume_early_cache=vanilla_only,
                               recrop_early_cache_len=early_late_split if vanilla_only else None,
                               preserve_final_step=preserve_final_step)

    # Collect indirect1 timing
    conf_timings.vanilla_indirect1_total_seconds = indirect_timings["total_seconds"]
    conf_timings.vanilla_indirect1_forward_seconds = indirect_timings["forward_seconds"]

    if dbg:
        debug_info["indirect_logits"] = dbg

    (vanilla_verbconf, dropout_verbconf,
     v_verbconf_dist, v_verbconf_top_score, v_verbconf_top_prob,
     d_verbconf_dist, d_verbconf_top_scores, d_verbconf_top_probs,
     dbg, _, verbconf_timings) = \
        dropout_verbalconf(llm, messages, cot_only_text, parsed_output, _coinflip_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                           gen_cache=gen_cache, base_tokens=base_tokens,
                           precomputed_early_cache=precomputed_early_cache,
                           consume_early_cache=(experimental_jackknife is None),
                           vanilla_only=vanilla_only,
                           preserve_final_step=preserve_final_step)

    # Collect verbconf timing
    conf_timings.vanilla_verbconf_total_seconds = verbconf_timings["total_seconds"]
    conf_timings.vanilla_verbconf_forward_seconds = verbconf_timings["forward_seconds"]
    conf_timings.vanilla_verbconf_joint_probs_seconds = verbconf_timings["joint_probs_seconds"]

    if dbg:
        debug_info["verbconf"] = dbg

    # --- Jackknife dropout (only when mode is set) ---
    timings["vanilla_coinflip_dropout"] = time.perf_counter() - _t_vanilla_start
    jackknife_scores = None
    jk_step_masks = None
    jk_kept_indices = None
    if experimental_jackknife == "default":
        _t_jk_start = time.perf_counter()
        if can_run_step_dropout:
            # Sample jackknife step masks once, reuse across all methods
            jk_kept_indices, jk_step_masks = _jackknife_sample_steps(
                parsed_output.cot_steps, nb_dropout_samples, experimental_grid=experimental_grid, preserve_final_step=preserve_final_step)

            # Use actual number of samples from the masks (nb_dropout_samples * 10 in grid mode)
            effective_nb_samples = len(jk_step_masks)

            _, _, jk_answer_probs, jk_answer_entropy, dbg, _, jk_masked_steps_dbg = \
                dropout_answerlogits(llm, messages, cot_only_text, parsed_output, effective_nb_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                                     gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                                     precomputed_early_cache=precomputed_early_cache,
                                     jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks)
            if dbg:
                debug_info["jackknife_answer_logits"] = dbg
            if jk_masked_steps_dbg:
                debug_info["jackknife_masked_steps"] = jk_masked_steps_dbg

            _, _, _, jk_ptrue1, jk_ptrue2, jk_ptrue3, dbg, _ = \
                dropout_indirectlogits(llm, messages, cot_only_text, parsed_output, effective_nb_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                                       gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                                       precomputed_early_cache=precomputed_early_cache,
                                       jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks)
            if dbg:
                debug_info["jackknife_indirect_logits"] = dbg

            (_, jk_verbconf,
             _, _, _,
             jk_verbconf_dist, jk_verbconf_top_scores, jk_verbconf_top_probs,
             dbg, _) = \
                dropout_verbalconf(llm, messages, cot_only_text, parsed_output, effective_nb_samples, use_fullstring, assistant_prefill, debug_conf=debug_conf,
                                   gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                                   precomputed_early_cache=precomputed_early_cache,
                                   consume_early_cache=True,
                                   jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks)
            if dbg:
                debug_info["jackknife_verbconf"] = dbg

            jackknife_scores = ConfidenceScores(
                answer_probabilities=jk_answer_probs,
                answer_entropy=jk_answer_entropy,
                indirect_ptrue1_probabilities=jk_ptrue1,
                indirect_ptrue2_probabilities=jk_ptrue2,
                indirect_ptrue3_probabilities=jk_ptrue3,
                verbconf_probabilities=jk_verbconf,
                verbconf_distribution=jk_verbconf_dist,
                verbconf_top_score=jk_verbconf_top_scores,
                verbconf_top_prob=jk_verbconf_top_probs,
                step_masks=jk_step_masks,
            )
        else:
            jackknife_scores = _empty_sampled_confidence_scores()
        timings["jackknife_default"] = time.perf_counter() - _t_jk_start

    elif experimental_jackknife == "true":
        _t_jkt_start = time.perf_counter()
        if can_run_step_dropout:
            jackknife_scores, jk_step_masks = _jackknife_true_forward(
                llm, messages, cot_only_text, parsed_output,
                nb_dropout_samples=nb_dropout_samples,
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
                debug_info=debug_info,
                normalized_content=normalized_content,
                experimental_grid=experimental_grid,
                preserve_final_step=preserve_final_step,
            )
        else:
            jackknife_scores = _empty_sampled_confidence_scores()
        timings["jackknife_true"] = time.perf_counter() - _t_jkt_start

    # --- Collect all prompts for debug ---
    if debug_conf:
        ans_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
        ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
        ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"
        verbconf_suffix = (
            f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
            "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
            "\nHow confident are you in your previous answer?"
            "\n<confidence>"
        )
        debug_info["all_prompts"] = {
            "answer_logits": _build_prompt_text(llm, messages,
                (assistant_prefill + cot_only_text + ans_suffix).strip()),
            "indirect_ptrue1": _build_prompt_text(llm, messages,
                (assistant_prefill + cot_only_text + ptrue1_suffix).strip()),
            "indirect_ptrue2": _build_prompt_text(llm, messages,
                (assistant_prefill + cot_only_text + ptrue2_suffix).strip()),
            "verbconf": _build_prompt_text(llm, messages,
                (assistant_prefill + cot_only_text + verbconf_suffix).strip()),
        }

    # --- Bootstrap (only when flag is set) ---
    bootstrap_scores = None
    if experimental_bootstrap:
        _t_bs_start = time.perf_counter()
        if len(parsed_output.cot_steps) >= 1:
            bootstrap_scores, bs_step_counts, bs_timings = _bootstrap_forward(
                llm, messages, cot_only_text, parsed_output,
                nb_dropout_samples=nb_dropout_samples,
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
                debug_info=debug_info,
                normalized_content=normalized_content,
            )
            # Populate bootstrap-specific timing fields
            conf_timings.bootstrap_meanprob_total_seconds = bs_timings["meanprob_total_seconds"]
            conf_timings.bootstrap_meanprob_forward_seconds = bs_timings["meanprob_forward_seconds"]
            conf_timings.bootstrap_indirect1_total_seconds = bs_timings["indirect1_total_seconds"]
            conf_timings.bootstrap_indirect1_forward_seconds = bs_timings["indirect1_forward_seconds"]
            conf_timings.bootstrap_verbconf_total_seconds = bs_timings["verbconf_total_seconds"]
            conf_timings.bootstrap_verbconf_forward_seconds = bs_timings["verbconf_forward_seconds"]
            conf_timings.bootstrap_verbconf_joint_probs_seconds = bs_timings["verbconf_joint_probs_seconds"]
        else:
            bootstrap_scores = _empty_sampled_confidence_scores()
        timings["bootstrap"] = time.perf_counter() - _t_bs_start

    # --- Fake Bootstrap (only when flag is set) ---
    fake_bootstrap_scores = None
    if experimental_fake_bootstrap:
        _t_fb_start = time.perf_counter()
        if can_run_step_dropout:
            fake_bootstrap_scores = _fake_bootstrap_forward(
                llm, messages, cot_only_text, parsed_output,
                nb_dropout_samples=nb_dropout_samples,
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
                debug_info=debug_info,
                normalized_content=normalized_content,
                gen_cache=gen_cache,
                base_tokens=base_tokens,
                precomputed_early_cache=precomputed_early_cache,
            )
        else:
            fake_bootstrap_scores = _empty_sampled_confidence_scores()
        timings["fake_bootstrap"] = time.perf_counter() - _t_fb_start

    # --- Fake Bootstrap Hook (only when flag is set) ---
    fake_bootstrap_hook_scores = None
    if experimental_fake_bootstrap_hook:
        _t_fbh_start = time.perf_counter()
        if can_run_step_dropout:
            fake_bootstrap_hook_scores = _fake_bootstrap_hook_forward(
                llm, messages, cot_only_text, parsed_output,
                nb_dropout_samples=nb_dropout_samples,
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
                debug_info=debug_info,
                normalized_content=normalized_content,
                gen_cache=gen_cache,
                base_tokens=base_tokens,
                precomputed_early_cache=precomputed_early_cache,
            )
        else:
            fake_bootstrap_hook_scores = _empty_sampled_confidence_scores()
        timings["fake_bootstrap_hook"] = time.perf_counter() - _t_fbh_start

    # --- Coinflip-true (only when flag is set) ---
    coinflip_true_scores = None
    if experimental_coinflip_true:
        _t_cft_start = time.perf_counter()
        if can_run_step_dropout:
            coinflip_true_scores, _cf_masks = _coinflip_true_forward(
                llm, messages, cot_only_text, parsed_output,
                nb_dropout_samples=nb_dropout_samples,
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
                debug_info=debug_info,
                normalized_content=normalized_content,
                preserve_final_step=preserve_final_step,
            )
        else:
            coinflip_true_scores = _empty_sampled_confidence_scores()
        timings["coinflip_true"] = time.perf_counter() - _t_cft_start

    # --- Hollow (only when flag is set) ---
    hollow_scores = None
    if experimental_hollow is not None:
        _t_hollow_start = time.perf_counter()
        if len(parsed_output.cot_steps) >= 1:
            hollow_scores = _hollow_forward(
                llm, messages, cot_only_text, parsed_output,
                assistant_prefill=assistant_prefill,
                debug_conf=debug_conf,
                debug_info=debug_info,
                normalized_content=normalized_content,
                hollow_mode=experimental_hollow,
            )
        else:
            hollow_scores = _empty_sampled_confidence_scores()
        timings["hollow"] = time.perf_counter() - _t_hollow_start

    timings["total"] = time.perf_counter() - _t_total_start
    if debug_conf:
        debug_info["timings_seconds"] = {k: round(v, 4) for k, v in timings.items()}

    return AllConfidenceData(
        vanilla_confidences=ConfidenceScores(
            answer_probabilities=vanilla_answer_probs,
            answer_entropy=vanilla_answer_entropy,
            indirect_ptrue1_probabilities=vanilla_ptrue1,
            indirect_ptrue2_probabilities=vanilla_ptrue2,
            indirect_ptrue3_probabilities=vanilla_ptrue3,
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
            indirect_ptrue3_probabilities=dropout_ptrue3,
            verbconf_probabilities=dropout_verbconf,
            verbconf_distribution=d_verbconf_dist,
            verbconf_top_score=d_verbconf_top_scores,
            verbconf_top_prob=d_verbconf_top_probs,
            step_masks=dropout_step_masks,
        ),
        jackknife_confidences=jackknife_scores,
        bootstrap_confidences=bootstrap_scores,
        fake_bootstrap_confidences=fake_bootstrap_scores,
        fake_bootstrap_hook_confidences=fake_bootstrap_hook_scores,
        coinflip_true_confidences=coinflip_true_scores,
        hollow_confidences=hollow_scores,
        debug_info=debug_info,
        timings=conf_timings,
    )


# Regex to match and capture step numbers for renumbering
_STEP_NUMBER_RE = re.compile(r'^(Step\s+)\d+(\s*:)', re.IGNORECASE)

# Regexes to strip <think>...</think> blocks from reconstructed prompts.
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL | re.IGNORECASE)
_STRAY_THINK_RE = re.compile(r'</?think>\s*', re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks plus prefill-opened thinking content.

    Three cases handled:
      1. Complete `<think>...</think>` blocks within `text` are removed.
      2. A lone `</think>` (no preceding `<think>` in `text`) means the opener
         was in the assistant prefill / chat template. Drop everything up to
         and including that closer — the entire span is thinking content.
      3. Any remaining stray `<think>` / `</think>` tags are stripped.
    """
    text = _THINK_BLOCK_RE.sub('', text)
    close_idx = text.find('</think>')
    if close_idx != -1 and '<think>' not in text[:close_idx]:
        text = text[close_idx + len('</think>'):]
    text = _STRAY_THINK_RE.sub('', text)
    return text.lstrip()


def _renumber_step(step_text: str, new_number: int) -> str:
    """Replace the step number in a step marker with new_number.

    Args:
        step_text: The original step text (e.g., "Step 5: Calculate the sum")
        new_number: The new step number to use (e.g., 2)

    Returns:
        The step text with the number replaced (e.g., "Step 2: Calculate the sum")
    """
    return _STEP_NUMBER_RE.sub(rf'\g<1>{new_number}\g<2>', step_text, count=1)


def _reconstruct_cot_text(normalized_content: str, cot_steps: list[str], kept_indices: list[int]) -> str:
    """Rebuild CoT text keeping only the steps at kept_indices, with renumbered steps.

    Preserves the non-step prefix (everything before the first step) verbatim.
    Renumbers kept steps to be consecutive (Step 1, Step 2, Step 3, ...).
    """
    if not cot_steps:
        return normalized_content

    # Find where the first step starts — everything before it is the prefix
    first_step_pos = normalized_content.find(cot_steps[0])
    if first_step_pos == -1:
        # Fallback: if we can't locate steps, return the full text
        return normalized_content
    prefix = normalized_content[:first_step_pos]

    # Collect kept step texts
    kept_step_texts = [cot_steps[i] for i in kept_indices if i < len(cot_steps)]

    # Renumber steps to be consecutive
    renumbered_steps = []
    step_counter = 1
    for step_text in kept_step_texts:
        if _STEP_NUMBER_RE.match(step_text):
            renumbered_steps.append(_renumber_step(step_text, step_counter))
            step_counter += 1
        else:
            # Not a numbered step (e.g., preamble), keep as-is
            renumbered_steps.append(step_text)

    return prefix + "\n".join(renumbered_steps)


def _jackknife_true_forward(
    llm: LLM,
    messages: list[dict],
    cot_only_text: str,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int,
    assistant_prefill: str,
    debug_conf: bool,
    debug_info: dict,
    normalized_content: str,
    experimental_grid: bool = False,
    preserve_final_step: bool = False,
) -> tuple[ConfidenceScores, list[list[int]]]:
    """Jackknife 'true' mode: rebuild prompts from sampled steps and run fresh forward passes.

    For each dropout sample, reconstructs the assistant content from the
    kept CoT steps, appends the method-specific suffix, builds the full
    prompt via _build_prompt_text, and runs a fresh model forward pass
    without any cache reuse or attention-mask modifications.

    Returns (ConfidenceScores, step_masks).
    """
    device = next(llm.model.parameters()).device

    # Sample which steps to keep — shared selection for all confidence methods
    kept_indices_per_sample, step_masks = _jackknife_sample_steps(
        parsed_output.cot_steps, nb_dropout_samples, experimental_grid=experimental_grid, preserve_final_step=preserve_final_step)

    # Use actual number of samples (may be nb_dropout_samples * 10 in grid mode)
    effective_nb_samples = len(kept_indices_per_sample)

    # Define suffixes for each confidence method
    ans_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
    ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
    ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"
    verbconf_suffix = (
        f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )

    # Token IDs for indirect logits
    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
    negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])
    yes_only_id = llm.tokenizer.encode(' Yes', add_special_tokens=False)
    no_only_id = llm.tokenizer.encode(' No', add_special_tokens=False)

    # Verbconf setup
    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.tensor([int(s) / 100 for s in verbconf_strings], dtype=torch.float64)

    # Accumulators for per-sample results
    all_answer_probs = []
    all_answer_entropy = []
    all_ptrue1 = []
    all_ptrue2 = []
    all_ptrue3 = []
    all_verbconf = []
    all_verbconf_dist = []
    all_verbconf_top_scores = []
    all_verbconf_top_probs = []

    # Debug prompt accumulators
    dbg_prompts_answer = []
    dbg_prompts_ptrue1 = []
    dbg_prompts_ptrue2 = []
    dbg_prompts_verbconf = []

    for sample_idx in range(effective_nb_samples):
        kept_indices = kept_indices_per_sample[sample_idx]

        # Reconstruct CoT text with only the kept steps
        reconstructed_cot = _reconstruct_cot_text(
            normalized_content, parsed_output.cot_steps, kept_indices)

        # Build the full assistant content for each method
        reconstructed_base = (assistant_prefill + reconstructed_cot).strip() if assistant_prefill else reconstructed_cot.strip()

        # --- answer_logits ---
        ans_content = (reconstructed_base + ans_suffix).strip()
        ans_prompt = _build_prompt_text(llm, messages, ans_content)
        ans_tokens = llm.tokenizer(ans_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_answer.append(ans_prompt)

        # Find answer token positions
        boxed_string = f"\\boxed{{{parsed_output.final_answer}}}"
        with torch.no_grad():
            ans_output = llm.model.forward(input_ids=ans_tokens)

        # Find answer tokens in the full sequence
        ans_token_ids = ans_tokens[0]
        boxed_start, boxed_end = find_token_indices_from_end(
            llm.tokenizer, ans_token_ids, boxed_string, llm.model_name)
        a_start, a_end = find_token_indices_from_end(
            llm.tokenizer, ans_token_ids[boxed_start:boxed_end],
            parsed_output.final_answer, llm.model_name)
        a_start += boxed_start
        a_end += boxed_start

        answer_toks = ans_token_ids[a_start:a_end]
        nb_answer_tokens = len(answer_toks)
        token_strings = [llm.tokenizer.decode([answer_toks[t].item()]) for t in range(nb_answer_tokens)]

        v_logits = ans_output.logits[0, a_start - 1:a_end - 1, :]
        v_probs = v_logits.double().softmax(dim=-1)
        sample_answer_probs = [{token_strings[t]: v_probs[t, answer_toks[t]].item()} for t in range(nb_answer_tokens)]
        all_answer_probs.append(sample_answer_probs)

        v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
        entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).cpu().numpy().tolist()
        sample_entropy = [{token_strings[t]: entropy_values[t]} for t in range(nb_answer_tokens)]
        all_answer_entropy.append(sample_entropy)

        # --- indirect_ptrue1 ---
        ptrue1_content = (reconstructed_base + ptrue1_suffix).strip()
        ptrue1_prompt = _build_prompt_text(llm, messages, ptrue1_content)
        ptrue1_tokens = llm.tokenizer(ptrue1_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_ptrue1.append(ptrue1_prompt)

        with torch.no_grad():
            ptrue1_output = llm.model.forward(input_ids=ptrue1_tokens)

        p1_pos = ptrue1_output.logits[0, -1, positive_true_ids].sum()
        p1_neg = ptrue1_output.logits[0, -1, negative_false_ids].sum()
        p1_sm = torch.softmax(torch.stack([p1_pos.double(), p1_neg.double()]), dim=0)
        all_ptrue1.append({"True": p1_sm[0].item(), "False": p1_sm[1].item()})

        # --- indirect_ptrue2 (also used for ptrue3) ---
        ptrue2_content = (reconstructed_base + ptrue2_suffix).strip()
        ptrue2_prompt = _build_prompt_text(llm, messages, ptrue2_content)
        ptrue2_tokens = llm.tokenizer(ptrue2_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_ptrue2.append(ptrue2_prompt)

        with torch.no_grad():
            ptrue2_output = llm.model.forward(input_ids=ptrue2_tokens)

        p2_pos = ptrue2_output.logits[0, -1, positive_yes_ids].sum()
        p2_neg = ptrue2_output.logits[0, -1, negative_no_ids].sum()
        p2_sm = torch.softmax(torch.stack([p2_pos.double(), p2_neg.double()]), dim=0)
        all_ptrue2.append({"Yes": p2_sm[0].item(), "No": p2_sm[1].item()})

        # ptrue3: Yes/No only (no synonyms) — reuse ptrue2 output
        p3_pos = ptrue2_output.logits[0, -1, yes_only_id[0]]
        p3_neg = ptrue2_output.logits[0, -1, no_only_id[0]]
        p3_sm = torch.softmax(torch.stack([p3_pos.double(), p3_neg.double()]), dim=0)
        all_ptrue3.append({"Yes": p3_sm[0].item(), "No": p3_sm[1].item()})

        # --- verbconf ---
        verbconf_content = (reconstructed_base + verbconf_suffix).strip()
        verbconf_prompt = _build_prompt_text(llm, messages, verbconf_content)
        verbconf_tokens = llm.tokenizer(verbconf_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_verbconf.append(verbconf_prompt)

        with torch.no_grad():
            verbconf_output = llm.model.forward(input_ids=verbconf_tokens)

        # Compute joint probs for multi-token numbers using the single-sample output
        vc_probs = _compute_verbconf_joint_probs(llm, verbconf_output, token_seqs, device)
        vc_probs_1d = vc_probs[0] if vc_probs.dim() > 1 else vc_probs
        sample_verbconf = (score_values * vc_probs_1d).sum().item()
        all_verbconf.append(sample_verbconf)
        all_verbconf_dist.append(vc_probs_1d.tolist())
        top_idx = vc_probs_1d.argmax().item()
        all_verbconf_top_scores.append(top_idx)
        all_verbconf_top_probs.append(vc_probs_1d[top_idx].item())

        torch.cuda.empty_cache()

    # Save debug prompts for "true" mode
    if debug_conf:
        debug_info["jackknife_true_prompts"] = {
            "answer_logits": dbg_prompts_answer,
            "indirect_ptrue1": dbg_prompts_ptrue1,
            "indirect_ptrue2": dbg_prompts_ptrue2,
            "verbconf": dbg_prompts_verbconf,
        }
        debug_info["jackknife_masked_steps"] = [
            {
                "sample_idx": i,
                "kept_step_indices": kept_indices_per_sample[i],
                "kept_steps": [parsed_output.cot_steps[j] for j in kept_indices_per_sample[i] if j < len(parsed_output.cot_steps)],
                "dropped_steps": [parsed_output.cot_steps[j] for j in range(len(parsed_output.cot_steps)) if j not in kept_indices_per_sample[i]],
            }
            for i in range(nb_dropout_samples)
        ]

    scores = ConfidenceScores(
        answer_probabilities=all_answer_probs,
        answer_entropy=all_answer_entropy,
        indirect_ptrue1_probabilities=all_ptrue1,
        indirect_ptrue2_probabilities=all_ptrue2,
        indirect_ptrue3_probabilities=all_ptrue3,
        verbconf_probabilities=all_verbconf,
        verbconf_distribution=all_verbconf_dist,
        verbconf_top_score=all_verbconf_top_scores,
        verbconf_top_prob=all_verbconf_top_probs,
        step_masks=step_masks,
    )

    return scores, step_masks


def _coinflip_true_forward(
    llm: LLM,
    messages: list[dict],
    cot_only_text: str,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int,
    assistant_prefill: str,
    debug_conf: bool,
    debug_info: dict,
    normalized_content: str,
    preserve_final_step: bool = False,
) -> tuple[ConfidenceScores, list[list[int]]]:
    """Coin-flip 'true' mode: rebuild prompts from sampled steps and run fresh forward passes.

    Like _jackknife_true_forward but each eligible step is independently kept
    with p=0.5 (Bernoulli) instead of a fixed-count mask.
    """
    device = next(llm.model.parameters()).device

    kept_indices_per_sample, step_masks = _coinflip_sample_steps(
        parsed_output.cot_steps, nb_dropout_samples, preserve_final_step=preserve_final_step)

    effective_nb_samples = len(kept_indices_per_sample)

    ans_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
    ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
    # ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"  # Not used
    verbconf_suffix = (
        f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )

    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    # positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])  # Not used
    # negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])  # Not used
    # yes_only_id = llm.tokenizer.encode(' Yes', add_special_tokens=False)  # Not used
    # no_only_id = llm.tokenizer.encode(' No', add_special_tokens=False)  # Not used

    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.tensor([int(s) / 100 for s in verbconf_strings], dtype=torch.float64)

    all_answer_probs = []
    all_answer_entropy = []
    all_ptrue1 = []
    # all_ptrue2 = []  # Not used
    # all_ptrue3 = []  # Not used
    all_verbconf = []
    all_verbconf_dist = []
    all_verbconf_top_scores = []
    all_verbconf_top_probs = []

    dbg_prompts_answer = []
    dbg_prompts_ptrue1 = []
    # dbg_prompts_ptrue2 = []  # Not used
    dbg_prompts_verbconf = []

    for sample_idx in range(effective_nb_samples):
        kept_indices = kept_indices_per_sample[sample_idx]

        reconstructed_cot = _reconstruct_cot_text(
            normalized_content, parsed_output.cot_steps, kept_indices)

        reconstructed_base = (assistant_prefill + reconstructed_cot).strip() if assistant_prefill else reconstructed_cot.strip()

        # --- answer_logits ---
        ans_content = (reconstructed_base + ans_suffix).strip()
        ans_prompt = _build_prompt_text(llm, messages, ans_content)
        ans_tokens = llm.tokenizer(ans_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_answer.append(ans_prompt)

        boxed_string = f"\\boxed{{{parsed_output.final_answer}}}"
        with torch.no_grad():
            ans_output = llm.model.forward(input_ids=ans_tokens)

        ans_token_ids = ans_tokens[0]
        boxed_start, boxed_end = find_token_indices_from_end(
            llm.tokenizer, ans_token_ids, boxed_string, llm.model_name)
        a_start, a_end = find_token_indices_from_end(
            llm.tokenizer, ans_token_ids[boxed_start:boxed_end],
            parsed_output.final_answer, llm.model_name)
        a_start += boxed_start
        a_end += boxed_start

        answer_toks = ans_token_ids[a_start:a_end]
        nb_answer_tokens = len(answer_toks)
        token_strings = [llm.tokenizer.decode([answer_toks[t].item()]) for t in range(nb_answer_tokens)]

        v_logits = ans_output.logits[0, a_start - 1:a_end - 1, :]
        v_probs = v_logits.double().softmax(dim=-1)
        sample_answer_probs = [{token_strings[t]: v_probs[t, answer_toks[t]].item()} for t in range(nb_answer_tokens)]
        all_answer_probs.append(sample_answer_probs)

        v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
        entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).cpu().numpy().tolist()
        sample_entropy = [{token_strings[t]: entropy_values[t]} for t in range(nb_answer_tokens)]
        all_answer_entropy.append(sample_entropy)

        # --- indirect_ptrue1 ---
        ptrue1_content = (reconstructed_base + ptrue1_suffix).strip()
        ptrue1_prompt = _build_prompt_text(llm, messages, ptrue1_content)
        ptrue1_tokens = llm.tokenizer(ptrue1_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_ptrue1.append(ptrue1_prompt)

        with torch.no_grad():
            ptrue1_output = llm.model.forward(input_ids=ptrue1_tokens)

        p1_pos = ptrue1_output.logits[0, -1, positive_true_ids].sum()
        p1_neg = ptrue1_output.logits[0, -1, negative_false_ids].sum()
        p1_sm = torch.softmax(torch.stack([p1_pos.double(), p1_neg.double()]), dim=0)
        all_ptrue1.append({"True": p1_sm[0].item(), "False": p1_sm[1].item()})

        # # --- indirect_ptrue2 (also used for ptrue3) ---
        # ptrue2_content = (reconstructed_base + ptrue2_suffix).strip()
        # ptrue2_prompt = _build_prompt_text(llm, messages, ptrue2_content)
        # ptrue2_tokens = llm.tokenizer(ptrue2_prompt, return_tensors="pt")["input_ids"].to(device)

        # if debug_conf:
        #     dbg_prompts_ptrue2.append(ptrue2_prompt)

        # with torch.no_grad():
        #     ptrue2_output = llm.model.forward(input_ids=ptrue2_tokens)

        # p2_pos = ptrue2_output.logits[0, -1, positive_yes_ids].sum()
        # p2_neg = ptrue2_output.logits[0, -1, negative_no_ids].sum()
        # p2_sm = torch.softmax(torch.stack([p2_pos.double(), p2_neg.double()]), dim=0)
        # all_ptrue2.append({"Yes": p2_sm[0].item(), "No": p2_sm[1].item()})

        # p3_pos = ptrue2_output.logits[0, -1, yes_only_id[0]]
        # p3_neg = ptrue2_output.logits[0, -1, no_only_id[0]]
        # p3_sm = torch.softmax(torch.stack([p3_pos.double(), p3_neg.double()]), dim=0)
        # all_ptrue3.append({"Yes": p3_sm[0].item(), "No": p3_sm[1].item()})

        # --- verbconf ---
        verbconf_content = (reconstructed_base + verbconf_suffix).strip()
        verbconf_prompt = _build_prompt_text(llm, messages, verbconf_content)
        verbconf_tokens = llm.tokenizer(verbconf_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_verbconf.append(verbconf_prompt)

        with torch.no_grad():
            verbconf_output = llm.model.forward(input_ids=verbconf_tokens)

        vc_probs = _compute_verbconf_joint_probs(llm, verbconf_output, token_seqs, device)
        vc_probs_1d = vc_probs[0] if vc_probs.dim() > 1 else vc_probs
        sample_verbconf = (score_values * vc_probs_1d).sum().item()
        all_verbconf.append(sample_verbconf)
        all_verbconf_dist.append(vc_probs_1d.tolist())
        top_idx = vc_probs_1d.argmax().item()
        all_verbconf_top_scores.append(top_idx)
        all_verbconf_top_probs.append(vc_probs_1d[top_idx].item())

        torch.cuda.empty_cache()

    if debug_conf:
        debug_info["coinflip_true_prompts"] = {
            "answer_logits": dbg_prompts_answer,
            "indirect_ptrue1": dbg_prompts_ptrue1,
            "indirect_ptrue2": dbg_prompts_ptrue2,
            "verbconf": dbg_prompts_verbconf,
        }
        debug_info["coinflip_true_masked_steps"] = [
            {
                "sample_idx": i,
                "kept_step_indices": kept_indices_per_sample[i],
                "kept_steps": [parsed_output.cot_steps[j] for j in kept_indices_per_sample[i] if j < len(parsed_output.cot_steps)],
                "dropped_steps": [parsed_output.cot_steps[j] for j in range(len(parsed_output.cot_steps)) if j not in kept_indices_per_sample[i]],
            }
            for i in range(effective_nb_samples)
        ]

    scores = ConfidenceScores(
        answer_probabilities=all_answer_probs,
        answer_entropy=all_answer_entropy,
        indirect_ptrue1_probabilities=all_ptrue1,
        indirect_ptrue2_probabilities=all_ptrue2,
        indirect_ptrue3_probabilities=all_ptrue3,
        verbconf_probabilities=all_verbconf,
        verbconf_distribution=all_verbconf_dist,
        verbconf_top_score=all_verbconf_top_scores,
        verbconf_top_prob=all_verbconf_top_probs,
        step_masks=step_masks,
    )

    return scores, step_masks


def _hollow_forward(
    llm: LLM,
    messages: list[dict],
    cot_only_text: str,
    parsed_output: ParsedOutput,
    assistant_prefill: str,
    debug_conf: bool,
    debug_info: dict,
    normalized_content: str,
    hollow_mode: int,
) -> ConfidenceScores:
    """Hollow confidence: drop every step except the last, run fresh forward passes.

    hollow_mode == 1: keep the last step text as-is.
    hollow_mode == 2: replace the last step with a canned sentence that
        references the final answer.

    In both modes the retained step is renumbered to "Step 1" via
    `_reconstruct_cot_text` (which renumbers kept steps consecutively).

    Returns a ConfidenceScores with a single sample per method (placed in the
    "dropout"-style list fields so downstream code can treat it uniformly).
    """
    device = next(llm.model.parameters()).device

    cot_steps = list(parsed_output.cot_steps)
    eligible_steps, first_is_reasoning = _dropout_eligible_steps(cot_steps)
    if not eligible_steps:
        return _empty_sampled_confidence_scores()

    # Index (in original cot_steps) of the last eligible step
    last_eligible_idx = len(cot_steps) - 1

    # For mode 2, splice a replacement step string into cot_steps at that index
    local_cot_steps = list(cot_steps)
    if hollow_mode == 2:
        replacement = (
            f"Step {last_eligible_idx + 1}: Considering the question, "
            f"{parsed_output.final_answer} seems to be the most appropriate answer."
        )
        local_cot_steps[last_eligible_idx] = replacement

    # Build kept_indices: preamble (if any) + last eligible step
    kept_indices = [last_eligible_idx]

    reconstructed_cot = _reconstruct_cot_text(
        normalized_content, local_cot_steps, kept_indices)
    reconstructed_base = (
        (assistant_prefill + reconstructed_cot).strip()
        if assistant_prefill else reconstructed_cot.strip()
    )

    ans_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
    ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
    ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"
    verbconf_suffix = (
        f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
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

    # --- answer_logits ---
    ans_content = (reconstructed_base + ans_suffix).strip()
    ans_prompt = _build_prompt_text(llm, messages, ans_content)
    ans_tokens = llm.tokenizer(ans_prompt, return_tensors="pt")["input_ids"].to(device)

    boxed_string = f"\\boxed{{{parsed_output.final_answer}}}"
    with torch.no_grad():
        ans_output = llm.model.forward(input_ids=ans_tokens)

    ans_token_ids = ans_tokens[0]
    boxed_start, boxed_end = find_token_indices_from_end(
        llm.tokenizer, ans_token_ids, boxed_string, llm.model_name)
    a_start, a_end = find_token_indices_from_end(
        llm.tokenizer, ans_token_ids[boxed_start:boxed_end],
        parsed_output.final_answer, llm.model_name)
    a_start += boxed_start
    a_end += boxed_start

    answer_toks = ans_token_ids[a_start:a_end]
    nb_answer_tokens = len(answer_toks)
    token_strings = [llm.tokenizer.decode([answer_toks[t].item()]) for t in range(nb_answer_tokens)]

    v_logits = ans_output.logits[0, a_start - 1:a_end - 1, :]
    v_probs = v_logits.double().softmax(dim=-1)
    sample_answer_probs = [{token_strings[t]: v_probs[t, answer_toks[t]].item()} for t in range(nb_answer_tokens)]

    v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
    entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).cpu().numpy().tolist()
    sample_entropy = [{token_strings[t]: entropy_values[t]} for t in range(nb_answer_tokens)]

    # --- indirect_ptrue1 ---
    ptrue1_content = (reconstructed_base + ptrue1_suffix).strip()
    ptrue1_prompt = _build_prompt_text(llm, messages, ptrue1_content)
    ptrue1_tokens = llm.tokenizer(ptrue1_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        ptrue1_output = llm.model.forward(input_ids=ptrue1_tokens)
    p1_pos = ptrue1_output.logits[0, -1, positive_true_ids].sum()
    p1_neg = ptrue1_output.logits[0, -1, negative_false_ids].sum()
    p1_sm = torch.softmax(torch.stack([p1_pos.double(), p1_neg.double()]), dim=0)
    sample_ptrue1 = {"True": p1_sm[0].item(), "False": p1_sm[1].item()}

    # --- indirect_ptrue2 / ptrue3 ---
    ptrue2_content = (reconstructed_base + ptrue2_suffix).strip()
    ptrue2_prompt = _build_prompt_text(llm, messages, ptrue2_content)
    ptrue2_tokens = llm.tokenizer(ptrue2_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        ptrue2_output = llm.model.forward(input_ids=ptrue2_tokens)
    p2_pos = ptrue2_output.logits[0, -1, positive_yes_ids].sum()
    p2_neg = ptrue2_output.logits[0, -1, negative_no_ids].sum()
    p2_sm = torch.softmax(torch.stack([p2_pos.double(), p2_neg.double()]), dim=0)
    sample_ptrue2 = {"Yes": p2_sm[0].item(), "No": p2_sm[1].item()}

    p3_pos = ptrue2_output.logits[0, -1, yes_only_id[0]]
    p3_neg = ptrue2_output.logits[0, -1, no_only_id[0]]
    p3_sm = torch.softmax(torch.stack([p3_pos.double(), p3_neg.double()]), dim=0)
    sample_ptrue3 = {"Yes": p3_sm[0].item(), "No": p3_sm[1].item()}

    # --- verbconf ---
    verbconf_content = (reconstructed_base + verbconf_suffix).strip()
    verbconf_prompt = _build_prompt_text(llm, messages, verbconf_content)
    verbconf_tokens = llm.tokenizer(verbconf_prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        verbconf_output = llm.model.forward(input_ids=verbconf_tokens)
    vc_probs = _compute_verbconf_joint_probs(llm, verbconf_output, token_seqs, device)
    vc_probs_1d = vc_probs[0] if vc_probs.dim() > 1 else vc_probs
    sample_verbconf = (score_values * vc_probs_1d).sum().item()
    sample_verbconf_dist = vc_probs_1d.tolist()
    top_idx = vc_probs_1d.argmax().item()
    sample_verbconf_top_score = top_idx
    sample_verbconf_top_prob = vc_probs_1d[top_idx].item()

    torch.cuda.empty_cache()

    if debug_conf:
        debug_info["hollow_mode"] = hollow_mode
        debug_info["hollow_prompts"] = {
            "answer_logits": ans_prompt,
            "indirect_ptrue1": ptrue1_prompt,
            "indirect_ptrue2": ptrue2_prompt,
            "verbconf": verbconf_prompt,
        }
        debug_info["hollow_kept_step"] = local_cot_steps[last_eligible_idx]

    return ConfidenceScores(
        answer_probabilities=[sample_answer_probs],
        answer_entropy=[sample_entropy],
        indirect_ptrue1_probabilities=[sample_ptrue1],
        indirect_ptrue2_probabilities=[sample_ptrue2],
        indirect_ptrue3_probabilities=[sample_ptrue3],
        verbconf_probabilities=[sample_verbconf],
        verbconf_distribution=[sample_verbconf_dist],
        verbconf_top_score=[sample_verbconf_top_score],
        verbconf_top_prob=[sample_verbconf_top_prob],
        step_masks=None,
    )


def _bootstrap_forward(
    llm: LLM,
    messages: list[dict],
    cot_only_text: str,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int,
    assistant_prefill: str,
    debug_conf: bool,
    debug_info: dict,
    normalized_content: str,
) -> tuple[ConfidenceScores, list[list[int]], dict]:
    """Bootstrap confidence: resample steps with replacement and run fresh forward passes.

    For each bootstrap sample, draws (n-1) steps with replacement from the
    first (n-1) eligible steps, sorts them, appends the final step, rebuilds
    the prompt, and runs all four confidence methods via fresh forward passes.

    Returns (ConfidenceScores, step_counts, timings_dict).
    """
    device = next(llm.model.parameters()).device

    # Bootstrap-specific timing accumulators
    bs_meanprob_total = 0.0
    bs_meanprob_forward = 0.0
    bs_indirect1_total = 0.0
    bs_indirect1_forward = 0.0
    bs_verbconf_total = 0.0
    bs_verbconf_forward = 0.0
    bs_verbconf_joint_probs = 0.0

    kept_indices_per_sample, step_counts = _bootstrap_sample_steps(
        parsed_output.cot_steps, nb_dropout_samples)

    effective_nb_samples = len(kept_indices_per_sample)

    ans_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
    ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
    # ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"  # Not used
    verbconf_suffix = (
        f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )

    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    # positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])  # Not used
    # negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])  # Not used
    # yes_only_id = llm.tokenizer.encode(' Yes', add_special_tokens=False)  # Not used
    # no_only_id = llm.tokenizer.encode(' No', add_special_tokens=False)  # Not used

    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.tensor([int(s) / 100 for s in verbconf_strings], dtype=torch.float64)

    all_answer_probs = []
    all_answer_entropy = []
    all_ptrue1 = []
    # all_ptrue2 = []  # Not used
    # all_ptrue3 = []  # Not used
    all_verbconf = []
    all_verbconf_dist = []
    all_verbconf_top_scores = []
    all_verbconf_top_probs = []

    dbg_prompts_answer = []
    dbg_prompts_ptrue1 = []
    # dbg_prompts_ptrue2 = []  # Not used
    dbg_prompts_verbconf = []

    for sample_idx in range(effective_nb_samples):
        kept_indices = kept_indices_per_sample[sample_idx]

        reconstructed_cot = _reconstruct_cot_text(
            normalized_content, parsed_output.cot_steps, kept_indices)

        reconstructed_base = (assistant_prefill + reconstructed_cot).strip() if assistant_prefill else reconstructed_cot.strip()

        # --- answer_logits ---
        _t_ans_start = time.perf_counter()
        _t_fwd_start = time.perf_counter()
        ans_content = (reconstructed_base + ans_suffix).strip()
        ans_prompt = _build_prompt_text(llm, messages, ans_content)
        ans_tokens = llm.tokenizer(ans_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_answer.append(ans_prompt)

        boxed_string = f"\\boxed{{{parsed_output.final_answer}}}"
        with torch.no_grad():
            ans_output = llm.model.forward(input_ids=ans_tokens)
        bs_meanprob_forward += time.perf_counter() - _t_fwd_start

        ans_token_ids = ans_tokens[0]
        boxed_start, boxed_end = find_token_indices_from_end(
            llm.tokenizer, ans_token_ids, boxed_string, llm.model_name)
        a_start, a_end = find_token_indices_from_end(
            llm.tokenizer, ans_token_ids[boxed_start:boxed_end],
            parsed_output.final_answer, llm.model_name)
        a_start += boxed_start
        a_end += boxed_start

        answer_toks = ans_token_ids[a_start:a_end]
        nb_answer_tokens = len(answer_toks)
        token_strings = [llm.tokenizer.decode([answer_toks[t].item()]) for t in range(nb_answer_tokens)]

        v_logits = ans_output.logits[0, a_start - 1:a_end - 1, :]
        v_probs = v_logits.double().softmax(dim=-1)
        sample_answer_probs = [{token_strings[t]: v_probs[t, answer_toks[t]].item()} for t in range(nb_answer_tokens)]
        all_answer_probs.append(sample_answer_probs)

        v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
        entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).cpu().numpy().tolist()
        sample_entropy = [{token_strings[t]: entropy_values[t]} for t in range(nb_answer_tokens)]
        all_answer_entropy.append(sample_entropy)
        bs_meanprob_total += time.perf_counter() - _t_ans_start

        # --- indirect_ptrue1 ---
        _t_ptrue1_start = time.perf_counter()
        _t_fwd_start = time.perf_counter()
        ptrue1_content = (reconstructed_base + ptrue1_suffix).strip()
        ptrue1_prompt = _build_prompt_text(llm, messages, ptrue1_content)
        ptrue1_tokens = llm.tokenizer(ptrue1_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_ptrue1.append(ptrue1_prompt)

        with torch.no_grad():
            ptrue1_output = llm.model.forward(input_ids=ptrue1_tokens)
        bs_indirect1_forward += time.perf_counter() - _t_fwd_start

        p1_pos = ptrue1_output.logits[0, -1, positive_true_ids].sum()
        p1_neg = ptrue1_output.logits[0, -1, negative_false_ids].sum()
        p1_sm = torch.softmax(torch.stack([p1_pos.double(), p1_neg.double()]), dim=0)
        all_ptrue1.append({"True": p1_sm[0].item(), "False": p1_sm[1].item()})
        bs_indirect1_total += time.perf_counter() - _t_ptrue1_start

        # # --- indirect_ptrue2 (also used for ptrue3) ---
        # ptrue2_content = (reconstructed_base + ptrue2_suffix).strip()
        # ptrue2_prompt = _build_prompt_text(llm, messages, ptrue2_content)
        # ptrue2_tokens = llm.tokenizer(ptrue2_prompt, return_tensors="pt")["input_ids"].to(device)

        # if debug_conf:
        #     dbg_prompts_ptrue2.append(ptrue2_prompt)

        # with torch.no_grad():
        #     ptrue2_output = llm.model.forward(input_ids=ptrue2_tokens)

        # p2_pos = ptrue2_output.logits[0, -1, positive_yes_ids].sum()
        # p2_neg = ptrue2_output.logits[0, -1, negative_no_ids].sum()
        # p2_sm = torch.softmax(torch.stack([p2_pos.double(), p2_neg.double()]), dim=0)
        # all_ptrue2.append({"Yes": p2_sm[0].item(), "No": p2_sm[1].item()})

        # p3_pos = ptrue2_output.logits[0, -1, yes_only_id[0]]
        # p3_neg = ptrue2_output.logits[0, -1, no_only_id[0]]
        # p3_sm = torch.softmax(torch.stack([p3_pos.double(), p3_neg.double()]), dim=0)
        # all_ptrue3.append({"Yes": p3_sm[0].item(), "No": p3_sm[1].item()})

        # --- verbconf ---
        _t_verbconf_start = time.perf_counter()
        _t_fwd_start = time.perf_counter()
        verbconf_content = (reconstructed_base + verbconf_suffix).strip()
        verbconf_prompt = _build_prompt_text(llm, messages, verbconf_content)
        verbconf_tokens = llm.tokenizer(verbconf_prompt, return_tensors="pt")["input_ids"].to(device)

        if debug_conf:
            dbg_prompts_verbconf.append(verbconf_prompt)

        with torch.no_grad():
            verbconf_output = llm.model.forward(input_ids=verbconf_tokens)
        bs_verbconf_forward += time.perf_counter() - _t_fwd_start

        _t_joint_start = time.perf_counter()
        vc_probs = _compute_verbconf_joint_probs(llm, verbconf_output, token_seqs, device)
        bs_verbconf_joint_probs += time.perf_counter() - _t_joint_start
        vc_probs_1d = vc_probs[0] if vc_probs.dim() > 1 else vc_probs
        sample_verbconf = (score_values * vc_probs_1d).sum().item()
        all_verbconf.append(sample_verbconf)
        all_verbconf_dist.append(vc_probs_1d.tolist())
        top_idx = vc_probs_1d.argmax().item()
        all_verbconf_top_scores.append(top_idx)
        all_verbconf_top_probs.append(vc_probs_1d[top_idx].item())
        bs_verbconf_total += time.perf_counter() - _t_verbconf_start

        torch.cuda.empty_cache()

    if debug_conf:
        debug_info["bootstrap_prompts"] = {
            "answer_logits": dbg_prompts_answer,
            "indirect_ptrue1": dbg_prompts_ptrue1,
            # "indirect_ptrue2": dbg_prompts_ptrue2,  # Commented out - not computed
            "verbconf": dbg_prompts_verbconf,
        }
        debug_info["bootstrap_samples"] = [
            {
                "sample_idx": i,
                "kept_step_indices": kept_indices_per_sample[i],
                "step_counts": step_counts[i],
                "kept_steps": [parsed_output.cot_steps[j] for j in kept_indices_per_sample[i] if j < len(parsed_output.cot_steps)],
            }
            for i in range(effective_nb_samples)
        ]

    scores = ConfidenceScores(
        answer_probabilities=all_answer_probs,
        answer_entropy=all_answer_entropy,
        indirect_ptrue1_probabilities=all_ptrue1,
        indirect_ptrue2_probabilities=[],  # Not computed
        indirect_ptrue3_probabilities=[],  # Not computed
        verbconf_probabilities=all_verbconf,
        verbconf_distribution=all_verbconf_dist,
        verbconf_top_score=all_verbconf_top_scores,
        verbconf_top_prob=all_verbconf_top_probs,
        step_masks=step_counts,
    )

    bs_timings = {
        "meanprob_total_seconds": bs_meanprob_total,
        "meanprob_forward_seconds": bs_meanprob_forward,
        "indirect1_total_seconds": bs_indirect1_total,
        "indirect1_forward_seconds": bs_indirect1_forward,
        "verbconf_total_seconds": bs_verbconf_total,
        "verbconf_forward_seconds": bs_verbconf_forward,
        "verbconf_joint_probs_seconds": bs_verbconf_joint_probs,
    }

    return scores, step_counts, bs_timings


def _fake_bootstrap_forward(
    llm: LLM,
    messages: list[dict],
    cot_only_text: str,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int,
    assistant_prefill: str,
    debug_conf: bool,
    debug_info: dict,
    normalized_content: str,
    gen_cache=None,
    base_tokens=None,
    precomputed_early_cache=None,
) -> ConfidenceScores:
    """Fake bootstrap: bootstrap sampling with attention mask weighting.

    Samples (n-1) steps with replacement, always keeps final step. Uses
    attention mask trick with weights (0 for unsampled, k for steps sampled k times).
    """
    step_counts = _fake_bootstrap_sample_steps(parsed_output.cot_steps, nb_dropout_samples)
    effective_nb_samples = len(step_counts)

    _, _, fb_answer_probs, fb_answer_entropy, dbg, _, fb_masked_steps_dbg = \
        dropout_answerlogits(llm, messages, cot_only_text, parsed_output, effective_nb_samples, False, assistant_prefill, debug_conf=debug_conf,
                             gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                             precomputed_early_cache=precomputed_early_cache,
                             jk_kept_indices=None, jk_step_masks=step_counts)
    if dbg:
        debug_info["fake_bootstrap_answer_logits"] = dbg
    if fb_masked_steps_dbg:
        debug_info["fake_bootstrap_masked_steps"] = fb_masked_steps_dbg

    _, _, _, fb_ptrue1, fb_ptrue2, fb_ptrue3, dbg, _ = \
        dropout_indirectlogits(llm, messages, cot_only_text, parsed_output, effective_nb_samples, False, assistant_prefill, debug_conf=debug_conf,
                               gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                               precomputed_early_cache=precomputed_early_cache,
                               jk_kept_indices=None, jk_step_masks=step_counts)
    if dbg:
        debug_info["fake_bootstrap_indirect_logits"] = dbg

    (_, fb_verbconf,
     _, _, _,
     fb_verbconf_dist, fb_verbconf_top_scores, fb_verbconf_top_probs,
     dbg, _) = \
        dropout_verbalconf(llm, messages, cot_only_text, parsed_output, effective_nb_samples, False, assistant_prefill, debug_conf=debug_conf,
                           gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                           precomputed_early_cache=precomputed_early_cache,
                           consume_early_cache=True,
                           jk_kept_indices=None, jk_step_masks=step_counts)
    if dbg:
        debug_info["fake_bootstrap_verbconf"] = dbg

    return ConfidenceScores(
        answer_probabilities=fb_answer_probs,
        answer_entropy=fb_answer_entropy,
        indirect_ptrue1_probabilities=fb_ptrue1,
        indirect_ptrue2_probabilities=fb_ptrue2,
        indirect_ptrue3_probabilities=fb_ptrue3,
        verbconf_probabilities=fb_verbconf,
        verbconf_distribution=fb_verbconf_dist,
        verbconf_top_score=fb_verbconf_top_scores,
        verbconf_top_prob=fb_verbconf_top_probs,
        step_masks=step_counts,
    )


def _fake_bootstrap_hook_forward(
    llm: LLM,
    messages: list[dict],
    cot_only_text: str,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int,
    assistant_prefill: str,
    debug_conf: bool,
    debug_info: dict,
    normalized_content: str,
    gen_cache=None,
    base_tokens=None,
    precomputed_early_cache=None,
) -> ConfidenceScores:
    """Fake bootstrap with forward hooks: directly scale attention weights.

    Uses forward hooks to multiply attention weights by k for steps sampled k times.
    """
    step_counts = _fake_bootstrap_sample_steps(parsed_output.cot_steps, nb_dropout_samples)
    effective_nb_samples = len(step_counts)

    _, _, fbh_answer_probs, fbh_answer_entropy, dbg, _, fbh_masked_steps_dbg = \
        dropout_answerlogits(llm, messages, cot_only_text, parsed_output, effective_nb_samples, False, assistant_prefill, debug_conf=debug_conf,
                             gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                             precomputed_early_cache=precomputed_early_cache,
                             jk_kept_indices=None, jk_step_masks=step_counts, use_hooks=True)
    if dbg:
        debug_info["fake_bootstrap_hook_answer_logits"] = dbg
    if fbh_masked_steps_dbg:
        debug_info["fake_bootstrap_hook_masked_steps"] = fbh_masked_steps_dbg

    _, _, _, fbh_ptrue1, fbh_ptrue2, fbh_ptrue3, dbg, _ = \
        dropout_indirectlogits(llm, messages, cot_only_text, parsed_output, effective_nb_samples, False, assistant_prefill, debug_conf=debug_conf,
                               gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                               precomputed_early_cache=precomputed_early_cache,
                               jk_kept_indices=None, jk_step_masks=step_counts, use_hooks=True)
    if dbg:
        debug_info["fake_bootstrap_hook_indirect_logits"] = dbg

    (_, fbh_verbconf,
     _, _, _,
     fbh_verbconf_dist, fbh_verbconf_top_scores, fbh_verbconf_top_probs,
     dbg, _) = \
        dropout_verbalconf(llm, messages, cot_only_text, parsed_output, effective_nb_samples, False, assistant_prefill, debug_conf=debug_conf,
                           gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=True,
                           precomputed_early_cache=precomputed_early_cache,
                           consume_early_cache=True,
                           jk_kept_indices=None, jk_step_masks=step_counts, use_hooks=True)
    if dbg:
        debug_info["fake_bootstrap_hook_verbconf"] = dbg

    return ConfidenceScores(
        answer_probabilities=fbh_answer_probs,
        answer_entropy=fbh_answer_entropy,
        indirect_ptrue1_probabilities=fbh_ptrue1,
        indirect_ptrue2_probabilities=fbh_ptrue2,
        indirect_ptrue3_probabilities=fbh_ptrue3,
        verbconf_probabilities=fbh_verbconf,
        verbconf_distribution=fbh_verbconf_dist,
        verbconf_top_score=fbh_verbconf_top_scores,
        verbconf_top_prob=fbh_verbconf_top_probs,
        step_masks=step_counts,
    )


def dropout_answerlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False, gen_cache=None, base_tokens=None, use_jackknife=False, precomputed_early_cache=None, consume_early_cache=False, jk_kept_indices=None, jk_step_masks=None, vanilla_only=False, recrop_early_cache_len=None, preserve_final_step=False, use_hooks=False):
    """Logit-based confidence on the answer tokens themselves."""
    _t_start = time.perf_counter()

    suffix_text = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
    # logger.info("dropout_answerlogits: generated_text='%s', suffix='%s'", generated_text[:100], suffix_text)

    late_tokens, vanilla_out, dropout_out, ans_start, ans_end, dropout_step_masks, masked_steps_debug, forward_timings = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=suffix_text, nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        consume_early_cache=consume_early_cache,
                        debug_conf=debug_conf,
                        jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks,
                        vanilla_only=vanilla_only,
                        recrop_early_cache_len=recrop_early_cache_len,
                        preserve_final_step=preserve_final_step,
                        use_hooks=use_hooks)

    answer_tokens = late_tokens[0, ans_start:ans_end]
    nb_answer_tokens = len(answer_tokens)
    token_strings = [llm.tokenizer.decode([answer_tokens[t].item()]) for t in range(nb_answer_tokens)]

    # --- Vanilla ---
    # Logit at position t-1 predicts token at position t
    v_logits = vanilla_out.logits[0, ans_start - 1:ans_end - 1, :]
    v_probs = v_logits.double().softmax(dim=-1)

    vanilla_answer_probs = [{token_strings[t]: v_probs[t, answer_tokens[t]].item()} for t in range(nb_answer_tokens)]

    v_logprobs = torch.nan_to_num(v_probs.log(), neginf=-99)
    entropy_values = (-(v_probs * v_logprobs).sum(dim=-1)).cpu().numpy().tolist()
    vanilla_answer_entropy = [{token_strings[t]: entropy_values[t]} for t in range(nb_answer_tokens)]

    # --- Dropout ---
    if dropout_out is not None:
        d_probs = dropout_out.logits[:, ans_start - 1:ans_end - 1, :].double().softmax(dim=-1)

        dropout_answer_probs = [[{token_strings[t]: d_probs[s, t, answer_tokens[t]].item()}
                                  for t in range(nb_answer_tokens)]
                                 for s in range(d_probs.shape[0])]

        d_logprobs = torch.nan_to_num(d_probs.log(), neginf=-99)
        d_entropy_values = (-(d_probs * d_logprobs).sum(dim=-1)).cpu().numpy().tolist()
        dropout_answer_entropy = [[{token_strings[t]: d_entropy_values[s][t]} for t in range(nb_answer_tokens)]
                                   for s in range(len(d_entropy_values))]
    else:
        dropout_answer_probs = []
        dropout_answer_entropy = []

    # --- Debug ---
    dbg = _debug_answer_logit_tokens(llm, late_tokens, ans_start, ans_end, v_probs, vanilla_answer_probs) if debug_conf else None

    # --- Timing ---
    total_time = time.perf_counter() - _t_start
    metric_timings = {
        "total_seconds": total_time,
        "forward_seconds": forward_timings["vanilla_forward_seconds"] + forward_timings["dropout_forward_seconds"],
    }

    return vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy, dbg, dropout_step_masks, masked_steps_debug, metric_timings


def dropout_indirectlogits(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False, gen_cache=None, base_tokens=None, use_jackknife=False, precomputed_early_cache=None, consume_early_cache=False, jk_kept_indices=None, jk_step_masks=None, vanilla_only=False, recrop_early_cache_len=None, preserve_final_step=False, use_hooks=False):
    """P(True) and P(Yes) probing after the answer."""
    _t_start = time.perf_counter()

    # logger.info("dropout_indirectlogits: generated_text='%s'", generated_text[:100])

    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
    negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])

    # ptrue1: "True/False:"
    ptrue1_suffix = f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}.\nTrue/False:"
    # logger.info("dropout_indirectlogits ptrue1: suffix='%s'", ptrue1_suffix)
    _, vanilla1, dropout1, _, _, dropout_step_masks, _, forward_timings1 = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=ptrue1_suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks,
                        vanilla_only=vanilla_only,
                        consume_early_cache=vanilla_only,
                        recrop_early_cache_len=recrop_early_cache_len,
                        preserve_final_step=preserve_final_step,
                        use_hooks=use_hooks)

    # ptrue2: "Is the answer \boxed{<X>}?"
    ptrue2_suffix = f"\nIs the answer \\boxed{{{parsed_output.final_answer}}}?"
    # logger.info("dropout_indirectlogits ptrue2: suffix='%s'", ptrue2_suffix)
    _, vanilla2, dropout2, _, _, _, _, forward_timings2 = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=ptrue2_suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        consume_early_cache=consume_early_cache or vanilla_only,
                        jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks,
                        vanilla_only=vanilla_only,
                        recrop_early_cache_len=recrop_early_cache_len,
                        preserve_final_step=preserve_final_step,
                        use_hooks=use_hooks)

    # Vanilla ptrue1
    v_pos1 = vanilla1.logits[0, -1, positive_true_ids].sum()
    v_neg1 = vanilla1.logits[0, -1, negative_false_ids].sum()
    v_softmax1 = torch.softmax(torch.stack([v_pos1.double(), v_neg1.double()]), dim=0)
    vanilla_ptrue1 = {"True": v_softmax1[0].item(), "False": v_softmax1[1].item()}

    # Vanilla ptrue2
    v_pos2 = vanilla2.logits[0, -1, positive_yes_ids].sum()
    v_neg2 = vanilla2.logits[0, -1, negative_no_ids].sum()
    v_softmax2 = torch.softmax(torch.stack([v_pos2.double(), v_neg2.double()]), dim=0)
    vanilla_ptrue2 = {"Yes": v_softmax2[0].item(), "No": v_softmax2[1].item()}

    # Vanilla ptrue3 (Yes/No only — no synonyms)
    yes_only_id = llm.tokenizer.encode(' Yes', add_special_tokens=False)
    no_only_id = llm.tokenizer.encode(' No', add_special_tokens=False)
    v_pos3 = vanilla2.logits[0, -1, yes_only_id[0]]
    v_neg3 = vanilla2.logits[0, -1, no_only_id[0]]
    v_softmax3 = torch.softmax(torch.stack([v_pos3.double(), v_neg3.double()]), dim=0)
    vanilla_ptrue3 = {"Yes": v_softmax3[0].item(), "No": v_softmax3[1].item()}

    # Dropout
    if dropout1 is not None:
        d_pos1 = dropout1.logits[:, -1, positive_true_ids].sum(-1)
        d_neg1 = dropout1.logits[:, -1, negative_false_ids].sum(-1)
        d_softmax1 = torch.softmax(torch.stack([d_pos1.double(), d_neg1.double()], dim=-1), dim=-1)
        dropout_ptrue1 = [{"True": d_softmax1[s, 0].item(), "False": d_softmax1[s, 1].item()}
                           for s in range(d_softmax1.shape[0])]

        d_pos2 = dropout2.logits[:, -1, positive_yes_ids].sum(-1)
        d_neg2 = dropout2.logits[:, -1, negative_no_ids].sum(-1)
        d_softmax2 = torch.softmax(torch.stack([d_pos2.double(), d_neg2.double()], dim=-1), dim=-1)
        dropout_ptrue2 = [{"Yes": d_softmax2[s, 0].item(), "No": d_softmax2[s, 1].item()}
                           for s in range(d_softmax2.shape[0])]

        # Dropout ptrue3 (Yes/No only — no synonyms)
        d_pos3 = dropout2.logits[:, -1, yes_only_id[0]]
        d_neg3 = dropout2.logits[:, -1, no_only_id[0]]
        d_softmax3 = torch.softmax(torch.stack([d_pos3.double(), d_neg3.double()], dim=-1), dim=-1)
        dropout_ptrue3 = [{"Yes": d_softmax3[s, 0].item(), "No": d_softmax3[s, 1].item()}
                           for s in range(d_softmax3.shape[0])]
    else:
        dropout_ptrue1 = []
        dropout_ptrue2 = []
        dropout_ptrue3 = []

    # --- Debug ---
    if debug_conf:
        dbg = {
            "ptrue1_true_false": _debug_indirect_logit_tokens(llm, vanilla1, positive_true_ids, negative_false_ids, "True/False"),
            "ptrue2_yes_no": _debug_indirect_logit_tokens(llm, vanilla2, positive_yes_ids, negative_no_ids, "Yes/No"),
            "ptrue3_yes_no_only": _debug_indirect_logit_tokens(llm, vanilla2, [yes_only_id[0]], [no_only_id[0]], "Yes/No (no synonyms)"),
        }
    else:
        dbg = None

    # --- Timing ---
    total_time = time.perf_counter() - _t_start
    forward_time = (forward_timings1["vanilla_forward_seconds"] +
                    forward_timings1["dropout_forward_seconds"] +
                    forward_timings2["vanilla_forward_seconds"] +
                    forward_timings2["dropout_forward_seconds"])

    metric_timings = {
        "total_seconds": total_time,
        "forward_seconds": forward_time,
    }

    return [vanilla_ptrue1], [vanilla_ptrue2], [vanilla_ptrue3], dropout_ptrue1, dropout_ptrue2, dropout_ptrue3, dbg, dropout_step_masks, metric_timings


def _slice_cache_batch(cache, start, end):
    """Return a deepcopied cache with the batch dimension sliced to [start:end].

    Slices first (cheap view), then deepcopies only the sliced tensors so peak
    memory scales with the chunk size rather than the full batch.
    """
    # Build a shallow shell sharing structure, then deepcopy after slicing.
    if hasattr(cache, 'key_cache'):
        # Save originals, swap in sliced views, deepcopy, then restore.
        orig_k = cache.key_cache
        orig_v = cache.value_cache
        sliced_k = [
            (k[start:end] if (k is not None and k.dim() == 4) else k)
            for k in orig_k
        ]
        sliced_v = [
            (v[start:end] if (v is not None and v.dim() == 4) else v)
            for v in orig_v
        ]
        cache.key_cache = sliced_k
        cache.value_cache = sliced_v
        try:
            sliced = copy.deepcopy(cache)
        finally:
            cache.key_cache = orig_k
            cache.value_cache = orig_v
        return sliced
    return copy.deepcopy(cache)


def _compute_verbconf_joint_probs(llm, model_output, token_seqs, device):
    """Compute joint probabilities for multi-token number sequences.

    For a number like 42 tokenized as [tok_4, tok_2], computes:
        P(42) = P(tok_4) * P(tok_2 | tok_4)
    Each prefix gets its own cloned KV cache so conditioning is exact.

    Processes the batch dimension in chunks of 10 to reduce peak memory.

    Args:
        model_output: forward-pass output with .logits and .past_key_values
        token_seqs: list of token-id lists, one per number (0-100)
        device: torch device

    Returns:
        probs: [batch, len(token_seqs)] normalized joint probabilities
    """

    batch_size = model_output.logits.shape[0]
    joint_logprobs = torch.zeros(batch_size, len(token_seqs), dtype=torch.float64)

    # --- depth 0: first-token log-probs from existing logits ---
    logprobs_0 = model_output.logits[:, -1, :].double().log_softmax(-1).cpu()
    for i, seq in enumerate(token_seqs):
        joint_logprobs[:, i] = logprobs_0[:, seq[0]]

    max_depth = max(len(seq) for seq in token_seqs)
    if max_depth <= 1:
        return joint_logprobs.softmax(-1)

    # --- depth >= 1: chunk the batch dimension to cap peak memory ---
    CHUNK_SIZE = 10
    num_chunks = (batch_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, batch_size)
        chunk_bs = end - start

        logger.info(
            "Processing verbconf chunk %d/%d (samples %d:%d)",
            chunk_idx + 1, num_chunks, start, end,
        )

        # Root cache for this chunk = slice of model_output.past_key_values
        _t0 = time.perf_counter()
        chunk_root_cache = _slice_cache_batch(model_output.past_key_values, start, end)
        logger.info(
            "slice+deepcopy(past_key_values) for verbconf chunk %d took %.4fs",
            chunk_idx, time.perf_counter() - _t0,
        )

        prefix_cache = {(): chunk_root_cache}

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

                tok_input = torch.full((chunk_bs, 1), feed_token, device=device)

                with torch.no_grad():
                    out = llm.model.forward(input_ids=tok_input, past_key_values=kv)

                logprobs_d = out.logits[:, -1, :].double().log_softmax(-1).cpu()

                for seq_idx, next_tok in entries:
                    joint_logprobs[start:end, seq_idx] += logprobs_d[:, next_tok]

                new_prefix_cache[prefix] = out.past_key_values

            prefix_cache = new_prefix_cache

        del prefix_cache
        torch.cuda.empty_cache()

    return joint_logprobs.softmax(-1)


def dropout_verbalconf(llm, messages, generated_text, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill="", debug_conf=False, gen_cache=None, base_tokens=None, use_jackknife=False, precomputed_early_cache=None, consume_early_cache=False, jk_kept_indices=None, jk_step_masks=None, vanilla_only=False, recrop_early_cache_len=None, preserve_final_step=False, use_hooks=False):
    """Verbalized confidence (0-100 score)."""
    _t_start = time.perf_counter()

    suffix = (
        f"\nThe answer is \\boxed{{{parsed_output.final_answer}}}."
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )
    # logger.info("dropout_verbalconf: generated_text='%s', suffix='%s'", generated_text[:100], suffix)

    _, vanilla_out, dropout_out, _, _, dropout_step_masks, _, forward_timings = \
        dropout_forward(llm, messages, generated_text, parsed_output,
                        suffix_text=suffix,
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill,
                        gen_cache=gen_cache, base_tokens=base_tokens, use_jackknife=use_jackknife,
                        precomputed_early_cache=precomputed_early_cache,
                        consume_early_cache=consume_early_cache or vanilla_only,
                        jk_kept_indices=jk_kept_indices, jk_step_masks=jk_step_masks,
                        vanilla_only=vanilla_only,
                        recrop_early_cache_len=None,
                        preserve_final_step=preserve_final_step,
                        use_hooks=use_hooks)

    verbconf_strings = ANSWER_TOKENS['VERBCONF']
    token_seqs = [llm.tokenizer.encode(s, add_special_tokens=False) for s in verbconf_strings]
    score_values = torch.tensor([int(s) / 100 for s in verbconf_strings], dtype=torch.float64)
    device = next(llm.model.parameters()).device

    # Vanilla
    _t_joint_start = time.perf_counter()
    v_probs = _compute_verbconf_joint_probs(llm, vanilla_out, token_seqs, device)
    joint_probs_time = time.perf_counter() - _t_joint_start

    v_probs_1d = v_probs[0] if v_probs.dim() > 1 else v_probs
    vanilla_verbconf = (score_values * v_probs_1d).sum().item()
    vanilla_distribution = v_probs_1d.tolist()
    vanilla_top_idx = v_probs_1d.argmax().item()
    vanilla_top_score = vanilla_top_idx          # score 0-100
    vanilla_top_prob = v_probs_1d[vanilla_top_idx].item()

    # Dropout
    if dropout_out is not None:
        _t_joint_dropout_start = time.perf_counter()
        d_probs = _compute_verbconf_joint_probs(llm, dropout_out, token_seqs, device)
        joint_probs_time += time.perf_counter() - _t_joint_dropout_start

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

    # --- Timing ---
    total_time = time.perf_counter() - _t_start
    forward_time = forward_timings["vanilla_forward_seconds"] + forward_timings["dropout_forward_seconds"]

    metric_timings = {
        "total_seconds": total_time,
        "forward_seconds": forward_time,
        "joint_probs_seconds": joint_probs_time,
    }

    return (
        [vanilla_verbconf], dropout_verbconf,
        vanilla_distribution, vanilla_top_score, vanilla_top_prob,
        dropout_distribution, dropout_top_scores, dropout_top_probs,
        dbg,
        dropout_step_masks,
        metric_timings,
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


def _build_prompt_text(llm, messages, full_assistant_content):
    """Build the full prompt text for confidence scoring (no tokenization).

    Applies chat template and model-specific post-processing, matching
    the approach in llm.generate_one() for consistency.
    """
    _t0 = time.perf_counter()
    conf_messages = copy.deepcopy(messages)
    logger.info("deepcopy(messages) in _build_prompt_text took %.4fs", time.perf_counter() - _t0)

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

    # For Qwen models with <think> tags, handle separately
    # Qwen's template auto-injects <think></think> which breaks continue_final_message.
    # Instead: render without assistant message, strip empty think block, append content directly.
    if "qwen" in llm.model_name.lower() and ("<think>" in full_assistant_content or "</think>" in full_assistant_content):
        base_messages = [m for m in conf_messages if m["role"] != "assistant"]
        prompt_text = llm.tokenizer.apply_chat_template(
            base_messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)
        prompt_text += full_assistant_content
        return prompt_text


    # Apply chat template to everything up to </think>
    prompt_text = llm.tokenizer.apply_chat_template(
        conf_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)

    return prompt_text


def _tokenize_for_confidence(llm, messages, full_assistant_content):
    """Tokenize a conversation with the full assistant response."""
    prompt_text = _build_prompt_text(llm, messages, full_assistant_content)
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
    debug_conf: bool = False,
    jk_kept_indices=None,
    jk_step_masks=None,
    vanilla_only: bool = False,
    recrop_early_cache_len: int | None = None,
    preserve_final_step: bool = False,
    use_hooks: bool = False,
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
        # logger.info("dropout_forward: full_content (with suffix) = '%s'", full_content)
    else:
        tokens = base_tokens
        full_content = (assistant_prefill + generated_text).strip()
        # logger.info("dropout_forward: full_content (no suffix) = '%s'", full_content)

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
            empty_cache = (
                Qwen3_5DynamicCache(config=llm.model.config)
                if "qwen" in llm.model_name.lower()
                else DynamicCache(config=llm.model.config)
            )
            early_output = llm.model(input_ids=early_tokens, past_key_values=empty_cache)
        early_cache = early_output.past_key_values

    # -- Vanilla forward (fresh full-sequence pass, no KV-cache reuse) ---------
    # The cached scheme below relied on `base_tokens == tokens[:early_late_split]`,
    # which can fail when BPE merges across the CoT/suffix boundary and yields
    # logits that don't match a full-sequence forward. The fresh full forward
    # matches the _bootstrap_forward / _jackknife_true_forward / _coinflip_true_forward
    # paths and removes that inconsistency. Logits are sliced to the late region
    # so downstream indexing with `ans_start`/`ans_end` (late-relative) is
    # unchanged.
    forward_timings = {
        "vanilla_forward_seconds": 0.0,
        "dropout_forward_seconds": 0.0,
    }

    _t_vanilla_start = time.perf_counter()
    with torch.no_grad():
        _full_vanilla_output = llm.model.forward(input_ids=tokens.to(device))
    forward_timings["vanilla_forward_seconds"] = time.perf_counter() - _t_vanilla_start

    class _VanillaForwardWrapper:
        __slots__ = ("logits", "past_key_values")

        def __init__(self, logits, past_key_values):
            self.logits = logits
            self.past_key_values = past_key_values

    vanilla_output = _VanillaForwardWrapper(
        logits=_full_vanilla_output.logits[:, early_late_split:, :],
        past_key_values=_full_vanilla_output.past_key_values,
    )

    # --- [OLD CACHED VANILLA SCHEME — commented out] ---------------------------
    # Why: this scheme built an `early_cache` from `base_tokens` (tokenization of
    # system+user+CoT only) and forwarded just `late_tokens` (the suffix) on top
    # of it, assuming `base_tokens == tokens[:early_late_split]` where `tokens`
    # is the tokenization of system+user+CoT+suffix. That assumption breaks
    # whenever BPE merges across the CoT/suffix boundary differ between the two
    # tokenizations, which silently produces logits that don't match a
    # full-sequence forward. Empirically this caused vanilla P(answer) and
    # ptrue2/3 to diverge dramatically from bootstrap-with-full-CoT on Qwen
    # (e.g. P(C) = 0.16 vs 0.98, ptrue2 Yes/No flipped). The fresh full forward
    # above removes the assumption entirely.
    # if suffix_text and gen_cache is not None and not vanilla_only:
    #     overlap_len = min(find_token_overlap(base_tokens[0], tokens[0]), gen_cache_len)
    #     nb_discarded = gen_cache_len - overlap_len
    #     logger.info(
    #         "KV cache reuse: overlap=%d, discarded=%d (of %d)",
    #         overlap_len, nb_discarded, gen_cache_len,
    #     )
    #
    #     if overlap_len > early_late_split:
    #         # Reuse more of the cache via overlap — forward only the tail
    #         _t0 = time.perf_counter()
    #         vanilla_cache = copy.deepcopy(gen_cache)
    #         logger.info("deepcopy(gen_cache) for vanilla_cache (suffix+overlap) took %.4fs", time.perf_counter() - _t0)
    #         crop_cache(vanilla_cache, overlap_len)
    #         remaining_tokens = tokens[:, overlap_len:].to(device)
    #         logger.info(
    #             "KV cache overlap boundary: cache_tail='%s' | remaining_head='%s'",
    #             llm.tokenizer.decode(tokens[0, overlap_len-3:overlap_len].tolist()),
    #             llm.tokenizer.decode(remaining_tokens[0, :3].tolist()),
    #         )
    #
    #         with torch.no_grad():
    #             vanilla_output = llm.model.forward(
    #                 input_ids=remaining_tokens,
    #                 past_key_values=vanilla_cache,
    #             )
    #     else:
    #         # Pathological case: overlap doesn't extend past split
    #         with torch.no_grad():
    #             if _owns_early_cache and not will_run_dropout:
    #                 logger.info("consuming early_cache directly for vanilla (suffix, no overlap)")
    #                 vanilla_output = llm.model.forward(
    #                     input_ids=late_tokens,
    #                     past_key_values=early_cache,
    #                 )
    #             else:
    #                 _t0 = time.perf_counter()
    #                 _early_cache_copy = copy.deepcopy(early_cache)
    #                 logger.info("deepcopy(early_cache) for vanilla (suffix, no overlap) took %.4fs", time.perf_counter() - _t0)
    #                 vanilla_output = llm.model.forward(
    #                     input_ids=late_tokens,
    #                     past_key_values=_early_cache_copy,
    #                 )
    # else:
    #     if gen_cache is not None:
    #         nb_discarded = gen_cache_len - early_late_split
    #         logger.info(
    #             "KV cache reuse (suffix=%s): reused=%d, discarded=%d (of %d)",
    #             bool(suffix_text), early_late_split, nb_discarded, gen_cache_len,
    #         )
    #         logger.info(
    #             "KV cache overlap boundary (suffix=%s): cache_tail='%s' | remaining_head='%s'",
    #             bool(suffix_text),
    #             llm.tokenizer.decode(tokens[0, early_late_split-3:early_late_split].tolist()),
    #             llm.tokenizer.decode(late_tokens[0, :3].tolist()),
    #         )
    #     with torch.no_grad():
    #         if _owns_early_cache and not will_run_dropout:
    #             logger.info("consuming early_cache directly for vanilla (suffix=%s)", bool(suffix_text))
    #             vanilla_output = llm.model.forward(
    #                 input_ids=late_tokens,
    #                 past_key_values=early_cache,
    #             )
    #         else:
    #             _t0 = time.perf_counter()
    #             _early_cache_copy = copy.deepcopy(early_cache)
    #             logger.info("deepcopy(early_cache) for vanilla (suffix=%s) took %.4fs", bool(suffix_text), time.perf_counter() - _t0)
    #             vanilla_output = llm.model.forward(
    #                 input_ids=late_tokens,
    #                 past_key_values=_early_cache_copy,
    #             )

    # -- Dropout forward --------------------------------------------------------
    dropout_output = None
    dropout_step_masks = None
    masked_steps_debug = None
    if will_run_dropout:
        if _owns_early_cache:
            logger.info("consuming early_cache directly for dropout_late_forward")
            _dropout_cache = early_cache
        else:
            _t0 = time.perf_counter()
            _dropout_cache = copy.deepcopy(early_cache)
            logger.info("deepcopy(early_cache) for dropout_late_forward took %.4fs", time.perf_counter() - _t0)

        _t_dropout_start = time.perf_counter()
        dropout_output, dropout_step_masks, masked_steps_debug = dropout_late_forward(
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
            debug_conf=debug_conf,
            jk_kept_indices=jk_kept_indices,
            jk_step_masks=jk_step_masks,
            preserve_final_step=preserve_final_step,
            use_hooks=use_hooks,
        )
        forward_timings["dropout_forward_seconds"] = time.perf_counter() - _t_dropout_start

    # In vanilla_only mode the early_cache was mutated in place by the vanilla
    # forward (extended with late_tokens). Crop it back so the next caller can
    # reuse it as a fresh early_cache. Skip on the final caller (recrop_len=None).
    if vanilla_only and recrop_early_cache_len is not None and precomputed_early_cache is not None:
        crop_cache(precomputed_early_cache, recrop_early_cache_len)

    return late_tokens, vanilla_output, dropout_output, answer_start_late, answer_end_late, dropout_step_masks, masked_steps_debug, forward_timings


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
    debug_conf=False,
    jk_kept_indices=None,
    jk_step_masks=None,
    preserve_final_step=False,
    use_hooks=False,
):
    """Batched forward with per-sample dropout attention masks or hooks.

    For each sample, a random subset of reasoning steps is masked so that
    the answer-region tokens cannot attend to those steps.

    When use_hooks=True, uses forward hooks to directly scale attention weights
    instead of attention masks.
    """
    steps, _ = _dropout_eligible_steps(cot_steps)
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
        # Jackknife: reuse pre-sampled step masks if provided, otherwise sample once
        if jk_step_masks is not None:
            step_weights = np.array(jk_step_masks)
            expected_shape = (nb_dropout_samples, len(steps))
            if step_weights.shape != expected_shape:
                raise ValueError(
                    "Jackknife step mask shape mismatch: "
                    f"expected {expected_shape}, got {step_weights.shape}"
                )
            # Check if weights are binary (0/1) or counts (0/1/2/...)
            is_binary = np.all((step_weights == 0) | (step_weights == 1))
            if is_binary:
                is_step_selected = step_weights.astype(bool)
            else:
                # Fake bootstrap: step_weights are counts (0, 1, 2, ...)
                is_step_selected = step_weights > 0
        else:
            # Sample jackknife masks
            k = len(steps)
            nb_keep = _jackknife_nb_keep(k)
            nb_mask = k - nb_keep
            is_step_selected = np.ones((nb_dropout_samples, k), dtype=bool)
            step_weights = np.ones((nb_dropout_samples, k), dtype=int)
            for i in range(nb_dropout_samples):
                masked_indices = np.random.choice(k, size=nb_mask, replace=False)
                is_step_selected[i, masked_indices] = False
                step_weights[i, masked_indices] = 0
    else:
        # Coin-flip: each step independently kept with probability = threshold
        is_step_selected = (np.random.random((nb_dropout_samples, len(steps))) <= threshold)
        if preserve_final_step and len(steps) > 0:
            is_step_selected[:, -1] = True
        step_weights = is_step_selected.astype(int)

    # Walk backwards through steps, finding each one's token range and masking.
    # Positions in early_token_ids directly correspond to early cache columns.
    remaining_ids = early_token_ids.clone()

    
    for step_id, step in reversed(list(enumerate(steps))):
        try:

            curstep = step
            curstr = llm.tokenizer.decode(remaining_ids.tolist())


            if curstep not in curstr:
                print(curstep in curstr)
                # breakpoint()
                logger.info("Step extraction UNSUCCESSFUL")

            step_start, step_end = find_token_indices_from_end(
                llm.tokenizer, remaining_ids, step, llm.model_name)
        except ConfidenceAlignmentError:
            raise
        except ValueError as exc:
            raise ConfidenceAlignmentError(
                "Unable to align dropout mask to token span "
                f"(step_id={step_id}, step={step!r}, remaining_len={len(remaining_ids)})."
            ) from exc

        for i in range(nb_dropout_samples):
            weight = step_weights[i, step_id]
            if weight == 0:
                # Step not sampled: mask it out completely
                late_mask[i, 0,
                          modify_start_late - 1:modify_end_late - 1,
                          step_start:step_end] = -10000.
            else:
                # Step sampled weight times: scale attention by weight
                # Attention mask is additive bias before softmax: attn = softmax(QK^T / sqrt(d) + mask)
                # To multiply attention weights by k, we add log(k) to the mask
                # This makes exp(score + log(k)) = k * exp(score)
                # So after softmax, this step's contribution is scaled by k relative to others
                scale_bias = np.log(weight) if weight > 0 else 0.
                late_mask[i, 0,
                          modify_start_late - 1:modify_end_late - 1,
                          step_start:step_end] = scale_bias

        remaining_ids = remaining_ids[:step_start + 1]

    # Capture debug info before dtype conversion (comparisons need float32)
    masked_steps_debug = None
    if debug_conf:
        masked_steps_debug = _debug_masked_text(
            llm, early_token_ids, late_mask, modify_start_late,
            nb_early_tokens, is_step_selected, steps, nb_dropout_samples, use_jackknife)

    # Hook-based implementation: register custom attention function that scales post-softmax weights
    if use_hooks:
        # Save current attn implementation to restore later
        _orig_attn_impl = llm.model.config._attn_implementation

        # Build step token ranges
        step_token_ranges = []
        remaining_ids_temp = early_token_ids.clone()
        for step_id, step in reversed(list(enumerate(steps))):
            try:
                s_start, s_end = find_token_indices_from_end(
                    llm.tokenizer, remaining_ids_temp, step, llm.model_name)
                step_token_ranges.insert(0, (s_start, s_end))
                remaining_ids_temp = remaining_ids_temp[:s_start + 1]
            except (ConfidenceAlignmentError, ValueError):
                step_token_ranges.insert(0, None)

        all_logits = []
        all_input_ids = []

        # Build causal attention mask for late tokens attending to early (cache) + late
        nb_late = late_tokens.shape[1]
        hook_mask = torch.cat([
            torch.ones(nb_late, nb_early_tokens),
            torch.ones(nb_late, nb_late).tril(),
        ], dim=1)
        hook_mask[hook_mask == 0] = -10000.
        hook_mask[hook_mask == 1] = 0.
        hook_mask = hook_mask.unsqueeze(0).unsqueeze(0).to(device=device, dtype=next(llm.model.parameters()).dtype)

        for sample_idx in range(nb_dropout_samples):
            sample_weights = step_weights[sample_idx]

            def _make_patched_eager(weights, ranges, m_start, m_end):
                def patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
                    key_states = repeat_kv(key, module.num_key_value_groups)
                    value_states = repeat_kv(value, module.num_key_value_groups)

                    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask

                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
                    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

                    # Scale post-softmax attention weights for bootstrap-sampled steps
                    for sid, rng in enumerate(ranges):
                        if rng is None:
                            continue
                        w = weights[sid]
                        s, e = rng
                        if w == 0:
                            attn_weights[:, :, m_start-1:m_end-1, s:e] = 0.0
                        elif w > 1:
                            attn_weights[:, :, m_start-1:m_end-1, s:e] *= float(w)

                    # Re-normalize answer rows
                    row_sums = attn_weights[:, :, m_start-1:m_end-1, :].sum(dim=-1, keepdim=True).clamp(min=1e-9)
                    attn_weights[:, :, m_start-1:m_end-1, :] = attn_weights[:, :, m_start-1:m_end-1, :] / row_sums

                    attn_output = torch.matmul(attn_weights, value_states)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    return attn_output, attn_weights
                return patched_eager

            patched_fn = _make_patched_eager(sample_weights, step_token_ranges, modify_start_late, modify_end_late)
            ALL_ATTENTION_FUNCTIONS["bootstrap_hook"] = patched_fn
            llm.model.set_attn_implementation("bootstrap_hook")

            sample_cache = copy.deepcopy(early_cache)

            with torch.no_grad():
                sample_output = llm.model.forward(
                    input_ids=late_tokens,
                    attention_mask=hook_mask,
                    past_key_values=sample_cache,
                )

            all_logits.append(sample_output.logits)
            all_input_ids.append(late_tokens)

            llm.model.set_attn_implementation(_orig_attn_impl)

            del sample_cache
            torch.cuda.empty_cache()

        late_output = type(sample_output)(
            logits=torch.cat(all_logits, dim=0),
        )
        late_output['input_ids'] = torch.cat(all_input_ids, dim=0)

        step_masks = step_weights.tolist() if hasattr(step_weights, 'tolist') else is_step_selected.astype(int).tolist()
        return late_output, step_masks, masked_steps_debug

    # Chunk processing to reduce memory: process 10 samples at a time
    CHUNK_SIZE = 10
    num_chunks = (nb_dropout_samples + CHUNK_SIZE - 1) // CHUNK_SIZE

    all_logits = []
    all_hidden_states = []
    all_input_ids = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, nb_dropout_samples)
        chunk_size = end_idx - start_idx

        logger.info(f"Processing dropout chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx}:{end_idx})")

        # Slice the mask for this chunk
        chunk_mask = late_mask[start_idx:end_idx]

        # Expand tokens for this chunk only
        chunk_tokens = late_tokens.expand(chunk_size, -1)

        # Deep copy cache and replicate only for this chunk
        _t0 = time.perf_counter()
        chunk_cache = copy.deepcopy(early_cache)
        logger.info(f"deepcopy(early_cache) for chunk {chunk_idx} took {time.perf_counter() - _t0:.4f}s")
        chunk_cache.reorder_cache(torch.tensor([0] * chunk_size))

        # Convert mask to correct dtype
        chunk_mask = chunk_mask.to(device=device, dtype=next(llm.model.parameters()).dtype)

        with torch.no_grad():
            chunk_output = llm.model.forward(
                input_ids=chunk_tokens,
                attention_mask=chunk_mask,
                past_key_values=chunk_cache,
                # output_hidden_states=True,
            )

        # Collect outputs
        all_logits.append(chunk_output.logits)
        # all_hidden_states.append(chunk_output.hidden_states)
        all_input_ids.append(chunk_tokens)

        # Free chunk cache memory
        del chunk_cache
        torch.cuda.empty_cache()

    # Concatenate all chunks
    late_output = type(chunk_output)(
        logits=torch.cat(all_logits, dim=0),
        # hidden_states=tuple(torch.cat([hs[i] for hs in all_hidden_states], dim=0)
        #                    for i in range(len(all_hidden_states[0]))),
    )
    late_output['input_ids'] = torch.cat(all_input_ids, dim=0)

    # Convert is_step_selected to binary masks (1 = kept, 0 = masked)
    step_masks = is_step_selected.astype(int).tolist()

    return late_output, step_masks, masked_steps_debug


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
    probs = logits.double().softmax(-1)
    pos_entries = [{"token": llm.tokenizer.decode([tid]), "id": tid,
                    "logit": round(logits[tid].item(), 4), "prob": round(probs[tid].item(), 6)}
                   for tid in positive_ids]
    neg_entries = [{"token": llm.tokenizer.decode([tid]), "id": tid,
                    "logit": round(logits[tid].item(), 4), "prob": round(probs[tid].item(), 6)}
                   for tid in negative_ids]
    pos_sum = logits[positive_ids].sum()
    neg_sum = logits[negative_ids].sum()
    normalized = torch.softmax(torch.stack([pos_sum.double(), neg_sum.double()]), dim=0)
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


def _debug_masked_text(llm, early_token_ids, late_mask, modify_start_late, nb_early_tokens, is_step_selected, steps, nb_dropout_samples, use_jackknife):
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
            "use_jacknife": use_jackknife,
        })

        # if samples[-1]["masked_text"] == "":
        #     breakpoint()

    return samples
