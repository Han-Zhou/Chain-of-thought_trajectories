import numpy as np
import torch
import copy
from collections import defaultdict
from llm import LLM
from utils.text_utils import find_token_indices_from_end
from utils.structures import ParsedOutput, GenerationResult, ConfidenceScores, AllConfidenceData


def crop_kv_cache(llm, cache, max_length):
    """Crop KV cache, with fallback for caches missing .crop() (e.g. Qwen 3.5)."""
    if "qwen3_5" in llm.model.config.model_type:
        for layer_idx in range(len(cache.key_cache)):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :max_length, :]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :max_length, :]
    else:
        cache.crop(max_length)


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
    generation_result: GenerationResult,
    parsed_output: ParsedOutput,
    nb_dropout_samples: int = 10,
    use_fullstring: bool = False,
    assistant_prefill: str = "",
) -> AllConfidenceData:
    vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy = \
        dropout_answerlogits(llm, generation_result, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill)

    vanilla_ptrue1, vanilla_ptrue2, dropout_ptrue1, dropout_ptrue2 = \
        dropout_indirectlogits(llm, generation_result, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill)

    vanilla_verbconf, dropout_verbconf = \
        dropout_verbalconf(llm, generation_result, parsed_output, nb_dropout_samples, use_fullstring, assistant_prefill)

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
    )



def dropout_answerlogits(llm, generation_result, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill=""):
    """Logit-based confidence on the answer tokens themselves."""
    late_tokens, vanilla_out, dropout_out, ans_start, ans_end = \
        dropout_forward(llm, generation_result, parsed_output,
                        suffix_text="", nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)

    answer_tokens = late_tokens[0, ans_start:ans_end]
    nb_answer_tokens = len(answer_tokens)

    # --- Vanilla ---
    # Logit at position t-1 predicts token at position t
    v_logits = vanilla_out.logits[0, ans_start - 1:ans_end - 1, :] # dimension: [nb_answer_tokens, vocab_size]
    v_probs = v_logits.softmax(dim=-1)

    breakpoint()



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

    return vanilla_answer_probs, vanilla_answer_entropy, dropout_answer_probs, dropout_answer_entropy


def dropout_indirectlogits(llm, generation_result, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill=""):
    """P(True) and P(Yes) probing after the answer."""
    positive_true_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
    negative_false_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    positive_yes_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
    negative_no_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])

    # ptrue1: "True/False:"
    _, vanilla1, dropout1, _, _ = \
        dropout_forward(llm, generation_result, parsed_output,
                        suffix_text="\nTrue/False:",
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)

    # ptrue2: "Is the answer <X> correct?"
    _, vanilla2, dropout2, _, _ = \
        dropout_forward(llm, generation_result, parsed_output,
                        suffix_text=f"\nIs the answer {parsed_output.final_answer} correct?",
                        nb_dropout_samples=nb_dropout_samples,
                        use_fullstring=use_fullstring, assistant_prefill=assistant_prefill)


    #  NOTE: need to make sure that for our models, the true/false/yes/no are only one token


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


    # breakpoint()


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


        # breakpoint()


    else:
        dropout_ptrue1 = []
        dropout_ptrue2 = []

    return [vanilla_ptrue1], [vanilla_ptrue2], dropout_ptrue1, dropout_ptrue2


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
    # Walk depth by depth. At each depth d, group sequences by their
    # token at depth d-1 (the prefix token that must be fed to the model).
    # Each unique prefix gets its own cloned KV cache for exact conditioning.

    # Build a mapping: prefix_token -> {cache, kv after feeding that token}
    # so deeper levels can chain off the correct cache.
    # Key: tuple of tokens up to depth d-1 (the full prefix fed so far)
    prefix_cache = {(): model_output.past_key_values}

    for d in range(1, max_depth):
        # Group sequences that still have a token at depth d by their prefix so far
        groups = defaultdict(list)  # prefix_tuple -> [(seq_index, token_at_depth_d)]
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

            # Store this cache in case even deeper levels need it
            new_prefix_cache[prefix] = out.past_key_values

        prefix_cache = new_prefix_cache

    return joint_logprobs.softmax(-1)


def dropout_verbalconf(llm, generation_result, parsed_output, nb_dropout_samples=10, use_fullstring=False, assistant_prefill=""):
    """Verbalized confidence (0-100 score)."""
    # NOTE may change this for gpt-oss
    suffix = (
        "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."
        "\nHow confident are you in your previous answer?"
        "\n<confidence>"
    )

    _, vanilla_out, dropout_out, _, _ = \
        dropout_forward(llm, generation_result, parsed_output,
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

    # breakpoint()

    # Dropout
    if dropout_out is not None:
        d_probs = _compute_verbconf_joint_probs(llm, dropout_out, token_seqs, device)
        dropout_verbconf = (score_values * d_probs).sum(-1).numpy().tolist()

        # breakpoint()

    else:
        dropout_verbconf = []

    return [vanilla_verbconf], dropout_verbconf


def dropout_forward(
    llm: LLM,
    generation_result: GenerationResult,
    parsed_output: ParsedOutput,
    suffix_text: str = "",
    nb_dropout_samples: int = 10,
    use_fullstring: bool = False,
    threshold: float = 0.5,
    assistant_prefill: str = "",
):
    """Core forward pass for dropout experiment. Crop the generation KV-cache, run a vanilla + dropout forward pass on the late (answer-region) tokens.
    """
    generated_ids = generation_result.generated_ids          # 1-D, on CPU
    prompt_end = generation_result.prompt_end_position
    generated_text = generation_result.generated_text
    device = next(llm.model.parameters()).device

    # answer_fullstring_start was computed on (assistant_prefill + generated_text).strip(),
    # so adjust for the prefill length to index into generated_text alone.
    offset = max(parsed_output.answer_fullstring_start - len(assistant_prefill), 0)
    fullstring_text = generated_text[offset:]
    fs_start, _ = find_token_indices_from_end(
        llm.tokenizer, generated_ids, fullstring_text)

    # Early / late split (late split is answer regin & onwards)
    # One token before "Final Answer" stays in the early cache as context (context token)
    early_late_split_gen = fs_start - 1           # index in generated_ids
    early_late_split_abs = prompt_end + early_late_split_gen   # absolute position


    breakpoint()


    # Build late token tensor: decode the tail, append suffix, retokenize together
    # (avoids incorrect tokens at the boundary from separate tokenization)
    late_text = llm.tokenizer.decode(
        generated_ids[early_late_split_gen:], skip_special_tokens=False)
    combined_text = late_text + suffix_text
    late_ids = torch.tensor(
        llm.tokenizer.encode(combined_text, add_special_tokens=False),
        dtype=generated_ids.dtype)


    breakpoint()

    # Recompute answer positions within the retokenized late_ids
    answer_start_late, answer_end_late = find_token_indices_from_end(
        llm.tokenizer, late_ids, parsed_output.final_answer)

    if use_fullstring:
        modify_start_late = 1                    # first "Final Answer" token
        modify_end_late = len(late_ids)          # cover everything incl. suffix
    else:
        modify_start_late = answer_start_late
        modify_end_late = answer_end_late

    late_tokens = late_ids.unsqueeze(0).to(device)           # [1, late_len]

    # Crop KV cache to the early portion
    kv_cache = copy.deepcopy(generation_result.past_key_values)
    crop_kv_cache(llm, kv_cache, early_late_split_abs)

    # -- Vanilla forward ------------------------------------------------------
    with torch.no_grad():
        vanilla_output = llm.model.forward(
            input_ids=late_tokens,
            past_key_values=copy.deepcopy(kv_cache),
            output_hidden_states=True,
        )

    # -- Dropout forward ------------------------------------------------------
    dropout_output = None
    if nb_dropout_samples > 0 and len(parsed_output.cot_steps) > 1:
        step_token_ids = torch.cat([generation_result.prompt_tail_ids, generated_ids[:early_late_split_gen]])
        tail_len = len(generation_result.prompt_tail_ids)
        dropout_output = dropout_late_forward(
            llm,
            parsed_output.cot_steps,
            step_token_ids,
            copy.deepcopy(kv_cache),
            late_tokens,
            modify_start_late,
            modify_end_late,
            early_late_split_abs,          # == nb_early_tokens
            prompt_end,
            nb_dropout_samples,
            threshold,
            tail_len,
        )

    return late_tokens, vanilla_output, dropout_output, answer_start_late, answer_end_late


def dropout_late_forward(
    llm,                  # LLM instance (need llm.model and llm.tokenizer)
    cot_steps,            # list of reasoning step strings from ParsedOutput
    step_token_ids,       # 1D tensor: prompt tail + generated token IDs containing the steps (before the answer region)
    early_cache,          # cropped KV cache covering [prompt + steps]
    late_tokens,          # [1, late_len] tensor: tokens to run forward on (answer region + suffix)
    modify_start_late,    # int: first token in late_tokens to apply dropout masking to
    modify_end_late,      # int: one-past-last token in late_tokens to apply dropout masking to
    nb_early_tokens,      # int: total length of the early cache (prompt + steps) == early_late_split_abs
    prompt_end,           # int: number of prompt tokens — used to convert step positions to absolute mask columns
    nb_dropout_samples,   # int: how many dropout samples to run in the batch
    threshold,            # float: probability of keeping each step (0.5 = 50% chance each step is masked)
    tail_len=0,           # int: number of prompt tail tokens prepended to step_token_ids
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
    # shape: [nb_dropout_samples, 1, nb_late, nb_early + nb_late]

    # Randomly select which steps to keep per sample
    is_step_selected = (np.random.random((nb_dropout_samples, len(steps))) <= threshold)

    # Walk backwards through steps, finding each one's token range and masking
    remaining_ids = step_token_ids.clone()
    for step_id, step in reversed(list(enumerate(steps))):
        try:
            step_start, step_end = find_token_indices_from_end(llm.tokenizer, remaining_ids, step)
        except ValueError:
            # Step text overlaps with the prompt region (not fully in generated tokens).
            # All earlier steps will too, so stop here.
            break
        # Absolute position in the full sequence (for mask column index)
        abs_start = prompt_end + step_start - tail_len
        abs_end = prompt_end + step_end - tail_len

        # If the step falls (partially) in the prompt region, skip it
        if abs_start < prompt_end:
            remaining_ids = remaining_ids[:step_start + 1]
            continue

        for i in range(nb_dropout_samples):
            # Rows = answer-region positions (shifted -1 for next-token prediction)
            # Cols = step positions in the early cache
            late_mask[i, 0,
                      modify_start_late - 1:modify_end_late - 1,
                      abs_start:abs_end] = 0. if is_step_selected[i, step_id] else -10000.

        remaining_ids = remaining_ids[:step_start + 1]

    # Expand for the dropout batch
    late_tokens_batch = late_tokens.expand(nb_dropout_samples, -1)
    early_cache.reorder_cache(torch.tensor([0] * nb_dropout_samples))
    late_mask = late_mask.to(device=device, dtype=next(llm.model.parameters()).dtype)


    breakpoint()


    with torch.no_grad():
        late_output = llm.model.forward(
            input_ids=late_tokens_batch,
            attention_mask=late_mask,
            past_key_values=early_cache,
            output_hidden_states=True,
        )
        late_output['input_ids'] = late_tokens_batch

    return late_output
