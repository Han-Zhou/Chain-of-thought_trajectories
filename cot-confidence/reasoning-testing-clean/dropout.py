import pickle
import numpy as np
import torch
import copy
from transformers import DynamicCache
from llm import LLM
from pathlib import Path
from text_utils import find_token_indices_from_end


# from transformers import DynamicCache

# # 1. Trim the cache to the first 80 tokens
# cache.crop(80)

# # 2. Trim your generated tokens to match (assuming 'output_ids' has shape [1, 100])
# new_input_ids = output_ids[:, :80]

# # 3. Resume generation
# # The model will use the 80 cached tokens and start generating from the 81st
# new_outputs = model.generate(
#     new_input_ids,
#     past_key_values=cache,
#     max_new_tokens=40, # Generate 20 to replace, plus 20 more
#     use_cache=True
# )



ANSWER_TOKENS = {' Yes': [' Yes',' yes',' YES',' Yeah', ' yeah', ' Yep', ' yep'],
                 ' No': [' No',' no', ' NO', ' Nah', ' nah', ' Nope', ' nope'],
                 ' True': [' True'],
                 ' False': [' False'],
                 'VERBCONF': [str(i) for i in range(0, 101)]}


def get_token_ids(tokenizer, tokens):
    labels = []
    for token in tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids)==1:
            labels += [token_ids[0]]
    return labels


def dropout_answerlogits(llm, query_prompt, steps, answer, nb_dropout_samples=100, use_fullstring=False):
    reasoning = "".join(steps)
    answer_string = f"\nThe answer is \\boxed{{{answer}}}."
    prompt = query_prompt + \
        [{'role': 'assistant', 'content': (reasoning + answer_string).strip()}]
    
    late_tokens, vanilla_output, dropout_output = dropout_forward(llm, prompt, steps, answer, answer_string, nb_dropout_samples, use_fullstring, threshold=0.5)

    boxed_start, boxed_end = find_token_indices_from_end(llm.tokenizer, late_tokens[0], '\\boxed{'+answer+'}')
    answer_start, answer_end = find_token_indices_from_end(llm.tokenizer, late_tokens[0, boxed_start:boxed_end], answer)
    answer_start, answer_end = answer_start+boxed_start, answer_end+boxed_start
    answer_tokens = late_tokens[0, answer_start:answer_end]
    
    answer_dropout_logits = dropout_output['logits'][:, answer_start-1:answer_end-1, :]
    answer_dropout_logprobs = torch.nan_to_num(answer_dropout_logits.log_softmax(dim=-1), neginf=-99)
    
    sample = {}
    # sample['answer_prompt'] = prompt
    sample['answer_tokens'] = llm.tokenizer.decode(answer_tokens)
    sample['answer_dropout_logits'] = torch.stack([answer_dropout_logits[:, t, answer_tokens[t]] for t in range(len(answer_tokens))], dim=-1).numpy()
    sample['answer_dropout_entropy'] = (-answer_dropout_logprobs.exp()*answer_dropout_logprobs).sum(-1).numpy()
    sample['answer_dropout_probs'] = torch.stack([answer_dropout_logprobs[:, t, answer_tokens[t]].float() for t in range(len(answer_tokens))], dim=-1).exp().numpy()
    return sample


def dropout_indirectlogit(llm, query_prompt, steps, answer, nb_dropout_samples=100, use_fullstring=False, string_format="ptrue"):
    reasoning = "".join(steps)
    if string_format=="ptrue":
        answer_string = f"\nThe answer is \\boxed{{{answer}}}.\nTrue/False:"
        positive_token_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' True'])
        negative_token_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' False'])
    elif string_format=="ptrue2":
        answer_string = f"\nIs the answer \\boxed{{{answer}}}?"
        positive_token_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' Yes'])
        negative_token_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS[' No'])
    else:
        raise ValueError(f"Unrecognized string format {string_format}")
        
    prompt = query_prompt + \
        [{'role': 'assistant', 'content': (reasoning + answer_string).strip()}]
    late_tokens, vanilla_output, dropout_output = dropout_forward(llm, prompt, steps, answer, answer_string, nb_dropout_samples, use_fullstring, threshold=0.5)
    
    dropout_positive_logits = dropout_output['logits'][:, -1, positive_token_ids].sum(-1)
    dropout_negative_logits = dropout_output['logits'][:, -1, negative_token_ids].sum(-1)
    
    sample = {}
    # sample[string_format+'_prompt'] = prompt
    sample[string_format+'_dropout_logits'] = (dropout_positive_logits - dropout_negative_logits).numpy()
    sample[string_format+'_dropout_probs'] = torch.softmax(torch.stack([dropout_positive_logits.float(), dropout_negative_logits.float()], dim=-1), dim=-1)[:, 0].numpy()
    return sample


def dropout_verbconf(llm, query_prompt, steps, answer, nb_dropout_samples=100, use_fullstring=False):
    reasoning = "".join(steps)
    answer_string = f"\nThe answer is \\boxed{{{answer}}}."\
                     "\nPlease respond with a score from 0 to 100 in <confidence> </confidence> tags."\
                     "\nHow confident are you in your previous answer?"\
                     "\n<confidence>"
    
    prompt = query_prompt + \
        [{'role': 'assistant', 'content': (reasoning + answer_string).strip()}]
    late_tokens, vanilla_output, dropout_output = dropout_forward(llm, prompt, steps, answer, answer_string, nb_dropout_samples, use_fullstring, threshold=0.5)

    verbconf_token_ids = get_token_ids(llm.tokenizer, ANSWER_TOKENS['VERBCONF'])
    verbconf_dropout_logits = dropout_output['logits'][:, -1, verbconf_token_ids]
    verbconf_dropout_probs = verbconf_dropout_logits.float().softmax(-1)
    
    sample = {}
    # sample['verbconf_prompt'] = prompt
    sample['verbconf_dropout_modeprobs'] = np.array([int(ANSWER_TOKENS['VERBCONF'][token_id])/100 for token_id in verbconf_dropout_logits.argmax(-1)], dtype=np.float32)
    sample['verbconf_dropout_meanprobs'] = (torch.FloatTensor([int(t)/100 for t in ANSWER_TOKENS['VERBCONF']])*verbconf_dropout_probs).sum(-1).numpy()
    return sample


def dropout_forward(llm, prompt, steps, answer, answer_string, nb_dropout_samples, use_fullstring, threshold=0.5):
    token_ids = llm.tokenizer.apply_chat_template([prompt], 
                                                   tokenize=True, return_dict=True, 
                                                   return_tensors="pt", padding=True,
                                                   chat_template=llm.chat_template)
    tokens = token_ids['input_ids']
    
    if use_fullstring:
        tokens_to_modify_start, tokens_to_modify_end = find_token_indices_from_end(llm.tokenizer, tokens[0], answer_string)
        early_late_split = tokens_to_modify_start-1
    else:
        boxed_start, boxed_end = find_token_indices_from_end(llm.tokenizer, tokens[0], '\\boxed{'+answer+'}')
        tokens_to_modify_start, tokens_to_modify_end = find_token_indices_from_end(llm.tokenizer, tokens[0, boxed_start:boxed_end], answer)
        tokens_to_modify_start, tokens_to_modify_end = tokens_to_modify_start+boxed_start, tokens_to_modify_end+boxed_start
        early_late_split = boxed_start-1
    
    early_tokens = tokens[:1, :early_late_split]
    with torch.no_grad():
        early_output = llm.model(input_ids=early_tokens, past_key_values=DynamicCache())

    late_tokens = tokens[:, early_late_split:]
    # Compute vanilla
    with torch.no_grad():
        vanilla_output = llm.model.forward(input_ids=late_tokens, early_cache=copy.deepcopy(early_output['past_key_values']), output_hidden_states=True)
        vanilla_output['input_ids'] = late_tokens
    # Compute dropout
    if nb_dropout_samples>0:
        dropout_output = dropout_late_forward(llm, steps, tokens, early_late_split, tokens_to_modify_start, 
                                              tokens_to_modify_end, nb_dropout_samples=nb_dropout_samples, 
                                              early_cache=copy.deepcopy(early_output['past_key_values']))
    
    return late_tokens, vanilla_output, dropout_output


def dropout_late_forward(llm, steps, early_tokens, early_cache, late_tokens, tokens_to_modify_start, tokens_to_modify_end, nb_dropout_samples, threshold=0.5):
    steps = steps[1:] # Always keep the first step
    # late_tokens = tokens[:, early_late_split:]
    nb_early_tokens = len(early_tokens[0])
    nb_late_tokens = len(late_tokens[0])
    
    late_mask = torch.cat([torch.ones(nb_late_tokens, nb_early_tokens), 
                           torch.ones(nb_late_tokens, nb_late_tokens).tril()], dim=1)
    late_mask[late_mask==0] = -10000.
    late_mask[late_mask==1] = 0.
    late_mask = late_mask.repeat(nb_dropout_samples, 1, 1).unsqueeze(1)
    
    # Construct a dropout attention mask
    is_step_selected = (np.random.random((nb_dropout_samples, len(steps)))<=threshold)
    step_tokens = torch.cat([
    for step_id, step in reversed(list(enumerate(steps))):
        step_start, step_end = find_token_indices_from_end(llm.tokenizer, step_tokens[0], step)
        for i in range(nb_dropout_samples):
            late_mask[i, 0, tokens_to_modify_start-1-early_late_split:tokens_to_modify_end-1-early_late_split, 
                            step_start:step_end] = 0. if is_step_selected[i, step_id] else -10000.
        step_tokens = step_tokens[:, :step_start+1]
    
    late_tokens = late_tokens.expand(nb_dropout_samples, -1)
    # late_mask = late_mask.expand(-1, len(llm.model.model.layers), -1, -1)
    early_cache.reorder_cache(torch.tensor([0] * nb_dropout_samples)) # Dunno what this does lol
    
    with torch.no_grad():
        late_output = llm.model.forward(input_ids=late_tokens, attention_mask=late_mask, past_key_values=early_cache, output_hidden_states=True)
        late_output['input_ids'] = late_tokens
    
    return late_output