def shorten_string(string):
    nb_brackets = 1
    shortened_string = ""
    for s in string:
        if s=='{':
            nb_brackets += 1
        if s=='}':
            nb_brackets -= 1
            if nb_brackets==0:
                break
        shortened_string += s
    return shortened_string


def find_token_indices_from_end(tokenizer, token_ids, string):
    prev_string = string
    string = tokenizer.encode(string, add_special_tokens=False)
    string = "".join(tokenizer.convert_ids_to_tokens(string))
    
    # breakpoint()

    start = len(token_ids)-1
    while start>0:
        text = "".join(tokenizer.convert_ids_to_tokens(token_ids[start:]))
        if string in text:
            break
        # breakpoint()
        start -= 1
    
    # breakpoint()
    if start==0:
        tokens = [tokenizer.decode(t) for t in token_ids[-10:]]
        all_tokens = tokenizer.decode(token_ids)
        breakpoint()
        raise ValueError(f"Cannot find '{string}' in '...{tokens}'")
    end = start+1
    while end<=len(token_ids):
        text = "".join(tokenizer.convert_ids_to_tokens(token_ids[start:end]))
        if string in text:
            break
        end += 1
    # breakpoint()
    if end==len(token_ids)+1:
        tokens = [tokenizer.decode(t) for t in token_ids[-10:]]
        all_tokens = tokenizer.decode(token_ids)
        breakpoint()
        raise ValueError(f"Cannot find '{string}' in '...{tokens}'")
    return start, end


def find_token_overlap(base_ids, suffix_ids):
    """Length of the longest common prefix between two 1D token ID tensors.

    Compares element-wise from index 0 and returns the count of leading
    tokens that are identical.  Used to determine how much of a generation
    KV cache can be reused after appending a suffix and re-tokenizing.
    """
    min_len = min(len(base_ids), len(suffix_ids))
    for i in range(min_len):
        if base_ids[i] != suffix_ids[i]:
            return i
    return min_len