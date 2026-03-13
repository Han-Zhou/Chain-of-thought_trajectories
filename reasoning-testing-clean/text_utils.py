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
    string = tokenizer.encode(string, add_special_tokens=False)
    string = "".join(tokenizer.convert_ids_to_tokens(string))
    
    start = len(token_ids)-1
    while start>0:
        text = "".join(tokenizer.convert_ids_to_tokens(token_ids[start:]))
        if string in text:
            break
        start -= 1
    if start==0:
        tokens = [tokenizer.decode(t) for t in token_ids[-10:]]
        raise ValueError(f"Cannot find '{string}' in '...{tokens}'")
    end = start+1
    while end<=len(token_ids):
        text = "".join(tokenizer.convert_ids_to_tokens(token_ids[start:end]))
        if string in text:
            break
        end += 1
    if end==len(token_ids)+1:
        tokens = [tokenizer.decode(t) for t in token_ids[-10:]]
        raise ValueError(f"Cannot find '{string}' in '...{tokens}'")
    return start, end