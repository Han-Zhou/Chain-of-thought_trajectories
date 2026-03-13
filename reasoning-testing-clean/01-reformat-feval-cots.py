import re
import json
import pickle
import argparse
import warnings
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from pathlib import Path


def extract_partial_cots(sampled_cot):
    steps = sampled_cot.split('\n\n')
    if len(steps)==1:
        steps = sampled_cot.split('\n')
        if len(steps)==1:
            steps = nltk.sent_tokenize(sampled_cot)
            if len(steps)==1:
                warnings.warn(f"\nIncapable of splitting CoT: '{sampled_cot}'")
    partial_cot = ""
    partial_cots = []
    for i in range(len(steps)):
        if i == 0:
            partial_cot = steps[0]
        elif i < len(steps)-1:
            partial_cot += " " + steps[i]
        else:
            partial_cot += "\n" + steps[i]
        partial_cots += [partial_cot]
    return partial_cots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_json_path'
    )
    args = parser.parse_args()
    # args = parser.parse_args(args=['/media/data1/didier-c/soumyacot/llama_70b_33_bigbench_causal_judgement_fs_cot_40.json'])
    print(args)

    with open(args.dataset_json_path, "r") as file:
        json_samples = json.load(file)
    dataset_name = json_samples['datatype']
    llm_name = json_samples['responses'][0]['request']['llm_type']
    
    examples = []
    for json_example in json_samples['responses']:
        test_question = 'Question: ' + json_example['data_entry']['question']
        test_answer = json_example['data_entry']['answer']
        
        fs_prompt, test_answer_prompt = json_example['request']['message'].split(test_question)
        
        system_prompt, fs_prompt = re.match(r"(.*Think step-by-step.)\n\n+(.*)", fs_prompt, flags=re.DOTALL).groups()
        system_prompt = json_example['request']['system_prompt']+' '+system_prompt
        
        fs_prompt = fs_prompt.strip()
        test_answer_prompt = test_answer_prompt.strip()
        if test_answer_prompt:
            test_answer_prompt += ' '
        
        # Get prompt
        fs_prompt = ['Question: '+s for s in re.split('(?:\n\n)?Question: ', fs_prompt, flags=re.DOTALL) if s]
        prompt = []
        for fs_example in fs_prompt[:-1]:
            query = re.match("(Question: .*)(Answer: .*)", fs_example, flags=re.DOTALL)
            if query:
                fs_question, cot_with_answer = query.groups()
            else:
                fs_question, cot_with_answer = re.match(r"(Question: .*?)\n(.*boxed.*)", fs_example, flags=re.DOTALL).groups()

            cot_with_answer, = re.match("(?:Answer: )?(?:Let\'s think step by step.[\n ]*)?(.*)", 
                                        cot_with_answer, flags=re.DOTALL).groups()
            cot_with_answer = 'Answer: Let\'s think step by step. ' + cot_with_answer
            fs_answer = re.findall(".*\\\\boxed{(.*)}.*", cot_with_answer)[0]
            prompt += [{'question': fs_question.rstrip(),
                       'answer': fs_answer,
                       'cot_with_answer': cot_with_answer,
                       'cot_answer': fs_answer}]
        
        # Get sampled CoTs
        sampled_cots = []
        for response in json_example['response']['responses']:
            sampled_cot, = re.match("(?:Let\'s think step by step.[\n ]*)?(.*)", response['message'], flags=re.DOTALL).groups()
            partial_cots = extract_partial_cots(sampled_cot)
            cot_answer = re.match(".*\\\\boxed{(.*)}", sampled_cot, flags=re.DOTALL)
            if response['finish_reason']=='stop' and cot_answer is not None:
                # This is good CoT
                answer_prompt = 'Answer: Let\'s think step by step.'
                partial_cots = [answer_prompt] + [answer_prompt+' '+partial_cot for partial_cot in partial_cots[:-1]]
                cot_answer, = cot_answer.groups()
                sampled_cots += [{'partial_cots': partial_cots,
                                  'cot_answer': cot_answer}]
        
        examples += [{'dataset_name': dataset_name,
                      'llm_name': llm_name,
                      'system_prompt': system_prompt,
                      'fs_prompt': prompt,
                      'question': test_question,
                      'answer': test_answer,
                      'sampled_cots': sampled_cots}]

    # DEBUG
    print(f"Processed {len(examples)} examples")
    print("Example:")
    example = examples[0]
    print(example['system_prompt'])
    print("---------------\n"+example['fs_prompt'][0]['question'])
    print("---------------\n"+example['fs_prompt'][0]['cot_with_answer'])
    print("---------------\n"+example['question'])
    print("---------------\n"+example['sampled_cots'][0]['partial_cots'][-1])
    print("\n\n")

    filename = Path(args.dataset_json_path).stem[:-len('_fs_cot_40')]
    with open(f"data/samples/{filename}.pkl", "wb") as file:
        pickle.dump(examples, file)
