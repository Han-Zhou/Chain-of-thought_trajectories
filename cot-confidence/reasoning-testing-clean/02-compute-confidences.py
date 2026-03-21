import os
import argparse
import pickle
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
from llm import LLM
from dropout import dropout_answerlogits, dropout_indirectlogit, dropout_verbconf


parser = argparse.ArgumentParser()
parser.add_argument(
    'sample_file',
)
parser.add_argument('-n', '--experiment_name', type=str, required=True)
parser.add_argument('-g', '--gpus', nargs="+", type=int, required=True)
parser.add_argument('-l', '--llm_path', type=str)
def pair(arg):
    return [int(x) for x in arg.split(':')]
parser.add_argument('-I', '--interval', type=pair)
# args = parser.parse_args()
args = parser.parse_args(args=['data/samples/llama_8b_31_aqua.pkl', 
                               '-g', *[str(i) for i in [3,4]], 
                               '-n', 'test',
                               '-l', "meta-llama/Meta-Llama-3-8B-Instruct"])
print(args)

# Load the samples
with open(args.sample_file, "rb") as file:
    dict_questions = pickle.load(file)
dataset_name = dict_questions[0]['dataset_name'].lower()
llm_name = args.llm_path if args.llm_path else dict_questions[0]['llm_name']

# Load the LLM
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in args.gpus])
llm = LLM(llm_name)
llm.name = Path(llm_name).name
llm.is_chat_model = 'instruct' in llm.name.lower() or 'chat' in llm.name.lower()
llm.chat_template = f"""{{%- set loop_messages = messages %}}
    {{%- for message in loop_messages %}}
        {{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim %}}
        {{%- if loop.index0 == 0 %}}
            {{%- set content = {'bos_token + ' if llm.tokenizer.bos_token is not None else ''}content %}}
        {{%- endif %}}
        {{%- if not (loop.last and message['role'] == 'assistant') %}}
            {{%- set content = content + '<|eot_id|>' %}}
        {{%- endif %}}
        {{{{- content }}}}
    {{%- endfor %}}
    {{%- if messages[-1]['role'] != 'assistant' %}}
      {{{{- '<|start_header_id|>assistant<|end_header_id|>\n' }}}}
    {{%- endif %}}"""

output_path = Path(f"data/confidences/{args.experiment_name}/{dataset_name}/{llm.name}")
output_path.mkdir(parents=True, exist_ok=True)


# Find which questions we are left to compute
start = max(args.interval[0], 0) if args.interval else 0
end = min(args.interval[1], len(dict_questions)) if args.interval else len(dict_questions)
computed_question_ids = sorted([int(s.name[len("question_"):len("question_")+4]) for s in output_path.glob("*.pkl")])
question_ids_to_compute = [i for i in range(start, end) if i not in computed_question_ids]

print(f"Extracting confidence data from questions {start} to {end} in '{dataset_name}'")
question_pbar = tqdm(question_ids_to_compute)
for question_id in question_pbar:
    dict_question = dict_questions[question_id]
    question = dict_question['question']
    correct_answer = dict_question['answer']
    prompt = [{'role': 'system', 'content': dict_question['system_prompt']}]
    # for fs_example in dict_question['fs_prompt']:
    #     prompt += [{'role': 'user', 'content': fs_example['question']},
    #                {'role': 'assistant', 'content': fs_example['cot_with_answer']}]
    prompt += [{'role': 'user', 'content': question},]

    question_data = defaultdict(list)
    with tqdm(leave=False, total=10) as sample_pbar:
        for sampled_cot in dict_question['sampled_cots']:
            if sampled_cot['cot_answer'] is not None:
                sampled_answer = sampled_cot['cot_answer']
                confidences = {'accuracy': sampled_answer==correct_answer}
                confidences |= dropout_answerlogits(llm, prompt, sampled_cot['steps'], 
                                                    sampled_answer, nb_dropout_samples=10, use_fullstring=False)
                confidences |= dropout_indirectlogit(llm, prompt, sampled_cot['steps'], 
                                                     sampled_answer, nb_dropout_samples=10, use_fullstring=False, 
                                                     string_format="ptrue")
                confidences |= dropout_indirectlogit(llm, prompt, sampled_cot['steps'], 
                                                     sampled_answer, nb_dropout_samples=10, use_fullstring=False, 
                                                     string_format="ptrue2")
                confidences |= dropout_verbconf(llm, prompt, sampled_cot['steps'], 
                                                sampled_answer, nb_dropout_samples=10, use_fullstring=False)
                raise Exception()

