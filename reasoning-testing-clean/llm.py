import os
os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]=""
import urllib3
import warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from transformers import StoppingCriteriaList, StopStringCriteria, StoppingCriteria
from peft import PeftConfig, AutoPeftModelForCausalLM
from constants import HF_CACHE, HF_TOKEN
os.environ["HF_HOME"] = HF_CACHE



class LLM():
    def __init__(self, path):
        peft = (Path(path)/"adapter_config.json").exists()
        print(f"Loading {'PEFT' if peft else 'full'} model \"{path}\"")
        a100 = is_flash_attn_2_available()
        if a100:
            print("Using Flash attention 2")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
        
        LMModel = AutoPeftModelForCausalLM if peft else AutoModelForCausalLM
        self.model = LMModel.from_pretrained(
            pretrained_model_name_or_path=path,
            cache_dir=HF_CACHE,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if a100 else torch.float16,
            device_map="auto",
            token=HF_TOKEN,
            # attn_implementation="flash_attention_2" if a100 else "",
            attn_implementation="eager",
            # attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            offload_state_dict=True,
        )
        if peft:
            # Merge and remove peft
            base_model = self.model.merge_and_unload(progressbar=True)
            self.model.delete_adapter('default')
            self.model = base_model
            del self.model.peft_config
            torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path,
            padding_side='left',
            cache_dir=HF_CACHE,
            token=HF_TOKEN,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.name = Path(path).stem

    def generate(self, prompt, chat=False, string_stopping_criteria=[], token_stopping_criteria="", **kwargs):
        if chat:
            token_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True, 
                                                           return_dict=True, 
                                                           continue_final_message=True, 
                                                           return_tensors="pt", padding=True)
        else:
            token_ids = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        stopping_criteria = []
        if string_stopping_criteria:
            stopping_criteria += [StopStringCriteria(self.tokenizer, string_stopping_criteria)]
        if token_stopping_criteria:
            stopping_criteria += [EosListStoppingCriteria(self.tokenizer.encode(token_stopping_criteria))]
        stopping_criteria = StoppingCriteriaList(stopping_criteria) if stopping_criteria else None
        
        with torch.no_grad():
            outputs = self.model.generate(**token_ids, pad_token_id=self.tokenizer.eos_token_id, 
                                          return_dict_in_generate=True, output_hidden_states=True,
                                          tokenizer=self.tokenizer,  stopping_criteria=stopping_criteria, **kwargs)
        outputs['input_ids'] = token_ids
        # outputs['strings'] = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=False)
        outputs['hidden_states'] = torch.cat([torch.stack(h, dim=2) for h in outputs['hidden_states']], dim=1)
        
        return outputs
    
    def forward(self, prompt, chat=False, **kwargs):
        if chat:
            token_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True, 
                                                           return_dict=True, 
                                                           continue_final_message=True, 
                                                           return_tensors="pt", padding=True)
        else:
            token_ids = self.tokenizer(prompt, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(**token_ids, output_hidden_states=True, **kwargs)
        outputs['input_ids'] = token_ids
        outputs['hidden_states'] = torch.stack(outputs['hidden_states'], dim=2)
        return outputs


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = []):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids
