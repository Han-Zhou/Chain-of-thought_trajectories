import os
from pyexpat import model
import re
# os.environ["CURL_CA_BUNDLE"]=""
# os.environ["REQUESTS_CA_BUNDLE"]=""
import copy
import logging
import urllib3
import warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
import numpy as np
import torch
from flash_attn import flash_attn_func
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList, StopStringCriteria, StoppingCriteria, DynamicCache, FineGrainedFP8Config, GPTQConfig, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
# from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig


from utils.enum import MODEL_DICT
from utils.structures import GenerationResult




load_dotenv()

logger = logging.getLogger(__name__)




# class _StopAfterBoxedAnswer(StoppingCriteria):
#     """Stop generation once '\\boxed{...}' with matched braces is produced."""
#
#     def __init__(self, tokenizer, prompt_len: int):
#         self.tokenizer = tokenizer
#         self.prompt_len = prompt_len
#
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         generated_ids = input_ids[0, self.prompt_len:]
#         text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
#         idx = text.find("\\boxed{")
#         if idx == -1:
#             return False
#         # Count braces from the opening { of \boxed{
#         num_open = 0
#         for i in range(idx + len("\\boxed"), len(text)):
#             if text[i] == "{":
#                 num_open += 1
#             elif text[i] == "}":
#                 num_open -= 1
#                 if num_open == 0:
#                     return True  # Matched braces — stop generation
#         return False


def _resolve_attn_implementation(model_name: str) -> str:
    """Return 'flash_attention_2' if model + hardware support it, else 'sdpa'."""
    # if not torch.cuda.is_available():
    #     logger.info("CUDA not available, falling back to sdpa")
    #     return "sdpa"

    # config = AutoConfig.from_pretrained(model_name)
    # model_class = AutoModelForCausalLM._model_mapping[type(config)]
    # # if not getattr(model_class, "_supports_flash_attn_2", False):
    # #     logger.info(f"{model_class.__name__} does not support flash_attention_2, falling back to sdpa")
# #     return "sdpa"

    # logger.info("Using flash_attention_2")
    # return "flash_attention_2"
    return "sdpa"


class LLM():
    def __init__(self, model_name: str, thinking: bool):

        logger.info(f"Loading model {model_name}...")
        self.model_name = model_name

        self._generation_attn_impl = _resolve_attn_implementation(model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # quantization_config = FineGrainedFP8Config(
        #     modules_to_not_convert=["linear_attn"],
        # )

        # quantization_config = FineGrainedFP8Config(
        #     modules_to_not_convert=["gdn"],
        #     weight_block_size=(128, 128) # Crucial for H100 hardware acceleration
        # )

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="sdpa",
        # )


        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="sdpa",
        # )

        # m = self.model
        # att = dir(m)


        # breakpoint()

        # root_config = AutoConfig.from_pretrained(model_name)
        # raw_quant = getattr(root_config, "quantization_config", None)
        # # # rescued_quant_config = GPTQConfig(**raw_quant) if raw_quant else None

        # rescued_quant_config = FineGrainedFP8Config(**raw_quant) if raw_quant else None


        # 2. Force the quantization config back into the model loader
        # self.model = Qwen3_5ForCausalLM.from_pretrained(
        #     model_name,
        #     # device_map="auto",
        #     device_map={"": 0},
        #     # torch_dtype=torch.bfloat16,
        #     attn_implementation="sdpa",
        #     # quantization_config=rescued_quant_config,
        #     # quantization_config=bnb_config,
        # )


        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            # device_map="auto",
            device_map={"": 0},
            # torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            # quantization_config=rescued_quant_config,
            # quantization_config=bnb_config,
        )







        # m = self.model
        att = dir(self.model)


        # 1. Extract the text configuration
        text_config = self.model.config.text_config

        # 2. Create an empty text-only skeleton on the "meta" device.
        # The "meta" device creates the structure without actually allocating RAM/VRAM for weights.
        with torch.device("meta"):
            text_model = Qwen3_5ForCausalLM._from_config(text_config)

        # 3. Graft the actual quantized modules from your multimodal model onto the skeleton
        text_model.model = self.model.model.language_model
        text_model.lm_head = self.model.lm_head

        # 4. Copy over the configuration files
        text_model.config = text_config
        text_model.generation_config = self.model.generation_config

        # (Optional) Ensure the new model points to the correct device
        # text_model.device = self.model.device
        

        self.model = text_model

        # # Check if scale tensors survived the graft
        # for name, param in self.model.named_parameters():
        #     if "scale" in name:
        #         logger.info(f"Found scale: {name} {param.shape}")
        #         break
        # else:
        #     logger.warning("No scale tensors found — FP8 descaling is broken")

        # for name, module in self.model.named_modules():
        #     if "q_proj" in name:
        #         logger.info(f"{name}: {module.__class__.__name__}")
        #         # Should print "FP8Linear" or "Float8Linear", NOT "Linear"
        #         print(f"weight dtype: {module.weight.dtype}") 
        #         break

        # logger.info(f"Device capability: {torch.cuda.get_device_capability()}")  # needs (9, 0) or higher = H100

        # breakpoint()



        # print(self.model.config.quantization_config)
        # print(self.model.dtype)                    # should show torch.float16 or similar
        # print(self.model.is_quantized)             # True if quantized
        # print(self.model.config.quantization_config)  # the actual GPTQ params




        # from bitsandbytes.nn import Linear4bit
        # for name, module in self.model.named_modules():
        #     if isinstance(module, Linear4bit):
        #         print(f"{name}: weight dtype={module.weight.dtype}, quant_type={module.weight.quant_type}")
        #         break
        # breakpoint()




        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        for name, module in self.model.named_modules():
            if "q_proj" in name and hasattr(module, 'weight'):
                print(f"weight dtype: {module.weight.dtype}")       # should be float8_e4m3fn
                if hasattr(module, 'weight_scale_inv'):
                    print(f"scale shape:  {module.weight_scale_inv.shape}")  # should be [32, 32] for 4096x4096
                # break

        print("=================================")

          # Check if layers are FP8Linear or plain Linear
        for name, module in self.model.named_modules():
            if "q_proj" in name:
                print(f"type:  {module.__class__.__name__}")
                print(f"weight dtype: {module.weight.dtype}")
                if hasattr(module, 'weight_scale_inv'):
                    print(f"scale shape:  {module.weight_scale_inv.shape}")
                else:
                    print("NO weight_scale — block dequant is NOT happening")
                # break


        # breakpoint()


        self.thinking = thinking
    

    def set_attn_implementation(self, impl: str):
        """Switch attention implementation using transformers' built-in API."""
        self.model.set_attn_implementation(impl)

    def generate_one(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        output_scores: bool = False,
        has_assistant_prefill: bool = True,
    ) -> GenerationResult:
        """Run one forward pass through model.generate() with a DynamicCache.

        Key HuggingFace arguments and why they are used here:

        past_key_values=DynamicCache()
            Passes a pre-initialised DynamicCache object into generate().
            Transformers fills it in-place during the forward passes; the
            populated cache is returned via outputs.past_key_values.
            This avoids re-computing attention keys/values on the prompt for any
            subsequent continuation (useful if you want to branch or re-use).

        return_dict_in_generate=True
            Makes generate() return a GenerateDecoderOnlyOutput (a structured
            object) instead of a raw tensor.  This gives access to:
            .sequences        – full token IDs (prompt + generated)
            .past_key_values  – the updated DynamicCache
            .scores           – per-step logits (if output_scores=True)

        do_sample / temperature
            temperature=0.0 → greedy decoding (deterministic, good for evals).
            temperature>0.0 → multinomial sampling.
        """

        device = next(self.model.parameters()).device

        if has_assistant_prefill:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,     # MUST be False if you provide the assistant role
                continue_final_message=True,      # MUST be True for Assistant Prefilling
            )
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,       # Let the template add the assistant header
                continue_final_message=False,
            )

        # Post processing
        # For qwen: strip the empty think block that Qwen injects
        if "qwen" in self.model_name.lower():
            prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)

        # For gpt: GPT-OSS templates default to <|channel|>final for the assistant prefill.
        # if we are using thinking mode, Switch the last occurrence to <|channel|>analysis so the model reasons before answering.
        if "gpt" in self.model_name.lower() and self.thinking:
            target = "<|channel|>final"
            last_idx = prompt_text.rfind(target)
            if last_idx != -1:
                prompt_text = prompt_text[:last_idx] + "<|channel|>analysis" + prompt_text[last_idx + len(target):]

        logger.info(prompt_text)

        # breakpoint()
            
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len: int = inputs["input_ids"].shape[1]

        # Qwen3.5 has a custom DynamicCache implementation that is required for correct attention behavior.
        if hasattr(self.model.config, "layer_types") and "linear_attention" in self.model.config.layer_types:
            cache = Qwen3_5DynamicCache(config=self.model.config)
        else:
            cache = DynamicCache()

        # cache = DynamicCache(config=self.model.config)

        # stop_criteria = StoppingCriteriaList([
        #     _StopAfterBoxedAnswer(self.tokenizer, prompt_len),
        # ])

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                past_key_values=cache,
                use_cache=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=output_scores,
                # stopping_criteria=stop_criteria,
            )

        total_len: int = outputs.sequences.shape[1]
        # prompt_positions - 1 is the end position of the prompt text
        prompt_positions = prompt_len
        # generated_positions - 1 is the end position of the generated text
        generated_positions = total_len

        generated_ids = outputs.sequences[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        generated_ids_tensor = outputs.sequences[0, prompt_len:].cpu()
        prompt_tail_ids = outputs.sequences[0, max(0, prompt_len - 10):prompt_len].cpu()

        return GenerationResult(
            prompt_text=prompt_text,
            generated_text=generated_text,
            prompt_end_position=prompt_positions,
            generated_end_position=generated_positions,
            past_key_values=outputs.past_key_values,
            prompt_tail_ids=prompt_tail_ids,
            scores=outputs.scores if output_scores else None,
            generated_ids=generated_ids_tensor,
        )


# class EosListStoppingCriteria(StoppingCriteria):
#     def __init__(self, eos_sequence = []):
#         self.eos_sequence = eos_sequence

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
#         return self.eos_sequence in last_ids




