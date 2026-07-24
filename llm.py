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
# from flash_attn import flash_attn_func
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList, StopStringCriteria, StoppingCriteria, FineGrainedFP8Config, GPTQConfig, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
# from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig


from utils.cache_compat import DynamicCache, Qwen3_5DynamicCache
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
    if not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to sdpa")
        return "sdpa"

    if "qwen" in model_name.lower():
        return "sdpa"
        # return "flash_attention_2"
    if "gpt" in model_name.lower():
        return "flash_attention_3"
    return "sdpa"


class LLM():
    def __init__(
        self,
        model_name: str,
        thinking: bool,
        backend: str = "hf",
        vllm_max_model_len: int | None = None,
        vllm_gpu_memory_utilization: float = 0.92,
        vllm_tensor_parallel_size: int = 1,
        vllm_max_num_seqs: int | None = None,
        vllm_language_model_only: bool = False,
    ):

        logger.info(f"Loading model {model_name} with backend {backend}...")
        self.model_name = model_name
        self.thinking = thinking
        self.backend = backend

        if backend == "vllm":
            # vLLM backend: lazy import and load
            try:
                from vllm import LLM as vLLM_LLM, SamplingParams

                vllm_kwargs = dict(
                    model=model_name,
                    dtype="bfloat16",
                    trust_remote_code=True,
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                    tensor_parallel_size=vllm_tensor_parallel_size,
                    language_model_only=vllm_language_model_only,
                )
                if vllm_max_model_len is not None:
                    vllm_kwargs["max_model_len"] = vllm_max_model_len
                if vllm_max_num_seqs is not None:
                    vllm_kwargs["max_num_seqs"] = vllm_max_num_seqs

                logger.info("vLLM engine parameters: %s", vllm_kwargs)
                self.vllm_model = vLLM_LLM(**vllm_kwargs)
                self.vllm_sampling_params_class = SamplingParams
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("vLLM backend loaded successfully")
                # HF model attributes set to None for vLLM backend
                self.model = None
                self._generation_attn_impl = None
            except ImportError:
                raise ImportError(
                    "vLLM backend requires the vllm package. "
                    "Install it with: pip install vllm"
                )
        else:
            # HuggingFace backend: existing logic
            self._generation_attn_impl = _resolve_attn_implementation(model_name)
            logger.info(f"Using attention implementation: {self._generation_attn_impl}")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation=self._generation_attn_impl,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # vLLM attributes set to None for HF backend
            self.vllm_model = None
            self.vllm_sampling_params_class = None
    

    def switch_attn_implementation(self, mode: str):
        """Switch attention implementation using transformers' built-in API.
            mode can be 'confidence' or 'cot'
        """
        if self.backend == "vllm":
            raise RuntimeError(
                "Cannot switch attention implementation with vLLM backend. "
                "Confidence scoring requires HuggingFace backend."
            )

        if mode == "confidence":
            if 'qwen' in self.model_name.lower():
                self.model.set_attn_implementation("sdpa")
            elif 'gpt' in self.model_name.lower():
                self.model.set_attn_implementation("eager")
            else:
                self.model.set_attn_implementation("sdpa")
        elif mode == "cot":
            self.model.set_attn_implementation(self._generation_attn_impl)
        else:
            raise ValueError(f"Unknown mode {mode} for attention implementation switch.")



    def generate_one(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        output_scores: bool = False,
        has_assistant_prefill: bool = True,
        stop_strings: list[str] | None = None,
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
        if self.backend == "vllm":
            return self._generate_one_vllm(messages, max_new_tokens, temperature, has_assistant_prefill, stop_strings)
        else:
            return self._generate_one_hf(messages, max_new_tokens, temperature, output_scores, has_assistant_prefill, stop_strings)

    def _generate_one_hf(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        output_scores: bool = False,
        has_assistant_prefill: bool = True,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        """HuggingFace backend implementation of generate_one."""

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

        # # Qwen3.5 has a custom DynamicCache implementation that is required for correct attention behavior.
        # if hasattr(self.model.config, "layer_types") and "linear_attention" in self.model.config.layer_types:
        #     cache = Qwen3_5DynamicCache(config=self.model.config)
        # else:
        #     cache = DynamicCache()

        # cache = DynamicCache(config=self.model.config)

        # stop_criteria = StoppingCriteriaList([
        #     _StopAfterBoxedAnswer(self.tokenizer, prompt_len),
        # ])

        if stop_strings:
            stop_criteria = StoppingCriteriaList([
                StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop_strings),
            ])
        else:
            stop_criteria = None

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                # past_key_values=cache,
                use_cache=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=output_scores,
                stopping_criteria=stop_criteria,
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

    def _generate_one_vllm(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        has_assistant_prefill: bool = True,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        """vLLM backend implementation of generate_one."""

        if has_assistant_prefill:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )

        # Post processing (same as HF backend)
        if "qwen" in self.model_name.lower():
            prompt_text = re.sub(r"<think>\s*</think>\s*", "", prompt_text)

        if "gpt" in self.model_name.lower() and self.thinking:
            target = "<|channel|>final"
            last_idx = prompt_text.rfind(target)
            if last_idx != -1:
                prompt_text = prompt_text[:last_idx] + "<|channel|>analysis" + prompt_text[last_idx + len(target):]

        logger.info(prompt_text)

        # Create sampling params
        sampling_params = self.vllm_sampling_params_class(
            temperature=temperature if temperature > 0.0 else 0.0,
            max_tokens=max_new_tokens,
            stop=stop_strings if stop_strings else None,
        )

        # Generate with vLLM
        outputs = self.vllm_model.generate([prompt_text], sampling_params)
        output = outputs[0]

        # Extract generated text
        generated_text = output.outputs[0].text

        # Compute token positions
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        generated_ids = output.outputs[0].token_ids
        total_len = prompt_len + len(generated_ids)

        # Convert to tensors for compatibility
        generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)
        prompt_tail_ids = torch.tensor(prompt_ids[max(0, prompt_len - 10):], dtype=torch.long)

        return GenerationResult(
            prompt_text=prompt_text,
            generated_text=generated_text,
            prompt_end_position=prompt_len,
            generated_end_position=total_len,
            past_key_values=None,  # vLLM does not expose KV cache
            prompt_tail_ids=prompt_tail_ids,
            scores=None,
            generated_ids=generated_ids_tensor,
        )

    def generate_raw(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        output_scores: bool = False,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        """Generate from an already-formatted prompt string, bypassing apply_chat_template."""
        if self.backend == "vllm":
            return self._generate_raw_vllm(prompt_text, max_new_tokens, temperature, stop_strings)
        else:
            return self._generate_raw_hf(prompt_text, max_new_tokens, temperature, output_scores, stop_strings)

    def _generate_raw_hf(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        output_scores: bool = False,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        """HuggingFace backend implementation of generate_raw."""

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len: int = inputs["input_ids"].shape[1]

        if stop_strings:
            stop_criteria = StoppingCriteriaList([
                StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop_strings),
            ])
        else:
            stop_criteria = None

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                use_cache=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=output_scores,
                stopping_criteria=stop_criteria,
            )

        total_len: int = outputs.sequences.shape[1]
        generated_ids = outputs.sequences[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        generated_ids_tensor = outputs.sequences[0, prompt_len:].cpu()
        prompt_tail_ids = outputs.sequences[0, max(0, prompt_len - 10):prompt_len].cpu()

        return GenerationResult(
            prompt_text=prompt_text,
            generated_text=generated_text,
            prompt_end_position=prompt_len,
            generated_end_position=total_len,
            past_key_values=outputs.past_key_values,
            prompt_tail_ids=prompt_tail_ids,
            scores=outputs.scores if output_scores else None,
            generated_ids=generated_ids_tensor,
        )

    def _generate_raw_vllm(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_strings: list[str] | None = None,
    ) -> GenerationResult:
        """vLLM backend implementation of generate_raw."""

        logger.info(prompt_text)

        # Create sampling params
        sampling_params = self.vllm_sampling_params_class(
            temperature=temperature if temperature > 0.0 else 0.0,
            max_tokens=max_new_tokens,
            stop=stop_strings if stop_strings else None,
        )

        # Generate with vLLM
        outputs = self.vllm_model.generate([prompt_text], sampling_params)
        output = outputs[0]

        # Extract generated text
        generated_text = output.outputs[0].text

        # Compute token positions
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        generated_ids = output.outputs[0].token_ids
        total_len = prompt_len + len(generated_ids)

        # Convert to tensors for compatibility
        generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)
        prompt_tail_ids = torch.tensor(prompt_ids[max(0, prompt_len - 10):], dtype=torch.long)

        return GenerationResult(
            prompt_text=prompt_text,
            generated_text=generated_text,
            prompt_end_position=prompt_len,
            generated_end_position=total_len,
            past_key_values=None,  # vLLM does not expose KV cache
            prompt_tail_ids=prompt_tail_ids,
            scores=None,
            generated_ids=generated_ids_tensor,
        )


# class EosListStoppingCriteria(StoppingCriteria):
#     def __init__(self, eos_sequence = []):
#         self.eos_sequence = eos_sequence

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
#         return self.eos_sequence in last_ids


