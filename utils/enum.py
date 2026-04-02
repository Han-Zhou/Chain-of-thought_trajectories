from enum import Enum
from types import MappingProxyType

MODEL_DICT = MappingProxyType({
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt": "openai/gpt-oss-20b",
    "qwen-fp8": "Qwen/Qwen3.5-27B-FP8",
    "qwen": "Qwen/Qwen3.5-27B",
    "qwen-gptq": "Qwen/Qwen3.5-27B-GPTQ-Int4"
})    


LETTERS = "ABCD"

