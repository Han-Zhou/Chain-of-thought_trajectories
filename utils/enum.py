from enum import Enum
from types import MappingProxyType

MODEL_DICT = MappingProxyType({
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt": "openai/gpt-oss-20b",
    "qwen": "Qwen/Qwen3-32B"
})    

