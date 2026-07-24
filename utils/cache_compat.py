"""Compatibility helpers for Hugging Face cache implementations."""

from transformers import Cache, DynamicCache

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DynamicCache,
    )
except ImportError:
    # Transformers >= 5.10 uses the unified, config-aware DynamicCache.
    Qwen3_5DynamicCache = DynamicCache

