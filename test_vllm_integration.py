#!/usr/bin/env python3
"""
Quick integration test for vLLM backend.

Tests:
1. vLLM backend loads successfully
2. generate_one() returns valid GenerationResult
3. generate_raw() works with stop strings
4. Confidence scoring with vLLM raises appropriate error
"""

import sys
import logging
from llm import LLM
from utils.structures import GenerationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vllm_loading():
    """Test that vLLM backend loads without errors."""
    logger.info("Test 1: Loading vLLM backend...")
    try:
        llm = LLM("meta-llama/Llama-3.1-8B-Instruct", thinking=False, backend="vllm")
        assert llm.backend == "vllm"
        assert llm.vllm_model is not None
        assert llm.model is None  # HF model should be None
        logger.info("✓ vLLM backend loaded successfully")
        return llm
    except ImportError as e:
        logger.error(f"✗ vLLM not installed: {e}")
        logger.info("Install with: pip install vllm")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Failed to load vLLM backend: {e}")
        raise


def test_generate_one(llm):
    """Test generate_one() with vLLM backend."""
    logger.info("\nTest 2: Testing generate_one()...")
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me think step-by-step.\nStep 1: "}
    ]

    result = llm.generate_one(
        messages,
        max_new_tokens=50,
        temperature=0.0,
        has_assistant_prefill=True,
    )

    assert isinstance(result, GenerationResult)
    assert result.generated_text is not None
    assert len(result.generated_text) > 0
    assert result.past_key_values is None  # vLLM doesn't expose KV cache
    assert result.prompt_end_position > 0
    assert result.generated_end_position > result.prompt_end_position

    logger.info(f"✓ Generated text: {result.generated_text[:100]}...")
    logger.info(f"  Prompt length: {result.prompt_end_position} tokens")
    logger.info(f"  Generated length: {result.generated_end_position - result.prompt_end_position} tokens")


def test_generate_raw(llm):
    """Test generate_raw() with stop strings."""
    logger.info("\nTest 3: Testing generate_raw() with stop strings...")
    prompt = "Count to 5: 1, 2, 3, "

    result = llm.generate_raw(
        prompt,
        max_new_tokens=20,
        temperature=0.0,
        stop_strings=["5"],
    )

    assert isinstance(result, GenerationResult)
    assert result.generated_text is not None
    assert "5" in result.generated_text or result.generated_text.strip().endswith("4")

    logger.info(f"✓ Generated with stop string: {result.generated_text}")


def test_confidence_error(llm):
    """Test that confidence scoring raises appropriate error."""
    logger.info("\nTest 4: Testing confidence scoring error handling...")
    try:
        llm.switch_attn_implementation("confidence")
        logger.error("✗ Should have raised RuntimeError")
        assert False
    except RuntimeError as e:
        assert "vLLM backend" in str(e)
        logger.info(f"✓ Correctly raised error: {e}")


def test_hf_backend_still_works():
    """Test that HF backend still works after changes."""
    logger.info("\nTest 5: Testing HF backend compatibility...")
    llm = LLM("meta-llama/Llama-3.1-8B-Instruct", thinking=False, backend="hf")
    assert llm.backend == "hf"
    assert llm.model is not None
    assert llm.vllm_model is None

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me think.\nStep 1: "}
    ]

    result = llm.generate_one(
        messages,
        max_new_tokens=50,
        temperature=0.0,
        has_assistant_prefill=True,
    )

    assert isinstance(result, GenerationResult)
    assert result.past_key_values is not None  # HF should have KV cache
    logger.info("✓ HF backend still works correctly")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("vLLM Integration Test Suite")
    logger.info("=" * 60)

    try:
        llm = test_vllm_loading()
        test_generate_one(llm)
        test_generate_raw(llm)
        test_confidence_error(llm)
        test_hf_backend_still_works()

        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        sys.exit(1)
