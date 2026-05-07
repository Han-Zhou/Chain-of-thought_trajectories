# vLLM Backend Integration

This document describes the vLLM backend integration for faster inference in the CoT evaluation pipeline.

## Overview

The pipeline now supports two backends:
- **HuggingFace (default)**: Full feature support including confidence scoring
- **vLLM**: 5-50x faster generation, but incompatible with confidence scoring

## Usage

### Basic Generation (No Confidence)

```bash
# Use vLLM for fast generation
python3 main.py --dataset math500 --model llama --backend vllm --sample_size 100

# Compare with HuggingFace
python3 main.py --dataset math500 --model llama --backend hf --sample_size 100
```

### With Confidence Scoring

```bash
# Automatically falls back to HuggingFace (with warning)
python3 main.py --dataset math500 --model llama --backend vllm --confidence --sample_size 10

# Explicit HuggingFace backend
python3 main.py --dataset math500 --model llama --backend hf --confidence --sample_size 10
```

### Multi-Sample CoT

```bash
# vLLM generates all samples in parallel (much faster)
python3 main.py --dataset math500 --model llama --backend vllm \
    --nb_cot_samples 5 --temperature 0.7 --sample_size 50
```

## Installation

```bash
pip install vllm>=0.6.0
```

vLLM will automatically detect and use available GPUs.

## Performance

Expected speedups with vLLM backend:

| Workload | Speedup |
|----------|---------|
| Single-sample generation (sequential) | 5-10x |
| Multi-sample CoT (N samples per prompt) | 10-20x |
| Batch processing (future optimization) | 20-50x |

Actual speedup depends on:
- Model size and quantization
- GPU memory and compute capability
- Prompt length and max_new_tokens
- Batch size (for future batched implementation)

## Limitations

### 1. Confidence Scoring Not Supported

vLLM does not expose KV cache or support custom attention masks, which are required for confidence scoring. When `--confidence` is set with `--backend vllm`, the system automatically falls back to HuggingFace.

**Why?** Confidence scoring requires:
- Direct access to KV cache for reuse across dropout samples
- Custom attention masks for step masking
- Attention implementation switching (eager/sdpa/flash)
- Forward hooks for bootstrap sampling

None of these are available in vLLM's API.

### 2. Multi-Phase Generation

The current implementation of `generate_qwen_outside_cot()` and `generate_continuation()` works with vLLM but processes entries sequentially. A future optimization could batch phase-1 across all entries, then batch phase-2, etc.

### 3. Quantization

vLLM supports FP8/AWQ/GPTQ quantization, but the current implementation uses the default dtype (bfloat16). Add quantization support if needed:

```python
# In llm.py, modify vLLM initialization:
self.vllm_model = vLLM_LLM(
    model=model_name,
    dtype="bfloat16",
    quantization="awq",  # or "fp8", "gptq"
    trust_remote_code=True,
)
```

## Architecture

### Backend Abstraction

The `LLM` class now supports backend selection via the `backend` parameter:

```python
llm = LLM(model_name, thinking, backend="vllm")  # or "hf"
```

Internally:
- `generate_one()` and `generate_raw()` dispatch to backend-specific implementations
- `_generate_one_vllm()` / `_generate_one_hf()` handle the actual generation
- `switch_attn_implementation()` raises an error for vLLM backend

### GenerationResult Compatibility

The `GenerationResult` dataclass now allows `past_key_values=None` for vLLM:

```python
@dataclass
class GenerationResult:
    ...
    past_key_values: DynamicCache | Qwen3_5DynamicCache | None
```

vLLM backend returns `None` for `past_key_values` since it doesn't expose the KV cache.

### Automatic Fallback

When `--backend vllm` is used with `--confidence`, the argument parser automatically switches to `backend="hf"` and logs a warning:

```
WARNING - vLLM backend is incompatible with confidence scoring. Falling back to HuggingFace backend.
```

## Testing

Run the integration test:

```bash
python3 test_vllm_integration.py
```

This tests:
1. vLLM backend loads successfully
2. `generate_one()` returns valid output
3. `generate_raw()` works with stop strings
4. Confidence scoring raises appropriate error
5. HuggingFace backend still works

## Future Optimizations

### Batch Processing (Phase 3)

The current implementation processes entries sequentially. A future optimization could:

1. Collect all prompts upfront
2. Call `vllm_model.generate(all_prompts, sampling_params)` once
3. Parse all outputs and write trajectory files

This would provide 20-50x speedup but requires:
- Higher memory usage (all prompts in memory)
- More complex error handling
- Restructuring the main loop

### Distributed Inference

vLLM supports tensor parallelism for multi-GPU inference:

```python
self.vllm_model = vLLM_LLM(
    model=model_name,
    dtype="bfloat16",
    tensor_parallel_size=4,  # Use 4 GPUs
    trust_remote_code=True,
)
```

## Troubleshooting

### ImportError: No module named 'vllm'

Install vLLM:
```bash
pip install vllm>=0.6.0
```

### CUDA out of memory

vLLM uses more GPU memory upfront for KV cache allocation. Try:
- Reducing `max_new_tokens`
- Using a smaller model
- Enabling quantization

### Different outputs between HF and vLLM

At `temperature=0.0`, outputs should be identical. If they differ:
- Check that both use the same tokenizer
- Verify stop strings are handled correctly
- Check for model-specific chat template differences

### Confidence scoring fails with vLLM

This is expected. Use `--backend hf` for confidence scoring, or omit `--confidence` when using vLLM.

## Files Modified

1. **`utils/structures.py`**: Made `past_key_values` optional
2. **`llm.py`**: Added backend selection and vLLM implementation
3. **`main.py`**: Added `--backend` CLI argument and validation
4. **`confidence.py`**: Added KV cache check with helpful error message

## Summary

The vLLM integration provides significant speedups for generation-only workloads while maintaining full backward compatibility with HuggingFace for confidence scoring. The implementation is minimally invasive and provides automatic fallback when needed.
