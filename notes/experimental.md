# Experimental Notes

## Prompt Types (`--type`)

### Type 1 (default)

The model is guided into chain-of-thought reasoning via **assistant prefilling**. The last message in the chat is an assistant turn that already contains the thinking token opener and the start of the first step:

```
assistant: <think>
Let's think step-by-step.
Step 1:
```

The model then continues generating from that point. In few-shot mode, the example reasoning steps are wrapped inside thinking token blocks (e.g. `<think>...</think>`).

This forces the model into a specific output format from the very first token, giving tight control over structure but also meaning the model never "decides" to reason on its own — it's told to.

### Type 2

There is **no assistant prefill**. Instead, `"Let's think step-by-step."` is appended to the end of the **user prompt**. The model receives a standard `add_generation_prompt` header and generates freely from there.

```
user: ... <original question> ...

Let's think step-by-step.
```

In few-shot mode, the example reasoning steps appear as **plain text** (no thinking token wrappers), so the model sees step-by-step reasoning as ordinary assistant output rather than something gated behind special tokens.

Because the model generates freely, it may emit a `<think>...</think>` block before its visible response. During parsing, the think block is **stripped** so that CoT steps and the final answer are extracted only from the visible output after `</think>`. The raw `generated_text` in the trajectory still contains the full output including the think block.

This tests whether the model can self-organize its CoT when simply asked to, rather than being structurally forced into it.

### Why the distinction matters

Type 1 and Type 2 isolate two variables:

1. **Structural forcing vs. natural generation** — does prefilling the assistant turn with `Step 1:` change the quality or calibration of the reasoning compared to letting the model start from scratch?
2. **Thinking tokens vs. plain text** — when few-shot examples place steps inside thinking blocks, the model may treat reasoning differently (e.g. hidden reasoning in Qwen's `<think>` mode) compared to seeing them as regular output.

Comparing trajectories from both types on the same dataset/model lets us measure the effect of these prompting choices on accuracy and confidence calibration.
