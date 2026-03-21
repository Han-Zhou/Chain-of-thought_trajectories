# Why Qwen Strips `<think>` from Non-Final Assistant Messages

## Official Rationale

From the [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-8B):

> "In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2. However, for frameworks that do not directly use the Jinja2 chat template, it is up to the developers to ensure that the best practice is followed."

## How It Works in the Template

The Jinja2 chat template first splits any assistant message content:

```jinja
{%- if '</think>' in content %}
    {%- set reasoning_content = content.split('</think>')[0]...split('<think>')[-1]... %}
    {%- set content = content.split('</think>')[-1].lstrip('\n') %}
{%- endif %}
```

Then conditionally re-injects the thinking block only for the final assistant turn:

```jinja
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>assistant\n' + content }}
{%- endif %}
```

`ns.last_query_index` is the index of the last user message. Assistant messages before it (i.e., few-shot examples) get thinking stripped; only the final assistant turn preserves it.

## Implied Design Reasons

- **Token efficiency** — past `<think>` blocks can be very long; re-including them wastes context
- **Reasoning is ephemeral scratchpad** — it was never meant to be part of "ground truth" conversation history
- **Stale reasoning prevention** — the model might anchor on outdated intermediate conclusions from earlier turns
- **Mirrors OpenAI API convention** — separates `content` from `reasoning_content`; only `content` goes back into history

## Relevant Issues

- **[Ollama #10448](https://github.com/ollama/ollama/issues/10448)** — confirmed stripping is required behavior, not a bug; Ollama 0.6.7 now does it automatically
- **[QwenLM/Qwen3 #1826](https://github.com/QwenLM/Qwen3/issues/1826)** — the stripping breaks KV-cache reuse (relevant to `DynamicCache` usage): when `enable_thinking=false`, the empty `<think></think>` injected at generation time is not injected into historical turns, so request N's token sequence is not a prefix of request N+1, causing full recompute (~90x slowdown: 36s vs 0.4s)
- **[Qwen/Qwen3-1.7B Discussion #9](https://huggingface.co/Qwen/Qwen3-1.7B/discussions/9)** — bug with multiple assistant messages + `enable_thinking=False` where subsequent turns incorrectly got `<think></think>` injected; one-line fix merged

## Implications for Few-Shot CoT Setup

The template will always strip `<think>` from few-shot assistant turns. Options:

1. **Put CoT steps outside the think block** — plain text before "Final Answer:" so the template doesn't strip it
2. **Embed few-shot examples in the system prompt** as raw text — the template won't strip content in system/user messages
3. **Bypass `apply_chat_template`** — construct the raw token string manually

## Sources

- [The 4 Things Qwen-3's Chat Template Teaches Us (HuggingFace Blog)](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [Qwen3-8B Model Card — Best Practices](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3: Think Deeper, Act Faster — Official Blog](https://qwenlm.github.io/blog/qwen3/)
- [QwenLM/Qwen3 Issue #1826](https://github.com/QwenLM/Qwen3/issues/1826)
- [Ollama Issue #10448](https://github.com/ollama/ollama/issues/10448)
- [QwenLM/Qwen3 Discussion #1429 — Best Practice for Disabling Thinking During SFT](https://github.com/QwenLM/Qwen3/discussions/1429)
