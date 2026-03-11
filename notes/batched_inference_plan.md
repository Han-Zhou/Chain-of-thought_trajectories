# Plan: Batched Inference in main.py

## Context

Currently `generate_trajectories()` processes one sample at a time (`batch_size=1`). This underutilises the GPU — the model sits idle between forward passes and the batch dimension is always 1. Adding proper batched inference will increase throughput significantly on any multi-sample run.

The dataloader already supports `batch_size > 1` (returns a list of dicts); the only missing piece is a `generate_batch()` function that tokenises multiple prompts together (with left-padding), calls `model.generate()` once per batch, and slices the outputs back into per-sample results.

---

## Changes

### 1. `main.py` — add `--batch_size` CLI argument (in `parse()`)

```python
args.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Number of samples per forward pass (batched inference).",
)
```

### 2. `main.py` — new `generate_batch()` function

Insert after `generate_one()`. Key steps:

1. **Set `tokenizer.padding_side = "left"`** — decoder-only models must left-pad so all sequences are right-aligned before generation.
2. **Set `tokenizer.pad_token` if `None`** — use `eos_token` as fallback.
3. Apply chat template to each message list → list of prompt strings.
4. Tokenise the full list with `padding=True` → `[B, max_input_len]` tensors.
5. Record `prompt_lens[i] = attention_mask[i].sum()` for each sample (actual non-padded length).
6. Call `model.generate(**inputs, ...)` once — `outputs.sequences` shape: `[B, max_input_len + max_new_tokens]`.
7. For each sample `i`:
   - `input_len = inputs["input_ids"].shape[1]` (max padded length, same for all)
   - `generated_ids = outputs.sequences[i, input_len:]`
   - `generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)`
   - `pad_len = input_len - prompt_lens[i]`
   - `prompt_positions = list(range(pad_len, input_len))`
   - `generated_positions = list(range(input_len, outputs.sequences.shape[1]))`
8. Return `list[GenerationResult]`.

`past_key_values` is set to `None` in batch mode — it is not consumed downstream.

### 3. `main.py` — update `generate_trajectories()`

```python
def generate_trajectories(model_name, dataloader, dataset_name, shot_mode, n_shots=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(...)

    trajectories = []
    for i, batch in enumerate(dataloader):
        entries = [batch] if isinstance(batch, dict) else batch   # normalise
        logger.info(f"Batch {i}/{len(dataloader)} — {len(entries)} sample(s)")

        messages_list = [
            load_messages(dataset_name, few_shot=(shot_mode == "few"), entry=e)
            for e in entries
        ]

        gens = (
            [generate_one(model, tokenizer, messages_list[0])]
            if len(entries) == 1
            else generate_batch(model, tokenizer, messages_list)
        )

        for entry, gen in zip(entries, gens):
            parsed = parse_output(gen.generated_text)
            trajectories.append({
                "id":                        entry["id"],
                "question":                  entry["question"],
                "ground_truth":              entry["answer"],
                "cot_steps":                 parsed.cot_steps,
                "raw_cot_block":             parsed.raw_cot_block,
                "generated_text":            gen.generated_text,
                "prompt_token_positions":    gen.prompt_token_positions,
                "generated_token_positions": gen.generated_token_positions,
            })

    return trajectories
```

### 4. `main.py` — pass `batch_size` through `main()`

```python
dataloader = make_dataloader(dataset, n=sample_size, batch_size=args.batch_size)
trajectories = generate_trajectories(
    model_name, dataloader,
    dataset_name=args.dataset,
    shot_mode=args.shot_mode,
)
```

### 5. `utils/structures.py` — relax `past_key_values` type

```python
past_key_values: DynamicCache | None   # None when running in batch mode
```

---

## Critical Files

| File | Change |
|------|--------|
| `main.py` | All changes above |
| `utils/structures.py` | `past_key_values: DynamicCache | None` |
| `dataloader/__init__.py` | No change needed — already accepts `batch_size` |
| `actual_run.sh` | Optionally add `--batch_size 4` |

---

## Verification

```bash
# batch_size=1 (regression check — must produce same output as before)
python main.py --dataset logiqa --model qwen --sample_size 4 --batch_size 1

# batch_size=4 (new batched path)
python main.py --dataset logiqa --model qwen --sample_size 8 --batch_size 4
```

Checks:
- Output JSON has correct number of entries
- `generated_text` is non-empty for each entry
- `prompt_token_positions` and `generated_token_positions` are non-overlapping
- No CUDA OOM with `batch_size=4` at the current model size (Qwen3.5-27B)
