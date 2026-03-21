"""
Verify that the single-token assumption in dropout_indirectlogits holds
for all three models.

For P(True) / P(Yes) probing, we assume that tokens like " True", " False",
" Yes", " No" (and their variants) each encode to exactly one token.
This script checks that assumption for every model in MODEL_DICT.

Usage:
    python -m tests.test_single_token_assumption
"""

from transformers import AutoTokenizer
from confidence import ANSWER_TOKENS
from utils.enum import MODEL_DICT


INDIRECT_KEYS = [" Yes", " No", " True", " False"]


def check_model(model_key: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"Model: {model_key} ({model_name})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_pass = True

    for key in INDIRECT_KEYS:
        variants = ANSWER_TOKENS[key]
        for token_str in variants:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            status = "OK" if len(ids) == 1 else "MULTI-TOKEN"
            if len(ids) != 1:
                all_pass = False
            decoded = [tokenizer.decode([tid]) for tid in ids]
            print(f"  {status:12s} | {token_str!r:10s} -> {ids}  {decoded}")

    if all_pass:
        print(f"\n  PASS: all indirect-logit tokens are single-token for {model_key}")
    else:
        print(f"\n  FAIL: some tokens are multi-token for {model_key}")

    return all_pass


def main():
    results = {}
    for model_key, model_name in MODEL_DICT.items():
        results[model_key] = check_model(model_key, model_name)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for model_key, passed in results.items():
        print(f"  {model_key:10s}: {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        print("\nAll models pass the single-token assumption.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\nWARNING: assumption violated for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
