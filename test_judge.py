"""Quick smoke test for the HLE LLM judge.

Loads .env, sends a single judge call, and prints the result.

Usage:
    python test_judge.py
    python test_judge.py --judge_model "o3-mini-2025-01-31"
    python test_judge.py --base_url "http://localhost:8000/v1" --api_key "dummy"
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Make eval/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

from hle_eval import _judge_response, _get_client, evaluate_hle_async, compute_hle_metrics, DEFAULT_JUDGE_MODEL


# A simple test case with a known correct answer
TEST_QUESTION = "What is 7 * 8?"
TEST_RESPONSE = "Let me think step by step. 7 * 8 = 56. The answer is 56."
TEST_CORRECT_ANSWER = "56"


async def main(args: argparse.Namespace) -> None:
    print("=== HLE Judge Smoke Test ===\n")
    print(f"OPENAI_BASE_URL : {os.environ.get('OPENAI_BASE_URL', '(not set)')}")
    print(f"OPENAI_API_KEY  : {'***' + os.environ['OPENAI_API_KEY'][-4:] if os.environ.get('OPENAI_API_KEY') else '(not set)'}")
    print(f"Judge model     : {args.judge_model}")
    print()

    client = _get_client(base_url=args.base_url, api_key=args.api_key)

    # --- Test 1: correct answer ---
    print("[Test 1] Correct answer")
    print(f"  Question : {TEST_QUESTION}")
    print(f"  Response : {TEST_RESPONSE}")
    print(f"  Expected : {TEST_CORRECT_ANSWER}")

    result = await _judge_response(
        TEST_QUESTION,
        TEST_CORRECT_ANSWER,
        TEST_RESPONSE,
        judge_model=args.judge_model,
        client=client,
    )

    if result is None:
        print("  FAILED - judge returned None (API error)\n")
    else:
        print(f"  Extracted: {result['model_answer']}")
        print(f"  Correct  : {result['correct']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Usage    : {result['usage']}")
        passed = result["correct"] == "yes"
        print(f"  {'PASS' if passed else 'FAIL'} - expected 'yes', got '{result['correct']}'")
        print(f"  Raw dict keys: {list(result.keys())}\n")

    # --- Test 2: wrong answer ---
    wrong_response = "I think the answer is 42."
    print("[Test 2] Wrong answer")
    print(f"  Question : {TEST_QUESTION}")
    print(f"  Response : {wrong_response}")
    print(f"  Expected : {TEST_CORRECT_ANSWER}")

    result2 = await _judge_response(
        TEST_QUESTION,
        TEST_CORRECT_ANSWER,
        wrong_response,
        judge_model=args.judge_model,
        client=client,
    )

    if result2 is None:
        print("  FAILED - judge returned None (API error)\n")
    else:
        print(f"  Extracted: {result2['model_answer']}")
        print(f"  Correct  : {result2['correct']}")
        print(f"  Reasoning: {result2['reasoning']}")
        print(f"  Usage    : {result2['usage']}")
        passed2 = result2["correct"] == "no"
        print(f"  {'PASS' if passed2 else 'FAIL'} - expected 'no', got '{result2['correct']}'")
        print(f"  Raw dict keys: {list(result2.keys())}\n")

    # --- Test 3: end-to-end evaluate_hle_async + compute_hle_metrics ---
    print("[Test 3] End-to-end: evaluate_hle_async -> compute_hle_metrics")
    trajectories = [
        {"question": TEST_QUESTION, "generated_text": TEST_RESPONSE, "ground_truth": TEST_CORRECT_ANSWER},
        {"question": TEST_QUESTION, "generated_text": wrong_response, "ground_truth": TEST_CORRECT_ANSWER},
    ]
    eval_results = []
    for traj in trajectories:
        r = await evaluate_hle_async(traj, judge_model=args.judge_model, client=client)
        eval_results.append(r)

    metrics = compute_hle_metrics(eval_results, total_questions=len(trajectories))
    print(f"  Metrics:\n{json.dumps(metrics, indent=4)}\n")

    # Verify token fields exist and are positive
    token_fields = ["total_prompt_tokens", "total_completion_tokens", "total_tokens"]
    all_present = all(f in metrics for f in token_fields)
    all_positive = all(metrics.get(f, 0) > 0 for f in token_fields)
    print(f"  Token fields present : {all_present}")
    print(f"  Token fields > 0     : {all_positive}")
    print(f"  {'PASS' if all_present and all_positive else 'FAIL'}\n")

    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for the HLE judge")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--base_url", default=None, help="Override OPENAI_BASE_URL")
    parser.add_argument("--api_key", default=None, help="Override OPENAI_API_KEY")
    args = parser.parse_args()
    asyncio.run(main(args))
