"""HLE (Humanity's Last Exam) evaluation via LLM judge.

Ported from /storage/backup/han/backup_workspace/hle/hle_eval/run_judge_results.py
and adapted to this repo's per-trajectory / batch-evaluation interfaces.
"""

import asyncio
import math
from typing import Literal

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel

from dataloader import load_dataset


DEFAULT_JUDGE_MODEL = "o3-mini-2025-01-31"
JUDGE_MAX_COMPLETION_TOKENS = 4096
CALIBRATION_BETA = 100
CI_Z_SCORE = 1.96

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=300.0, max_retries=1)


async def _judge_response(
    question: str,
    correct_answer: str,
    response: str,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    client: AsyncOpenAI | None = None,
) -> dict | None:
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    client = client or _get_client()
    try:
        completion = await client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
    except Exception:
        return None

    content = completion.choices[0].message.parsed
    return {
        "correct_answer": correct_answer,
        "model_answer": content.extracted_final_answer,
        "reasoning": content.reasoning,
        "correct": content.correct,
        "confidence": content.confidence,
    }


async def evaluate_hle_async(
    trajectory: dict,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    client: AsyncOpenAI | None = None,
) -> dict:
    question = trajectory.get("question")
    generated_text = trajectory.get("generated_text")
    ground_truth = trajectory.get("ground_truth", "")

    if not question or not generated_text:
        return {
            "prediction": None,
            "ground_truth": ground_truth,
            "score": 0.0,
            "metric": "accuracy",
            "extraction_failed": True,
            "judge_response": None,
        }

    judge_response = await _judge_response(
        question,
        str(ground_truth),
        generated_text,
        judge_model=judge_model,
        client=client,
    )
    if judge_response is None:
        return {
            "prediction": None,
            "ground_truth": ground_truth,
            "score": 0.0,
            "metric": "accuracy",
            "extraction_failed": True,
            "judge_response": None,
        }

    score = 1.0 if judge_response["correct"] == "yes" else 0.0
    prediction = judge_response["model_answer"]
    extraction_failed = prediction == "None"

    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "score": score,
        "metric": "accuracy",
        "extraction_failed": extraction_failed,
        "judge_response": judge_response,
    }


def evaluate_hle(trajectory: dict, *, judge_model: str = DEFAULT_JUDGE_MODEL) -> dict:
    return asyncio.run(evaluate_hle_async(trajectory, judge_model=judge_model))


async def evaluate_hle_batch_async(
    trajectories: list[dict],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    num_workers: int = 100,
) -> list[dict]:
    client = _get_client()
    semaphore = asyncio.Semaphore(num_workers)

    async def bound_eval(trajectory: dict) -> dict:
        async with semaphore:
            return await evaluate_hle_async(
                trajectory,
                judge_model=judge_model,
                client=client,
            )

    tasks = [bound_eval(trajectory) for trajectory in trajectories]
    return await asyncio.gather(*tasks)


def calib_err(confidence, correct, p: str = "2", beta: int = CALIBRATION_BETA):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]

    if len(confidence) == 0:
        return 0.0

    bins = [[i * beta, min((i + 1) * beta, len(confidence))] for i in range((len(confidence) + beta - 1) // beta)]

    cerr = 0.0
    total_examples = len(confidence)
    for start, end in bins:
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in {"infty", "infinity", "max"}:
                cerr = np.maximum(cerr, difference)
            else:
                raise AssertionError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = np.sqrt(cerr)

    return float(cerr)


def compute_hle_metrics(results: list[dict], total_questions: int) -> dict:
    correct = []
    confidence = []
    extraction_failures = 0
    judge_failures = 0

    for result in results:
        if result.get("extraction_failed"):
            extraction_failures += 1
        judge_response = result.get("judge_response")
        if judge_response is None:
            judge_failures += 1
            continue
        correct.append(judge_response["correct"] == "yes")
        confidence.append(judge_response["confidence"] / 100)

    correct_arr = np.array(correct, dtype=float)
    confidence_arr = np.array(confidence, dtype=float)

    accuracy = round(100 * correct_arr.sum() / total_questions, 2) if total_questions else 0.0
    ci = round(CI_Z_SCORE * math.sqrt(accuracy * (100 - accuracy) / total_questions), 2) if total_questions else 0.0
    calibration_error = 100 * round(calib_err(confidence_arr, correct_arr, p="2", beta=CALIBRATION_BETA), 2)

    return {
        "metric": "accuracy",
        "total_questions": total_questions,
        "evaluated_count": len(results),
        "judge_successes": len(correct),
        "judge_failures": judge_failures,
        "extraction_failures": extraction_failures,
        "accuracy": accuracy,
        "confidence_interval_95": ci,
        "calibration_error": calibration_error,
    }


def load_hle_total_questions() -> int:
    return len(load_dataset("hle"))
