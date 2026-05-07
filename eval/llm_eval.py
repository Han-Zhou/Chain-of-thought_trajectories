"""LLM-as-judge re-evaluation for trajectories marked incorrect.

For each traj_*.json in a directory, if trajectory['correct'] is False,
call gpt-5-mini to re-judge whether the model's final_answer matches the
ground truth. Tracks token usage & estimated cost, and writes a summary.

Usage:
    python -m eval.llm_eval <trajectory_dir>
    python eval/llm_eval.py <trajectory_dir>
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys

from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_JUDGE_MODEL = "gpt-5-mini"
JUDGE_MAX_COMPLETION_TOKENS = 4096

# Third-party API provider (matches ~/.codex/config.toml "yibu" provider)
DEFAULT_JUDGE_BASE_URL = "https://yibuapi.com/v1"
DEFAULT_JUDGE_API_KEY_ENV = "YIBU_API_KEY"

# gpt-5-mini pricing (USD per 1M tokens); override via env if needed.
PRICE_PER_M_PROMPT = float(os.environ.get("GPT5_MINI_PRICE_PROMPT", "0.5"))
PRICE_PER_M_COMPLETION = float(os.environ.get("GPT5_MINI_PRICE_COMPLETION", "4.0"))


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

Answer 'yes' if in the context of the question, [response] matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, non-equivalency, or if the extracted answer is incorrect.

Put your final answer in \\boxed{{}}.

"""


_BOXED_RE = re.compile(r"\\boxed\{\s*(yes|no)\s*\}", re.IGNORECASE)


def _extract_boxed_yes_no(text: str) -> str | None:
    if not text:
        return None
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].lower()
    # Fallback: look for trailing yes/no
    m = re.search(r"\b(yes|no)\b\s*\.?\s*$", text.strip(), re.IGNORECASE)
    return m.group(1).lower() if m else None


def _get_client() -> AsyncOpenAI:
    base_url = (
        os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_JUDGE_BASE_URL
    )
    api_key = (
        os.environ.get(DEFAULT_JUDGE_API_KEY_ENV)
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        raise RuntimeError(
            f"No judge API key found. Set ${DEFAULT_JUDGE_API_KEY_ENV} or $OPENAI_API_KEY."
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=300.0, max_retries=1)


def _normalize_correct(correct) -> bool | None:
    """Collapse `correct` (bool, dict, or other) into a tri-state bool.

    - bool -> itself
    - dict -> True iff it looks like a passing result (e.g. HotpotQA em=True)
    - anything else -> None (unknown)
    """
    if isinstance(correct, bool):
        return correct
    if isinstance(correct, dict):
        if "em" in correct:
            return bool(correct["em"])
        if "score" in correct and isinstance(correct["score"], (int, float, bool)):
            return bool(correct["score"])
        return None
    return None


def _extract_fields(data: dict) -> tuple[str | None, str | None, str | None, bool | None, object, str | None]:
    """Return (question, ground_truth, final_answer, correct_bool, correct_raw, generated_text)."""
    traj = data.get("trajectory") if isinstance(data.get("trajectory"), dict) else data
    question = data.get("question") or traj.get("question")
    ground_truth = traj.get("ground_truth")
    final_answer = traj.get("final_answer")
    correct_raw = traj.get("correct")
    correct = _normalize_correct(correct_raw)
    generated_text = traj.get("generated_text") or final_answer
    return question, ground_truth, final_answer, correct, correct_raw, generated_text


async def _judge(
    client: AsyncOpenAI,
    question: str,
    correct_answer: str,
    response: str,
    judge_model: str,
) -> dict | None:
    prompt = JUDGE_PROMPT.format(
        question=question, correct_answer=correct_answer, response=response
    )
    try:
        completion = await client.chat.completions.create(
            model=judge_model,
            max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print(f"[judge_error] {type(e).__name__}: {e}", file=sys.stderr)
        return {"error": f"{type(e).__name__}: {e}"}

    msg = completion.choices[0].message
    content = msg.content or ""
    verdict = _extract_boxed_yes_no(content)
    usage = completion.usage
    return {
        "raw_response": content,
        "correct": verdict if verdict else "no",
        "parse_ok": verdict is not None,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
    }


async def _process_file(
    client: AsyncOpenAI,
    path: str,
    judge_model: str,
    semaphore: asyncio.Semaphore,
    write: bool,
) -> dict:
    with open(path) as f:
        data = json.load(f)
    question, gt, final_answer, correct, correct_raw, generated_text = _extract_fields(data)

    record = {
        "file": os.path.basename(path),
        "original_correct": correct,
        "original_correct_raw": correct_raw,
        "ground_truth": gt,
        "final_answer": final_answer,
        "judged": False,
    }

    if correct is not False:
        record["skipped_reason"] = "original_correct_not_false"
        return record
    if not question or gt is None or not generated_text:
        record["skipped_reason"] = "missing_fields"
        return record

    async with semaphore:
        result = await _judge(client, question, str(gt), str(generated_text), judge_model)

    if result is None or "error" in (result or {}):
        record["judge_error"] = result.get("error") if result else "unknown"
        return record

    record["judged"] = True
    record["judge_response"] = result
    record["judge_correct"] = result["correct"] == "yes"

    if write:
        traj = data.get("trajectory") if isinstance(data.get("trajectory"), dict) else data
        traj["llm_judge"] = {
            "model": judge_model,
            "correct": result["correct"],
            "raw_response": result["raw_response"],
            "parse_ok": result["parse_ok"],
            "usage": result["usage"],
        }
        # Overwrite the correct field with the judge's verdict
        traj["correct"] = result["correct"] == "yes"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return record


async def _with_index(index: int, task: asyncio.Task) -> tuple[int, dict]:
    return index, await task


async def run_async(
    directory: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    num_workers: int = 32,
    write: bool = True,
) -> dict:
    files = sorted(glob.glob(os.path.join(directory, "traj_*.json")))
    if not files:
        raise SystemExit(f"No traj_*.json files found in {directory}")

    client = _get_client()
    semaphore = asyncio.Semaphore(num_workers)
    tasks = [
        asyncio.create_task(_process_file(client, p, judge_model, semaphore, write))
        for p in files
    ]
    indexed_tasks = [
        asyncio.create_task(_with_index(idx, task)) for idx, task in enumerate(tasks)
    ]
    records: list[dict | None] = [None] * len(tasks)

    if tqdm is not None:
        iterator = tqdm(
            asyncio.as_completed(indexed_tasks),
            total=len(indexed_tasks),
            desc="Judging trajectories",
            unit="file",
        )
        for done_task in iterator:
            idx, result = await done_task
            records[idx] = result
    else:
        completed = 0
        total = len(indexed_tasks)
        print(f"[progress] judging {total} files...")
        for done_task in asyncio.as_completed(indexed_tasks):
            idx, result = await done_task
            records[idx] = result
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"[progress] {completed}/{total} files processed")

    # Typing guard: all entries are filled after awaiting all tasks above.
    records = [r for r in records if r is not None]

    total_files = len(records)
    judged = [r for r in records if r.get("judged")]
    errored = [r for r in records if r.get("judge_error")]
    judge_yes = sum(1 for r in judged if r["judge_correct"])
    originally_correct = sum(1 for r in records if r["original_correct"] is True)
    originally_incorrect = sum(1 for r in records if r["original_correct"] is False)

    total_prompt = sum(r["judge_response"]["usage"]["prompt_tokens"] for r in judged)
    total_completion = sum(r["judge_response"]["usage"]["completion_tokens"] for r in judged)
    total_tokens = sum(r["judge_response"]["usage"]["total_tokens"] for r in judged)
    est_cost = (
        total_prompt / 1_000_000 * PRICE_PER_M_PROMPT
        + total_completion / 1_000_000 * PRICE_PER_M_COMPLETION
    )

    new_correct_count = originally_correct + judge_yes
    accuracy = round(100 * new_correct_count / total_files, 2) if total_files else 0.0

    summary = {
        "directory": directory,
        "judge_model": judge_model,
        "total_files": total_files,
        "originally_correct": originally_correct,
        "originally_incorrect": originally_incorrect,
        "judged": len(judged),
        "judge_errors": len(errored),
        "judge_error_sample": errored[0]["judge_error"] if errored else None,
        "judge_rescued": judge_yes,
        "new_correct_count": new_correct_count,
        "accuracy_after_judge": accuracy,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(est_cost, 6),
        "records": records,
    }

    if write:
        out_path = os.path.join(directory, "llm_judge_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ok] wrote summary to {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Path to a trajectory directory containing traj_*.json")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true", help="Do not modify trajectory files or write summary")
    args = parser.parse_args()

    summary = asyncio.run(
        run_async(
            args.directory,
            judge_model=args.judge_model,
            num_workers=args.num_workers,
            write=not args.dry_run,
        )
    )
    printable = {k: v for k, v in summary.items() if k != "records"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
