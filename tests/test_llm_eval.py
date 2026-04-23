"""Unit tests for eval.llm_eval (LLM-as-judge re-evaluation)."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.llm_eval import _extract_boxed_yes_no, _extract_fields, _judge, run_async


def _make_traj(correct, final_answer="42", ground_truth="42", question="q?"):
    return {
        "index": 0,
        "question": question,
        "trajectory": {
            "question": question,
            "ground_truth": ground_truth,
            "final_answer": final_answer,
            "generated_text": f"The answer is \\boxed{{{final_answer}}}",
            "correct": correct,
        },
    }


class TestExtractFields:
    def test_reads_nested_trajectory(self):
        data = _make_traj(False, final_answer="7", ground_truth="8", question="what?")
        q, gt, fa, c, gen = _extract_fields(data)
        assert q == "what?"
        assert gt == "8"
        assert fa == "7"
        assert c is False
        assert "7" in gen

    def test_falls_back_to_top_level(self):
        data = {
            "question": "qq",
            "ground_truth": "g",
            "final_answer": "a",
            "generated_text": "txt",
            "correct": True,
        }
        q, gt, fa, c, gen = _extract_fields(data)
        assert q == "qq" and gt == "g" and fa == "a" and c is True and gen == "txt"


class TestExtractBoxedYesNo:
    def test_yes(self):
        assert _extract_boxed_yes_no("reasoning... \\boxed{yes}") == "yes"

    def test_no(self):
        assert _extract_boxed_yes_no("nope \\boxed{no}") == "no"

    def test_case_insensitive(self):
        assert _extract_boxed_yes_no("\\boxed{YES}") == "yes"

    def test_takes_last(self):
        assert _extract_boxed_yes_no("\\boxed{no} then \\boxed{yes}") == "yes"

    def test_fallback_trailing(self):
        assert _extract_boxed_yes_no("the answer matches, yes") == "yes"

    def test_none(self):
        assert _extract_boxed_yes_no("no boxed verdict here maybe") is None
        assert _extract_boxed_yes_no("") is None


def _fake_completion(content, p=50, c=20):
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    usage = MagicMock(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)
    return MagicMock(choices=[choice], usage=usage)


class TestJudge:
    def test_judge_yes(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("reasoning... \\boxed{yes}")
        )
        out = asyncio.run(_judge(client, "q", "42", "resp", "gpt-5-mini"))
        assert out["correct"] == "yes"
        assert out["parse_ok"] is True
        assert "\\boxed{yes}" in out["raw_response"]
        assert out["usage"]["total_tokens"] == 70

    def test_judge_no(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("mismatch \\boxed{no}")
        )
        out = asyncio.run(_judge(client, "q", "42", "resp", "gpt-5-mini"))
        assert out["correct"] == "no"
        assert out["parse_ok"] is True

    def test_judge_unparseable_defaults_no(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_fake_completion("I cannot decide")
        )
        out = asyncio.run(_judge(client, "q", "42", "resp", "gpt-5-mini"))
        assert out["correct"] == "no"
        assert out["parse_ok"] is False

    def test_judge_api_error(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
        out = asyncio.run(_judge(client, "q", "42", "resp", "gpt-5-mini"))
        assert "error" in out


class TestRunAsync:
    def test_end_to_end(self, tmp_path):
        files = {
            "traj_0.json": _make_traj(True, final_answer="42", ground_truth="42"),
            "traj_1.json": _make_traj(False, final_answer="forty-two", ground_truth="42"),
            "traj_2.json": _make_traj(False, final_answer="99", ground_truth="42"),
        }
        for name, data in files.items():
            (tmp_path / name).write_text(json.dumps(data))

        async def fake_create(**kwargs):
            content = kwargs["messages"][0]["content"]
            if "forty-two" in content:
                return _fake_completion("matches \\boxed{yes}", p=10, c=5)
            return _fake_completion("nope \\boxed{no}", p=10, c=5)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        with patch("eval.llm_eval._get_client", return_value=mock_client):
            summary = asyncio.run(
                run_async(str(tmp_path), judge_model="gpt-5-mini", num_workers=4, write=True)
            )

        assert summary["total_files"] == 3
        assert summary["originally_correct"] == 1
        assert summary["originally_incorrect"] == 2
        assert summary["judged"] == 2
        assert summary["judge_rescued"] == 1
        assert summary["new_correct_count"] == 2
        assert summary["total_prompt_tokens"] == 20
        assert summary["total_completion_tokens"] == 10
        assert summary["estimated_cost_usd"] >= 0

        assert os.path.exists(tmp_path / "llm_judge_summary.json")

        rescued = json.loads((tmp_path / "traj_1.json").read_text())
        assert rescued["trajectory"]["llm_judge"]["correct"] == "yes"
        assert rescued["trajectory"]["llm_judge"]["parse_ok"] is True
        assert "raw_response" in rescued["trajectory"]["llm_judge"]
        assert rescued["trajectory"]["correct_llm_judge"] is True

        not_rescued = json.loads((tmp_path / "traj_2.json").read_text())
        assert not_rescued["trajectory"]["llm_judge"]["correct"] == "no"
        assert "correct_llm_judge" not in not_rescued["trajectory"]

        orig = json.loads((tmp_path / "traj_0.json").read_text())
        assert "llm_judge" not in orig["trajectory"]

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            asyncio.run(run_async(str(tmp_path)))
