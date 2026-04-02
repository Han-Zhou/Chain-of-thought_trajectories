"""Unit tests for the eval package."""

import asyncio

import numpy as np
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from eval.extractors import extract_mcq_letter, extract_text_answer, extract_boxed_or_text
from eval.comparators import exact_match, normalized_text_match, qa_f1_score
from eval.registry import evaluate_one, EVAL_REGISTRY, UNSUPPORTED
from eval.hle_eval import calib_err, compute_hle_metrics, evaluate_hle_async


# ===================================================================
# Extractor tests
# ===================================================================

class TestExtractMCQLetter:
    def test_basic(self):
        assert extract_mcq_letter("Step 1: blah\n\\boxed{B}") == "B"

    def test_lowercase(self):
        assert extract_mcq_letter("reasoning...\n\\boxed{c}") == "C"

    def test_with_brackets(self):
        assert extract_mcq_letter("reasoning...\n\\boxed{[D]}") == "D"

    def test_with_parens(self):
        assert extract_mcq_letter("reasoning...\n\\boxed{(A)}") == "A"

    def test_no_marker(self):
        assert extract_mcq_letter("The answer is probably B") is None

    def test_no_letter(self):
        assert extract_mcq_letter("\\boxed{Yes}") is None


class TestExtractTextAnswer:
    def test_basic(self):
        assert extract_text_answer("Reasoning...\n\\boxed{Yes}") == "Yes"

    def test_multiword(self):
        result = extract_text_answer("Blah\n\\boxed{The Shining}")
        assert result == "The Shining"

    def test_last_occurrence(self):
        text = "\\boxed{wrong}\nMore reasoning\n\\boxed{correct}"
        assert extract_text_answer(text) == "correct"

    def test_no_marker(self):
        assert extract_text_answer("Just some text without answer") is None

    def test_empty_after_marker(self):
        assert extract_text_answer("\\boxed{   }") is None

    def test_case(self):
        assert extract_text_answer("reasoning\n\\boxed{Delhi}") == "Delhi"


class TestExtractBoxedOrText:
    def test_boxed(self):
        text = "Step 1: ...\n\\boxed{42}"
        assert extract_boxed_or_text(text) == "42"

    def test_boxed_fraction(self):
        text = "reasoning\n\\boxed{\\frac{1}{2}}"
        assert extract_boxed_or_text(text) == "\\frac{1}{2}"

    def test_no_boxed_fallback_dollar(self):
        text = "reasoning\n\\boxed{$3x + 5$}"
        assert extract_boxed_or_text(text) == "3x + 5"

    def test_no_boxed_fallback_plain(self):
        text = "reasoning\n\\boxed{42}"
        assert extract_boxed_or_text(text) == "42"

    def test_no_marker(self):
        assert extract_boxed_or_text("No answer here") is None


# ===================================================================
# Comparator tests
# ===================================================================

class TestExactMatch:
    def test_same(self):
        assert exact_match("A", "A") is True

    def test_case_insensitive(self):
        assert exact_match("a", "A") is True

    def test_whitespace(self):
        assert exact_match("  B ", "B") is True

    def test_different(self):
        assert exact_match("A", "B") is False

    def test_yes_no(self):
        assert exact_match("Yes", "yes") is True
        assert exact_match("No", "Yes") is False


class TestNormalizedTextMatch:
    def test_same(self):
        assert normalized_text_match("The Shining", "the shining") is True

    def test_articles(self):
        assert normalized_text_match("The Matrix", "Matrix") is True

    def test_punctuation(self):
        assert normalized_text_match("hello, world!", "hello world") is True

    def test_different(self):
        assert normalized_text_match("Star Wars", "Star Trek") is False


class TestQaF1Score:
    def test_perfect(self):
        assert qa_f1_score("the cat", "the cat") == 1.0

    def test_partial(self):
        score = qa_f1_score("the cat sat", "a cat sat on mat")
        assert 0 < score < 1

    def test_no_overlap(self):
        assert qa_f1_score("hello", "world") == 0


# ===================================================================
# Registry / evaluate_one tests
# ===================================================================

class TestEvaluateOne:
    def test_logiqa_correct(self):
        traj = {"generated_text": "Step 1: blah\n\\boxed{A}", "ground_truth": "A"}
        result = evaluate_one(traj, "logiqa")
        assert result["score"] == 1.0
        assert result["prediction"] == "A"
        assert not result["extraction_failed"]

    def test_logiqa_wrong(self):
        traj = {"generated_text": "Step 1: blah\n\\boxed{B}", "ground_truth": "A"}
        result = evaluate_one(traj, "logiqa")
        assert result["score"] == 0.0

    def test_extraction_failure(self):
        traj = {"generated_text": "No answer marker here", "ground_truth": "A"}
        result = evaluate_one(traj, "logiqa")
        assert result["extraction_failed"] is True
        assert result["score"] == 0.0

    def test_unsupported_dataset(self):
        traj = {"generated_text": "...", "ground_truth": "..."}
        result = evaluate_one(traj, "cs1qa")
        assert result is None

    def test_unknown_dataset(self):
        traj = {"generated_text": "...", "ground_truth": "..."}
        result = evaluate_one(traj, "nonexistent")
        assert result is None

    def test_bigbench_causal(self):
        traj = {"generated_text": "Reasoning\n\\boxed{Yes}", "ground_truth": "Yes"}
        result = evaluate_one(traj, "bigbench_causal")
        assert result["score"] == 1.0

    def test_hotpotqa(self):
        traj = {"generated_text": "Reasoning\n\\boxed{Delhi}", "ground_truth": "Delhi"}
        result = evaluate_one(traj, "hotpotqa")
        assert result["score"] == 1.0
        assert result["metric"] == "f1"

    def test_math500_plain(self):
        traj = {"generated_text": "reasoning\n\\boxed{42}", "ground_truth": "42"}
        result = evaluate_one(traj, "math500")
        assert result["prediction"] == "42"

    def test_all_supported_datasets_in_registry(self):
        expected = {"logiqa", "college_math", "bigbench_causal", "bigbench_movie",
                    "hotpotqa", "math500", "olympiadbench"}
        assert set(EVAL_REGISTRY.keys()) == expected

    def test_unsupported_set(self):
        assert UNSUPPORTED == {"cs1qa"}


# ===================================================================
# HLE evaluation tests
# ===================================================================

class TestHLERegistry:
    """Test that HLE is properly registered as a special-case evaluator."""

    def test_hle_not_unsupported(self):
        assert "hle" not in UNSUPPORTED

    def test_hle_not_in_registry(self):
        assert "hle" not in EVAL_REGISTRY

    @patch("eval.registry.evaluate_hle")
    def test_evaluate_one_dispatches_to_hle(self, mock_hle):
        mock_hle.return_value = {
            "prediction": "D",
            "ground_truth": "D",
            "score": 1.0,
            "metric": "accuracy",
            "extraction_failed": False,
            "judge_response": {
                "correct_answer": "D",
                "model_answer": "D",
                "reasoning": "match",
                "correct": "yes",
                "confidence": 95,
            },
        }
        traj = {
            "question": "What is the answer?",
            "generated_text": "The answer is D",
            "ground_truth": "D",
        }
        result = evaluate_one(traj, "hle")
        mock_hle.assert_called_once_with(traj)
        assert result["score"] == 1.0
        assert result["prediction"] == "D"


class TestHLEJudgeMocked:
    """Test evaluate_hle_async with mocked judge calls."""

    def test_correct_answer(self):
        mock_judge = AsyncMock(return_value={
            "correct_answer": "B",
            "model_answer": "B",
            "reasoning": "exact match",
            "correct": "yes",
            "confidence": 90,
        })
        with patch("eval.hle_eval._judge_response", mock_judge):
            result = asyncio.run(evaluate_hle_async({
                "question": "Test question?",
                "generated_text": "Answer is B",
                "ground_truth": "B",
            }))
        assert result["score"] == 1.0
        assert result["prediction"] == "B"
        assert not result["extraction_failed"]
        assert result["judge_response"]["confidence"] == 90

    def test_incorrect_answer(self):
        mock_judge = AsyncMock(return_value={
            "correct_answer": "A",
            "model_answer": "C",
            "reasoning": "no match",
            "correct": "no",
            "confidence": 80,
        })
        with patch("eval.hle_eval._judge_response", mock_judge):
            result = asyncio.run(evaluate_hle_async({
                "question": "Test?",
                "generated_text": "I think C",
                "ground_truth": "A",
            }))
        assert result["score"] == 0.0
        assert result["prediction"] == "C"

    def test_missing_question(self):
        with patch("eval.hle_eval._judge_response", AsyncMock()):
            result = asyncio.run(evaluate_hle_async({
                "generated_text": "Some text",
                "ground_truth": "A",
            }))
        assert result["extraction_failed"] is True
        assert result["score"] == 0.0
        assert result["judge_response"] is None

    def test_missing_generated_text(self):
        with patch("eval.hle_eval._judge_response", AsyncMock()):
            result = asyncio.run(evaluate_hle_async({
                "question": "Test?",
                "ground_truth": "A",
            }))
        assert result["extraction_failed"] is True
        assert result["judge_response"] is None

    def test_judge_failure(self):
        mock_judge = AsyncMock(return_value=None)
        with patch("eval.hle_eval._judge_response", mock_judge):
            result = asyncio.run(evaluate_hle_async({
                "question": "Test?",
                "generated_text": "Some answer",
                "ground_truth": "A",
            }))
        assert result["extraction_failed"] is True
        assert result["judge_response"] is None

    def test_extraction_failed_when_none_answer(self):
        mock_judge = AsyncMock(return_value={
            "correct_answer": "A",
            "model_answer": "None",
            "reasoning": "no answer found",
            "correct": "no",
            "confidence": 100,
        })
        with patch("eval.hle_eval._judge_response", mock_judge):
            result = asyncio.run(evaluate_hle_async({
                "question": "Test?",
                "generated_text": "I don't know",
                "ground_truth": "A",
            }))
        assert result["extraction_failed"] is True


class TestCalibErr:
    """Test the calibration error computation."""

    def test_perfect_calibration(self):
        confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        err = calib_err(confidence, correct, p="2", beta=5)
        # With one bin, difference is |mean_conf - mean_correct| = |0.7 - 0.6| = 0.1
        assert abs(err - 0.1) < 1e-6

    def test_all_correct_high_confidence(self):
        confidence = np.array([0.95, 0.98, 0.99])
        correct = np.array([1.0, 1.0, 1.0])
        err = calib_err(confidence, correct, p="2", beta=3)
        # mean_conf ≈ 0.973, mean_correct = 1.0, diff ≈ 0.027
        assert err < 0.05

    def test_empty_input(self):
        err = calib_err(np.array([]), np.array([]), p="2", beta=100)
        assert err == 0.0

    def test_l1_norm(self):
        confidence = np.array([0.5, 0.5])
        correct = np.array([0.0, 1.0])
        err = calib_err(confidence, correct, p="1", beta=2)
        # mean_conf=0.5, mean_correct=0.5, diff=0
        assert abs(err) < 1e-6

    def test_linfty_norm(self):
        confidence = np.array([1.0, 0.0])
        correct = np.array([0.0, 1.0])
        err = calib_err(confidence, correct, p="infty", beta=2)
        # one bin: mean_conf=0.5, mean_correct=0.5, diff=0
        assert abs(err) < 1e-6

    def test_last_bin_included(self):
        """Verify the fix for the upstream bug that dropped the last bin."""
        # 5 elements with beta=3 -> bins: [0,3), [3,5]
        # Last bin should be included.
        confidence = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        correct = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        err = calib_err(confidence, correct, p="1", beta=3)
        # Bin 1: mean_conf=0.2, mean_correct=0.0, diff=0.2, weight=3/5
        # Bin 2: mean_conf=0.925, mean_correct=1.0, diff=0.075, weight=2/5
        # L1: 3/5 * 0.2 + 2/5 * 0.075 = 0.12 + 0.03 = 0.15
        assert abs(err - 0.15) < 1e-6


class TestComputeHLEMetrics:
    """Test the aggregate metrics computation."""

    def test_basic_metrics(self):
        results = [
            {
                "prediction": "A",
                "ground_truth": "A",
                "score": 1.0,
                "metric": "accuracy",
                "extraction_failed": False,
                "judge_response": {
                    "correct_answer": "A",
                    "model_answer": "A",
                    "reasoning": "match",
                    "correct": "yes",
                    "confidence": 90,
                },
            },
            {
                "prediction": "B",
                "ground_truth": "C",
                "score": 0.0,
                "metric": "accuracy",
                "extraction_failed": False,
                "judge_response": {
                    "correct_answer": "C",
                    "model_answer": "B",
                    "reasoning": "no match",
                    "correct": "no",
                    "confidence": 80,
                },
            },
        ]
        metrics = compute_hle_metrics(results, total_questions=10)
        # accuracy = 100 * 1/10 = 10.0
        assert metrics["accuracy"] == 10.0
        assert metrics["total_questions"] == 10
        assert metrics["evaluated_count"] == 2
        assert metrics["judge_successes"] == 2
        assert metrics["judge_failures"] == 0
        assert metrics["extraction_failures"] == 0
        assert metrics["confidence_interval_95"] > 0

    def test_with_judge_failures(self):
        results = [
            {
                "prediction": None,
                "ground_truth": "A",
                "score": 0.0,
                "metric": "accuracy",
                "extraction_failed": True,
                "judge_response": None,
            },
        ]
        metrics = compute_hle_metrics(results, total_questions=5)
        assert metrics["accuracy"] == 0.0
        assert metrics["judge_failures"] == 1
        assert metrics["extraction_failures"] == 1

    def test_denominator_is_total_questions(self):
        """Verify accuracy uses total dataset size, not evaluated count."""
        results = [
            {
                "prediction": "A",
                "ground_truth": "A",
                "score": 1.0,
                "metric": "accuracy",
                "extraction_failed": False,
                "judge_response": {
                    "correct_answer": "A",
                    "model_answer": "A",
                    "reasoning": "match",
                    "correct": "yes",
                    "confidence": 100,
                },
            },
        ]
        # 1 correct out of 100 total = 1% accuracy
        metrics = compute_hle_metrics(results, total_questions=100)
        assert metrics["accuracy"] == 1.0
        assert metrics["evaluated_count"] == 1
