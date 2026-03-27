"""Unit tests for the eval package."""

import pytest
from eval.extractors import extract_mcq_letter, extract_text_answer, extract_boxed_or_text
from eval.comparators import exact_match, normalized_text_match, qa_f1_score
from eval.registry import evaluate_one, EVAL_REGISTRY, UNSUPPORTED


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
        result = evaluate_one(traj, "bfcl")
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
        assert UNSUPPORTED == {"bfcl", "codeqa", "cs1qa", "hle"}
