"""Targeted tests for CodeQA evaluation."""

from unittest.mock import patch

from eval.codeqa_eval import compute_codeqa_metrics, evaluate_codeqa


class TestEvaluateCodeQA:
    @patch("eval.codeqa_eval.compute_meteor_score")
    def test_includes_meteor_and_normalizes_inputs(self, mock_meteor):
        mock_meteor.return_value = 0.123456

        result = evaluate_codeqa({
            "generated_text": "  Foo   BAR  ",
            "ground_truth": ["baz", "foo bar"],
        })

        mock_meteor.assert_called_once_with("foo bar", ["baz", "foo bar"])
        assert result["prediction"] == "foo bar"
        assert result["score"] == 1.0
        assert result["metric"] == "f1"
        assert result["scores"]["em"] == 100.0
        assert result["scores"]["meteor"] == 12.3456

    @patch("eval.codeqa_eval.compute_meteor_score")
    def test_partial_overlap_keeps_primary_f1(self, mock_meteor):
        mock_meteor.return_value = 0.5

        result = evaluate_codeqa({
            "generated_text": "count marks in string",
            "ground_truth": "marks in string",
        })

        assert result["metric"] == "f1"
        assert 0 < result["score"] < 1.0
        assert result["scores"]["meteor"] == 50.0
        assert result["scores"]["f1"] > 0.0


class TestComputeCodeQAMetrics:
    def test_compute_codeqa_metrics_means(self):
        metrics = compute_codeqa_metrics([
            {
                "scores": {
                    "em": 0.0,
                    "bleu": 10.0,
                    "rouge_l": 20.0,
                    "meteor": 30.0,
                    "precision": 40.0,
                    "recall": 50.0,
                    "f1": 60.0,
                }
            },
            {
                "scores": {
                    "em": 100.0,
                    "bleu": 30.0,
                    "rouge_l": 40.0,
                    "meteor": 50.0,
                    "precision": 60.0,
                    "recall": 70.0,
                    "f1": 80.0,
                }
            },
        ])

        assert metrics["em_mean"] == 50.0
        assert metrics["bleu_mean"] == 20.0
        assert metrics["rouge_l_mean"] == 30.0
        assert metrics["meteor_mean"] == 40.0
        assert metrics["precision_mean"] == 50.0
        assert metrics["recall_mean"] == 60.0
        assert metrics["f1_mean"] == 70.0
