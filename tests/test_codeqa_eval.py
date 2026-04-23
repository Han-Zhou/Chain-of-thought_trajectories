"""Targeted tests for CodeQA evaluation."""

from eval.codeqa_eval import compute_codeqa_metrics, evaluate_codeqa


class TestEvaluateCodeQA:
    def test_normalizes_inputs(self):
        result = evaluate_codeqa({
            "generated_text": "  Foo   BAR  ",
            "ground_truth": ["baz", "foo bar"],
        })

        assert result["prediction"] == "foo bar"
        assert result["score"] == 1.0
        assert result["metric"] == "f1"
        assert result["scores"]["em"] == 100.0
        assert "meteor" not in result["scores"]
        assert "bleu" not in result["scores"]
        assert "rouge_l" not in result["scores"]

    def test_partial_overlap_keeps_primary_f1(self):
        result = evaluate_codeqa({
            "generated_text": "count marks in string",
            "ground_truth": "marks in string",
        })

        assert result["metric"] == "f1"
        assert 0 < result["score"] < 1.0
        assert result["scores"]["f1"] > 0.0


class TestComputeCodeQAMetrics:
    def test_compute_codeqa_metrics_means(self):
        metrics = compute_codeqa_metrics([
            {
                "scores": {
                    "em": 0.0,
                    "precision": 40.0,
                    "recall": 50.0,
                    "f1": 60.0,
                }
            },
            {
                "scores": {
                    "em": 100.0,
                    "precision": 60.0,
                    "recall": 70.0,
                    "f1": 80.0,
                }
            },
        ])

        assert metrics["em_mean"] == 50.0
        assert metrics["precision_mean"] == 50.0
        assert metrics["recall_mean"] == 60.0
        assert metrics["f1_mean"] == 70.0
        assert "bleu_mean" not in metrics
        assert "rouge_l_mean" not in metrics
        assert "meteor_mean" not in metrics
