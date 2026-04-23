"""Unit tests for BFCL-specific parsing and evaluation."""

import pytest

from eval.bfcl_eval import (
    default_decode_ast_prompting,
    evaluate_bfcl,
    is_function_calling_format_output,
    multiple_function_checker,
    parse_json_function_call,
)


def make_schema(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "parameters": {
            "type": "dict",
            "properties": properties,
            "required": required,
        },
    }


def make_trajectory(
    generated_text: str,
    ground_truth: list[dict],
    func_description: list[dict],
    test_category: str = "simple_python",
) -> dict:
    return {
        "generated_text": generated_text,
        "ground_truth": ground_truth,
        "func_description": func_description,
        "test_category": test_category,
    }


class TestBFCLParsing:
    def test_default_decode_ast_prompting_wraps_bare_call(self):
        decoded = default_decode_ast_prompting('math.factorial(number=5)')
        assert decoded == [{"math.factorial": {"number": 5}}]

    def test_parse_json_function_call_extracts_embedded_payload(self):
        text = (
            'analysis\n'
            '[{"function": "math.factorial", "parameters": {"number": 5}}]\n'
            'done'
        )
        decoded = parse_json_function_call(text)
        assert decoded == [{"math.factorial": {"number": 5}}]

    def test_is_function_calling_format_output_validates_shape(self):
        assert is_function_calling_format_output([{"f": {"x": 1}}]) is True
        assert is_function_calling_format_output({"f": {"x": 1}}) is False
        assert is_function_calling_format_output([{"f": 1}]) is False
        assert is_function_calling_format_output([{"f": {"x": 1}, "g": {}}]) is False


class TestEvaluateBFCL:
    def test_simple_python_success(self):
        traj = make_trajectory(
            generated_text='[math.factorial(number=5)]',
            ground_truth=[{"math.factorial": {"number": [5]}}],
            func_description=[
                make_schema(
                    "math.factorial",
                    {"number": {"type": "integer"}},
                    ["number"],
                )
            ],
        )

        result = evaluate_bfcl(traj)

        assert result["score"] == 1.0
        assert result["prediction"] == [{"math.factorial": {"number": 5}}]
        assert result["metric"] == "accuracy"
        assert result["extraction_failed"] is False

    def test_unquoted_lambda_argument_is_normalized(self):
        traj = make_trajectory(
            generated_text='[estimate_derivative(function=lambda x: x**2, x=5)]',
            ground_truth=[{"estimate_derivative": {"function": ["lambda x: x**2"], "x": [5]}}],
            func_description=[
                make_schema(
                    "estimate_derivative",
                    {
                        "function": {"type": "any"},
                        "x": {"type": "integer"},
                    },
                    ["function", "x"],
                )
            ],
        )

        result = evaluate_bfcl(traj)

        assert result["score"] == 1.0
        assert result["prediction"] == [
            {"estimate_derivative": {"function": "lambda x: x ** 2", "x": 5}}
        ]
        assert result["extraction_failed"] is False

    def test_json_fallback_success(self):
        traj = make_trajectory(
            generated_text=(
                'Calls: '
                '[{"function": "math.factorial", "parameters": {"number": 5}}]'
            ),
            ground_truth=[{"math.factorial": {"number": [5]}}],
            func_description=[
                make_schema(
                    "math.factorial",
                    {"number": {"type": "integer"}},
                    ["number"],
                )
            ],
        )

        result = evaluate_bfcl(traj)

        assert result["score"] == 1.0
        assert result["prediction"] == [{"math.factorial": {"number": 5}}]
        assert result["extraction_failed"] is False

    def test_optional_parameter_can_be_omitted(self):
        traj = make_trajectory(
            generated_text='[triangle.area(base=10, height=5)]',
            ground_truth=[
                {
                    "triangle.area": {
                        "base": [10],
                        "height": [5],
                        "unit": ["units", ""],
                    }
                }
            ],
            func_description=[
                make_schema(
                    "triangle.area",
                    {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"},
                    },
                    ["base", "height"],
                )
            ],
        )

        result = evaluate_bfcl(traj)

        assert result["score"] == 1.0
        assert result["prediction"] == [{"triangle.area": {"base": 10, "height": 5}}]
        assert result["extraction_failed"] is False

    def test_parallel_calls_match_without_order(self):
        traj = make_trajectory(
            generated_text=(
                '[mail.send(to="sam@example.com"), '
                'calendar.create_event(title="Standup")]'
            ),
            ground_truth=[
                {"calendar.create_event": {"title": ["Standup"]}},
                {"mail.send": {"to": ["sam@example.com"]}},
            ],
            func_description=[
                make_schema(
                    "calendar.create_event",
                    {"title": {"type": "string"}},
                    ["title"],
                ),
                make_schema(
                    "mail.send",
                    {"to": {"type": "string"}},
                    ["to"],
                ),
            ],
            test_category="parallel",
        )

        result = evaluate_bfcl(traj)

        assert result["score"] == 1.0
        assert result["prediction"] == [
            {"mail.send": {"to": "sam@example.com"}},
            {"calendar.create_event": {"title": "Standup"}},
        ]
        assert result["extraction_failed"] is False

    def test_non_call_text_marks_extraction_failure(self):
        traj = make_trajectory(
            generated_text='I cannot complete this request.',
            ground_truth=[{"math.factorial": {"number": [5]}}],
            func_description=[
                make_schema(
                    "math.factorial",
                    {"number": {"type": "integer"}},
                    ["number"],
                )
            ],
        )

        result = evaluate_bfcl(traj)

        assert result["prediction"] is None
        assert result["score"] == 0.0
        assert result["extraction_failed"] is True


class TestBFCLPipelineWiring:
    """Regression tests for the on-disk trajectory shape produced by main.py.

    Reproduces traj_0.json from the qwen-bfcl-100 run: prediction holds the
    bracketed final answer, generated_text is the full prompt+CoT+\\boxed{...}
    blob, and ground_truth is a JSON string (as serialized by BFCLDataLoader).
    """

    SCHEMA = make_schema(
        "calculate_triangle_area",
        {
            "base": {"type": "integer"},
            "height": {"type": "integer"},
            "unit": {"type": "string"},
        },
        ["base", "height"],
    )
    GROUND_TRUTH_LIST = [
        {
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            }
        }
    ]
    PREDICTION = '[calculate_triangle_area(base=10, height=5, unit="units")]'
    GENERATED_TEXT = (
        "<think>\nStep 1: pick the function.\n</think>\n"
        "Therefore the answer is \\boxed{"
        '[calculate_triangle_area(base=10, height=5, unit="units")]}'
    )

    def test_uses_prediction_field_when_generated_text_is_unparseable(self):
        traj = {
            "generated_text": self.GENERATED_TEXT,
            "prediction": self.PREDICTION,
            "ground_truth": self.GROUND_TRUTH_LIST,
            "func_description": [self.SCHEMA],
            "test_category": "simple_python",
        }

        result = evaluate_bfcl(traj)

        assert result["extraction_failed"] is False
        assert result["score"] == 1.0
        assert result["prediction"] == [
            {"calculate_triangle_area": {"base": 10, "height": 5, "unit": "units"}}
        ]

    def test_ground_truth_accepted_as_json_string(self):
        import json as _json

        traj = {
            "generated_text": self.GENERATED_TEXT,
            "prediction": self.PREDICTION,
            "ground_truth": _json.dumps(self.GROUND_TRUTH_LIST),
            "func_description": [self.SCHEMA],
            "test_category": "simple_python",
        }

        result = evaluate_bfcl(traj)

        assert result["extraction_failed"] is False
        assert result["score"] == 1.0

    def test_without_prediction_falls_back_to_generated_text(self):
        traj = {
            "generated_text": self.PREDICTION,
            "ground_truth": self.GROUND_TRUTH_LIST,
            "func_description": [self.SCHEMA],
            "test_category": "simple_python",
        }

        result = evaluate_bfcl(traj)

        assert result["extraction_failed"] is False
        assert result["score"] == 1.0

    def test_real_traj0_payload_scores_correct(self):
        """End-to-end against the exact traj_0.json from the failing run."""
        import json as _json
        from pathlib import Path

        path = Path(
            "trajectories/417-qwen-bfcl-100-defaultjk_qwen_thinking_bfcl_zero_0_100_"
            "type1_conf_jackknife_default_mod-none/traj_0.json"
        )
        if not path.exists():
            pytest.skip(f"trajectory not present: {path}")

        envelope = _json.loads(path.read_text())
        on_disk = envelope["trajectory"]

        traj = {
            "generated_text": on_disk["generated_text"],
            "prediction": on_disk["final_answer"],
            "ground_truth": on_disk["ground_truth"],
            "func_description": [self.SCHEMA],
            "test_category": "simple_python",
        }

        result = evaluate_bfcl(traj)

        assert result["extraction_failed"] is False
        assert result["score"] == 1.0


class TestBFCLKnownBugs:
    # @pytest.mark.xfail(
    #     strict=True,
    #     reason="multiple_function_checker only validates the first call",
    # )
    # def test_multiple_checker_validates_every_call(self):
    #     funcs = [
    #         make_schema(
    #             "counter.increment",
    #             {"amount": {"type": "integer"}},
    #             ["amount"],
    #         )
    #     ]
    #     model_output = [
    #         {"counter.increment": {"amount": 1}},
    #         {"counter.increment": {"amount": 999}},
    #     ]
    #     possible_answers = [
    #         {"counter.increment": {"amount": [1]}},
    #         {"counter.increment": {"amount": [2]}},
    #     ]

    #     result = multiple_function_checker(funcs, model_output, possible_answers)

    #     assert result["valid"] is False

    @pytest.mark.xfail(
        strict=True,
        reason="evaluate_bfcl does not unwrap boxed final answers before parsing",
    )
    def test_boxed_final_answer_is_accepted(self):
        traj = make_trajectory(
            generated_text='Step 1: choose the function\n\\boxed{[math.factorial(number=5)]}',
            ground_truth=[{"math.factorial": {"number": [5]}}],
            func_description=[
                make_schema(
                    "math.factorial",
                    {"number": {"type": "integer"}},
                    ["number"],
                )
            ],
        )

        result = evaluate_bfcl(traj)

        assert result["score"] == 1.0
        assert result["extraction_failed"] is False
