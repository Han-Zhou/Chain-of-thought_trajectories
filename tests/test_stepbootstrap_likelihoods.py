import json

import pytest
import torch

from score_stepbootstrap_likelihoods import (
    StepSpan,
    find_reasoning_step_spans,
    load_stepbootstrap_data,
    map_tokens_to_steps,
    _numbered_source_step_indices,
    selected_token_log_likelihoods,
    summarize_step_log_likelihoods,
)


def test_loads_nested_stepbootstrap_schema(tmp_path):
    input_path = tmp_path / "trajectory.json"
    input_path.write_text(
        json.dumps(
            {
                "trajectory": {
                    "debug_info": {
                        "bootstrap_prompts": {"answer_logits": ["complete prompt"]},
                        "bootstrap_samples": [
                            {
                                "sample_idx": 0,
                                "kept_steps": ["Step 1: reason"],
                            }
                        ],
                    }
                }
            }
        )
    )

    prompts, samples = load_stepbootstrap_data(input_path)

    assert prompts == ["complete prompt"]
    assert samples[0]["sample_idx"] == 0


def test_rejects_prompt_sample_count_mismatch(tmp_path):
    input_path = tmp_path / "trajectory.json"
    input_path.write_text(
        json.dumps(
            {
                "bootstrap_prompts": {"answer_logits": ["one", "two"]},
                "bootstrap_samples": [{"kept_steps": ["Step 1: reason"]}],
            }
        )
    )

    with pytest.raises(ValueError, match="count mismatch"):
        load_stepbootstrap_data(input_path)


def test_finds_final_reasoning_chain_and_excludes_answer():
    prompt = (
        "system\nLabel each reasoning step as 'Step 1:', 'Step 2:'.\n"
        "user\nExample:\nStep 1: This is not the assistant chain.\n"
        "assistant\nLet's think step-by-step.\n"
        "Step 1: First sampled reason.\n"
        "Step 2: Second sampled reason.\n"
        "Therefore the answer is \\boxed{C}."
    )

    spans = find_reasoning_step_spans(prompt, expected_step_count=2)
    texts = [prompt[span.start : span.end] for span in spans]

    assert texts == [
        "Step 1: First sampled reason.",
        "Step 2: Second sampled reason.",
    ]
    assert all("answer" not in text.lower() for text in texts)


def test_step_detection_supports_repeated_bootstrap_content():
    prompt = (
        "assistant\n"
        "Step 1: repeated reason.\n"
        "Step 2: repeated reason.\n"
        "Step 3: final reason.\n"
        "The answer is \\boxed{A}."
    )

    spans = find_reasoning_step_spans(prompt, expected_step_count=3)

    assert [span.display_number for span in spans] == [1, 2, 3]
    assert prompt[spans[0].start : spans[0].end].endswith("repeated reason.")
    assert prompt[spans[1].start : spans[1].end].endswith("repeated reason.")


def test_unnumbered_preamble_is_excluded_from_step_metadata():
    sample = {
        "kept_steps": [
            "Let's think step-by-step.",
            "Step 1: first reason.",
            "Step 2: second reason.",
        ],
        "kept_step_indices": [0, 1, 2],
        "step_counts": [1, 1],
    }

    source_indices = _numbered_source_step_indices(sample, sample_idx=0)

    assert source_indices == [1, 2]


def test_maps_boundary_overlapping_tokens_to_best_step():
    spans = [StepSpan(1, 5, 12), StepSpan(2, 13, 20)]
    offsets = [(0, 5), (4, 8), (8, 12), (12, 15), (15, 20), (20, 21)]

    mapped = map_tokens_to_steps(offsets, spans)

    assert mapped == [[1, 2], [3, 4]]


def test_causal_shift_scores_token_t_from_logits_t_minus_one():
    input_ids = torch.tensor([0, 1, 2])
    logits = torch.tensor(
        [
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [9.0, 0.0, 0.0],
        ]
    )

    scores = selected_token_log_likelihoods(logits, input_ids, [1, 2])
    expected = torch.stack(
        [
            torch.log_softmax(logits[0], dim=-1)[1],
            torch.log_softmax(logits[1], dim=-1)[2],
        ]
    )

    assert torch.allclose(scores, expected)


def test_all_cot_mean_is_weighted_by_token_count():
    token_scores = torch.tensor([-1.0, -3.0, -5.0, -7.0])

    step_means, all_cot_mean = summarize_step_log_likelihoods(token_scores, [1, 3])

    assert step_means == pytest.approx([-1.0, -5.0])
    assert all_cot_mean == pytest.approx(-4.0)
    assert all_cot_mean != pytest.approx(sum(step_means) / len(step_means))
