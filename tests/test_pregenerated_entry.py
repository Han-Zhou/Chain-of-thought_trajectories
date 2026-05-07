import json

from main import (
    _build_dataset_lookup,
    _load_pregenerated_envelope,
    _merge_pregenerated_entry,
    _resolve_pregenerated_source_entry,
)


def test_merge_pregenerated_entry_uses_pregenerated_identity_and_keeps_prompt_fields():
    dataset_entry = {
        "id": "simple_python_0",
        "question": "Dataset row question",
        "answer": "dataset answer",
        "context": None,
        "choices": None,
        "metadata": {
            "functions": [{"name": "calculate_triangle_area"}],
            "bfcl_type": "simple_python",
        },
    }
    traj_saved = {
        "id": "simple_python_327",
        "question": "Pregenerated row question",
        "ground_truth": "pregenerated answer",
        "generated_text": "Step 1\n\\boxed{42}",
        "prompt_end_position": 10,
        "generated_end_position": 20,
    }

    merged = _merge_pregenerated_entry(dataset_entry, traj_saved)

    assert merged["id"] == "simple_python_327"
    assert merged["question"] == "Pregenerated row question"
    assert merged["answer"] == "pregenerated answer"
    assert merged["metadata"] == dataset_entry["metadata"]


def test_load_pregenerated_envelope_supports_sample0_suffix(tmp_path):
    path = tmp_path / "traj_7_sample_0.json"
    payload = {
        "index": 7,
        "question": "Question",
        "trajectory": {
            "id": "row_7",
            "question": "Question",
            "ground_truth": "Answer",
            "generated_text": "Reasoning\n\\boxed{A}",
            "prompt_end_position": 12,
            "generated_end_position": 34,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_pregenerated_envelope(str(tmp_path), 7)

    assert loaded == payload


def test_merge_pregenerated_entry_resolves_dataset_fields_by_identity_not_index():
    wrong_index_entry = {
        "id": "simple_python_0",
        "question": "Wrong index question",
        "answer": "wrong index answer",
        "metadata": {"functions": [{"name": "wrong"}], "bfcl_type": "wrong_type"},
    }
    matching_dataset_entry = {
        "id": "simple_python_327",
        "question": "Pregenerated row question",
        "answer": "pregenerated answer",
        "metadata": {"functions": [{"name": "correct"}], "bfcl_type": "simple_python"},
    }
    traj_saved = {
        "id": "simple_python_327",
        "question": "Pregenerated row question",
        "ground_truth": "pregenerated answer",
        "generated_text": "Reasoning\n\\boxed{42}",
        "prompt_end_position": 10,
        "generated_end_position": 20,
    }

    lookup = _build_dataset_lookup([wrong_index_entry, matching_dataset_entry])
    merged = _merge_pregenerated_entry(wrong_index_entry, traj_saved, dataset_lookup=lookup)

    assert merged["id"] == "simple_python_327"
    assert merged["question"] == "Pregenerated row question"
    assert merged["answer"] == "pregenerated answer"
    assert merged["metadata"] == matching_dataset_entry["metadata"]


def test_resolve_pregenerated_source_entry_uses_codeqa_pickle_entries_by_index():
    live_dataset_entry = {
        "id": "live-row",
        "question": "Wrong live question",
        "answer": "wrong live answer",
        "context": "wrong live context",
        "choices": ["wrong live choice"],
        "metadata": {"origin": "live"},
    }
    codeqa_pickle_entries = [
        {
            "id": "None",
            "question": "Question from codeqa pickle",
            "answer": "ground truth from codeqa pickle",
            "context": "context from codeqa pickle",
            "choices": ["choice from codeqa pickle"],
            "metadata": {"origin": "pickle"},
        }
    ]
    traj_saved = {
        "id": "None",
        "question": "Question from pregenerated trajectory",
        "ground_truth": "ground truth from pregenerated trajectory",
        "generated_text": "Reasoning\n\\boxed{42}",
        "prompt_end_position": 10,
        "generated_end_position": 20,
    }

    merged = _resolve_pregenerated_source_entry(
        "codeqa",
        live_dataset_entry,
        0,
        traj_saved,
        dataset_lookup=None,
        pregenerated_source_entries=codeqa_pickle_entries,
    )

    assert merged["id"] == "None"
    assert merged["question"] == "Question from pregenerated trajectory"
    assert merged["answer"] == "ground truth from pregenerated trajectory"
    assert merged["context"] == "context from codeqa pickle"
    assert merged["choices"] == ["choice from codeqa pickle"]
    assert merged["metadata"] == {"origin": "pickle"}


def test_resolve_pregenerated_source_entry_raises_when_codeqa_pickle_index_missing():
    traj_saved = {
        "id": "None",
        "question": "Question from pregenerated trajectory",
        "ground_truth": "ground truth from pregenerated trajectory",
        "generated_text": "Reasoning\n\\boxed{42}",
        "prompt_end_position": 10,
        "generated_end_position": 20,
    }

    try:
        _resolve_pregenerated_source_entry(
            "codeqa",
            {"id": "unused"},
            2,
            traj_saved,
            dataset_lookup=None,
            pregenerated_source_entries=[{"id": "None", "question": "q", "answer": "a"}],
        )
    except IndexError as exc:
        assert "Pregenerated codeqa index 2 exceeds pickle size 1" in str(exc)
    else:
        raise AssertionError("Expected IndexError for missing codeqa pickle index")
