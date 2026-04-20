from utils.model_utils import (
    clean_phase1_generated_text,
    phase1_stop_strings_for_model,
)


def test_llama_phase1_stop_strings_include_answer_leak_guards():
    stop_strings = phase1_stop_strings_for_model("meta-llama/Llama-3.1-8B-Instruct", thinking=False)

    assert stop_strings is not None
    assert "\nAnswer:" in stop_strings
    assert "\n\nA\n" in stop_strings


def test_qwen_phase1_stop_strings_only_use_think_close():
    stop_strings = phase1_stop_strings_for_model("Qwen/Qwen3.5-27B", thinking=True)

    assert stop_strings == ["</think>"]


def test_non_llama_non_qwen_phase1_stop_strings_are_disabled():
    stop_strings = phase1_stop_strings_for_model("openai/gpt-oss-20b", thinking=False)

    assert stop_strings is None


def test_llama_phase1_cleanup_strips_matched_stop_string():
    text = "Step 1: Compare options.\nAnswer:"
    stop_strings = ["\nAnswer:"]

    assert clean_phase1_generated_text("meta-llama/Llama-3.1-8B-Instruct", text, stop_strings) == "Step 1: Compare options."


def test_non_llama_phase1_cleanup_leaves_text_unchanged():
    text = "Step 1: Compare options.\nAnswer:"
    stop_strings = ["\nAnswer:"]

    assert clean_phase1_generated_text("Qwen/Qwen3.5-27B", text, stop_strings) == text
