from prompts.load import load_messages


def test_experimental_extra_box_appends_cot_only_for_llama():
    entry = {
        "question": (
            "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
            "Options:\n"
            "(A) The Front Page\n"
            "(B) Maelstrom\n"
            "(C) The Lion King\n"
            "(D) Lamerica"
        )
    }

    messages = load_messages(
        dataset="bigbench_movie",
        few_shot=False,
        entry=entry,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        thinking=False,
        prompt_type=1,
        experimental_extra_box=True,
    )

    system_prompt = messages[0]["content"]

    # MCQ answer-format instruction always present
    assert "output ONLY the single letter" in system_prompt
    # COT_ONLY appended for extra_box
    assert "Stop immediately after your last numbered reasoning step." in system_prompt
    # \boxed instruction stripped
    assert "\\boxed{your answer}" not in system_prompt


def test_experimental_extra_box_appends_cot_only_for_qwen():
    entry = {
        "question": (
            "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
            "Options:\n"
            "(A) The Front Page\n"
            "(B) Maelstrom\n"
            "(C) The Lion King\n"
            "(D) Lamerica"
        )
    }

    messages = load_messages(
        dataset="bigbench_movie",
        few_shot=False,
        entry=entry,
        model_name="Qwen/Qwen3.5-27B",
        thinking=False,
        prompt_type=1,
        experimental_extra_box=True,
    )

    system_prompt = messages[0]["content"]

    # MCQ answer-format instruction always present
    assert "output ONLY the single letter" in system_prompt
    # COT_ONLY appended for extra_box (now for ALL models, not just llama)
    assert "Stop immediately after your last numbered reasoning step." in system_prompt
    # \boxed instruction stripped
    assert "\\boxed{your answer}" not in system_prompt


def test_bfcl_uses_correct_prompt_with_extra_box():
    entry = {
        "question": "Find the area of a triangle with a base of 10 units and height of 5 units.",
        "metadata": {
            "functions": [{"name": "calculate_triangle_area", "parameters": {"base": "int", "height": "int"}}]
        },
    }

    messages = load_messages(
        dataset="bfcl",
        few_shot=False,
        entry=entry,
        model_name="Qwen/Qwen3.5-27B",
        thinking=False,
        prompt_type=1,
        experimental_extra_box=True,
    )

    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    # BFCL user prompt must contain the functions, not bigbench_movie text
    assert "calculate_triangle_area" in user_prompt
    assert "movies" not in user_prompt
    # BFCL answer-format instruction
    assert "func_name" in system_prompt
    # COT_ONLY appended
    assert "Stop immediately after your last numbered reasoning step." in system_prompt
    # \boxed instruction stripped
    assert "\\boxed{your answer}" not in system_prompt


def test_math_dataset_gets_math_answer_format():
    entry = {"question": "Solve x^2 = 4."}

    messages = load_messages(
        dataset="math500",
        few_shot=False,
        entry=entry,
        model_name="Qwen/Qwen3.5-27B",
        thinking=False,
        prompt_type=1,
    )

    system_prompt = messages[0]["content"]
    assert "exact numerical or symbolic answer" in system_prompt


def test_no_extra_box_keeps_boxed_instruction():
    entry = {
        "question": (
            "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
            "Options:\n"
            "(A) The Front Page\n"
            "(B) Maelstrom\n"
            "(C) The Lion King\n"
            "(D) Lamerica"
        )
    }

    messages = load_messages(
        dataset="bigbench_movie",
        few_shot=False,
        entry=entry,
        model_name="Qwen/Qwen3.5-27B",
        thinking=False,
        prompt_type=1,
        experimental_extra_box=False,
    )

    system_prompt = messages[0]["content"]
    # \boxed instruction present when extra_box is off
    assert "\\boxed{your answer}" in system_prompt
    # MCQ instruction present
    assert "output ONLY the single letter" in system_prompt
    # COT_ONLY NOT present
    assert "Stop immediately after your last numbered reasoning step." not in system_prompt
