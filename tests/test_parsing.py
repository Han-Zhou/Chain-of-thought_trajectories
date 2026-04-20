from parsing import parse_output


class TestParseOutput:
    def test_think_wrapper_steps_exclude_wrapper_tokens(self):
        text = (
            "<think>\n"
            "Let's think step-by-step.\n"
            "Step 1: Inspect the examples.\n"
            "Step 2: Choose the best match.\n"
            "\n"
            "Final Answer seems to be D.\n"
            "</think>\n"
            "Therefore the answer is \\boxed{D}"
        )

        parsed = parse_output(text)

        assert parsed.cot_steps == [
            "Step 1: Inspect the examples.",
            "Step 2: Choose the best match.",
        ]
        assert all("<think>" not in step for step in parsed.cot_steps)
        assert all("</think>" not in step for step in parsed.cot_steps)
        assert all("Final Answer seems to be" not in step for step in parsed.cot_steps)
        assert all("Therefore the answer is" not in step for step in parsed.cot_steps)
        assert parsed.final_answer == "D"
        assert parsed.raw_cot_block.startswith("<think>\nLet's think step-by-step.")
        assert parsed.raw_cot_block.endswith("Final Answer seems to be D.\n</think>\nTherefore the answer is")

    def test_think_wrapper_without_step_markers_uses_cleaned_fallback(self):
        text = (
            "<think>\n"
            "First observation.\n\n"
            "Second observation.\n"
            "</think>\n"
            "\\boxed{Yes}"
        )

        parsed = parse_output(text)

        assert parsed.cot_steps == ["First observation.", "Second observation."]
        assert parsed.final_answer == "Yes"
        assert parsed.raw_cot_block == "<think>\nFirst observation.\n\nSecond observation.\n</think>"

    def test_non_think_preamble_behavior_is_unchanged(self):
        text = (
            "Let's solve this carefully.\n"
            "Step 1: Inspect the evidence.\n"
            "Step 2: Return the answer.\n"
            "\\boxed{A}"
        )

        parsed = parse_output(text)

        assert parsed.cot_steps == [
            "Let's solve this carefully.",
            "Step 1: Inspect the evidence.",
            "Step 2: Return the answer.",
        ]
        assert parsed.final_answer == "A"

    def test_strips_trailing_answer_label_from_last_step(self):
        text = (
            "Let's think step-by-step.\n"
            "Step 1: Compare the options.\n"
            "Step 2: Option B is the closest fit.\n\n"
            "Answer: B\n"
            "Therefore the answer is \\boxed{B}"
        )

        parsed = parse_output(
            text,
            extra_box_suffix="\nTherefore the answer is \\boxed{",
            llama_answer_cleanup=True,
        )

        assert parsed.cot_steps == [
            "Let's think step-by-step.",
            "Step 1: Compare the options.",
            "Step 2: Option B is the closest fit.",
        ]
        assert all("Answer:" not in step for step in parsed.cot_steps)
        assert parsed.final_answer == "B"

    def test_strips_trailing_final_answer_sentence_from_last_step(self):
        text = (
            "Let's think step-by-step.\n"
            "Step 1: Review the themes.\n"
            "Step 2: Option D matches best.\n\n"
            "The final answer is: D\n"
            "Therefore the answer is \\boxed{D}"
        )

        parsed = parse_output(
            text,
            extra_box_suffix="\nTherefore the answer is \\boxed{",
            llama_answer_cleanup=True,
        )

        assert parsed.cot_steps == [
            "Let's think step-by-step.",
            "Step 1: Review the themes.",
            "Step 2: Option D matches best.",
        ]
        assert all("final answer" not in step.lower() for step in parsed.cot_steps)
        assert parsed.final_answer == "D"

    def test_strips_trailing_bare_answer_line_from_last_step(self):
        text = (
            "Let's think step-by-step.\n"
            "Step 1: Compare genres.\n"
            "Step 2: Option C is the only strong match.\n\n"
            "C\n"
            "Therefore the answer is \\boxed{C}"
        )

        parsed = parse_output(
            text,
            extra_box_suffix="\nTherefore the answer is \\boxed{",
            llama_answer_cleanup=True,
        )

        assert parsed.cot_steps == [
            "Let's think step-by-step.",
            "Step 1: Compare genres.",
            "Step 2: Option C is the only strong match.",
        ]
        assert all(step.strip() != "C" for step in parsed.cot_steps)
        assert parsed.final_answer == "C"

    def test_strips_trailing_bare_mcq_letter_even_if_it_disagrees_with_boxed_answer(self):
        text = (
            "Let's think step-by-step.\n"
            "Step 1: Compare the options.\n"
            "Step 2: Option C is the strongest match.\n\n"
            "A\n"
            "Therefore the answer is \\boxed{C}"
        )

        parsed = parse_output(
            text,
            extra_box_suffix="\nTherefore the answer is \\boxed{",
            llama_answer_cleanup=True,
        )

        assert parsed.cot_steps == [
            "Let's think step-by-step.",
            "Step 1: Compare the options.",
            "Step 2: Option C is the strongest match.",
        ]
        assert all(step.strip() != "A" for step in parsed.cot_steps)
        assert parsed.final_answer == "C"

    def test_generic_cleanup_keeps_same_line_answer_label_behavior(self):
        text = (
            "Let's think step-by-step.\n"
            "Step 1: Compare the options.\n"
            "Step 2: Option B is the closest fit.\n"
            "Answer: B\n"
            "Therefore the answer is \\boxed{B}"
        )

        parsed = parse_output(
            text,
            extra_box_suffix="\nTherefore the answer is \\boxed{",
            llama_answer_cleanup=False,
        )

        assert parsed.cot_steps == [
            "Let's think step-by-step.",
            "Step 1: Compare the options.",
            "Step 2: Option B is the closest fit.\nAnswer: B",
        ]
        assert parsed.final_answer == "B"
