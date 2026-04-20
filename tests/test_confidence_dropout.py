import pytest
import torch

from confidence import ConfidenceAlignmentError, dropout_late_forward


def _normalize_token_ids(token_ids):
    if isinstance(token_ids, torch.Tensor):
        if token_ids.dim() == 0:
            return [token_ids.item()]
        return token_ids.tolist()
    if isinstance(token_ids, int):
        return [token_ids]
    return list(token_ids)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]

    def convert_ids_to_tokens(self, token_ids):
        return [chr(tok) for tok in _normalize_token_ids(token_ids)]

    def decode(self, token_ids):
        return "".join(self.convert_ids_to_tokens(token_ids))


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, past_key_values=None, output_hidden_states=False):
        del past_key_values, output_hidden_states
        return {"logits": torch.zeros(input_ids.shape[0], input_ids.shape[1], 1, device=input_ids.device)}


class DummyCache:
    def reorder_cache(self, indices):
        self.indices = indices


class DummyLLM:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel()
        self.model_name = "dummy-model"


def _char_ids(text: str) -> torch.Tensor:
    return torch.tensor([ord(ch) for ch in text], dtype=torch.long)


def _run_dropout(cot_steps, early_text, jk_step_masks):
    llm = DummyLLM()
    early_token_ids = _char_ids(early_text)
    late_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    _, step_masks, masked_steps_debug = dropout_late_forward(
        llm=llm,
        cot_steps=cot_steps,
        early_token_ids=early_token_ids,
        early_cache=DummyCache(),
        late_tokens=late_tokens,
        modify_start_late=1,
        modify_end_late=2,
        nb_early_tokens=early_token_ids.numel(),
        nb_dropout_samples=1,
        threshold=0.5,
        use_jackknife=True,
        debug_conf=True,
        jk_step_masks=jk_step_masks,
    )
    return step_masks, masked_steps_debug


def test_dropout_late_forward_jackknife_masks_step1_when_first_item_is_reasoning():
    cot_steps = ["Step 1: alpha", "Step 2: beta", "Step 3: gamma"]
    step_masks, masked_steps_debug = _run_dropout(
        cot_steps=cot_steps,
        early_text="Prompt:\n" + "\n".join(cot_steps),
        jk_step_masks=[[0, 1, 0]],
    )

    assert step_masks == [[0, 1, 0]]
    assert masked_steps_debug[0]["dropped_steps"] == ["Step 1: alpha", "Step 3: gamma"]
    assert "Step 1: alpha" in masked_steps_debug[0]["masked_text"]
    assert "Step 3: gamma" in masked_steps_debug[0]["masked_text"]
    assert "Step 2: beta" not in masked_steps_debug[0]["masked_text"]


def test_dropout_late_forward_jackknife_keeps_preamble_out_of_optional_mask_columns():
    cot_steps = ["Let's think step-by-step.", "Step 1: alpha", "Step 2: beta"]
    step_masks, masked_steps_debug = _run_dropout(
        cot_steps=cot_steps,
        early_text="Prompt:\n" + "\n".join(cot_steps),
        jk_step_masks=[[0, 1]],
    )

    assert step_masks == [[0, 1]]
    assert masked_steps_debug[0]["dropped_steps"] == ["Step 1: alpha"]
    assert "Let's think step-by-step." not in masked_steps_debug[0]["masked_text"]
    assert "Step 1: alpha" in masked_steps_debug[0]["masked_text"]
    assert "Step 2: beta" not in masked_steps_debug[0]["masked_text"]


def test_dropout_late_forward_raises_on_step_alignment_failure():
    llm = DummyLLM()
    early_token_ids = _char_ids("Prompt:\nStep 1: alpha\nStep 3: gamma")
    late_tokens = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ConfidenceAlignmentError, match="Step text not found"):
        dropout_late_forward(
            llm=llm,
            cot_steps=["Step 1: alpha", "Step 2: beta", "Step 3: gamma"],
            early_token_ids=early_token_ids,
            early_cache=DummyCache(),
            late_tokens=late_tokens,
            modify_start_late=1,
            modify_end_late=2,
            nb_early_tokens=early_token_ids.numel(),
            nb_dropout_samples=1,
            threshold=0.5,
            use_jackknife=True,
            debug_conf=True,
            jk_step_masks=[[1, 0, 1]],
        )
