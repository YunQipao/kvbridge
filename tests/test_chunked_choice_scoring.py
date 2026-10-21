"""Suffix likelihood equivalence and bounded FP32 conversion, CPU/offline only."""

import socket
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import experiments.eval_mmlu_redux as evaluation


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scoring tests must not use GPU, network or model downloads")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(evaluation, "load_causal_lm", forbidden)
    monkeypatch.setattr(evaluation, "load_tokenizer", forbidden)


def reference_scores(logits, targets, starts):
    lp = F.log_softmax(logits.float(), dim=-1)
    return torch.stack([
        lp[i, start:start + len(ids)].gather(-1, torch.tensor(ids)[:, None]).sum()
        for i, (ids, start) in enumerate(zip(targets, starts))])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk_size", [1, 4, 8])
def test_scores_and_predictions_match_full_reference(dtype, chunk_size):
    generator = torch.Generator().manual_seed(73)
    logits = torch.randn(4, 19, 31, generator=generator).to(dtype)
    targets = [[3], [5, 1], [2, 8, 7, 6, 4], [12, 3, 7, 2, 8, 9, 0, 1, 5]]
    starts = [6, 3, 4, 7]
    expected = reference_scores(logits, targets, starts)
    actual = torch.tensor(evaluation._chunked_continuation_scores(
        logits, targets, logit_starts=starts, chunk_size=chunk_size))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert int(actual.argmax()) == int(expected.argmax())


def test_prompt_and_padding_rows_are_not_scored():
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(2, 13, 17, generator=generator)
    targets, starts = [[4, 5, 6], [7]], [5, 8]
    expected = evaluation._chunked_continuation_scores(logits, targets, logit_starts=starts)
    modified = logits.clone()
    for i, (ids, start) in enumerate(zip(targets, starts)):
        modified[i, :start] = torch.randn(start, 17, generator=generator) * 100
        modified[i, start + len(ids):] = 1000
    actual = evaluation._chunked_continuation_scores(
        modified, targets, logit_starts=starts, chunk_size=4)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))


def test_first_continuation_token_uses_previous_position():
    # Three prompt tokens occupy positions 0..2; continuation [4,5] starts
    # at token position 3. The prediction rows are positions [2,3], not [3,4].
    logits = torch.zeros(1, 6, 8)
    logits[0, 2, 4] = 8
    logits[0, 3, 5] = 8
    logits[0, 4, 5] = -8
    targets = [[4, 5]]
    actual = evaluation._chunked_continuation_scores(logits, targets, logit_starts=[2])
    expected = reference_scores(logits, targets, [2])
    torch.testing.assert_close(torch.tensor(actual), expected)
    wrong = reference_scores(logits, targets, [3])
    assert actual[0] > float(wrong[0]) + 5


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_only_selected_small_rows_are_converted_to_float(monkeypatch, chunk_size):
    generator = torch.Generator().manual_seed(41)
    logits = torch.randn(3, 23, 29, generator=generator).to(torch.bfloat16)
    targets, starts = [[3], [4, 5, 6, 7, 8], [9, 10]], [8, 5, 9]
    expected = reference_scores(logits, targets, starts)
    original_float = torch.Tensor.float
    converted_rows = []
    def limited_float(tensor, *args, **kwargs):
        assert tensor.ndim == 2, "full [batch, sequence, vocab] FP32 conversion is forbidden"
        assert 1 <= tensor.shape[0] <= chunk_size
        assert tensor.shape[1] == logits.shape[-1]
        converted_rows.append(tensor.shape[0])
        return original_float(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "float", limited_float)
    actual = evaluation._chunked_continuation_scores(
        logits, targets, logit_starts=starts, chunk_size=chunk_size)
    assert sum(converted_rows) == sum(map(len, targets))
    torch.testing.assert_close(torch.tensor(actual), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_cached_score_keeps_bridge_token_alignment_and_variable_lengths(monkeypatch, chunk_size):
    logits = torch.zeros(2, 3, 8, dtype=torch.bfloat16)
    logits[0, 0, 6] = 5
    logits[0, 1:, 6] = -10  # unscored padding for one-token choice
    targets = [[6], [4, 5, 6]]
    expected = reference_scores(logits, targets, [0, 0])
    cache, expanded = object(), object()
    prefill_calls, suffix_calls = [], []
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [1, 2, 3] if add_special_tokens else {" a": [6], " b": [4, 5, 6]}[text]
            return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
    def factory(ids):
        prefill_calls.append(ids.tolist())
        return cache
    def expand(value, batch):
        assert value is cache and batch == 2
        return expanded
    def model(ids, *, past_key_values, attention_mask, position_ids):
        assert past_key_values is expanded, "target must continue the supplied cache"
        assert ids.tolist() == [[3, 0, 0], [3, 4, 5]]
        assert position_ids.tolist() == [[2, 3, 4], [2, 3, 4]]
        assert attention_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        suffix_calls.append(True)
        return SimpleNamespace(logits=logits)
    monkeypatch.setattr(evaluation, "_expand", expand)
    original_float = torch.Tensor.float
    def suffix_rows_only(tensor, *args, **kwargs):
        assert tensor.ndim == 2 and tensor.shape[0] <= chunk_size
        return original_float(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "float", suffix_rows_only)
    prediction, scores, prefix_tokens, choice_tokens = evaluation._score(
        model, Tokenizer(), "prompt", ["a", "b"], factory, "cpu", score_chunk_size=chunk_size)
    torch.testing.assert_close(torch.tensor(scores), expected, atol=1e-5, rtol=1e-5)
    assert prediction == int(expected.argmax())
    assert prefill_calls == [[[1, 2]]] and suffix_calls == [True]
    assert prefix_tokens == 2 and choice_tokens == 4


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_invalid_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError, match="positive integer"):
        evaluation._chunked_continuation_scores(torch.zeros(1, 1, 3), [[1]], chunk_size=chunk_size)
