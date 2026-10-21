from pathlib import Path

import pytest

from kvbridge.protocol import validate_mapped_evaluation_accounting
from experiments.eval_paper_tasks import _mapped_prefix_prefill_guard


def test_mapped_evaluation_has_zero_target_prefix_prefill_and_method_guard():
    source = (Path(__file__).resolve().parents[1] / "experiments/eval_paper_tasks.py").read_text()
    assert 'assert_mapper_method(mapper, "cachebridge" if args.mode == "cachebridge" else args.mode)' in source
    assert "validate_mapped_evaluation_accounting(accounting)" in source
    assert "--target-tokenizer" in source
    assert "args.target_tokenizer or args.target" in source


def test_mapped_evaluation_accounting_rejects_target_prefix_prefill():
    validate_mapped_evaluation_accounting({"target_prefix_prefill_tokens": 0})
    with pytest.raises(ValueError, match="target prefix prefill"):
        validate_mapped_evaluation_accounting({"target_prefix_prefill_tokens": 1})


def test_mapped_target_forward_without_transferred_cache_fails():
    import torch
    accounting = {"target_prefix_prefill_tokens": 0}
    guard = _mapped_prefix_prefill_guard(accounting)
    with pytest.raises(ValueError, match="attempted target prefix prefill"):
        guard(None, (torch.ones((1, 7), dtype=torch.long),),
              {"past_key_values": None})
    assert accounting["target_prefix_prefill_tokens"] == 7
