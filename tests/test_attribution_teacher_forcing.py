import torch

from experiments.eval_attribution import (
    build_teacher_forcing_inputs,
    teacher_forcing_accounting,
)


def test_teacher_forcing_does_not_repeat_prefix_last_token():
    prefix = torch.tensor([[10, 11, 12]])
    continuation = torch.tensor([[20, 21, 22]])
    bridge, forced, targets = build_teacher_forcing_inputs(prefix, continuation)
    assert bridge.tolist() == [[12]]
    assert forced.tolist() == [[20, 21]]
    assert targets.tolist() == [[20, 21, 22]]
    assert teacher_forcing_accounting(3, 3) == {
        "mapped_prefix_tokens": 2,
        "bridge_tokens": 1,
        "teacher_forced_tokens": 2,
        "target_prefix_prefill_tokens": 0,
    }
