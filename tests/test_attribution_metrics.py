import torch
from experiments.eval_attribution import mean_nll, r2_metric


def test_perfect_mapping_has_unit_r2():
    x = torch.randn(20, 4)
    assert r2_metric(x, x) == 1.0


def test_identical_logits_have_identical_nll():
    logits = torch.randn(8, 5)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    assert mean_nll(logits, targets) == mean_nll(logits, targets)
