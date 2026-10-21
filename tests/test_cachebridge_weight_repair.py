import pytest
import torch

from kvbridge.cachebridge import aggregate_boundary_sensitivities, repair_weights


def test_boundary_normalization_has_equal_boundary_mass():
    raw = torch.tensor([[[9.0, 1.0, 0.0], [1.0, 1.0, 1.0]]])
    weights = aggregate_boundary_sensitivities(raw, feature_width=2)
    assert torch.allclose(weights.mean(-1), torch.ones(1))
    assert torch.isfinite(weights).all()


def test_repair_weights_matches_uniform_when_raw_is_uniform():
    raw = torch.ones(2, 7)
    assert torch.allclose(repair_weights(raw, 3), raw)


def test_repair_weights_has_mean_one_and_ess_floor():
    raw = torch.tensor([[100.0, 1, 1, 1, 1, 1, 1, 1]])
    weights = repair_weights(raw, feature_width=2)
    ess = weights.sum(-1).square() / weights.square().sum(-1)
    assert torch.allclose(weights.mean(-1), torch.ones(1))
    assert ess.item() >= 4.0 - 1e-6


def test_repair_weights_padding_and_zero_sensitivity_are_finite():
    raw = torch.zeros(1, 5)
    mask = torch.tensor([[True, True, True, False, False]])
    weights = repair_weights(raw, feature_width=2, valid_mask=mask)
    assert torch.equal(weights[..., 3:], torch.zeros(1, 2))
    assert torch.allclose(weights[..., :3].mean(-1), torch.ones(1))
    assert torch.isfinite(weights).all()


def test_repair_weights_rejects_negative_or_empty_inputs():
    with pytest.raises(ValueError):
        repair_weights(torch.tensor([[1.0, -1.0]]), 2)
    with pytest.raises(ValueError):
        repair_weights(torch.ones(1, 2), 2,
                       valid_mask=torch.zeros(1, 2, dtype=torch.bool))
