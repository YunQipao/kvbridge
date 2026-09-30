import torch

from kvbridge.cachebridge import (
    compute_key_sensitivity,
    compute_value_sensitivity,
    make_log_causal_boundaries,
    make_observation_indices,
    repair_weights,
)


def tiny_attention():
    probs = torch.tensor([[[[1., 0., 0.],
                            [.5, .5, 0.],
                            [1/3, 1/3, 1/3]]]])
    q = torch.tensor([[[[1., 0.], [0., 2.], [1., 1.]]]])
    v = torch.tensor([[[[1., 0.], [0., 1.], [2., 2.]]]])
    out = torch.tensor([[[[1., 0.], [.5, .5], [1., 1.]]]])
    causal = torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=torch.bool)
    return probs, q, v, out, causal


def test_hand_computed_value_and_key_sensitivity():
    probs, q, v, out, causal = tiny_attention()
    value = compute_value_sensitivity(probs, q, v, out, causal_mask=causal)
    key = compute_key_sensitivity(probs, q, v, out, causal_mask=causal)
    assert torch.allclose(value, torch.tensor([[[49/36, 13/36, 1/9]]]), atol=1e-6)
    assert torch.allclose(key, torch.tensor([[[13/36, 13/36, 2/9]]]), atol=1e-6)


def test_padding_is_excluded_and_causal_invisible_entries_do_not_contribute():
    probs, q, v, out, causal = tiny_attention()
    altered = probs.clone()
    altered[..., 0, 2] = 100.
    mask = torch.tensor([[True, True, False]])
    base = compute_value_sensitivity(probs, q, v, out, valid_mask=mask,
                                     causal_mask=causal)
    changed = compute_value_sensitivity(altered, q, v, out, valid_mask=mask,
                                        causal_mask=causal)
    assert torch.equal(base[..., 2], torch.zeros_like(base[..., 2]))
    assert torch.allclose(base, changed)


def test_shapes_finite_nonnegative_deterministic_and_k_differs_from_v():
    probs, q, v, out, causal = tiny_attention()
    k1 = compute_key_sensitivity(probs, q, v, out, causal_mask=causal)
    k2 = compute_key_sensitivity(probs, q, v, out, causal_mask=causal)
    val = compute_value_sensitivity(probs, q, v, out, causal_mask=causal)
    assert k1.shape == val.shape == (1, 1, 3)
    assert torch.isfinite(k1).all() and torch.isfinite(val).all()
    assert (k1 >= 0).all() and (val >= 0).all()
    assert torch.equal(k1, k2)
    assert not torch.allclose(k1, val)


def test_boundaries_are_unique_monotone_and_short_sequences_are_explicit():
    boundaries = make_log_causal_boundaries(1024, num_boundaries=32, start_token=12)
    assert len(boundaries) == 32
    assert boundaries == sorted(set(boundaries))
    assert boundaries[0] >= 12 and boundaries[-1] <= 1023
    assert len(make_log_causal_boundaries(5, num_boundaries=32, start_token=12)) == 0


def test_observation_indices_are_deterministic_and_respect_padding():
    mask = torch.tensor([True, True, False, True, True, False, True])
    expected = torch.tensor([0, 3, 6])
    assert torch.equal(make_observation_indices(7, stride=3, valid_mask=mask), expected)
    assert torch.equal(make_observation_indices(7, stride=3, valid_mask=mask), expected)


def test_repair_weights_preserves_mean_and_ess_floor():
    sensitivity = torch.tensor([[100., 1., 1., 1., 1., 1., 1., 1.]], dtype=torch.float64)
    w = repair_weights(sensitivity, feature_width=2)
    assert torch.allclose(w.mean(-1), torch.ones(1, dtype=w.dtype))
    assert (w.sum(-1).square() / w.square().sum(-1) >= 4 - 1e-10).all()
    mask = torch.tensor([[True] * 7 + [False]])
    w_pad = repair_weights(sensitivity, feature_width=2, valid_mask=mask)
    assert w_pad[0, -1] == 0
    assert torch.allclose(w_pad[0, :-1].mean(), torch.ones((), dtype=w_pad.dtype))
