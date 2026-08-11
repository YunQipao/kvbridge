"""End-to-end transfer against a mapper whose answer is known exactly.

With identity weights, zero means and equal RoPE thetas, `transfer` must
return the source cache unchanged: strip -> identity -> re-apply. Anything
that breaks (head/feature reshaping, layer selection order, position
handling) shows up as a mismatch here without needing a real model.
"""

import pytest
import torch
from transformers import DynamicCache

from kvbridge.cache_compat import cache_layer_kv
from kvbridge.mapper import Mapper
from kvbridge.transfer import transfer

H, D, T = 4, 32, 48
THETA = 10_000.0


def make_cache(n_layers, batch=1, seed=0):
    torch.manual_seed(seed)
    cache = DynamicCache()
    for layer in range(n_layers):
        k = torch.randn(batch, H, T, D)
        v = torch.randn(batch, H, T, D)
        cache.update(k, v, layer)
    return cache


def identity_mapper(n_target_layers, source_of, theta_t=THETA):
    """Target layer i copies source layer `source_of(i)` verbatim."""
    eye = torch.eye(H * D)
    zero = torch.zeros(H * D)
    layers = [{"source_layers": [source_of(i)],
               "W_k": eye.clone(), "mu_xk": zero.clone(), "mu_yk": zero.clone(),
               "W_v": eye.clone(), "mu_xv": zero.clone(), "mu_yv": zero.clone()}
              for i in range(n_target_layers)]
    return Mapper(layers, {"source": "src", "target": "tgt", "n_kv_heads": H,
                           "head_dim": D, "rope_theta_source": THETA,
                           "rope_theta_target": theta_t, "top_k": 1})


def test_identity_mapper_round_trips_the_cache():
    cache = make_cache(4)
    mapper = identity_mapper(4, lambda i: i).to(dtype=torch.float32)
    out = transfer(cache, mapper)
    for layer in range(4):
        sk, sv = cache_layer_kv(cache, layer)
        ok, ov = cache_layer_kv(out, layer)
        assert torch.allclose(sk, ok, atol=1e-4), f"keys differ at layer {layer}"
        assert torch.allclose(sv, ov, atol=1e-5), f"values differ at layer {layer}"


def test_layer_selection_is_honoured():
    """A mapper that reads source layer 3 for every target layer must produce
    four copies of layer 3, not four copies of the layer index."""
    cache = make_cache(4)
    mapper = identity_mapper(4, lambda _: 3).to(dtype=torch.float32)
    out = transfer(cache, mapper)
    _, sv3 = cache_layer_kv(cache, 3)
    for layer in range(4):
        assert torch.allclose(cache_layer_kv(out, layer)[1], sv3, atol=1e-5)


def test_target_layer_count_can_exceed_source():
    cache = make_cache(4)
    mapper = identity_mapper(9, lambda i: i % 4).to(dtype=torch.float32)
    out = transfer(cache, mapper)
    assert len(out.layers) == 9 if hasattr(out, "layers") else True
    assert cache_layer_kv(out, 8)[0].shape == (1, H, T, D)


def test_mid_sequence_slices_round_trip():
    """`positions` says where the cached tokens really live, and is used for
    both the strip and the re-apply -- so mapping a mid-sequence slice at its
    true offset is position-invariant. This is what lets `hybrid` map the
    middle of a context and splice it between two natively prefilled regions."""
    cache = make_cache(2)
    mapper = identity_mapper(2, lambda i: i).to(dtype=torch.float32)
    shifted = transfer(cache, mapper, positions=torch.arange(100, 100 + T))
    for layer in range(2):
        sk, sv = cache_layer_kv(cache, layer)
        ok, ov = cache_layer_kv(shifted, layer)
        assert torch.allclose(sk, ok, atol=1e-4)
        assert torch.allclose(sv, ov, atol=1e-5)


def test_positions_matter_when_thetas_differ():
    """With different source/target thetas the two rotations no longer cancel,
    so the same cache mapped at a different offset gives different keys."""
    cache = make_cache(1)
    mapper = identity_mapper(1, lambda i: i, theta_t=500_000.0).to(dtype=torch.float32)
    at_zero = cache_layer_kv(transfer(cache, mapper), 0)[0]
    at_100 = cache_layer_kv(
        transfer(cache, mapper, positions=torch.arange(100, 100 + T)), 0)[0]
    assert not torch.allclose(at_zero, at_100, atol=1e-2)


def test_theta_mismatch_is_applied():
    """Source and target with different rope_theta must not round-trip: the
    keys come back rotated in the target's convention."""
    cache = make_cache(1)
    mapper = identity_mapper(1, lambda i: i, theta_t=500_000.0).to(dtype=torch.float32)
    out = transfer(cache, mapper)
    assert not torch.allclose(cache_layer_kv(cache, 0)[0],
                              cache_layer_kv(out, 0)[0], atol=1e-2)


def test_rejects_batched_cache():
    mapper = identity_mapper(2, lambda i: i).to(dtype=torch.float32)
    with pytest.raises(ValueError, match="batch"):
        transfer(make_cache(2, batch=3), mapper)


def test_rejects_cache_from_the_wrong_model():
    """A cache with fewer layers than the mapper selects is the most likely
    misuse (source and target swapped, or the wrong mapper loaded)."""
    mapper = identity_mapper(4, lambda i: i).to(dtype=torch.float32)
    with pytest.raises(ValueError, match="source cache has"):
        transfer(make_cache(2), mapper)


def test_rejects_position_length_mismatch():
    mapper = identity_mapper(1, lambda i: i).to(dtype=torch.float32)
    with pytest.raises(ValueError, match="positions"):
        transfer(make_cache(1), mapper, positions=torch.arange(T - 1))
