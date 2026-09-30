import pytest
import torch
from transformers import DynamicCache

from kvbridge.cachebridge import cachebridge_metadata
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.mapper import Mapper
from kvbridge.transfer import transfer


def _mapper():
    meta = cachebridge_metadata(
        source_model="synthetic/source", target_model="synthetic/target",
        num_source_layers=2, num_target_layers=1, num_source_kv_heads=2,
        num_target_kv_heads=2, head_dim=4, head_assignment=[1, 0],
        selected_source_layers=[[1, 0]], rope_theta_source=10000.,
        rope_theta_target=10000.)
    weight = torch.zeros(2, 8, 4)
    weight[:, :4, :] = torch.eye(4)
    zeros_x = torch.zeros(2, 8)
    zeros_y = torch.zeros(2, 4)
    return Mapper([{"source_layers": [1, 0],
                    "W_k": weight.clone(), "W_v": weight.clone(),
                    "mu_xk": zeros_x.clone(), "mu_xv": zeros_x.clone(),
                    "mu_yk": zeros_y.clone(), "mu_yv": zeros_y.clone()}], meta)


def _cache():
    cache = DynamicCache()
    for layer in range(2):
        k = torch.zeros(1, 2, 3, 4)
        v = torch.zeros_like(k)
        for head in range(2):
            k[:, head] = 10 * layer + head + 1
            v[:, head] = 10 * layer + head + 1
        cache.update(k, v, layer)
    return cache


def test_transfer_uses_only_selected_layer_and_assigned_head():
    out = transfer(_cache(), _mapper())
    k, v = cache_layer_kv(out, 0)
    assert k.shape == v.shape == (1, 2, 3, 4)
    assert torch.allclose(v[0, 0], torch.full((3, 4), 12.))
    assert torch.allclose(v[0, 1], torch.full((3, 4), 11.))


def test_unknown_method_cannot_fallback_to_full_head():
    mapper = _mapper()
    mapper.meta["method"] = "unknown"
    with pytest.raises(ValueError, match="unsupported mapper method"):
        transfer(_cache(), mapper)
