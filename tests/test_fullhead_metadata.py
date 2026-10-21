import pytest
import torch
from kvbridge.mapper import Mapper


def meta(method, weighting):
    return {"method": method, "source": "s", "target": "t",
            "num_source_layers": 2, "num_target_layers": 1,
            "num_source_kv_heads": 2, "num_target_kv_heads": 2,
            "n_kv_heads": 2, "head_dim": 2, "head_assignment": [0, 1],
            "selected_source_layers": [[0]], "ridge_lambda": .01,
            "weighting_method": weighting, "token_stride": 4,
            "num_boundaries": 32, "boundary_start_token": 12,
            "rope_theta_source": 1., "rope_theta_target": 1.}


def layer():
    return {"source_layers": [0], "W_k": torch.zeros(2, 4, 2),
            "W_v": torch.zeros(2, 4, 2), "mu_xk": torch.zeros(2, 4),
            "mu_xv": torch.zeros(2, 4), "mu_yk": torch.zeros(2, 2),
            "mu_yv": torch.zeros(2, 2)}


def test_full_head_requires_uniform_weighting():
    Mapper([layer()], meta("full_head", "uniform"))
    with pytest.raises(ValueError, match="weighting_method=uniform"):
        Mapper([layer()], meta("full_head", "attn_repair"))
