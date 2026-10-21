import pytest
import torch

from kvbridge.cachebridge import cachebridge_metadata
from kvbridge.mapper import Mapper


def test_cachebridge_artifact_rejects_wrong_head_local_width():
    meta = cachebridge_metadata(
        source_model="s", target_model="t", num_source_layers=2,
        num_target_layers=1, num_source_kv_heads=2, num_target_kv_heads=2,
        head_dim=3, head_assignment=[0, 1], selected_source_layers=[[0, 1]],
        rope_theta_source=10000, rope_theta_target=10000)
    bad = {"source_layers": [0, 1],
           "W_k": torch.zeros(2, 12, 3), "W_v": torch.zeros(2, 6, 3),
           "mu_xk": torch.zeros(2, 6), "mu_xv": torch.zeros(2, 6),
           "mu_yk": torch.zeros(2, 3), "mu_yv": torch.zeros(2, 3)}
    with pytest.raises(ValueError, match="invalid shape"):
        Mapper([bad], meta)


@pytest.mark.parametrize("debug_flags", [
    {"debug_run": True},
    {"smoke_source_layers": 1},
    {"layer_selection_method": "fixed_debug_support"},
])
def test_paper_validator_rejects_debug_support(monkeypatch, debug_flags):
    from types import SimpleNamespace

    from scripts import validate_mapper

    meta = cachebridge_metadata(
        source_model="s", target_model="t", num_source_layers=2,
        num_target_layers=1, num_source_kv_heads=2, num_target_kv_heads=2,
        head_dim=3, head_assignment=[0, 1], selected_source_layers=[[0]],
        rope_theta_source=10000, rope_theta_target=10000)
    meta.update(debug_flags)
    monkeypatch.setattr(validate_mapper.Mapper, "load",
                        lambda *args, **kwargs: SimpleNamespace(meta=meta))
    with pytest.raises(RuntimeError, match="debug/smoke support"):
        validate_mapper.main(["--mapper", "unused", "--paper-faithful"])
