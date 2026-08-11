"""Mapper serialization: a saved mapper must load back identically, and a
mapper written by a future format version must refuse to load rather than
produce silently wrong caches.
"""

import json

import pytest
import torch

from kvbridge.mapper import Mapper

H, D, K = 4, 32, 2
META = {"source": "org/small", "target": "org/big", "n_kv_heads": H,
        "head_dim": D, "rope_theta_source": 10_000.0,
        "rope_theta_target": 1_000_000.0, "top_k": K, "lam": 0.01}


def make_mapper(n_layers=3):
    torch.manual_seed(0)
    p, q = K * H * D, H * D
    layers = [{"source_layers": [i, i + 1],
               "W_k": torch.randn(p, q), "mu_xk": torch.randn(p), "mu_yk": torch.randn(q),
               "W_v": torch.randn(p, q), "mu_xv": torch.randn(p), "mu_yv": torch.randn(q)}
              for i in range(n_layers)]
    return Mapper(layers, META)


def test_round_trip_preserves_weights_and_selection(tmp_path):
    mapper = make_mapper()
    mapper.save(tmp_path / "m", dtype=torch.float32)
    loaded = Mapper.load(tmp_path / "m")

    assert loaded.source == mapper.source and loaded.target == mapper.target
    assert loaded.rope_theta_target == mapper.rope_theta_target
    assert len(loaded.layers) == len(mapper.layers)
    for a, b in zip(mapper.layers, loaded.layers):
        assert a["source_layers"] == b["source_layers"]
        for key in ("W_k", "mu_xk", "mu_yk", "W_v", "mu_xv", "mu_yv"):
            assert torch.equal(a[key], b[key]), key


def test_bfloat16_is_the_default_on_disk(tmp_path):
    make_mapper().save(tmp_path / "m")
    assert Mapper.load(tmp_path / "m").dtype == torch.bfloat16


def test_from_pretrained_accepts_a_local_directory(tmp_path):
    make_mapper().save(tmp_path / "m")
    assert Mapper.from_pretrained(str(tmp_path / "m")).top_k == K


def test_rejects_unknown_format_version(tmp_path):
    make_mapper().save(tmp_path / "m")
    path = tmp_path / "m" / "mapper.json"
    meta = json.loads(path.read_text())
    meta["format_version"] = Mapper.FORMAT_VERSION + 1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version"):
        Mapper.load(tmp_path / "m")


def test_rejects_incomplete_metadata():
    with pytest.raises(ValueError, match="missing required keys"):
        Mapper([], {"source": "org/small"})


def test_to_is_idempotent_and_returns_self():
    mapper = make_mapper()
    assert mapper.to(dtype=torch.float32) is mapper
    mapper.to(dtype=torch.bfloat16).to(dtype=torch.bfloat16)
    assert mapper.dtype == torch.bfloat16
