import torch

from kvbridge.cachebridge import cachebridge_metadata
from kvbridge.mapper import Mapper


def make_cachebridge_mapper():
    meta = cachebridge_metadata(
        source_model="source", target_model="target",
        num_source_layers=3, num_target_layers=2,
        num_source_kv_heads=2, num_target_kv_heads=2, head_dim=4,
        head_assignment=[0, 1], selected_source_layers=[[0, 2], [1, 2]],
        rope_theta_source=10000.0, rope_theta_target=10000.0)
    layers = []
    for chosen in meta["selected_source_layers"]:
        p = len(chosen) * 4
        layers.append({
            "source_layers": chosen,
            "W_k": torch.randn(2, p, 4), "mu_xk": torch.randn(2, p),
            "mu_yk": torch.randn(2, 4), "W_v": torch.randn(2, p, 4),
            "mu_xv": torch.randn(2, p), "mu_yv": torch.randn(2, 4),
        })
    meta["rope_theta_source"] = 10000.0
    meta["rope_theta_target"] = 10000.0
    return Mapper(layers, meta)


def test_cachebridge_metadata_and_tensors_round_trip(tmp_path):
    mapper = make_cachebridge_mapper()
    mapper.save(tmp_path / "cb", dtype=torch.float32)
    loaded = Mapper.load(tmp_path / "cb")
    assert loaded.meta["method"] == "cachebridge"
    assert loaded.meta["format_version"] == 2
    assert loaded.meta["head_assignment"] == [0, 1]
    assert loaded.meta["selected_source_layers"] == [[0, 2], [1, 2]]
    for before, after in zip(mapper.layers, loaded.layers):
        assert before["source_layers"] == after["source_layers"]
        for key in ("W_k", "mu_xk", "mu_yk", "W_v", "mu_xv", "mu_yv"):
            assert torch.equal(before[key], after[key])


def test_old_full_head_artifact_remains_compatible(tmp_path):
    p, q = 2 * 2 * 4, 2 * 4
    meta = {"source": "old-source", "target": "old-target", "n_kv_heads": 2,
            "head_dim": 4, "rope_theta_source": 10000.0,
            "rope_theta_target": 10000.0, "top_k": 2, "lam": 0.01}
    layer = {"source_layers": [0, 1], "W_k": torch.randn(p, q),
             "mu_xk": torch.randn(p), "mu_yk": torch.randn(q),
             "W_v": torch.randn(p, q), "mu_xv": torch.randn(p),
             "mu_yv": torch.randn(q)}
    Mapper([layer], meta).save(tmp_path / "old", dtype=torch.float32)
    loaded = Mapper.load(tmp_path / "old")
    assert loaded.meta["format_version"] == 1
    assert loaded.meta.get("method") is None
    assert loaded.source == "old-source"
