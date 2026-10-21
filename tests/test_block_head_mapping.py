"""Offline CPU contracts for explicit Qwen2.5 block-head mapping."""

from types import SimpleNamespace

import pytest
import torch

from kvbridge.cachebridge import build_head_local_features, cachebridge_metadata
from kvbridge.geometry import build_head_map, check_kv_compatibility
from kvbridge.mapper import Mapper
from kvbridge.transfer import transfer
from kvbridge.cache_compat import cache_layer_kv
from transformers import DynamicCache


def config(heads, layers=4, dim=16):
    return SimpleNamespace(num_hidden_layers=layers, hidden_size=heads * dim,
        num_attention_heads=heads, num_key_value_heads=heads,
        head_dim=dim, rope_theta=1_000_000., rope_scaling=None)


def test_matched_and_block_maps_are_explicit():
    assert build_head_map(8, 8, "matched") == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert build_head_map(8, 4, "block") == [[0, 1], [2, 3], [4, 5], [6, 7]]
    with pytest.raises(ValueError, match="equal"):
        build_head_map(8, 4, "matched")
    with pytest.raises(ValueError, match="source KV heads"):
        build_head_map(4, 8, "block")


def test_geometry_mode_controls_compatibility():
    source, target = config(8), config(4)
    source.num_hidden_layers, target.num_hidden_layers = 48, 28
    assert not check_kv_compatibility(source, target, "matched")[0]
    assert check_kv_compatibility(source, target)[0]
    compatible, src, tgt = check_kv_compatibility(source, target, "block")
    assert compatible and src["head_map"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert tgt["head_map_mode"] == "block"


def test_block_feature_width_and_order():
    source = torch.arange(4 * 5 * 8 * 16, dtype=torch.float32).reshape(4, 5, 8, 16)
    got = build_head_local_features(source, [2, 0, 3], 2,
                                    head_map=[[0, 1], [2, 3], [4, 5], [6, 7]])
    expected = torch.cat([source[layer, :, [4, 5], :].reshape(5, -1)
                          for layer in [2, 0, 3]], dim=-1)
    assert got.shape == (5, 3 * 2 * 16)
    torch.testing.assert_close(got, expected)


def _block_mapper(method="head_local"):
    meta = cachebridge_metadata(source_model="s", target_model="t",
        num_source_layers=1, num_target_layers=1, num_source_kv_heads=8,
        num_target_kv_heads=4, head_dim=2, head_assignment=[0, 2, 4, 6],
        head_map=[[0, 1], [2, 3], [4, 5], [6, 7]], head_map_mode="block",
        selected_source_layers=[[0]], rope_theta_source=1_000_000.,
        rope_theta_target=1_000_000.)
    meta["method"] = method
    p = 4
    eye = torch.eye(p)[:, :2]
    entry = {"source_layers": [0], "W_k": eye[None].expand(4, -1, -1).clone(),
             "W_v": eye[None].expand(4, -1, -1).clone(),
             "mu_xk": torch.zeros(4, p), "mu_xv": torch.zeros(4, p),
             "mu_yk": torch.zeros(4, 2), "mu_yv": torch.zeros(4, 2)}
    return Mapper([entry], meta)


def test_mapper_save_load_preserves_block_map(tmp_path):
    mapper = _block_mapper()
    mapper.save(tmp_path, dtype=torch.float32)
    loaded = Mapper.load(tmp_path, dtype=torch.float32)
    assert loaded.meta["head_map"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert loaded.meta["head_map_mode"] == "block"


def test_block_transfer_reads_only_each_assigned_source_block():
    mapper = _block_mapper()
    cache = DynamicCache()
    values = torch.arange(8 * 3 * 2, dtype=torch.float32).reshape(1, 8, 3, 2)
    cache.update(values.clone(), values.clone(), 0)
    out = transfer(cache, mapper)
    key, value = cache_layer_kv(out, 0)
    for target, block in enumerate([[0, 1], [2, 3], [4, 5], [6, 7]]):
        expected = values[0, block].transpose(0, 1).reshape(3, -1)[:, :2]
        torch.testing.assert_close(value[0, target], expected)
    # A changed source head outside a target block cannot affect that target.
    values[:, 7] += 10000
    cache2 = DynamicCache(); cache2.update(values, values.clone(), 0)
    out2 = transfer(cache2, mapper)
    torch.testing.assert_close(cache_layer_kv(out2, 0)[1][0, 0], value[0, 0])


def test_mapper_validates_layers_independently_of_head_count():
    mapper = _block_mapper()
    meta = dict(mapper.meta, num_target_layers=7, selected_source_layers=[[0]] * 7)
    loaded = Mapper([mapper.layers[0]] * 7, meta)
    assert len(loaded.layers) == 7


def test_block_feature_rejects_out_of_range_target_head():
    source = torch.zeros(1, 3, 8, 2)
    with pytest.raises(ValueError, match="target_head"):
        build_head_local_features(source, [0], 4,
                                  head_map=build_head_map(8, 4, "block"))


@pytest.mark.parametrize("backend", ["torch_bounded", "fused_equivalent"])
def test_block_and_full_head_fit_widths_and_ess(tmp_path, monkeypatch, backend):
    import kvbridge.formal as formal

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    generator = torch.Generator().manual_seed(19)
    src = {name: [torch.randn(30, 8, 2, generator=generator) for _ in range(3)]
           for name in ("keys", "values")}
    tgt = {name: [torch.randn(30, 4, 2, generator=generator)]
           for name in ("keys", "values")}
    tgt.update(weights_k=[torch.ones(4, 30)], weights_v=[torch.ones(4, 30)])
    torch.save(src, source / "source_00000.pt")
    torch.save(tgt, target / "target_00000.pt")
    widths = []
    repair = formal.repair_weights

    def tracked_repair(weights, *, feature_width):
        widths.append(feature_width)
        return repair(weights, feature_width=feature_width)

    monkeypatch.setattr(formal, "repair_weights", tracked_repair)
    block = formal.fit_one_target_layer(source, target, 0, [2, 0, 1], 4,
        [0, 2, 4, 6], 2, head_map=build_head_map(8, 4, "block"), backend=backend,
        stats_dtype=torch.float64)
    assert widths == [12] * 8  # k=3, two source heads, d=2; both K and V.
    assert block["W_k"].shape == (4, 12, 2)
    full = formal.fit_one_target_layer_full_head(source, target, 0, [2, 0, 1], 4, 2,
        source_panel_heads=8, backend=backend, stats_dtype=torch.float64)
    assert full["W_k"].shape == (4, 48, 2)
    expected = torch.cat([src["keys"][layer].reshape(30, -1)
                          for layer in [2, 0, 1]], dim=-1).double().mean(0)
    torch.testing.assert_close(full["mu_xk"], expected[None].expand(4, -1))
