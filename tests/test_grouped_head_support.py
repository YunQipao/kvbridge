"""Grouped KV-head CPU regression tests; no real models, network, or CUDA."""

import json
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from kvbridge import model_utils
from kvbridge.cache_compat import cache_layer_kv
from kvbridge.cachebridge import build_head_local_features, cachebridge_metadata
from kvbridge.formal import fit_one_target_layer
from kvbridge.geometry import build_head_map, check_kv_compatibility
from kvbridge.mapper import Mapper
from kvbridge.rope import apply_rope, strip_rope
from kvbridge.transfer import transfer


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network/model download/CUDA is forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(model_utils, "snapshot_download", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def _config(heads, dim=4):
    return SimpleNamespace(num_hidden_layers=2, hidden_size=heads * dim,
        num_attention_heads=heads, num_key_value_heads=heads, head_dim=dim,
        rope_theta=1000000., rope_scaling=None)


def test_auto_geometry_and_explicit_matched():
    assert build_head_map(8, 4) == GROUPS
    assert build_head_map(8, 8) == [[h] for h in range(8)]
    assert check_kv_compatibility(_config(8), _config(4))[0]
    assert not check_kv_compatibility(_config(8), _config(4, 8))[0]
    assert not check_kv_compatibility(_config(4), _config(8))[0]
    assert not check_kv_compatibility(_config(8), _config(3))[0]
    assert not check_kv_compatibility(_config(8), _config(4), "matched")[0]


def test_preflight_reports_grouped_assignment(monkeypatch, capsys):
    import scripts.check_model_geometry as preflight
    monkeypatch.delenv("HEAD_MAP_MODE", raising=False)
    monkeypatch.setattr(preflight, "load_config", lambda name, **kw: _config(8 if name == "s" else 4))
    assert preflight.main(["--source", "s", "--target", "t", "--local-files-only"]) == 0
    output = capsys.readouterr().out
    assert "KV geometry compatible: YES" in output
    assert "assignment_mode: grouped_contiguous" in output
    assert "group_size: 2" in output
    assert f"head_assignment: {json.dumps(GROUPS)}" in output


@pytest.mark.parametrize("backend", ["torch_bounded", "fused_equivalent"])
def test_grouped_assignment_fit_feature_order(tmp_path, backend):
    generator = torch.Generator().manual_seed(31)
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); target.mkdir()
    src = [torch.randn(40, 8, 4, generator=generator, dtype=torch.float64) for _ in range(2)]
    stacked = torch.stack(src)
    features = [build_head_local_features(stacked, [1, 0], h, head_assignment=GROUPS)
                for h in range(4)]
    weights = torch.randn(4, 16, 4, generator=generator, dtype=torch.float64)
    y = torch.stack([features[h] @ weights[h] + h for h in range(4)], dim=1)
    torch.save(dict(keys=src, values=src), source / "source_00000.pt")
    torch.save(dict(keys=[y], values=[y], weights_k=[torch.ones(4, 40)],
                    weights_v=[torch.ones(4, 40)]), target / "target_00000.pt")
    got = fit_one_target_layer(source, target, 0, [1, 0], 4, GROUPS, 4,
        ridge_lambda=0., uniform_weights=True, stats_dtype=torch.float64, backend=backend)
    for head, block in enumerate(GROUPS):
        manual = torch.cat([src[layer][:, block].reshape(40, -1) for layer in [1, 0]], -1)
        torch.testing.assert_close(features[head], manual)
        torch.testing.assert_close(got["W_k"][head], weights[head], atol=1e-9, rtol=1e-9)


def _mapper(assignment, source_heads, selected):
    generator = torch.Generator().manual_seed(41)
    group_size = len(assignment[0]) if isinstance(assignment[0], list) else 1
    p = len(selected) * group_size * 4
    meta = cachebridge_metadata(source_model="s", target_model="t",
        num_source_layers=2, num_target_layers=1, num_source_kv_heads=source_heads,
        num_target_kv_heads=4, head_dim=4, head_assignment=assignment,
        selected_source_layers=[selected], rope_theta_source=1000000., rope_theta_target=1000000.)
    entry = {"source_layers": selected}
    for kind in ("k", "v"):
        entry[f"W_{kind}"] = torch.randn(4, p, 4, generator=generator)
        entry[f"mu_x{kind}"] = torch.randn(4, p, generator=generator)
        entry[f"mu_y{kind}"] = torch.randn(4, 4, generator=generator)
    return Mapper([entry], meta)


def test_grouped_transfer_matches_manual_affine_and_metadata_roundtrip(tmp_path):
    mapper = _mapper(GROUPS, 8, [1, 0])
    assert mapper.meta["head_assignment"] == GROUPS
    # A new artifact need not carry the additional legacy head_map fields.
    meta = dict(mapper.meta)
    meta.pop("head_map"); meta.pop("head_map_mode")
    mapper = Mapper(mapper.layers, meta)
    mapper.save(tmp_path, dtype=torch.float32)
    mapper = Mapper.load(tmp_path)
    assert mapper.meta["head_assignment"] == GROUPS
    generator = torch.Generator().manual_seed(57)
    cache = DynamicCache()
    for layer in range(2):
        key, value = [torch.randn(1, 8, 5, 4, generator=generator) for _ in range(2)]
        cache.update(key, value, layer)
    positions = torch.arange(5)
    output = transfer(cache, mapper, layer_devices=["cpu"])
    entry = mapper.layers[0]
    for component, kind in enumerate(("k", "v")):
        expected = []
        for head, group in enumerate(GROUPS):
            parts = []
            for layer in [1, 0]:
                tensor = cache_layer_kv(cache, layer)[component][0]
                if kind == "k":
                    tensor = strip_rope(tensor, positions, 1000000.)
                parts.append(tensor[group].transpose(0, 1).reshape(5, -1))
            x = torch.cat(parts, -1)
            expected.append((x - entry[f"mu_x{kind}"][head]) @ entry[f"W_{kind}"][head]
                            + entry[f"mu_y{kind}"][head])
        expected = torch.stack(expected).unsqueeze(0)
        if kind == "k":
            expected = apply_rope(expected[0], positions, 1000000.).unsqueeze(0)
        torch.testing.assert_close(cache_layer_kv(output, 0)[component], expected)


def test_grouped_mapper_validator_accepts_feature_width(tmp_path):
    from scripts.validate_mapper import main
    mapper = _mapper(GROUPS, 8, [1, 0])
    mapper.save(tmp_path, dtype=torch.float32)
    assert main(["--mapper", str(tmp_path)]) == 0


def test_harness_prompt_and_choices_are_preserved():
    from experiments.eval_mmlu_redux import _prompt
    row = dict(prompt="Native question\nAnswer:", choices=["A", "B"], answer=0)
    assert _prompt(row) == (row["prompt"], row["choices"])
    example = dict(prompt="Train\nAnswer:", choices=["A", "B"], answer=1)
    assert _prompt(row, [example]) == ("Train\nAnswer: B\n\n" + row["prompt"], row["choices"])


def test_legacy_flat_assignment_without_head_map_still_loads(tmp_path):
    mapper = _mapper([0, 1, 2, 3], 4, [0])
    mapper.save(tmp_path, dtype=torch.float32)
    path = tmp_path / "mapper.json"
    meta = json.loads(path.read_text())
    meta.pop("head_map"); meta.pop("head_map_mode")
    path.write_text(json.dumps(meta))
    loaded = Mapper.load(tmp_path)
    assert loaded.meta["head_assignment"] == [0, 1, 2, 3]
    assert loaded.meta["head_map"] == [[0], [1], [2], [3]]


@pytest.mark.parametrize("mode", ["target_native", "source_native", "full_head", "head_local", "cachebridge"])
@pytest.mark.parametrize("device_map", ["balanced_low_0", "none"])
def test_mmlu_eval_dispatch_and_zero_target_prefix(monkeypatch, tmp_path, mode, device_map):
    import experiments.eval_mmlu_redux as evaluation

    loads, forwards, transfers = [], [], []
    class Model(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.embed = torch.nn.Embedding(16, 4)
            self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(
                q_proj=torch.nn.Linear(4, 4)))])
        def get_input_embeddings(self):
            return self.embed
        def to(self, *args, **kwargs):
            pytest.fail("do not move dispatched models")
        def forward(self, ids, past_key_values=None, **kwargs):
            assert ids.device == self.embed.weight.device
            forwards.append((self.name, past_key_values is None))
            if past_key_values is None:
                cache = DynamicCache()
                tensor = torch.ones(1, 1, ids.shape[1], 4)
                cache.update(tensor, tensor.clone(), 0)
                return SimpleNamespace(past_key_values=cache)
            assert kwargs["attention_mask"].device == ids.device
            assert kwargs["position_ids"].device == ids.device
            return SimpleNamespace(logits=torch.zeros(ids.shape[0], ids.shape[1], 16))
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [1, 2, 3] if add_special_tokens else [4]
            return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
    def load(name, **kwargs):
        assert kwargs["device_map"] == (None if device_map == "none" else device_map)
        assert kwargs["local_files_only"]
        loads.append(name)
        return Model(name)
    def map_cache(cache, mapper, **kwargs):
        assert kwargs == ({} if device_map == "none" else {"layer_devices": [torch.device("cpu")]})
        transfers.append(cache)
        return cache
    monkeypatch.setattr(evaluation, "load_causal_lm", load)
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(evaluation.Mapper, "load", lambda *a, **kw: SimpleNamespace(meta={"method": mode}))
    monkeypatch.setattr(evaluation, "transfer", map_cache)
    monkeypatch.setattr(evaluation, "_rows", lambda *a, **kw: [dict(
        subject="toy", question="Question?", choices=["a", "b", "c", "d"], answer=0)])
    output = tmp_path / "eval.json"
    argv = ["eval", "--mode", mode, "--source", "source", "--target", "target",
        "--device", "cpu", "--device-map", device_map, "--local-files-only",
        "--protocol", "simple", "--num-eval-samples", "1", "--output", str(output)]
    mapped = mode in ("full_head", "head_local", "cachebridge")
    if mapped:
        argv.extend(["--mapper", "mock"])
    monkeypatch.setattr(sys, "argv", argv)
    evaluation.main()
    result = json.loads(output.read_text())
    assert result["total"] == 1
    if mapped:
        assert loads == ["target", "source"]
        assert forwards == [("source", True), ("target", False)]
        assert result["target_prefix_prefill_tokens"] == 0
        assert len(transfers) == 1
    else:
        name = "target" if mode == "target_native" else "source"
        assert loads == [name]
        assert forwards == [(name, True), (name, False)]
    assert (tmp_path / "mmlu_redux_summary.csv").is_file()
