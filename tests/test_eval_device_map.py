"""Offline CPU/mock contracts for sharded evaluation. No real models or CUDA."""

import json
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from kvbridge import model_utils
from kvbridge.cache_compat import cache_layer_kv


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests must not access network, download models, or initialize CUDA")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(model_utils, "snapshot_download", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


@pytest.mark.parametrize("device_map", [None, "none"])
def test_loader_single_device_path_retained(monkeypatch, device_map):
    calls = []
    class Model:
        config = SimpleNamespace(model_type="qwen3")
        def to(self, device):
            calls.append(("to", device))
            return self
        def eval(self):
            return self
    def load(path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["dtype"] is torch.float32
        assert "device_map" not in kwargs
        calls.append(("load", path))
        return Model()
    monkeypatch.setattr(model_utils.AutoModelForCausalLM, "from_pretrained", load)
    model_utils.load_causal_lm("mock-qwen", device="cpu", device_map=device_map,
                               local_files_only=True)
    assert calls == [("load", "mock-qwen"), ("to", "cpu")]


@pytest.mark.parametrize("device_map", ["auto", "balanced", "balanced_low_0", "sequential"])
@pytest.mark.parametrize("dtype, expected", [("auto", "auto"), ("bfloat16", torch.bfloat16)])
def test_sharded_loader_passes_map_and_never_moves_model(monkeypatch, device_map, dtype, expected):
    calls = []
    class Model:
        config = SimpleNamespace(model_type="qwen3")
        def to(self, *args, **kwargs):
            pytest.fail("Accelerate-dispatched model must never receive model.to()")
        def eval(self):
            return self
    def load(path, **kwargs):
        calls.append(kwargs)
        assert kwargs["device_map"] == device_map
        assert kwargs["dtype"] == expected
        assert kwargs["local_files_only"] is True
        assert kwargs["attn_implementation"] == "sdpa"
        return Model()
    monkeypatch.setattr(model_utils.AutoModelForCausalLM, "from_pretrained", load)
    model_utils.load_causal_lm("mock-qwen", device="cpu", device_map=device_map,
        dtype=dtype, local_files_only=True, attn_implementation="sdpa")
    assert len(calls) == 1


def test_dtype_keyword_fallback_preserves_dispatch(monkeypatch):
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    model.eval = lambda: model
    def load(path, **kwargs):
        calls.append(kwargs)
        assert kwargs["device_map"] == "balanced_low_0"
        if "dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'dtype'")
        assert kwargs["torch_dtype"] == "auto"
        return model  # deliberately has no .to()
    monkeypatch.setattr(model_utils.AutoModelForCausalLM, "from_pretrained", load)
    assert model_utils.load_causal_lm("mock", device_map="balanced_low_0") is model
    assert len(calls) == 2


def test_input_device_comes_from_embeddings_not_first_map_entry():
    embeddings = torch.nn.Embedding(8, 4)
    model = SimpleNamespace(get_input_embeddings=lambda: embeddings,
        hf_device_map={"lm_head": "meta", "model.embed_tokens": "cpu"})
    assert model_utils.get_model_input_device(model) == embeddings.weight.device
    assert model_utils.model_input_device(model) == embeddings.weight.device


def test_offloaded_embeddings_use_execution_hook_and_reject_unresolved_meta():
    embeddings = torch.nn.Embedding(8, 4, device="meta")
    embeddings._hf_hook = SimpleNamespace(execution_device="cpu")
    model = SimpleNamespace(get_input_embeddings=lambda: embeddings)
    assert model_utils.get_model_input_device(model) == torch.device("cpu")
    del embeddings._hf_hook
    with pytest.raises(RuntimeError, match="meta module"):
        model_utils.get_model_input_device(model)


def test_target_layer_devices_follow_attention_execution_hooks():
    projections = [torch.nn.Linear(4, 4) for _ in range(2)]
    projections[1]._hf_hook = SimpleNamespace(execution_device="cpu")
    layers = [SimpleNamespace(self_attn=SimpleNamespace(q_proj=p)) for p in projections]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    assert model_utils.get_model_layer_devices(model) == [torch.device("cpu")] * 2


def test_candidate_cache_expansion_preserves_each_layer_device():
    from experiments.eval_mmlu_redux import _expand
    cache = DynamicCache()
    for layer, device in enumerate(("cpu", "meta")):
        tensor = torch.empty(1, 1, 3, 2, device=device)
        cache.update(tensor, tensor.clone(), layer)
    expanded = _expand(cache, 4)
    for layer, device in enumerate(("cpu", "meta")):
        key, value = cache_layer_kv(expanded, layer)
        assert key.device.type == value.device.type == device
        assert key.shape == value.shape == (4, 1, 3, 2)


@pytest.mark.parametrize("mode", ["target_native", "full_head", "head_local", "cachebridge"])
def test_eval_dispatches_all_models_and_keeps_zero_mapped_prefix(monkeypatch, tmp_path, mode):
    import experiments.eval_paper_tasks as paper
    loads, forwards, maps = [], [], []
    class Model(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.embeddings = torch.nn.Embedding(16, 4)
            self.model = SimpleNamespace(layers=[SimpleNamespace(
                self_attn=SimpleNamespace(q_proj=torch.nn.Linear(4, 4)))])
        def get_input_embeddings(self):
            return self.embeddings
        def to(self, *args, **kwargs):
            pytest.fail("evaluation must not move a dispatched source or target model")
        def forward(self, ids, past_key_values=None, use_cache=False, **kwargs):
            assert ids.device == self.embeddings.weight.device
            forwards.append((self.name, past_key_values is None))
            if past_key_values is None:
                cache = DynamicCache()
                key = torch.ones(1, 1, ids.shape[1], 2)
                cache.update(key, key.clone(), 0)
                return SimpleNamespace(past_key_values=cache)
            assert kwargs["attention_mask"].device == ids.device
            assert kwargs["position_ids"].device == ids.device
            return SimpleNamespace(logits=torch.zeros(ids.shape[0], ids.shape[1], 16))
    def load(name, **kwargs):
        assert kwargs["device_map"] == "balanced_low_0"
        loads.append(name)
        return Model(name)
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = [1, 2, 3] if add_special_tokens else [4, 5]
            return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)
    def load_mapper(path, **kwargs):
        assert kwargs["device"] == "cpu"
        return SimpleNamespace(meta={"method": mode})
    def map_cache(cache, mapper, *, layer_devices):
        assert layer_devices == [torch.device("cpu")]
        maps.append(cache)
        return cache
    monkeypatch.setattr(paper, "load_causal_lm", load)
    monkeypatch.setattr(paper, "load_tokenizer", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(paper.Mapper, "load", load_mapper)
    monkeypatch.setattr(paper, "transfer", map_cache)
    monkeypatch.setattr(paper, "load_rows", lambda *a: [
        {"prompt": "prompt", "choices": ["a", "b"], "answer": 0}])
    output = tmp_path / "eval.json"
    args = ["eval", "--task", "HellaSwag", "--mode", mode,
        "--source", "source", "--target", "target", "--device", "cpu",
        "--device-map", "balanced_low_0", "--local-files-only",
        "--protocol", "simple", "--num-eval-samples", "1", "--output", str(output)]
    if mode != "target_native":
        args.extend(["--mapper", "mock"])
    monkeypatch.setattr(sys, "argv", args)
    paper.main()
    result = json.loads(output.read_text())
    assert result["total"] == 1
    assert loads == (["target"] if mode == "target_native" else ["target", "source"])
    if mode != "target_native":
        assert result["target_prefix_prefill_tokens"] == 0
        assert result["source_prefix_prefill_tokens"] == 2
        assert forwards == [("source", True), ("target", False)]
        assert len(maps) == 1
    else:
        assert result["target_prefix_prefill_tokens"] == 2
        assert forwards == [("target", True), ("target", False)]


@pytest.mark.parametrize("method", ["cachebridge", "head_local", "full_head", "legacy"])
def test_staged_layer_transfer_matches_legacy_without_moving_mapper(monkeypatch, method):
    from kvbridge.cachebridge import cachebridge_metadata
    from kvbridge.mapper import Mapper, TENSOR_KEYS
    from kvbridge.transfer import transfer
    heads, dim, tokens = 2, 4, 7
    if method == "legacy":
        meta = dict(source="s", target="t", n_kv_heads=heads, head_dim=dim,
                    rope_theta_source=10000., rope_theta_target=100000.)
    else:
        meta = cachebridge_metadata(source_model="s", target_model="t",
            num_source_layers=3, num_target_layers=2, num_source_kv_heads=heads,
            num_target_kv_heads=heads, head_dim=dim, head_assignment=[1, 0],
            selected_source_layers=[[2, 0], [1, 2]], rope_theta_source=10000.,
            rope_theta_target=100000., weighting_method="attn_repair" if method == "cachebridge" else "uniform")
        meta["method"] = method
    generator = torch.Generator().manual_seed(71)
    layers = []
    for chosen in ([2, 0], [1, 2]):
        p = len(chosen) * dim * (heads if method in ("full_head", "legacy") else 1)
        shapes = ((p, heads*dim), (p,), (heads*dim,)) if method == "legacy" else (
            (heads, p, dim), (heads, p), (heads, dim))
        entry = {"source_layers": chosen}
        for suffix in ("k", "v"):
            for key, shape in zip((f"W_{suffix}", f"mu_x{suffix}", f"mu_y{suffix}"), shapes):
                entry[key] = torch.randn(*shape, generator=generator)
        layers.append(entry)
    mapper = Mapper(layers, meta)
    before = [{key: entry[key].clone() for key in TENSOR_KEYS} for entry in layers]
    cache = DynamicCache()
    for layer in range(3):
        cache.update(torch.randn(1, heads, tokens, dim, generator=generator),
                     torch.randn(1, heads, tokens, dim, generator=generator), layer)
    positions = torch.arange(100, 100 + tokens)
    expected = transfer(cache, mapper, positions)
    def forbidden(*args, **kwargs):
        pytest.fail("staged transfer must not call Mapper.to()")
    monkeypatch.setattr(mapper, "to", forbidden)
    actual = transfer(cache, mapper, positions, layer_devices=["cpu", "cpu"])
    for layer in range(2):
        for left, right in zip(cache_layer_kv(actual, layer), cache_layer_kv(expected, layer)):
            torch.testing.assert_close(left, right)
        for key in TENSOR_KEYS:
            torch.testing.assert_close(mapper.layers[layer][key], before[layer][key])
            assert mapper.layers[layer][key].device.type == "cpu"
    with pytest.raises(ValueError, match="one device per target"):
        transfer(cache, mapper, layer_devices=["cpu"])


def test_staged_transfer_does_not_materialize_unselected_source_layers(monkeypatch):
    from kvbridge.mapper import Mapper
    from kvbridge.transfer import transfer
    # Meta is a CPU-only device stand-in: touching data from this unused
    # source layer would fail. Only source layer 1 belongs in the feature panel.
    cache = DynamicCache()
    cache.update(torch.empty(1, 1, 3, 2, device="meta"),
                 torch.empty(1, 1, 3, 2, device="meta"), 0)
    key, value = torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2)
    cache.update(key, value, 1)
    entry = {"source_layers": [1], "W_k": torch.eye(2), "W_v": torch.eye(2),
        "mu_xk": torch.zeros(2), "mu_xv": torch.zeros(2),
        "mu_yk": torch.zeros(2), "mu_yv": torch.zeros(2)}
    mapper = Mapper([entry], dict(source="s", target="t", n_kv_heads=1,
        head_dim=2, rope_theta_source=10000., rope_theta_target=10000.))
    def forbidden(*args, **kwargs):
        pytest.fail("staged transfer must not gather the complete mapper")
    monkeypatch.setattr(mapper, "to", forbidden)
    actual = transfer(cache, mapper, positions=torch.arange(3), layer_devices=["cpu"])
    output_key, output_value = cache_layer_kv(actual, 0)
    torch.testing.assert_close(output_key, key)
    torch.testing.assert_close(output_value, value)
    assert cache_layer_kv(cache, 0)[0].device.type == "meta"
