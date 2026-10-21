"""CPU-only offline coverage. Never loads checkpoint weights or downloads data."""

import json
import socket
import sys
from types import SimpleNamespace, ModuleType

import pytest
import torch

from kvbridge.geometry import geometry_dict, check_kv_compatibility, text_config
from kvbridge import model_utils
from kvbridge.rope import apply_rope, strip_rope, rope_spec, get_rope_theta
from kvbridge.formal import capture_native_ministral_query, check_tokenizer_alignment
from kvbridge.protocol import validate_mapped_evaluation_accounting


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Ministral unit tests must not use network, Hub or CUDA")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(model_utils, "snapshot_download", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def geometry(layers, hidden):
    return SimpleNamespace(model_type="ministral3", num_hidden_layers=layers,
        hidden_size=hidden, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, rope_parameters={"rope_type": "yarn", "rope_theta": 1e6})


def test_nested_text_geometry_and_eight_identity_heads():
    source = SimpleNamespace(model_type="mistral3", text_config=geometry(26, 3072),
                             vision_config=geometry(999, 999))
    target = SimpleNamespace(model_type="mistral3", text_config=geometry(40, 5120))
    compatible, s, t = check_kv_compatibility(source, target)
    assert compatible
    assert (s["num_hidden_layers"], s["hidden_size"], t["num_hidden_layers"], t["hidden_size"]) == (26, 3072, 40, 5120)
    assert s["head_dim"] == t["head_dim"] == 128
    assert list(range(s["num_key_value_heads"])) == list(range(8))
    assert get_rope_theta(source) == 1e6
    assert s["rope_parameters"]["rope_type"] == "yarn"


def test_qwen_config_and_rope_keep_original_path():
    qwen = SimpleNamespace(model_type="qwen3", rope_theta=10000.,
                           text_config=geometry(999, 999))
    assert text_config(qwen) is qwen
    assert rope_spec(qwen) == 10000.
    x = torch.randn(2, 7, 8)
    positions = torch.arange(7)
    torch.testing.assert_close(strip_rope(apply_rope(x, positions, 10000.), positions, 10000.), x)
    plain = SimpleNamespace(config=qwen)
    assert model_utils.unwrap_text_model(plain) is plain


def test_composite_unwrap_reuses_decoder_and_lm_head(monkeypatch):
    decoder = torch.nn.Module()
    decoder.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    head = torch.nn.Linear(4, 8)
    config = SimpleNamespace(model_type="mistral3", text_config=geometry(1, 4))
    composite = SimpleNamespace(config=config, model=SimpleNamespace(language_model=decoder),
                                get_output_embeddings=lambda: head)
    class Causal(torch.nn.Module):
        def set_output_embeddings(self, value):
            self.lm_head = value
    def make(cfg):
        assert cfg is config.text_config
        return Causal()
    monkeypatch.setattr(model_utils.AutoModelForCausalLM, "from_config", make)
    causal = model_utils.unwrap_text_model(composite)
    assert causal.model is decoder and causal.lm_head is head
    assert next(causal.model.parameters()) is next(decoder.parameters())
    with pytest.raises(ValueError, match="structure"):
        model_utils.unwrap_text_model(SimpleNamespace(config=config,
            model=SimpleNamespace(), get_output_embeddings=lambda: head))


def test_local_text_weight_namespace_is_verified(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.embed_tokens.weight": "shard.safetensors",
        "lm_head.weight": "shard.safetensors",
        "model.vision_tower.fake.weight": "vision.safetensors"}}))
    config = SimpleNamespace(model_type="mistral3", text_config=geometry(26, 3072))
    kwargs = model_utils.ministral_load_kwargs(tmp_path, config)
    assert kwargs["config"] is config.text_config
    assert kwargs["output_loading_info"] is True
    assert kwargs["key_mapping"] == {r"^model\.language_model\.": "model."}
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {"vision.weight": "v"}}')
    with pytest.raises(ValueError, match="namespace"):
        model_utils.ministral_load_kwargs(tmp_path, config)


def test_native_query_capture_needs_no_q_norm_and_restores(monkeypatch):
    module = ModuleType("toy_ministral_native")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    def eager(attention, query, *args, **kwargs):
        calls.append(query)
        return query.square().sum()
    module.eager_attention_forward = eager
    Attention = type("Attention", (), {"__module__": module.__name__})
    attention, other = Attention(), Attention()
    assert not hasattr(attention, "q_norm")
    captured = []
    handle = capture_native_ministral_query(attention, captured.append)
    q = torch.randn(1, 32, 1, 4) * 1.7  # already transformed native query
    try:
        module.eager_attention_forward(other, q)
        result = module.eager_attention_forward(attention, q)
        assert len(captured) == 1 and captured[0] is q
        torch.testing.assert_close(result, q.square().sum())
    finally:
        handle.remove()
    assert module.eager_attention_forward is eager


def yarn_config(**overrides):
    configs = pytest.importorskip("transformers.models.ministral3.configuration_ministral3")
    params = dict(rope_type="yarn", rope_theta=1e6, factor=16.,
                  original_max_position_embeddings=16, beta_fast=32., beta_slow=1.,
                  mscale=1., mscale_all_dim=1., llama_4_scaling_beta=0.1)
    params.update(overrides)
    return configs.Ministral3Config(vocab_size=32, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=8,
        num_key_value_heads=8, head_dim=4, max_position_embeddings=256,
        rope_parameters=params)


@pytest.mark.parametrize("attention_factor", [1., 1.7])
def test_yarn_roundtrip_matches_native_at_nonzero_positions(attention_factor):
    config = yarn_config(attention_factor=attention_factor)
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding, apply_rotary_pos_emb
    x = torch.randn(8, 5, 4)
    positions = torch.tensor([0, 12, 17, 1023, 18000])
    spec = rope_spec(config)
    cos, sin = Ministral3RotaryEmbedding(config)(x[None], positions[None])
    expected, _ = apply_rotary_pos_emb(x[None], x[None], cos, sin)
    actual = apply_rope(x, positions, spec)
    torch.testing.assert_close(actual, expected[0])
    torch.testing.assert_close(strip_rope(actual, positions, spec), x, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(actual, apply_rope(x, positions, get_rope_theta(config)))


@pytest.mark.parametrize("method", ["cachebridge", "head_local", "full_head"])
def test_yarn_mapper_metadata_roundtrip_and_transfer(tmp_path, method):
    from transformers import DynamicCache
    from kvbridge.cachebridge import cachebridge_metadata
    from kvbridge.mapper import Mapper
    from kvbridge.transfer import transfer
    from kvbridge.cache_compat import cache_layer_kv
    source, target = yarn_config(attention_factor=1.7), yarn_config(factor=8.)
    meta = cachebridge_metadata(source_model="local/Ministral-3-3B", target_model="local/Ministral-3-14B",
        num_source_layers=1, num_target_layers=1, num_source_kv_heads=8,
        num_target_kv_heads=8, head_dim=4, head_assignment=list(range(8)),
        selected_source_layers=[[0]], rope_theta_source=1e6, rope_theta_target=1e6)
    meta.update(method=method, top_k=20, rope_config_source=rope_spec(source), rope_config_target=rope_spec(target))
    if method != "cachebridge":
        meta["weighting_method"] = "uniform"
    weight = torch.eye(4).expand(8, 4, 4).clone()
    mu_x = torch.zeros(8, 4)
    if method == "full_head":
        weight = torch.stack([torch.eye(32)[:, h*4:(h+1)*4] for h in range(8)])
        mu_x = torch.zeros(8, 32)
    entry = {"source_layers": [0], "W_k": weight, "W_v": weight.clone(),
             "mu_xk": mu_x, "mu_xv": mu_x.clone(),
             "mu_yk": torch.zeros(8, 4), "mu_yv": torch.zeros(8, 4)}
    mapper = Mapper([entry], meta)
    mapper.save(tmp_path, dtype=torch.float32)
    restored = Mapper.load(tmp_path, device="cpu", dtype=torch.float32)
    assert restored.meta["rope_config_source"] == meta["rope_config_source"]
    assert restored.meta["top_k"] == 20
    positions = torch.tensor([12, 17, 1023])
    content, values = torch.randn(8, 3, 4), torch.randn(8, 3, 4)
    cache = DynamicCache()
    cache.update(apply_rope(content, positions, meta["rope_config_source"])[None], values[None], 0)
    key, value = cache_layer_kv(transfer(cache, restored, positions), 0)
    torch.testing.assert_close(key[0], apply_rope(content, positions, meta["rope_config_target"]), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(value[0], values)


def test_native_cpu_incremental_capture_uses_post_transform_query(monkeypatch):
    config = yarn_config(attention_factor=1.7)
    from transformers.models.ministral3 import modeling_ministral3 as native
    import kvbridge.formal as formal
    model = native.Ministral3ForCausalLM(config).eval()
    model.set_attn_implementation("sdpa")
    assert not hasattr(model.model.layers[0].self_attn, "q_norm")
    original_eager = native.eager_attention_forward
    observed = []
    def eager(module, query, *args, **kwargs):
        observed.append(query.detach().clone())
        return original_eager(module, query, *args, **kwargs)
    monkeypatch.setattr(native, "eager_attention_forward", eager)
    original_sensitivity = formal.compute_gqa_sensitivities
    checked = []
    def sensitivity(probs, q, *args, **kwargs):
        torch.testing.assert_close(q, observed[-1].float())
        checked.append(q.shape)
        return original_sensitivity(probs, q, *args, **kwargs)
    monkeypatch.setattr(formal, "compute_gqa_sensitivities", sensitivity)
    keys, values, wk, wv = formal._capture_target_weights(
        model, torch.arange(24)[None] % config.vocab_size, 4, [12, 17, 23])
    assert checked == [torch.Size([1, 8, 1, 4])] * 3
    assert keys[0].shape == values[0].shape == (6, 8, 4)
    assert wk[0].shape == wv[0].shape == (8, 6)
    assert native.eager_attention_forward is eager
    assert model.config._attn_implementation == "sdpa"


def test_local_ministral_load_uses_only_causal_model(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "mistral3"}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.language_model.embed_tokens.weight": "text.safetensors",
        "lm_head.weight": "text.safetensors"}}))
    text = geometry(26, 3072)
    config = SimpleNamespace(model_type="mistral3", text_config=text)
    monkeypatch.setattr(model_utils, "load_config", lambda *a, **kw: config)
    class Model(torch.nn.Module):
        config = text
    calls = []
    def load(path, **kwargs):
        assert kwargs["config"] is text
        assert kwargs["local_files_only"] is True
        assert kwargs["output_loading_info"] is True
        calls.append(kwargs)
        return Model(), {"missing_keys": [], "mismatched_keys": []}
    monkeypatch.setattr(model_utils.AutoModelForCausalLM, "from_pretrained", load)
    result = model_utils.load_causal_lm(tmp_path, local_files_only=True)
    assert result.config is text and len(calls) == 1
    def incomplete(*args, **kwargs):
        return Model(), {"missing_keys": ["lm_head.weight"], "mismatched_keys": []}
    monkeypatch.setattr(model_utils.AutoModelForCausalLM, "from_pretrained", incomplete)
    with pytest.raises(RuntimeError, match="incomplete Ministral text weight load"):
        model_utils.load_causal_lm(tmp_path, local_files_only=True)


def test_native_tokenizer_backend_fallback_remains_local(tmp_path, monkeypatch):
    import transformers
    (tmp_path / "tekken.json").write_text('{}')
    def fail(*a, **kw):
        raise ValueError("native backend required")
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained", fail)
    sentinel = object()
    class Backend:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == str(tmp_path) and kwargs["local_files_only"] is True
            return sentinel
    monkeypatch.setattr(transformers, "MistralCommonBackend", Backend, raising=False)
    assert model_utils.load_tokenizer(tmp_path, local_files_only=True) is sentinel


def test_ministral_token_mismatch_and_target_prefix_prefill_are_rejected():
    with pytest.raises(RuntimeError, match="tokenizer IDs differ"):
        check_tokenizer_alignment([torch.tensor([1, 2])], [torch.tensor([1, 3])])
    source = lambda *a, **kw: SimpleNamespace(input_ids=[1, 2])
    target = lambda *a, **kw: SimpleNamespace(input_ids=[1, 3])
    with pytest.raises(ValueError, match="tokenizer IDs differ"):
        model_utils.assert_text_tokenizer_alignment(source, target, "prefix")
    from experiments.eval_paper_tasks import _mapped_prefix_prefill_guard
    accounting = {"target_prefix_prefill_tokens": 0}
    cache = object()
    _mapped_prefix_prefill_guard(accounting)(None, (torch.ones(1, 1),), {"past_key_values": cache})
    validate_mapped_evaluation_accounting(accounting)
    assert accounting["target_prefix_prefill_tokens"] == 0
    with pytest.raises(ValueError, match="target prefix prefill"):
        validate_mapped_evaluation_accounting({"target_prefix_prefill_tokens": 12})
