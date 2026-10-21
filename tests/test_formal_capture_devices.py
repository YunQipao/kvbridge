from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")

from kvbridge.formal import _capture_target_weights, capture_source_shard
from kvbridge.cachebridge import compute_gqa_sensitivities
from kvbridge.rope import strip_rope


class _Cache:
    def __init__(self, keys, values):
        self.key_cache = keys
        self.value_cache = values


def test_capture_source_shard_builds_sampling_indices_on_cache_device(monkeypatch):
    arange = torch.arange
    devices = []

    def tracked_arange(*args, **kwargs):
        devices.append(torch.device(kwargs["device"]))
        return arange(*args, **kwargs)

    class Model:
        config = SimpleNamespace(num_hidden_layers=1, rope_theta=10000.0)

        def __call__(self, ids, **kwargs):
            key = torch.zeros(1, 2, ids.shape[1], 4)
            return SimpleNamespace(past_key_values=_Cache([key], [key.clone()]))

    monkeypatch.setattr(torch, "arange", tracked_arange)
    ids = SimpleNamespace(shape=(1, 9), device=torch.device("meta"))
    shard = capture_source_shard(Model(), ids, 4)
    assert shard["keys"][0].shape == shard["values"][0].shape == (3, 2, 4)
    assert shard["token_count"] == 9
    assert devices and all(device.type == "cpu" for device in devices)


class _ToyTarget:
    """Native token-indexed Q/K/V, causal softmax and a growing CPU cache."""

    def __init__(self, dtype=torch.float32, allow_full=False, projected_query=False):
        self.allow_full = allow_full
        self.projected_query = projected_query
        self.config = SimpleNamespace(
            num_hidden_layers=2, rope_theta=10000.0, rope_scaling=None,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            hidden_size=16, _attn_implementation="sdpa")
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(
            q_norm=torch.nn.Identity())) for _ in range(2)])
        if projected_query:
            for layer in self.model.layers:
                del layer.self_attn.q_norm
                layer.self_attn.q_proj = torch.nn.Identity()
        generator = torch.Generator().manual_seed(17)
        self.fixtures = [tuple(torch.randn(1, heads, 8, 4, generator=generator).to(dtype)
                               for heads in (4, 2, 2)) for _ in range(2)]
        self.calls = []
        self.refs = []
        self.fail_boundary = False

    def set_attn_implementation(self, name):
        self.config._attn_implementation = name

    def __call__(self, ids, *, past_key_values, cache_position, position_ids,
                 output_attentions, use_cache, return_dict, logits_to_keep):
        start, end = int(cache_position[0]), int(cache_position[-1]) + 1
        assert torch.equal(ids[0], cache_position)
        assert torch.equal(position_ids, cache_position[None])
        assert use_cache and return_dict and logits_to_keep == 1
        assert start == (past_key_values.key_cache[0].shape[2] if past_key_values else 0)
        assert all(ref() is None for ref in self.refs)
        assert self.config._attn_implementation == ("eager" if output_attentions else "sdpa")
        if output_attentions:
            assert self.allow_full or end - start == 1
            if self.fail_boundary:
                raise RuntimeError("mock boundary failure")
        self.calls.append((start, end, output_attentions))
        keys, values, rows = [], [], []
        for index, (q, k, v) in enumerate(self.fixtures):
            query = q[:, :, start:end].transpose(1, 2).contiguous().clone()
            attention = self.model.layers[index].self_attn
            if self.projected_query:
                attention.q_proj(query.reshape(1, end - start, -1))
            else:
                attention.q_norm(query)
            self.refs.append(weakref.ref(query))
            old_k = past_key_values.key_cache[index] if past_key_values else k[:, :, :0]
            old_v = past_key_values.value_cache[index] if past_key_values else v[:, :, :0]
            keys.append(torch.cat((old_k, k[:, :, start:end]), dim=2))
            values.append(torch.cat((old_v, v[:, :, start:end]), dim=2))
            if output_attentions:
                scores = q[:, :, start:end].float() @ keys[-1][:, [0, 0, 1, 1]].float().transpose(-1, -2)
                causal = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
                scores = scores.masked_fill(~causal[None, None], float("-inf"))
                row = (scores / 2).softmax(-1).to(q.dtype)
                rows.append(row)
                self.refs.append(weakref.ref(row))
        return SimpleNamespace(past_key_values=_Cache(keys, values),
                               attentions=tuple(rows) if output_attentions else None)


def _full_reference(model, stride, boundaries):
    keep = torch.arange(0, 8, stride)
    causal = torch.tril(torch.ones(8, 8, dtype=torch.bool))[None, None]
    result = []
    for q, k, v in model.fixtures:
        scores = q.float() @ k[:, [0, 0, 1, 1]].float().transpose(-1, -2) / 2
        probs = scores.masked_fill(~causal, float("-inf")).softmax(-1).to(q.dtype).float()
        output = probs @ v[:, [0, 0, 1, 1]].float()
        raw_k, raw_v = [], []
        for boundary in boundaries:
            if boundary >= 8:
                raw_k.append(torch.zeros(2, keep.numel()))
                raw_v.append(torch.zeros(2, keep.numel()))
                continue
            wk, wv = compute_gqa_sensitivities(
                probs[:, :, boundary:boundary + 1], q[:, :, boundary:boundary + 1].float(),
                v.float(), output[:, :, boundary:boundary + 1],
                query_to_kv=[0, 0, 1, 1], valid_mask=torch.ones(1, 8, dtype=torch.bool),
                causal_mask=causal[:, :, boundary:boundary + 1].expand(1, 4, 1, 8))
            eligible = keep < boundary
            raw_k.append(wk.index_select(-1, keep)[0].masked_fill(~eligible[None], 0))
            raw_v.append(wv.index_select(-1, keep)[0].masked_fill(~eligible[None], 0))

        def combine(rows):
            raw = torch.stack(rows)
            mass = raw.sum(-1, keepdim=True)
            return torch.where(mass > 0, raw / mass.clamp_min(1e-30),
                               torch.zeros_like(raw)).mean(0)

        sampled_k = strip_rope(k[0].float(), torch.arange(8), model.config.rope_theta)[:, keep]
        result.append((sampled_k.transpose(0, 1), v[0].float()[:, keep].transpose(0, 1),
                       combine(raw_k), combine(raw_v), probs, output))
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("boundaries", [[0, 2, 5, 8], [2], [0, 1, 7], [8]])
def test_incremental_boundary_capture_matches_full_reference(monkeypatch, dtype, boundaries):
    import kvbridge.formal as formal

    model = _ToyTarget(dtype=dtype)
    expected = _full_reference(model, 2, boundaries)
    sensitivity = formal.compute_gqa_sensitivities
    boundary_calls = []

    def checked_sensitivity(probs, q, v, output, **kwargs):
        boundary = model.calls[-1][0]
        layer = len(boundary_calls) % 2
        reference = expected[layer]
        assert probs.shape == (1, 4, 1, boundary + 1)
        assert output.shape == q.shape == (1, 4, 1, 4)
        torch.testing.assert_close(probs, reference[4][:, :, boundary:boundary + 1, :boundary + 1])
        torch.testing.assert_close(output, reference[5][:, :, boundary:boundary + 1],
                                   atol=1e-6, rtol=1e-5)
        mask = kwargs["causal_mask"]
        assert mask.shape == probs.shape and mask.stride(1) == 0
        boundary_calls.append((boundary, layer))
        return sensitivity(probs, q, v, output, **kwargs)

    original_float = torch.Tensor.float

    def checked_float(tensor, *args, **kwargs):
        assert tuple(tensor.shape) != (1, 4, 8, 8), "full attention cast"
        return original_float(tensor, *args, **kwargs)

    monkeypatch.setattr(formal, "compute_gqa_sensitivities", checked_sensitivity)
    monkeypatch.setattr(torch.Tensor, "float", checked_float)
    actual = _capture_target_weights(model, torch.arange(8)[None], 2, boundaries)
    for layer, reference in enumerate(expected):
        for tensors, expected_tensor in zip(actual, reference[:4]):
            torch.testing.assert_close(tensors[layer], expected_tensor, atol=1e-6, rtol=1e-5)
            assert tensors[layer].device.type == "cpu"
    assert model.calls[0][0] == 0 and model.calls[-1][1] == 8
    assert all(a[1] == b[0] for a, b in zip(model.calls, model.calls[1:]))
    assert [start for start, _, capture in model.calls if capture] == [c for c in boundaries if c < 8]
    assert all(end - start == 1 for start, end, capture in model.calls if capture)
    assert model.config._attn_implementation == "sdpa"
    assert all(ref() is None for ref in model.refs)
    assert all(not layer.self_attn.q_norm._forward_hooks for layer in model.model.layers)


def test_boundary_self_position_is_excluded_from_sampled_weights():
    model = _ToyTarget()
    _, _, wk, wv = _capture_target_weights(model, torch.arange(8)[None], 2, [2])
    for weights in wk + wv:
        assert torch.count_nonzero(weights[:, 1:]) == 0
        assert torch.count_nonzero(weights[:, 0]) > 0


@pytest.mark.parametrize("projected_query", [False, True])
@pytest.mark.parametrize("boundaries", [[0, 2, 4, 7, 8], [8]])
def test_full_capture_matches_incremental_and_reference(projected_query, boundaries):
    full = _ToyTarget(allow_full=True, projected_query=projected_query)
    chunked = _ToyTarget(projected_query=projected_query)
    expected = _full_reference(full, 2, boundaries)
    actual = _capture_target_weights(full, torch.arange(8)[None], 2, boundaries,
                                     capture_mode="full")
    incremental = _capture_target_weights(chunked, torch.arange(8)[None], 2, boundaries,
                                          capture_mode="boundary_chunked")
    for layer, reference in enumerate(expected):
        for index, expected_tensor in enumerate(reference[:4]):
            torch.testing.assert_close(actual[index][layer], expected_tensor, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual[index][layer], incremental[index][layer],
                                       atol=1e-6, rtol=1e-5)
    assert full.calls == [(0, 8, True)]  # No duplicate forward / cache append.
    assert full.config._attn_implementation == "sdpa"
    assert all(not (getattr(layer.self_attn, "q_norm", None) or layer.self_attn.q_proj)._forward_hooks
               for layer in full.model.layers)


def test_attention_implementation_and_hooks_restored_on_failure():
    model = _ToyTarget()
    model.config._attn_implementation = "eager"
    model.fail_boundary = True
    with pytest.raises(RuntimeError, match="mock boundary failure"):
        _capture_target_weights(model, torch.arange(8)[None], 2, [2])
    assert model.config._attn_implementation == "eager"
    assert all(not layer.self_attn.q_norm._forward_hooks for layer in model.model.layers)


@pytest.mark.parametrize("capture_mode", [None, "boundary_chunked"])
def test_target_stage_loads_sdpa_without_model_download(monkeypatch, tmp_path, capture_mode):
    import kvbridge.formal as formal

    model = _ToyTarget(allow_full=capture_mode is None)
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return model

    monkeypatch.setattr(formal, "load_causal_lm", load)
    monkeypatch.setattr(formal, "model_input_device", lambda model: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    options = {} if capture_mode is None else {"target_capture_mode": capture_mode}
    formal.capture_stage("mock", [torch.arange(8)], tmp_path, source=False,
                         device="cpu", stride=2, boundaries=[2, 5], local_files_only=True,
                         **options)
    assert loads[0]["attn_implementation"] == "sdpa"
    shard = torch.load(tmp_path / "target_00000.pt", weights_only=False)
    assert set(shard) == {"keys", "values", "weights_k", "weights_v", "token_count"}
    assert shard["token_count"] == 8
    if capture_mode is None:
        assert model.calls == [(0, 8, True)]
