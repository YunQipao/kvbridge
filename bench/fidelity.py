"""Diagnostics: where the mapped cache differs from a real prefill.

Two measurements, both cheap and generation-free:

- **kv_fidelity** -- per-target-layer, per-position-bucket cosine between the
  mapped and real K/V. This is the map that says where capacity should go:
  a layer whose K cosine is 0.95 does not need 12 source layers, and a
  position bucket that droops says the problem is length, not depth.
- **attn_cosine** -- cosine between the target's *attention outputs* under the
  mapped and real caches. On the measured pair this tracked benchmark accuracy
  at r=+0.92 where NLL-type metrics did not (see the top-level README), so it
  is the metric to tune k by when a full benchmark run is too expensive.

Run these before any expensive eval: they cost one prefill each and usually
tell you whether a change is worth benchmarking at all.
"""

import torch

from kvbridge.cache_compat import cache_layer_kv, cache_num_layers

from .engine import Strategy

REAL = Strategy("real")


def _cos(a, b, dim=-1):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=dim)


def _bucketize(values, n_positions, n_buckets):
    """Mean of `values` (one per position) over n_buckets position ranges."""
    edges = [round(i * n_positions / n_buckets) for i in range(n_buckets + 1)]
    return [float(values[edges[i]:edges[i + 1]].mean()) if edges[i + 1] > edges[i]
            else float("nan") for i in range(n_buckets)]


@torch.no_grad()
def kv_fidelity(engine, ctx_ids, strategy, n_buckets=8):
    """Per-layer and per-position-bucket K/V cosine, mapped vs real."""
    real = engine.build_cache(REAL, ctx_ids)
    approx = engine.build_cache(strategy, ctx_ids)
    T = ctx_ids.shape[1]

    per_layer, k_bucket_acc, v_bucket_acc = [], None, None
    for layer in range(cache_num_layers(real)):
        rk, rv = cache_layer_kv(real, layer)
        ak, av = cache_layer_kv(approx, layer)
        # [B, H, T, d] -> per-position mean over heads
        kc = _cos(ak, rk).mean(dim=(0, 1))
        vc = _cos(av, rv).mean(dim=(0, 1))
        per_layer.append({"layer": layer, "k": float(kc.mean()), "v": float(vc.mean())})
        k_bucket_acc = kc if k_bucket_acc is None else k_bucket_acc + kc
        v_bucket_acc = vc if v_bucket_acc is None else v_bucket_acc + vc

    n = cache_num_layers(real)
    return {
        "n_ctx": T,
        "per_layer": per_layer,
        "mean_k": sum(x["k"] for x in per_layer) / n,
        "mean_v": sum(x["v"] for x in per_layer) / n,
        "k_by_position": _bucketize(k_bucket_acc / n, T, n_buckets),
        "v_by_position": _bucketize(v_bucket_acc / n, T, n_buckets),
    }


@torch.no_grad()
def attn_cosine(engine, ctx_ids, probe_ids, strategy):
    """Cosine of the target's per-layer attention outputs, mapped vs real.

    `probe_ids` are the tokens the target actually attends *with* -- the
    post-handoff tokens. Their attention output is what the mapped cache
    ultimately has to reproduce; K/V cosine is an upstream proxy for it.
    """
    model = engine.target
    layers = model.model.layers
    captured = []

    def hook(_module, _inp, out):
        captured.append((out[0] if isinstance(out, tuple) else out).detach())

    def run(cache):
        captured.clear()
        handles = [l.self_attn.register_forward_hook(hook) for l in layers]
        try:
            model(probe_ids, past_key_values=cache, use_cache=True)
        finally:
            for h in handles:
                h.remove()
        return list(captured)

    real_out = run(engine.build_cache(REAL, ctx_ids))
    approx_out = run(engine.build_cache(strategy, ctx_ids))
    per_layer = [float(_cos(a, b).mean()) for a, b in zip(approx_out, real_out)]
    return {"per_layer": per_layer, "mean": sum(per_layer) / len(per_layer)}
