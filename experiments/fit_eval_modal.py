# The pipeline that produced the tables in the README: fit a cross-model KV
# mapper on Modal, sweep top_k, and evaluate every k on held-out NLL, the
# lm-eval benchmark suite, attention-output cosine, and wall-clock timing.
#
# Method (after Heo et al., NVIDIA, arXiv:2608.03893):
#   1. Calibration corpus through SOURCE and TARGET, capture per-layer KV.
#      Paper protocol (App. E): 500 FineWeb-Edu sequences x 1024 tokens,
#      stride-4 subsampled to ~128K decorrelated token observations per head.
#   2. Strip RoPE from keys -> position-free content space.
#   3. Per target layer: select the top-k most predictive source layers (R^2
#      screen over ALL KV heads), then fit SEPARATE ridge regressions for K and
#      for V. Features = all KV heads of the chosen layers, concatenated. Ridge
#      is centered, lambda = 0.01, bias from the means and unregularized.
#   4. Transfer: strip source RoPE -> linear map -> re-apply target RoPE ->
#      inject as a DynamicCache -> target generates without prefilling.
#
# Stages, fanned out one container per k:
#   prepare     capture calibration KV (cached per MODEL, so a source capture is
#               reused across pairs), rank source layers, materialize held-out
#               blocks. Everything lands on the `kvbridge-out` volume.
#   run_k       fit + held-out NLL + QA smoke test for one k. Fitting happens
#               before models load, so the whole GPU is free for the solve.
#   bench_k     HellaSwag / ARC-C / MMLU log-likelihood scoring and WikiText-2
#               prefix perplexity against the mapped cache, lm-eval conventions.
#   timing_all  every k sequentially in ONE container, so timings are
#               comparable; mappers load one at a time (bf16 k=24 is ~3.6 GB).
#
# Results persist to the volume per stage and short-circuit on rerun, so a dead
# local orchestrator costs nothing and a rerun is a volume read.
#
# Scoring detail worth knowing: a KV cache carries no logits, so the first
# continuation token is unscoreable from a cache of the FULL context (fatal for
# MMLU, whose continuation is a single letter). The cache therefore holds
# ctx[:-1] and the input row is [last ctx token] + ending[:-1]; all endings of a
# question are scored in ONE batched forward against a batch-expanded cache.
#
# Evaluation:
#   a. Held-out NLL of continuations under {mapped, real prefill, no context}.
#      Primary metric: (nll_none - nll_mapped) / (nll_none - nll_real).
#   b. Attention-output cosine vs ground-truth KV, per layer and position bucket.
#   c. QA smoke test on a synthetic document (needle substring checks).
#   d. Timing with CUDA synchronization, warmup, median of 5 trials.
#
# Usage (run from the repo root):
#   modal run experiments/fit_eval_modal.py
#   modal run experiments/fit_eval_modal.py --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B
#   modal run experiments/fit_eval_modal.py --source ... --target ... --skip-timing
#   POC_TOP_K_SWEEP=16,20,24 modal run experiments/fit_eval_modal.py --source ... --target ...
#   POC_BENCH_SAMPLES=0 modal run experiments/fit_eval_modal.py ...  # 0 = full sets
#   POC_GPU=A100-40GB modal run experiments/fit_eval_modal.py ...    # faster solves
#
# (Sweep and GPU config are read from env vars at import time, because the
#  decorators need them before Modal parses CLI flags.)
#
# Requires a matched-KV pair: same num_key_value_heads and head_dim. All dense
# Qwen3 models share 8 KV heads x 128 dims.

import os

import modal

APP_NAME = "kvbridge-fit-eval"
GPU_TYPE = os.environ.get("POC_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("POC_TIMEOUT_SECONDS", "7200"))
CONTEXT_LENGTHS = tuple(
    int(x) for x in os.environ.get("POC_CONTEXT_LENGTHS", "512,1024,2048,4096").split(",")
)
TOP_K_SWEEP = tuple(
    int(x) for x in os.environ.get("POC_TOP_K_SWEEP", "4,8,12,16,20,24").split(",")
)
CONTINUATION_LENGTH = 128
CALIB_STRIDE = 4          # paper App. E: stride-4 subsampling
RIDGE_CHUNK = 8192        # rows per chunk when accumulating X'X
POSITION_BUCKETS = 4      # cosine diagnostic buckets over the prefix
COSINE_MAX_PREFIX = int(os.environ.get("POC_COSINE_MAX_PREFIX", "2048"))
BENCH_SAMPLES = int(os.environ.get("POC_BENCH_SAMPLES", "2000"))  # 0 = full sets

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.51,<5",
        "datasets>=2.19",
        "safetensors>=0.4",
        "accelerate>=0.30",
    )
    # Sweep config is read at import time, so it must exist in the container too.
    .env({
        "POC_CONTEXT_LENGTHS": ",".join(str(x) for x in CONTEXT_LENGTHS),
        "POC_TOP_K_SWEEP": ",".join(str(x) for x in TOP_K_SWEEP),
        "POC_COSINE_MAX_PREFIX": str(COSINE_MAX_PREFIX),
        "POC_BENCH_SAMPLES": str(BENCH_SAMPLES),
        # Reduces fragmentation, which is what turned a tight-but-feasible
        # allocation into an OOM during evaluation.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("kvbridge-out", create_if_missing=True)


# -----------------------------------------------------------------------------
# Volume artifact paths. Captures are keyed per MODEL (not per pair) so a
# model's calibration KV is reused by every pair it appears in.
# -----------------------------------------------------------------------------

def _short(model_id):
    return model_id.split("/")[-1]


def capture_paths(model_id, n_calib, seq_len):
    tag = f"{_short(model_id)}_n{n_calib}_L{seq_len}_s{CALIB_STRIDE}"
    return f"/out/cap_{tag}_keys.pt", f"/out/cap_{tag}_vals.pt"


def ranking_path(source, target, n_calib, seq_len, lam):
    return (f"/out/rank_{_short(source)}_to_{_short(target)}"
            f"_n{n_calib}_L{seq_len}_s{CALIB_STRIDE}_lam{lam}.pt")


def fit_cache_path(source, target, n_calib, seq_len, lam, k):
    return (f"/out/fitcache_{_short(source)}_to_{_short(target)}"
            f"_n{n_calib}_L{seq_len}_s{CALIB_STRIDE}_lam{lam}_k{k}.pt")


def heldout_path(source, n_calib, n_heldout, max_tokens):
    # Held-out blocks start where the calibration documents end in the stream,
    # so the key must include n_calib as well.
    return f"/out/heldout_{_short(source)}_c{n_calib}_h{n_heldout}_T{max_tokens}.pt"


def baseline_cache_path(model_id, n_samples):
    return f"/out/benchbase_{_short(model_id)}_s{n_samples}_v1.pt"


def _ctx_tag():
    return "-".join(str(p) for p in CONTEXT_LENGTHS)


def eval_result_path(source, target, n_calib, seq_len, lam, k, n_heldout):
    """run_k results persist per (pair, settings, k, heldout config, contexts)
    so a dead orchestrator loses nothing and reruns are volume reads."""
    return (f"/out/evalres_{_short(source)}_to_{_short(target)}"
            f"_n{n_calib}_L{seq_len}_s{CALIB_STRIDE}_lam{lam}_k{k}"
            f"_h{n_heldout}_ctx{_ctx_tag()}.pt")


def bench_result_path(source, target, n_calib, seq_len, lam, k, n_samples):
    return (f"/out/benchres_{_short(source)}_to_{_short(target)}"
            f"_n{n_calib}_L{seq_len}_s{CALIB_STRIDE}_lam{lam}_k{k}_s{n_samples}_v1.pt")


# -----------------------------------------------------------------------------
# Cache API compatibility (DynamicCache internals changed across 4.5x)
#   <= ~4.53: cache.key_cache[l] / cache.value_cache[l]
#   >= ~4.54: cache.layers[l].keys / cache.layers[l].values
# -----------------------------------------------------------------------------

def cache_layer_kv(cache, layer_idx):
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def cache_num_layers(cache):
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


# -----------------------------------------------------------------------------
# RoPE helpers (Llama/Qwen half-rotation convention)
# -----------------------------------------------------------------------------

def _build_cos_sin(positions, head_dim, rope_theta, device, dtype):
    import torch
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x):
    import torch
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def get_rope_theta(config):
    """rope_theta moved under config.rope_parameters in transformers >= 4.57."""
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and params.get("rope_theta") is not None:
        return float(params["rope_theta"])
    raise AttributeError("could not locate rope_theta on config")


def strip_rope(k, positions, rope_theta):
    """Inverse RoPE. k: [H, T, d] post-rotation. Returns content-space keys."""
    cos, sin = _build_cos_sin(positions, k.shape[-1], rope_theta, k.device, k.dtype)
    return k * cos[None] - _rotate_half(k) * sin[None]


def apply_rope(k, positions, rope_theta):
    """Forward RoPE. k: [H, T, d] content-space."""
    cos, sin = _build_cos_sin(positions, k.shape[-1], rope_theta, k.device, k.dtype)
    return k * cos[None] + _rotate_half(k) * sin[None]


# -----------------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------------

def load_corpus(tok, n_calib, n_heldout, max_eval_tokens, need_calib, need_heldout):
    """Stream FineWeb-Edu once: the first n_calib long documents are the
    calibration set, and held-out token blocks start strictly after them."""
    import torch
    from datasets import load_dataset

    heldout_token_count = n_heldout * max_eval_tokens
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    calib, heldout_tokens = [], []
    n_calib_seen = 0
    for row in ds:
        text = row["text"]
        if n_calib_seen < n_calib:
            if len(text) > 3000:
                n_calib_seen += 1
                if need_calib:
                    calib.append(text)
            continue
        if not need_heldout:
            break
        remaining = heldout_token_count - len(heldout_tokens)
        heldout_tokens.extend(tok(text, add_special_tokens=False).input_ids[:remaining])
        if len(heldout_tokens) >= heldout_token_count:
            break
        heldout_tokens.append(tok.eos_token_id)

    heldout = None
    if need_heldout:
        if len(heldout_tokens) < heldout_token_count:
            raise RuntimeError("FineWeb-Edu stream ended before held-out blocks were filled")
        heldout = torch.tensor(heldout_tokens, dtype=torch.long).reshape(n_heldout, max_eval_tokens)
    return calib, heldout


# -----------------------------------------------------------------------------
# Activation capture
# -----------------------------------------------------------------------------

def collect_kv(model, tok, texts, seq_len, device, rope_theta, desc="", stride=CALIB_STRIDE):
    """Returns (keys, vals): lists over layers of CPU fp16 tensors [N, H, d].
    Keys are RoPE-stripped (content space).

    Positions are subsampled with the given stride (paper App. E uses stride 4)
    so the retained observations are decorrelated. The stride offset is rotated
    per sequence so every position phase is represented across the corpus."""
    import torch
    n_layers = model.config.num_hidden_layers
    keys = [[] for _ in range(n_layers)]
    vals = [[] for _ in range(n_layers)]

    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
        T = ids.shape[1]
        if T < 32:
            continue
        pos = torch.arange(T, device=device)
        keep = torch.arange(i % stride, T, stride, device=device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        cache = out.past_key_values
        for l in range(n_layers):
            k, v = cache_layer_kv(cache, l)
            k, v = k[0], v[0]                                   # [H, T, d]
            k_content = strip_rope(k.float(), pos, rope_theta)
            keys[l].append(k_content[:, keep].transpose(0, 1).half().cpu())   # [n_keep, H, d]
            vals[l].append(v.float()[:, keep].transpose(0, 1).half().cpu())
        del out, cache
        if (i + 1) % 50 == 0:
            print(f"  [{desc}] {i+1}/{len(texts)} sequences", flush=True)

    keys = [torch.cat(x, dim=0) for x in keys]
    vals = [torch.cat(x, dim=0) for x in vals]
    return keys, vals


# -----------------------------------------------------------------------------
# Centered ridge regression (paper: lambda = 0.01, bias unregularized)
# -----------------------------------------------------------------------------

def ridge_centered(X, Y, lam=0.01):
    """X: [N, p], Y: [N, q] float32 on GPU. Center both, solve
    (Xc'Xc / N + lam I) W = Xc'Yc / N. Normalizing keeps lambda meaningful
    and representable in float32. Bias is implied by the means and therefore
    unregularized. Returns (W [p, q], mu_x [p], mu_y [q])."""
    import torch
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    scale = 1.0 / X.shape[0]
    XtX = (Xc.T @ Xc) * scale
    XtX += lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX, (Xc.T @ Yc) * scale)
    return W, mu_x, mu_y


def ridge_streaming(src_list, tgt_layer, chosen, lam, device, chunk=RIDGE_CHUNK):
    """Centered ridge for a large concatenated design, accumulating the normal
    equations over row chunks so large k never materializes the full [N, p]
    design on the GPU.

    Precision split: moments accumulate in float32 and the solve runs in
    float64. Accumulating in float64 was measured at 31x slower than float32 on
    an A10G (consumer-class cards run fp64 at ~1/32 rate), while the solve
    itself is a negligible share of the total, so the split keeps the
    conditioning benefit where it matters and pays almost nothing for it.

    src_list: list over source layers of CPU fp16 [N, H, d]
    tgt_layer: CPU fp16 [N, H, d] response for one target layer
    chosen: source layer indices to concatenate as features

    Returns (W [p, q], mu_x [p], mu_y [q], r2). r2 is in-sample, computed from
    the same accumulated moments (no second pass over the data)."""
    import torch
    N = tgt_layer.shape[0]
    p = sum(src_list[sl].shape[1] * src_list[sl].shape[2] for sl in chosen)
    q = tgt_layer.shape[1] * tgt_layer.shape[2]
    acc = torch.float32

    XtX = torch.zeros(p, p, device=device, dtype=acc)
    XtY = torch.zeros(p, q, device=device, dtype=acc)
    sum_x = torch.zeros(p, device=device, dtype=torch.float64)
    sum_y = torch.zeros(q, device=device, dtype=torch.float64)
    sum_yy = torch.zeros(q, device=device, dtype=torch.float64)

    for n0 in range(0, N, chunk):
        n1 = min(n0 + chunk, N)
        Xb = torch.cat([_flat(src_list[sl], n0, n1, device) for sl in chosen], dim=-1)
        Yb = _flat(tgt_layer, n0, n1, device)
        XtX.addmm_(Xb.T, Xb)
        XtY.addmm_(Xb.T, Yb)
        # First moments stay in float64: they are cheap (O(p)) and centering
        # accuracy directly sets the bias term.
        sum_x += Xb.sum(dim=0).double()
        sum_y += Yb.sum(dim=0).double()
        sum_yy += (Yb.double() ** 2).sum(dim=0)
        del Xb, Yb

    XtX = XtX.double()
    XtY = XtY.double()
    mu_x = sum_x / N
    mu_y = sum_y / N
    # Centered moments from raw moments: Xc'Xc = X'X - N mu_x mu_x'
    XtX -= N * torch.outer(mu_x, mu_x)
    XtY -= N * torch.outer(mu_x, mu_y)
    XtX /= N
    XtY /= N
    XtX.diagonal().add_(lam)
    W = torch.linalg.solve(XtX, XtY)

    # In-sample R^2 from moments: ss_res = tr(Yc'Yc) - 2 tr(W' Xc'Yc) + tr(W' Xc'Xc W)
    # Using the *unregularized* centered moments recovered from the solve inputs.
    XtX.diagonal().sub_(lam)
    ss_tot = float((sum_yy - N * mu_y ** 2).sum())
    ss_res = float(
        (sum_yy - N * mu_y ** 2).sum()
        - 2.0 * N * (W * XtY).sum()
        + N * (W * (XtX @ W)).sum()
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return W.float(), mu_x.float(), mu_y.float(), r2


def r2_score(X, W, mu_x, mu_y, Y):
    resid = (Y - mu_y) - (X - mu_x) @ W
    ss_res = (resid ** 2).sum()
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def _flat(t, n0, n1, device):
    """CPU fp16 [N, H, d] -> GPU fp32 [n, H*d] slice."""
    x = t[n0:n1].to(device).float()
    return x.reshape(x.shape[0], -1)


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------

def rank_source_layers(src_keys, src_vals, tgt_k_layer, tgt_v_layer, idx, lam, device):
    """Rank source layers by mean per-head K/V ridge R^2 on a subsample.

    idx is an index tensor striding across the WHOLE corpus, so the ranking sees
    every calibration document rather than the first few (paper S 2.3 probes the
    full corpus). This ranking is the single largest quality contributor per the
    paper's ablation (S 4.3), so representative sampling matters."""
    Yk = tgt_k_layer[idx].to(device).float()
    Yv = tgt_v_layer[idx].to(device).float()
    n_heads = Yk.shape[1]
    scores = []
    for sl in range(len(src_keys)):
        Xk = src_keys[sl][idx].to(device).float()
        Xv = src_vals[sl][idx].to(device).float()
        score = 0.0
        for h in range(n_heads):
            Wk, mxk, myk = ridge_centered(Xk[:, h], Yk[:, h], lam)
            Wv, mxv, myv = ridge_centered(Xv[:, h], Yv[:, h], lam)
            score += r2_score(Xk[:, h], Wk, mxk, myk, Yk[:, h])
            score += r2_score(Xv[:, h], Wv, mxv, myv, Yv[:, h])
        scores.append(score / (2 * n_heads))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order, scores


def fit_mapper(src_keys, src_vals, tgt_keys, tgt_vals, rankings, top_k, lam, device):
    """Per target layer: choose source layers, then fit SEPARATE centered ridge
    maps for K and V. Features = concat of ALL KV heads from chosen layers.

    Mapper entry per target layer:
      source_layers, W_k [p, Hd], mu_xk, mu_yk, W_v, mu_xv, mu_yv
    """
    layers, diag = [], []

    for tl in range(len(tgt_keys)):
        chosen = rankings[tl][:top_k]
        W_k, mu_xk, mu_yk, r2k = ridge_streaming(src_keys, tgt_keys[tl], chosen, lam, device)
        W_v, mu_xv, mu_yv, r2v = ridge_streaming(src_vals, tgt_vals[tl], chosen, lam, device)

        layers.append({
            "source_layers": chosen,
            "W_k": W_k.cpu(), "mu_xk": mu_xk.cpu(), "mu_yk": mu_yk.cpu(),
            "W_v": W_v.cpu(), "mu_xv": mu_xv.cpu(), "mu_yv": mu_yv.cpu(),
        })
        diag.append({"k": r2k, "v": r2v})
        if tl % 6 == 0 or tl == len(tgt_keys) - 1:
            print(f"  target layer {tl:2d}: sources={sorted(chosen)}  "
                  f"R^2 K={r2k:.3f} V={r2v:.3f}", flush=True)

    return {"layers": layers, "r2": diag}


# -----------------------------------------------------------------------------
# Transfer
# -----------------------------------------------------------------------------

def transfer_cache(src_cache, mapper, positions, theta_src, theta_tgt, n_heads, head_dim,
                   device, dtype, compute_dtype=None):
    """Map a source cache (batch 1) to a target DynamicCache.

    One matmul per target layer per cache type, run in compute_dtype (bf16 by
    default) so it lands on tensor cores. The paper measures the mapper as a
    batched bf16 matmul per target layer (App. G); fp32 eager math was the main
    reason our per-k mapping cost scaled badly."""
    import torch
    from transformers import DynamicCache

    if compute_dtype is None:
        compute_dtype = mapper.get("compute_dtype", torch.bfloat16)

    n_src = cache_num_layers(src_cache)
    T = positions.shape[0]
    src_K, src_V = [], []
    for l in range(n_src):
        k, v = cache_layer_kv(src_cache, l)
        k, v = k[0].float(), v[0].float()                       # [H, T, d]
        kc = strip_rope(k, positions, theta_src)
        src_K.append(kc.transpose(0, 1).reshape(T, -1).to(compute_dtype))   # [T, H*d]
        src_V.append(v.transpose(0, 1).reshape(T, -1).to(compute_dtype))

    tgt = DynamicCache()
    for tl, e in enumerate(mapper["layers"]):
        chosen = e["source_layers"]
        Xk = torch.cat([src_K[sl] for sl in chosen], dim=-1)
        Yk = torch.addmm(e["mu_yk"], Xk - e["mu_xk"], e["W_k"])
        Xv = torch.cat([src_V[sl] for sl in chosen], dim=-1)
        Yv = torch.addmm(e["mu_yv"], Xv - e["mu_xv"], e["W_v"])
        Kt = Yk.float().reshape(T, n_heads, head_dim).transpose(0, 1)    # [H, T, d]
        Vt = Yv.float().reshape(T, n_heads, head_dim).transpose(0, 1)
        Kt = apply_rope(Kt, positions, theta_tgt)
        tgt.update(Kt.to(dtype).unsqueeze(0), Vt.to(dtype).unsqueeze(0), tl)
    return tgt


def mapper_to_device(mapper, device, compute_dtype=None):
    """Copy mapper tensors once so repeated transfers do not include host-to-device
    I/O, casting to the matmul compute dtype at the same time."""
    import torch
    if compute_dtype is None:
        compute_dtype = torch.bfloat16
    tensor_keys = ("W_k", "mu_xk", "mu_yk", "W_v", "mu_xv", "mu_yv")
    layers = []
    for entry in mapper["layers"]:
        moved = {"source_layers": entry["source_layers"]}
        moved.update({key: entry[key].to(device=device, dtype=compute_dtype)
                      for key in tensor_keys})
        layers.append(moved)
    return {"layers": layers, "r2": mapper["r2"], "compute_dtype": compute_dtype}


# -----------------------------------------------------------------------------
# Attention-output cosine (paper S 4.5)
#
# R^2 measures channel-wise reconstruction, weighting all dimensions equally.
# Attention does not: it scores K against the target's queries and weights V by
# the resulting pattern. The paper finds attention-output cosine correlates with
# downstream retention at r=+0.57 while calibration R^2 gives r=-0.20, so this
# is the scalar to steer by. Bucketing by position localizes the length decay.
# -----------------------------------------------------------------------------

def attention_output_cosine(model, ids, mapped_cache, real_cache, positions,
                            n_buckets=POSITION_BUCKETS):
    """Cosine between attention output computed from mapped KV and from
    ground-truth KV, averaged over layers and heads.

    Queries come from the target's own forward pass on the same tokens, so the
    only thing that differs between the two conditions is the KV cache. Returns
    (overall_cosine, per_layer list, per_bucket list over query-position bins).
    """
    import torch
    import torch.nn.functional as F

    cfg = model.config
    n_q_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    rep = n_q_heads // n_kv_heads
    T = positions.shape[0]

    # Capture each layer's query projection output (post-norm, post-RoPE queries
    # are what attention actually uses, so we recompute RoPE on them here).
    queries = {}

    def make_hook(idx):
        def hook(module, args, output):
            queries[idx] = output.detach()
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i)))
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    causal = torch.tril(torch.ones(T, T, device=ids.device, dtype=torch.bool))
    bucket_edges = [round(T * i / n_buckets) for i in range(n_buckets + 1)]

    per_layer = []
    bucket_sums = [0.0] * n_buckets
    bucket_counts = [0] * n_buckets

    # A full [n_q_heads, T, T] score tensor is 2 GB at T=4096 in fp32, per
    # cache, which does not fit alongside both models. Process a few query heads
    # at a time so peak scratch stays bounded regardless of context length.
    heads_per_chunk = max(1, min(n_q_heads, int(2 ** 26 / max(T * T, 1))))

    with torch.no_grad():
        for tl in range(cfg.num_hidden_layers):
            q = queries[tl][0].reshape(T, n_q_heads, head_dim).transpose(0, 1).float()
            # Qwen3 applies q_norm before RoPE; use the layer's own norm module.
            q_norm = getattr(model.model.layers[tl].self_attn, "q_norm", None)
            if q_norm is not None:
                q = q_norm(q.to(queries[tl].dtype)).float()
            q = apply_rope(q, positions, get_rope_theta(cfg))

            k_m, v_m = cache_layer_kv(mapped_cache, tl)
            k_r, v_r = cache_layer_kv(real_cache, tl)
            cos_rows = []
            for h0 in range(0, n_q_heads, heads_per_chunk):
                h1 = min(h0 + heads_per_chunk, n_q_heads)
                qc = q[h0:h1]
                # Map query-head range onto the corresponding KV heads.
                kv0, kv1 = h0 // rep, (h1 - 1) // rep + 1
                outs = []
                for kk, vv in ((k_m, v_m), (k_r, v_r)):
                    kc = kk[0, kv0:kv1].float().repeat_interleave(rep, dim=0)
                    vc = vv[0, kv0:kv1].float().repeat_interleave(rep, dim=0)
                    off = h0 - kv0 * rep
                    kc = kc[off:off + (h1 - h0)]
                    vc = vc[off:off + (h1 - h0)]
                    scores = (qc @ kc.transpose(1, 2)) / (head_dim ** 0.5)
                    scores = scores.masked_fill(~causal, float("-inf"))
                    outs.append(torch.softmax(scores, dim=-1) @ vc)
                    del scores, kc, vc
                cos_rows.append(F.cosine_similarity(outs[0], outs[1], dim=-1))
                del outs
            cos = torch.cat(cos_rows, dim=0)                       # [Hq, T]
            del cos_rows, q

            per_layer.append(float(cos.mean()))
            for b in range(n_buckets):
                lo, hi = bucket_edges[b], bucket_edges[b + 1]
                if hi > lo:
                    bucket_sums[b] += float(cos[:, lo:hi].mean())
                    bucket_counts[b] += 1
            del cos
            queries.pop(tl, None)

    per_bucket = [bucket_sums[b] / bucket_counts[b] if bucket_counts[b] else float("nan")
                  for b in range(n_buckets)]
    overall = sum(per_layer) / len(per_layer)
    return overall, per_layer, per_bucket


# -----------------------------------------------------------------------------
# Timing (CUDA-synchronized, warmup + median of trials)
# -----------------------------------------------------------------------------

def timed_median(fn, trials=5):
    import time

    import torch
    fn()                                   # warmup
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


# -----------------------------------------------------------------------------
# Held-out evaluation: continuation NLL under three cache conditions
# -----------------------------------------------------------------------------

def continuation_nll(model, cont_ids, cache, total_prefix_len, device):
    """Mean NLL (nats/token) of cont_ids[1:] given cont_ids[:-1] + cache.
    Identical conditioning structure across conditions for fair comparison."""
    import torch
    import torch.nn.functional as F
    inp = cont_ids[:, :-1]
    tgt = cont_ids[:, 1:]
    attn = torch.ones(1, total_prefix_len + inp.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True)
    logits = out.logits[0]
    return float(F.cross_entropy(logits.float(), tgt[0]))


def heldout_eval(m_src, m_tgt, mappers, blocks, theta_s, theta_t, n_heads, head_dim,
                 prefix_lengths, cont_len, device, dtype, cosine_blocks=4,
                 cosine_max_prefix=COSINE_MAX_PREFIX, tag=""):
    import copy
    import math

    import torch
    results = {}
    for prefix_len in prefix_lengths:
        mapped_nll = {k: [] for k in mappers}
        cos_overall = {k: [] for k in mappers}
        cos_bucket = {k: [] for k in mappers}
        real_nll, none_nll = [], []
        for bi, block in enumerate(blocks):
            ids = block[:prefix_len + cont_len].unsqueeze(0).to(device)
            pre, cont = ids[:, :prefix_len], ids[:, prefix_len:]
            pos = torch.arange(prefix_len, device=device)

            with torch.no_grad():
                src_cache = m_src(pre, use_cache=True).past_key_values
                real_cache = m_tgt(pre, use_cache=True).past_key_values
            # continuation_nll appends the continuation to whatever cache it is
            # given, so the cosine diagnostic needs a prefix-length snapshot.
            # Cosine cost grows with T^2, so cap it: it is a diagnostic, and the
            # position buckets it reports are already informative at 2K.
            want_cos = bi < cosine_blocks and prefix_len <= cosine_max_prefix
            real_snapshot = copy.deepcopy(real_cache) if want_cos else None
            for k, mapper in mappers.items():
                mapped = transfer_cache(src_cache, mapper, pos, theta_s, theta_t,
                                        n_heads, head_dim, device, dtype)
                if want_cos:
                    ov, _, buckets = attention_output_cosine(
                        m_tgt, pre, copy.deepcopy(mapped), copy.deepcopy(real_snapshot), pos)
                    cos_overall[k].append(ov)
                    cos_bucket[k].append(buckets)
                mapped_nll[k].append(
                    continuation_nll(m_tgt, cont, mapped, prefix_len, device)
                )
                del mapped
            del real_snapshot
            real_nll.append(continuation_nll(m_tgt, cont, real_cache, prefix_len, device))
            none_nll.append(continuation_nll(m_tgt, cont, None, 0, device))
            del src_cache, real_cache
            # At 8K/16K prefixes three caches plus both models leave little
            # headroom on a 24 GB card; releasing between blocks avoids
            # fragmentation-driven OOM.
            if prefix_len >= 8192:
                torch.cuda.empty_cache()

        real = sum(real_nll) / len(real_nll)
        none = sum(none_nll) / len(none_nll)
        denom = none - real
        results[prefix_len] = {}
        for k in mappers:
            mapped = sum(mapped_nll[k]) / len(mapped_nll[k])
            retained = (none - mapped) / denom if denom > 1e-6 else float("nan")
            n_cos = len(cos_overall[k])
            buckets = ([sum(b[i] for b in cos_bucket[k]) / n_cos
                        for i in range(len(cos_bucket[k][0]))] if n_cos else [])
            results[prefix_len][k] = {
                "mapped": mapped, "real": real, "none": none,
                "benefit_retained": retained, "n": len(blocks),
                "ppl": {"mapped": math.exp(mapped), "real": math.exp(real),
                        "none": math.exp(none)},
                "attn_cosine": sum(cos_overall[k]) / n_cos if n_cos else float("nan"),
                "attn_cosine_by_position": buckets,
            }
            if tag:
                ev = results[prefix_len][k]
                print(f"{tag} prefix={prefix_len:4d}: benefit={ev['benefit_retained']*100:6.1f}% "
                      f"ppl {ev['ppl']['mapped']:.2f}/{ev['ppl']['real']:.2f}/{ev['ppl']['none']:.2f} "
                      f"cos={ev['attn_cosine']:.3f}", flush=True)
    return results


# -----------------------------------------------------------------------------
# QA smoke test
# -----------------------------------------------------------------------------

EVAL_DOC = (
    "Meeting notes, Project Heliotrope, March 12.\n"
    "Attendees: Mara Voss (lead), Deniz Okafor, Priya Ramanathan.\n"
    "The launch date was moved from June 3 to August 21 because the thermal "
    "regulator failed its second stress test at 340 kelvin. Deniz will source a "
    "replacement regulator from Calyx Dynamics for 48,000 dollars. Priya's "
    "simulation showed the backup battery lasts 91 minutes under peak load, "
    "which exceeds the 75-minute requirement. Mara approved the revised budget "
    "of 2.3 million dollars. The next review is scheduled for April 9.\n"
)

EVAL_QA = [
    ("Question: Why was the launch date moved?\nAnswer:", "thermal regulator"),
    ("Question: What is the new launch date?\nAnswer:", "August 21"),
    ("Question: How long does the backup battery last under peak load?\nAnswer:", "91"),
    ("Question: Which company is supplying the replacement regulator?\nAnswer:", "Calyx"),
]


def qa_smoke(m_src, m_tgt, tok, mapper_gpu, theta_s, theta_t, n_heads, head_dim,
             device, dtype, tag=""):
    import copy

    import torch
    doc_ids = tok(EVAL_DOC, return_tensors="pt").input_ids.to(device)
    P = doc_ids.shape[1]
    pos = torch.arange(P, device=device)
    with torch.no_grad():
        src_cache = m_src(doc_ids, use_cache=True).past_key_values
        real_doc = m_tgt(doc_ids, use_cache=True).past_key_values
    mapped_doc = transfer_cache(src_cache, mapper_gpu, pos, theta_s, theta_t,
                                n_heads, head_dim, device, dtype)

    def answer(cache, question, has_prefix):
        q_ids = tok(question, return_tensors="pt").input_ids.to(device)
        full = torch.cat([doc_ids, q_ids], dim=1) if has_prefix else q_ids
        attn = torch.ones_like(full)
        with torch.no_grad():
            out = m_tgt.generate(input_ids=full, attention_mask=attn, past_key_values=cache,
                                 max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, full.shape[1]:], skip_special_tokens=True).strip()

    score = {"mapped": 0, "real": 0, "none": 0, "n": len(EVAL_QA)}
    for q, needle in EVAL_QA:
        for name, cache, pref in [("mapped", copy.deepcopy(mapped_doc), True),
                                  ("real", copy.deepcopy(real_doc), True),
                                  ("none", None, False)]:
            a = answer(cache, q, pref)
            hit = needle.lower() in a.lower()
            score[name] += int(hit)
            print(f"  {tag}[{name:6s}] {'HIT ' if hit else 'miss'} | "
                  f"{q.splitlines()[0][10:]:50s} -> {a[:70]}", flush=True)
    return score


# -----------------------------------------------------------------------------
# Stage 1: prepare shared artifacts (captures, rankings, held-out blocks)
# -----------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=TIMEOUT_SECONDS, memory=65536, volumes={"/out": vol})
def prepare(source: str, target: str, n_calib: int = 500, seq_len: int = 1024,
            lam: float = 0.01, n_heldout: int = 48):
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    max_eval_tokens = max(CONTEXT_LENGTHS) + CONTINUATION_LENGTH

    src_paths = capture_paths(source, n_calib, seq_len)
    tgt_paths = capture_paths(target, n_calib, seq_len)
    rank_p = ranking_path(source, target, n_calib, seq_len, lam)
    held_p = heldout_path(source, n_calib, n_heldout, max_eval_tokens)

    need_src = not all(os.path.exists(p) for p in src_paths)
    need_tgt = not all(os.path.exists(p) for p in tgt_paths)
    need_rank = not os.path.exists(rank_p)
    need_held = not os.path.exists(held_p)
    done = {"captured_source": need_src, "captured_target": need_tgt,
            "ranked": need_rank, "built_heldout": need_held}
    if not any((need_src, need_tgt, need_rank, need_held)):
        print("prepare: all artifacts cached, nothing to do")
        return done

    tok = AutoTokenizer.from_pretrained(source)
    calib = None
    if need_src or need_tgt or need_held:
        print(f"== Corpus: {n_calib} calib docs + {n_heldout} held-out blocks (FineWeb-Edu) ==")
        calib, heldout = load_corpus(tok, n_calib, n_heldout, max_eval_tokens,
                                     need_calib=need_src or need_tgt,
                                     need_heldout=need_held)
        if need_held:
            torch.save(heldout, held_p)

    def capture(model_id, paths):
        print(f"== Capturing calibration KV for {model_id} ==")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        theta = get_rope_theta(model.config)
        t0 = time.time()
        keys, vals = collect_kv(model, tok, calib, seq_len, device, theta, _short(model_id))
        print(f"captured {keys[0].shape[0]} observations "
              f"({len(keys)} layers) in {time.time()-t0:.0f}s", flush=True)
        torch.save(keys, paths[0])
        torch.save(vals, paths[1])
        del model
        torch.cuda.empty_cache()
        return keys, vals

    src_kv = capture(source, src_paths) if need_src else None
    tgt_kv = capture(target, tgt_paths) if need_tgt else None

    if need_rank:
        if src_kv is None:
            src_kv = (torch.load(src_paths[0], map_location="cpu", weights_only=False),
                      torch.load(src_paths[1], map_location="cpu", weights_only=False))
        if tgt_kv is None:
            tgt_kv = (torch.load(tgt_paths[0], map_location="cpu", weights_only=False),
                      torch.load(tgt_paths[1], map_location="cpu", weights_only=False))
        src_keys, src_vals = src_kv
        tgt_keys, tgt_vals = tgt_kv
        n_obs = src_keys[0].shape[0]
        # Stride across the WHOLE corpus rather than taking a contiguous slice.
        rank_idx = torch.linspace(0, n_obs - 1, min(16384, n_obs)).long()
        print(f"== Ranking source layers once (per-head K/V R^2, lam={lam}) ==")
        t0 = time.time()
        rankings, scores = [], []
        for tl in range(len(tgt_keys)):
            order, s = rank_source_layers(src_keys, src_vals, tgt_keys[tl], tgt_vals[tl],
                                          rank_idx, lam, device)
            rankings.append(order)
            scores.append(s)
            if tl % 6 == 0 or tl == len(tgt_keys) - 1:
                print(f"  target layer {tl:2d}: top sources={order[:8]}", flush=True)
        torch.save({"rankings": rankings, "scores": scores,
                    "n_source_layers": len(src_keys)}, rank_p)
        print(f"ranking done in {time.time()-t0:.0f}s")

    vol.commit()
    return done


# -----------------------------------------------------------------------------
# Stage 2: fit + evaluate ONE k (spawned in parallel across the sweep)
# -----------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=TIMEOUT_SECONDS, memory=65536, volumes={"/out": vol})
def run_k(k: int, source: str, target: str, n_calib: int = 500, seq_len: int = 1024,
          lam: float = 0.01, n_heldout: int = 48):
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    tag = f"[k={k}]"

    res_p = eval_result_path(source, target, n_calib, seq_len, lam, k, n_heldout)
    if os.path.exists(res_p):
        print(f"{tag} reusing persisted eval result", flush=True)
        return torch.load(res_p, weights_only=False)

    # ---------------- Fit (models NOT loaded yet: the whole GPU belongs to the
    # ridge solve, which peaks around 11 GB at k=24) ----------------
    fit_p = fit_cache_path(source, target, n_calib, seq_len, lam, k)
    if os.path.exists(fit_p):
        blob = torch.load(fit_p, map_location="cpu", weights_only=False)
        mapper, fit_summary = blob["mapper"], blob["summary"]
        print(f"{tag} reusing cached fit "
              f"(R^2 K={fit_summary['r2_k']:.3f} V={fit_summary['r2_v']:.3f})", flush=True)
    else:
        rank_blob = torch.load(ranking_path(source, target, n_calib, seq_len, lam),
                               map_location="cpu", weights_only=False)
        rankings = rank_blob["rankings"]
        assert k <= rank_blob["n_source_layers"], \
            f"top_k={k} exceeds {rank_blob['n_source_layers']} source layers"
        src_paths = capture_paths(source, n_calib, seq_len)
        tgt_paths = capture_paths(target, n_calib, seq_len)
        print(f"{tag} loading captured KV from volume", flush=True)
        src_keys = torch.load(src_paths[0], map_location="cpu", weights_only=False)
        src_vals = torch.load(src_paths[1], map_location="cpu", weights_only=False)
        tgt_keys = torch.load(tgt_paths[0], map_location="cpu", weights_only=False)
        tgt_vals = torch.load(tgt_paths[1], map_location="cpu", weights_only=False)

        print(f"{tag} == Fitting mapper (separate K/V ridge, all heads) ==", flush=True)
        t0 = time.time()
        mapper = fit_mapper(src_keys, src_vals, tgt_keys, tgt_vals, rankings, k, lam, device)
        mk = sum(x["k"] for x in mapper["r2"]) / len(mapper["r2"])
        mv = sum(x["v"] for x in mapper["r2"]) / len(mapper["r2"])
        fit_summary = {"r2_k": mk, "r2_v": mv, "seconds": time.time() - t0}
        torch.save({"mapper": mapper, "summary": fit_summary}, fit_p)
        vol.commit()
        print(f"{tag} fit done in {fit_summary['seconds']:.0f}s | "
              f"mean R^2 K={mk:.3f} V={mv:.3f}", flush=True)
        del src_keys, src_vals, tgt_keys, tgt_vals
        torch.cuda.empty_cache()

    # ---------------- Evaluate ----------------
    print(f"{tag} loading {source} and {target}", flush=True)
    tok = AutoTokenizer.from_pretrained(source)
    m_src = AutoModelForCausalLM.from_pretrained(source, dtype=dtype).to(device).eval()
    m_tgt = AutoModelForCausalLM.from_pretrained(target, dtype=dtype).to(device).eval()
    cs, ct = m_src.config, m_tgt.config
    assert cs.num_key_value_heads == ct.num_key_value_heads, "KV head count mismatch"
    assert cs.head_dim == ct.head_dim, "head_dim mismatch"
    H, d = cs.num_key_value_heads, cs.head_dim
    theta_s, theta_t = get_rope_theta(cs), get_rope_theta(ct)

    mapper_gpu = mapper_to_device(mapper, device)
    max_eval_tokens = max(CONTEXT_LENGTHS) + CONTINUATION_LENGTH
    heldout = torch.load(heldout_path(source, n_calib, n_heldout, max_eval_tokens),
                         map_location="cpu", weights_only=False)

    print(f"{tag} == Held-out eval ({n_heldout} blocks, "
          f"continuation {CONTINUATION_LENGTH}) ==", flush=True)
    quality = heldout_eval(m_src, m_tgt, {k: mapper_gpu}, heldout, theta_s, theta_t, H, d,
                           CONTEXT_LENGTHS, CONTINUATION_LENGTH, device, dtype, tag=tag)

    print(f"{tag} == QA smoke test ==", flush=True)
    qa = qa_smoke(m_src, m_tgt, tok, mapper_gpu, theta_s, theta_t, H, d, device, dtype, tag=tag)

    result = {"k": k, "fit": fit_summary,
              "quality": {p: quality[p][k] for p in CONTEXT_LENGTHS}, "qa": qa}
    torch.save(result, res_p)
    vol.commit()
    return result


# -----------------------------------------------------------------------------
# Stage 3: timing sweep -- ALL ks in one container on one physical GPU, so the
# numbers are comparable across k. Mappers load one at a time: at k=24 a bf16
# mapper is ~3.6 GB, and six at once would not fit beside both models.
# -----------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=TIMEOUT_SECONDS, memory=32768, volumes={"/out": vol})
def timing_all(ks: list, source: str, target: str, n_calib: int = 500, seq_len: int = 1024,
               lam: float = 0.01, n_heldout: int = 48):
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"== Timing: loading {source} and {target} ==")
    m_src = AutoModelForCausalLM.from_pretrained(source, dtype=dtype).to(device).eval()
    m_tgt = AutoModelForCausalLM.from_pretrained(target, dtype=dtype).to(device).eval()
    cs, ct = m_src.config, m_tgt.config
    H, d = cs.num_key_value_heads, cs.head_dim
    theta_s, theta_t = get_rope_theta(cs), get_rope_theta(ct)

    max_eval_tokens = max(CONTEXT_LENGTHS) + CONTINUATION_LENGTH
    block = torch.load(heldout_path(source, n_calib, n_heldout, max_eval_tokens),
                       map_location="cpu", weights_only=False)[0]

    print("== Timing sweep (warmup + median of 5, CUDA-synchronized) ==")
    results = {}
    for prefix in CONTEXT_LENGTHS:
        ids = block[:prefix].unsqueeze(0).to(device)

        def source_prefill(ids=ids):
            with torch.no_grad():
                m_src(ids, use_cache=True)

        def target_prefill(ids=ids):
            with torch.no_grad():
                m_tgt(ids, use_cache=True)

        results[prefix] = {"source_prefill": timed_median(source_prefill),
                           "target_prefill": timed_median(target_prefill),
                           "mapping": {}}
        print(f"  prefix={prefix:5d}: source={results[prefix]['source_prefill']*1e3:.1f}ms "
              f"target={results[prefix]['target_prefill']*1e3:.1f}ms", flush=True)
        torch.cuda.empty_cache()

    # One mapper AND one source cache resident at a time: a 16K source cache is
    # ~1.9 GB and the k=24 mapper ~3.6 GB, which together with both models is
    # already most of an A10G. Rebuilding the source cache per (k, prefix)
    # costs a few seconds total and buys the headroom.
    for k in ks:
        blob = torch.load(fit_cache_path(source, target, n_calib, seq_len, lam, k),
                          map_location="cpu", weights_only=False)
        mapper_gpu = mapper_to_device(blob["mapper"], device)
        del blob
        for prefix in CONTEXT_LENGTHS:
            ids = block[:prefix].unsqueeze(0).to(device)
            with torch.no_grad():
                src_cache = m_src(ids, use_cache=True).past_key_values
            pos = torch.arange(prefix, device=device)

            def map_cache(src_cache=src_cache, mapper_gpu=mapper_gpu, pos=pos):
                transfer_cache(src_cache, mapper_gpu, pos, theta_s, theta_t,
                               H, d, device, dtype)

            try:
                results[prefix]["mapping"][k] = timed_median(map_cache)
                print(f"  k={k:2d} prefix={prefix:5d}: "
                      f"map {results[prefix]['mapping'][k]*1e3:.1f}ms", flush=True)
            except torch.cuda.OutOfMemoryError:
                results[prefix]["mapping"][k] = float("nan")
                print(f"  k={k:2d} prefix={prefix:5d}: OOM during mapping timing", flush=True)
            del src_cache
            torch.cuda.empty_cache()
        del mapper_gpu
        torch.cuda.empty_cache()
    return results


# -----------------------------------------------------------------------------
# Benchmark evaluation (paper S 4.1 protocol)
#
# HellaSwag / ARC-Challenge / MMLU 5-shot are multiple-choice log-likelihood
# benchmarks: each answer choice is scored under the model and the argmax is
# compared to gold -- no generation. Prompt formats and the length-normalized
# acc_norm metric follow lm-eval-harness defaults, which is the paper's
# protocol. The transfer condition prefILLS each question's context through
# the SOURCE model, maps that cache, and the TARGET scores the choices.
# -----------------------------------------------------------------------------

def _stride_sample(n_total, n_want):
    """Deterministic, evenly spaced subsample; <= 0 means the full set. The
    test splits are ordered (MMLU 'all' by subject), so striding stratifies."""
    if n_want <= 0 or n_want >= n_total:
        return list(range(n_total))
    step = n_total / n_want
    return [int(i * step) for i in range(n_want)]


def _hs_preprocess(text):
    import re
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_bench_data(tok, n_samples):
    """Tokenized multiple-choice items per benchmark, deterministic given
    (tok, n_samples). Continuation ids are tok(ctx + " " + choice) minus the
    ctx ids, so boundary tokenization matches a full-sequence forward
    (lm-eval's convention, " " being its target delimiter). acc_norm
    normalizes by the choice's character length, also per lm-eval."""
    from collections import defaultdict

    from datasets import load_dataset

    def encode(query, choices, gold):
        ctx = tok(query, add_special_tokens=False).input_ids
        conts, lens = [], []
        for ch in choices:
            whole = tok(query + " " + ch, add_special_tokens=False).input_ids
            cont = whole[len(ctx):] or tok(" " + ch, add_special_tokens=False).input_ids
            conts.append(cont)
            lens.append(max(1, len(ch)))
        return {"ctx": ctx, "conts": conts, "lens": lens, "gold": gold}

    out = {}

    ds = load_dataset("hellaswag", split="validation")
    items = []
    for i in _stride_sample(len(ds), n_samples):
        row = ds[i]
        query = _hs_preprocess(
            row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
        items.append(encode(query, [_hs_preprocess(e) for e in row["endings"]],
                            int(row["label"])))
    out["hellaswag"] = {"floor": 0.25, "metric": "acc_norm", "items": items}

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for i in _stride_sample(len(ds), n_samples):
        row = ds[i]
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        items.append(encode("Question: " + row["question"] + "\nAnswer:",
                            list(row["choices"]["text"]),
                            labels.index(row["answerKey"])))
    out["arc_challenge"] = {"floor": 0.25, "metric": "acc_norm", "items": items}

    LETTERS = ["A", "B", "C", "D"]

    def mmlu_q(row, with_answer):
        s = row["question"].strip() + "\n"
        for letter, choice in zip(LETTERS, row["choices"]):
            s += f"{letter}. {choice}\n"
        s += "Answer:"
        return s + (f" {LETTERS[int(row['answer'])]}\n\n" if with_answer else "")

    dev = defaultdict(list)
    for row in load_dataset("cais/mmlu", "all", split="dev"):
        dev[row["subject"]].append(row)
    shots = {s: ("The following are multiple choice questions (with answers) about "
                 f"{s.replace('_', ' ')}.\n\n" + "".join(mmlu_q(r, True) for r in rows[:5]))
             for s, rows in dev.items()}
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    for i in _stride_sample(len(ds), n_samples):
        row = ds[i]
        items.append(encode(shots[row["subject"]] + mmlu_q(row, False),
                            LETTERS, int(row["answer"])))
    out["mmlu"] = {"floor": 0.25, "metric": "acc", "items": items}
    return out


def load_wikitext_chunks(tok, chunk_len=2048):
    """Paper App. D: non-overlapping 2048-token chunks of the WikiText-2 test
    split, second half scored against a prefix cache of the first half."""
    import torch
    from datasets import load_dataset
    text = "".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, add_special_tokens=False).input_ids
    n = len(ids) // chunk_len
    return torch.tensor(ids[:n * chunk_len], dtype=torch.long).reshape(n, chunk_len)


def expand_cache(cache, batch):
    """Batch-expand a batch-1 DynamicCache. The scoring forward mutates the
    expansion, so callers treat it as disposable."""
    from transformers import DynamicCache
    out = DynamicCache()
    for l in range(cache_num_layers(cache)):
        k, v = cache_layer_kv(cache, l)
        out.update(k.expand(batch, -1, -1, -1).contiguous(),
                   v.expand(batch, -1, -1, -1).contiguous(), l)
    return out


def score_mc_suite(model, tok, bench_data, make_cache, device, tag=""):
    """Accuracy per benchmark, each item's choices scored in ONE batched
    forward conditioned on make_cache(ctx[:-1]).

    make_cache(prefix_ids) must return a cache holding exactly prefix_ids for
    `model` -- the transfer path maps a source prefill, the standalone path
    prefills the scoring model itself, and everything downstream is identical.
    The cache stops at ctx[:-1] because a cache carries no logits: feeding
    [last ctx token] + ending[:-1] yields a logit for every ending token,
    including single-token MMLU answers."""
    import time

    import torch
    import torch.nn.functional as F

    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    results = {}
    for name, bench in bench_data.items():
        t0 = time.time()
        n_acc = n_norm = 0
        for it_i, item in enumerate(bench["items"]):
            ctx = torch.tensor(item["ctx"], dtype=torch.long, device=device)
            cache = make_cache(ctx[:-1])
            past = ctx.shape[0] - 1
            rows = [[item["ctx"][-1]] + c[:-1] for c in item["conts"]]
            B, L = len(rows), max(len(r) for r in rows)
            inp = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros(B, past + L, dtype=torch.long, device=device)
            mask[:, :past] = 1
            for b, r in enumerate(rows):
                inp[b, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
                mask[b, past:past + len(r)] = 1
            pos = (past + torch.arange(L, device=device)).unsqueeze(0).expand(B, L)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=mask, position_ids=pos,
                            past_key_values=expand_cache(cache, B), use_cache=True)
            logprobs = F.log_softmax(out.logits.float(), dim=-1)
            lls = [float(logprobs[b, :len(c)].gather(
                       -1, torch.tensor(c, device=device).unsqueeze(-1)).sum())
                   for b, c in enumerate(item["conts"])]
            n_acc += max(range(B), key=lambda b: lls[b]) == item["gold"]
            n_norm += max(range(B), key=lambda b: lls[b] / item["lens"][b]) == item["gold"]
            if (it_i + 1) % 500 == 0:
                print(f"{tag} {name}: {it_i + 1}/{len(bench['items'])}", flush=True)
        n = len(bench["items"])
        results[name] = {"acc": n_acc / n, "acc_norm": n_norm / n, "n": n,
                         "metric": bench["metric"], "floor": bench["floor"]}
        print(f"{tag} {name}: acc={n_acc / n:.4f} acc_norm={n_norm / n:.4f} "
              f"(n={n}, {time.time() - t0:.0f}s)", flush=True)
    return results


def wikitext_prefix_ppl(model, chunks, make_cache, device, prefix_len=1024, tag=""):
    """Paper App. D prefix perplexity. The cache holds tokens [0, prefix_len-1)
    and the input starts at the last prefix token, so all of the second half's
    tokens get a logit."""
    import math

    import torch
    import torch.nn.functional as F

    nll, count = 0.0, 0
    for chunk in chunks:
        ids = chunk.to(device)
        cache = make_cache(ids[:prefix_len - 1])
        inp = ids[prefix_len - 1:-1].unsqueeze(0)
        tgt = ids[prefix_len:]
        mask = torch.ones(1, (prefix_len - 1) + inp.shape[1], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=mask, past_key_values=cache, use_cache=True)
        nll += float(F.cross_entropy(out.logits[0].float(), tgt, reduction="sum"))
        count += tgt.shape[0]
    ppl = math.exp(nll / count)
    print(f"{tag} wikitext2 prefix-ppl: {ppl:.3f} ({len(chunks)} chunks)", flush=True)
    return {"ppl": ppl, "n_chunks": int(len(chunks)), "prefix_len": prefix_len}


# -----------------------------------------------------------------------------
# Stage 4: benchmark evaluation -- one container per k for the mapped
# condition, plus standalone baselines cached on the volume per model.
# -----------------------------------------------------------------------------

@app.function(gpu=GPU_TYPE, timeout=TIMEOUT_SECONDS, memory=32768, volumes={"/out": vol})
def bench_k(k: int, source: str, target: str, n_calib: int = 500, seq_len: int = 1024,
            lam: float = 0.01, n_samples: int = 2000):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    tag = f"[bench k={k}]"

    res_p = bench_result_path(source, target, n_calib, seq_len, lam, k, n_samples)
    if os.path.exists(res_p):
        print(f"{tag} reusing persisted bench result", flush=True)
        return torch.load(res_p, weights_only=False)

    fit_p = fit_cache_path(source, target, n_calib, seq_len, lam, k)
    if not os.path.exists(fit_p):
        return {"k": k, "error": f"fit cache missing ({os.path.basename(fit_p)}); "
                                 "rerun after the sweep has fitted this k"}
    mapper = torch.load(fit_p, map_location="cpu", weights_only=False)["mapper"]

    print(f"{tag} loading {source} and {target}", flush=True)
    tok = AutoTokenizer.from_pretrained(source)
    m_src = AutoModelForCausalLM.from_pretrained(source, dtype=dtype).to(device).eval()
    m_tgt = AutoModelForCausalLM.from_pretrained(target, dtype=dtype).to(device).eval()
    cs, ct = m_src.config, m_tgt.config
    H, d = cs.num_key_value_heads, cs.head_dim
    theta_s, theta_t = get_rope_theta(cs), get_rope_theta(ct)
    mapper_gpu = mapper_to_device(mapper, device)

    def make_mapped(prefix_ids):
        pos = torch.arange(prefix_ids.shape[0], device=device)
        with torch.no_grad():
            src_cache = m_src(prefix_ids.unsqueeze(0), use_cache=True).past_key_values
        return transfer_cache(src_cache, mapper_gpu, pos, theta_s, theta_t,
                              H, d, device, dtype)

    bench_data = load_bench_data(tok, n_samples)
    mc = score_mc_suite(m_tgt, tok, bench_data, make_mapped, device, tag=tag)
    wt = wikitext_prefix_ppl(m_tgt, load_wikitext_chunks(tok), make_mapped, device, tag=tag)
    result = {"k": k, "mc": mc, "wikitext": wt, "n_samples": n_samples}
    torch.save(result, res_p)
    vol.commit()
    return result


@app.function(gpu=GPU_TYPE, timeout=TIMEOUT_SECONDS, memory=32768, volumes={"/out": vol})
def bench_baseline(model_id: str, n_samples: int = 2000):
    """Standalone accuracy for one model: its own prefill feeds the same
    scoring path the mapped condition uses, so the only difference between
    conditions is where the cache came from. k-independent, cached on the
    volume per (model, n_samples). Target = retention denominator; source =
    the 'small model alone' reference the cascade must beat."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = baseline_cache_path(model_id, n_samples)
    if os.path.exists(path):
        print(f"[base {_short(model_id)}] reusing cached baseline", flush=True)
        return torch.load(path, weights_only=False)

    device = "cuda"
    dtype = torch.float16
    tag = f"[base {_short(model_id)}]"
    print(f"{tag} loading {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    def make_self(prefix_ids):
        with torch.no_grad():
            return model(prefix_ids.unsqueeze(0), use_cache=True).past_key_values

    bench_data = load_bench_data(tok, n_samples)
    mc = score_mc_suite(model, tok, bench_data, make_self, device, tag=tag)
    wt = wikitext_prefix_ppl(model, load_wikitext_chunks(tok), make_self, device, tag=tag)
    res = {"model": model_id, "mc": mc, "wikitext": wt, "n_samples": n_samples}
    torch.save(res, path)
    vol.commit()
    return res


@app.function(timeout=600, volumes={"/out": vol})
def save_summary(name: str, payload: dict):
    """Merge top-level keys into any existing summary so stage-skipping reruns
    (e.g. a --skip-nll bench-only run) do not clobber earlier results."""
    import json
    path = f"/out/{name}"
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(payload)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    vol.commit()
    return path


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main(source: str = "Qwen/Qwen3-0.6B", target: str = "Qwen/Qwen3-1.7B",
         n_calib: int = 500, seq_len: int = 1024, lam: float = 0.01,
         n_heldout: int = 48, bench_samples: int = BENCH_SAMPLES,
         skip_nll: bool = False, skip_bench: bool = False, skip_timing: bool = False):
    ks = tuple(sorted(set(TOP_K_SWEEP)))
    max_eval_tokens = max(CONTEXT_LENGTHS) + CONTINUATION_LENGTH

    # Pre-check the volume so cached stages cost no GPU container. Falls back
    # to running prepare (itself idempotent) if the listing fails.
    artifacts = [*capture_paths(source, n_calib, seq_len),
                 *capture_paths(target, n_calib, seq_len),
                 ranking_path(source, target, n_calib, seq_len, lam),
                 heldout_path(source, n_calib, n_heldout, max_eval_tokens)]
    names = None
    try:
        names = {os.path.basename(e.path) for e in vol.listdir("/")}
        prep_needed = not all(os.path.basename(p) in names for p in artifacts)
        cached_fits = [k for k in ks if os.path.basename(
            fit_cache_path(source, target, n_calib, seq_len, lam, k)) in names]
    except Exception as e:
        print(f"volume listing unavailable ({e}); running prepare unconditionally")
        prep_needed, cached_fits = True, []

    print(f"pair: {source} -> {target} | k sweep: {list(ks)} | GPU: {GPU_TYPE} | "
          f"contexts: {list(CONTEXT_LENGTHS)}")
    print(f"cached fits: {cached_fits or 'none'} | prepare needed: {prep_needed} | "
          f"stages: nll={not skip_nll} bench={not skip_bench} timing={not skip_timing}")

    # Benchmarks and baselines depend only on fit caches, not on prepare's
    # captures/held-out blocks -- spawn them first so they overlap prepare.
    base_handles, bench_handles = [], {}
    if not skip_bench:
        base_handles = [bench_baseline.spawn(model_id=m, n_samples=bench_samples)
                        for m in (target, source)]
        benchable = list(ks) if names is None else [k for k in ks if k in cached_fits]
        skipped = [k for k in ks if k not in benchable]
        if skipped:
            print(f"bench: skipping k={skipped} (no cached fit yet; this run fits them -- "
                  f"rerun with --skip-nll --skip-timing to bench them cheaply)")
        bench_handles = {k: bench_k.spawn(k=k, source=source, target=target, n_calib=n_calib,
                                          seq_len=seq_len, lam=lam, n_samples=bench_samples)
                         for k in benchable}

    if prep_needed and not (skip_nll and skip_timing):
        print(f"prepare: {prepare.remote(source=source, target=target, n_calib=n_calib, seq_len=seq_len, lam=lam, n_heldout=n_heldout)}")

    run_handles = {}
    if not skip_nll:
        run_handles = {k: run_k.spawn(k=k, source=source, target=target, n_calib=n_calib,
                                      seq_len=seq_len, lam=lam, n_heldout=n_heldout)
                       for k in ks}
    print(f"spawned {len(run_handles)} run_k + {len(bench_handles)} bench_k + "
          f"{len(base_handles)} baseline containers")

    results = {k: h.get() for k, h in run_handles.items()}
    timing = None
    if not skip_timing:
        timing_ks = list(ks) if not skip_nll else (cached_fits or list(ks))
        print("running timing sweep (single container, all ks) ...")
        timing = timing_all.remote(ks=timing_ks, source=source, target=target, n_calib=n_calib,
                                   seq_len=seq_len, lam=lam, n_heldout=n_heldout)
    base_target = base_handles[0].get() if base_handles else None
    base_source = base_handles[1].get() if base_handles else None
    bench_results = {}
    for k, h in bench_handles.items():
        r = h.get()
        if "error" in r:
            print(f"bench k={k}: {r['error']}")
        else:
            bench_results[k] = r

    # ---------------- Combined report ----------------
    if results:
        print("\n== Fit (mean in-sample R^2) ==")
        print("   k |   K R2 |   V R2 | fit s")
        for k in ks:
            fs = results[k]["fit"]
            print(f"  {k:2d} | {fs['r2_k']:6.3f} | {fs['r2_v']:6.3f} | {fs.get('seconds', float('nan')):5.0f}")

        print(f"\n== Held-out: context benefit retained (%) (n={n_heldout} blocks) ==")
        print("prefix | " + " | ".join(f" k={k:<2d}" for k in ks) + " | real ppl | none ppl")
        for prefix in CONTEXT_LENGTHS:
            cells = " | ".join(
                f"{results[k]['quality'][prefix]['benefit_retained']*100:5.1f}" for k in ks)
            ev0 = results[ks[0]]["quality"][prefix]
            print(f"{prefix:6d} | {cells} |    {ev0['ppl']['real']:5.2f} |    {ev0['ppl']['none']:5.2f}")

        print("\n== Held-out: mapped ppl ==")
        print("prefix | " + " | ".join(f" k={k:<2d}" for k in ks))
        for prefix in CONTEXT_LENGTHS:
            cells = " | ".join(f"{results[k]['quality'][prefix]['ppl']['mapped']:5.2f}" for k in ks)
            print(f"{prefix:6d} | {cells}")

        cos_prefixes = [p for p in CONTEXT_LENGTHS if p <= COSINE_MAX_PREFIX]
        if cos_prefixes:
            print(f"\n== Attention-output cosine (prefix <= {COSINE_MAX_PREFIX}) ==")
            print("prefix | " + " | ".join(f" k={k:<2d}" for k in ks))
            for prefix in cos_prefixes:
                cells = " | ".join(f"{results[k]['quality'][prefix]['attn_cosine']:5.3f}" for k in ks)
                buckets = results[ks[-1]]["quality"][prefix]["attn_cosine_by_position"]
                print(f"{prefix:6d} | {cells} | k={ks[-1]} buckets: "
                      + " ".join(f"{b:.3f}" for b in buckets))

        print("\n== QA smoke test ==")
        for k in ks:
            qa = results[k]["qa"]
            print(f"  k={k:2d}: mapped {qa['mapped']}/{qa['n']} | "
                  f"real {qa['real']}/{qa['n']} | none {qa['none']}/{qa['n']}")

    if timing:
        tks = sorted(next(iter(timing.values()))["mapping"])
        print("\n== Timing (median ms) ==")
        print("prefix | src prefill | tgt prefill | " + " | ".join(f" k={k:<2d}" for k in tks))
        for prefix in CONTEXT_LENGTHS:
            row = timing[prefix]
            cells = " | ".join(f"{row['mapping'][k]*1e3:5.1f}" for k in tks)
            print(f"{prefix:6d} | {row['source_prefill']*1e3:11.1f} | "
                  f"{row['target_prefill']*1e3:11.1f} | {cells}")

    bench_payload = None
    if bench_results and base_target:
        names3 = ["hellaswag", "arc_challenge", "mmlu"]

        def acc_of(res, nm):
            b = res["mc"][nm]
            return b[b["metric"]]

        print(f"\n== Benchmarks (~{bench_samples if bench_samples > 0 else 'all'}/bench; "
              f"HS+ARC acc_norm, MMLU 5-shot acc; lm-eval prompt conventions) ==")
        for label, base in (("target standalone", base_target), ("source standalone", base_source)):
            if base:
                print(f"  {label:17s}: " + " | ".join(
                    f"{nm} {acc_of(base, nm)*100:5.2f}" for nm in names3)
                    + f" | wt2 ppl {base['wikitext']['ppl']:6.2f}")
        print("\n   k | hs ret | arc ret | mmlu ret | mean acc | fn ret | wt2 ppl")
        crit = {}
        for k in sorted(bench_results):
            r = bench_results[k]
            accs = [acc_of(r, nm) for nm in names3]
            bases = [acc_of(base_target, nm) for nm in names3]
            floors = [r["mc"][nm]["floor"] for nm in names3]
            rets = [a / b for a, b in zip(accs, bases)]
            fns = [(a - f) / (b - f) for a, b, f in zip(accs, bases, floors)]
            crit[k] = sum(accs) / len(accs)
            print(f"  {k:2d} | {rets[0]*100:6.1f} | {rets[1]*100:7.1f} | {rets[2]*100:8.1f}"
                  f" | {crit[k]*100:8.2f} | {sum(fns)/len(fns)*100:6.1f}"
                  f" | {r['wikitext']['ppl']:7.2f}")
        best = max(crit, key=lambda kk: (crit[kk], kk))
        print(f"\n  selected k = {best} (argmax mean log-likelihood accuracy over "
              f"HS/ARC-C/MMLU, ties toward larger k -- paper S 4.1 criterion)")
        bench_payload = {"baseline_target": base_target, "baseline_source": base_source,
                         "per_k": {str(k): v for k, v in bench_results.items()},
                         "selected_k": best, "n_samples": bench_samples}

    ctx_tag = "-".join(str(p) for p in CONTEXT_LENGTHS)
    name = (f"summary_{_short(source)}_to_{_short(target)}"
            f"_n{n_calib}_L{seq_len}_lam{lam}_ctx{ctx_tag}.json")
    payload = {"source": source, "target": target, "n_calib": n_calib,
               "seq_len": seq_len, "lam": lam, "gpu": GPU_TYPE, "ks": list(ks)}
    if results:
        payload["n_heldout"] = n_heldout
        payload["contexts"] = list(CONTEXT_LENGTHS)
        payload["results"] = {str(k): results[k] for k in ks}
    if timing:
        payload["timing"] = {str(p): timing[p] for p in timing}
    if bench_payload:
        payload["bench"] = bench_payload
    path = save_summary.remote(name=name, payload=payload)
    print(f"\nsummary saved to volume kvbridge-out as {os.path.basename(path)}")
