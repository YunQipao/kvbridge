"""Offline fitting: calibrate a linear mapper for one model pair.

Implements the closed-form pipeline of "Cross-Model KV Cache Transfer in LLM
Families" (Heo et al., NVIDIA, arXiv:2608.03893): capture per-layer KV on a
small calibration corpus, strip RoPE from keys, select the top-k most
predictive source layers per target layer, and fit separate centered ridge
regressions for K and V (the bias comes from the means and is not regularized).

Precision: normal equations accumulate in float32 on the compute device; the
centering and solve run in float64 on the CPU. Consumer GPUs run fp64 at
~1/32 rate and Apple MPS has no fp64 at all, so the CPU solve is both the
portable and the fast choice; it matches a full-fp64 fit to ~5e-4 relative.

This module needs enough memory for one model at a time plus the captured
activations (~30 GB CPU RAM for a Qwen3 1.7B->4B fit at the default 500x1024
calibration). Model forwards run on CUDA, MPS, or CPU.
"""

import time

import torch

from .cache_compat import cache_layer_kv
from .mapper import Mapper
from .rope import get_rope_theta, strip_rope

RIDGE_CHUNK = 8192


def _log(verbose, msg):
    if verbose:
        print(msg, flush=True)


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _head_dim(config):
    hd = getattr(config, "head_dim", None)
    return int(hd) if hd else config.hidden_size // config.num_attention_heads


def _solve_fp32(A, B):
    """fp32 solve that routes through CPU on backends without linalg support."""
    if A.device.type == "mps":
        return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)
    return torch.linalg.solve(A, B)


def default_calibration_texts(n_calib=500, min_chars=3000):
    """Stream FineWeb-Edu (the paper's calibration domain). Requires the
    optional ``datasets`` dependency (``pip install kvbridge[fit]``)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "default calibration texts need the `datasets` package; "
            "install kvbridge[fit] or pass calibration_texts=") from e
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    texts = []
    for row in ds:
        if len(row["text"]) > min_chars:
            texts.append(row["text"])
            if len(texts) == n_calib:
                break
    return texts


def collect_kv(model, tok, texts, seq_len=1024, stride=4, device="cuda",
               verbose=True, desc=""):
    """Run texts through ``model`` and capture per-layer KV.

    Returns (keys, vals): lists over layers of CPU fp16 tensors [N, H, d],
    keys RoPE-stripped (content space). Positions are stride-subsampled
    (paper App. E uses stride 4) with the offset rotated per sequence so
    every position phase is represented.
    """
    rope_theta = get_rope_theta(model.config)
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim = _head_dim(model.config)

    # Tokenize everything first so the capture buffers can be preallocated and
    # filled in place. Appending per-document tensors and torch.cat-ing at the
    # end briefly holds the fragments AND the consolidated copy -- a +100%
    # peak-RAM transient that evicts a 24 GB machine for no benefit.
    all_ids, all_keep = [], []
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=seq_len).input_ids
        T = ids.shape[1]
        if T < 32:
            continue
        all_ids.append(ids)
        all_keep.append(torch.arange(i % stride, T, stride))
    n_total = sum(len(k) for k in all_keep)

    shape = (n_total, n_kv_heads, head_dim)
    keys = [torch.empty(shape, dtype=torch.float16) for _ in range(n_layers)]
    vals = [torch.empty(shape, dtype=torch.float16) for _ in range(n_layers)]

    n0 = 0
    for i, (ids, keep) in enumerate(zip(all_ids, all_keep)):
        ids = ids.to(device)
        keep = keep.to(device)
        pos = torch.arange(ids.shape[1], device=device)
        n1 = n0 + len(keep)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        cache = out.past_key_values
        for l in range(n_layers):
            k, v = cache_layer_kv(cache, l)
            k, v = k[0], v[0]                                   # [H, T, d]
            k_content = strip_rope(k.float(), pos, rope_theta)
            keys[l][n0:n1] = k_content[:, keep].transpose(0, 1).half().cpu()
            vals[l][n0:n1] = v.float()[:, keep].transpose(0, 1).half().cpu()
        n0 = n1
        del out, cache
        if (i + 1) % 50 == 0:
            _log(verbose, f"  [{desc}] {i + 1}/{len(all_ids)} sequences")

    return keys, vals


def _flat(t, n0, n1, device):
    """CPU fp16 [N, H, d] -> device fp32 [n, H*d] slice."""
    x = t[n0:n1].to(device).float()
    return x.reshape(x.shape[0], -1)


def ridge_centered(X, Y, lam=0.01):
    """Small centered ridge, fp32, in memory. X: [N, p], Y: [N, q].
    Returns (W, mu_x, mu_y); bias is implied by the means (unregularized)."""
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    scale = 1.0 / X.shape[0]
    XtX = (Xc.T @ Xc) * scale
    XtX += lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    W = _solve_fp32(XtX, (Xc.T @ Yc) * scale)
    return W, mu_x, mu_y


def r2_score(X, W, mu_x, mu_y, Y):
    resid = (Y - mu_y) - (X - mu_x) @ W
    ss_res = (resid ** 2).sum()
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def ridge_streaming(src_list, tgt_layer, chosen, lam=0.01, device="cpu",
                    chunk=RIDGE_CHUNK):
    """Centered ridge for a large concatenated design, accumulating the
    normal equations over row chunks so the full [N, p] design never
    materializes. Moments accumulate fp32 on ``device``; centering and the
    solve run fp64 on CPU.

    src_list: list over source layers of CPU fp16 [N, H, d]
    tgt_layer: CPU fp16 [N, H, d] response for one target layer
    chosen: source layer indices to concatenate as features

    Returns (W [p, q], mu_x [p], mu_y [q], r2) as fp32 CPU tensors; r2 is
    in-sample, computed from the same accumulated moments.
    """
    N = tgt_layer.shape[0]
    p = sum(src_list[sl].shape[1] * src_list[sl].shape[2] for sl in chosen)
    q = tgt_layer.shape[1] * tgt_layer.shape[2]

    XtX = torch.zeros(p, p, device=device, dtype=torch.float32)
    XtY = torch.zeros(p, q, device=device, dtype=torch.float32)
    sum_x = torch.zeros(p, dtype=torch.float64)
    sum_y = torch.zeros(q, dtype=torch.float64)
    sum_yy = torch.zeros(q, dtype=torch.float64)

    for n0 in range(0, N, chunk):
        n1 = min(n0 + chunk, N)
        Xb = torch.cat([_flat(src_list[sl], n0, n1, device) for sl in chosen], dim=-1)
        Yb = _flat(tgt_layer, n0, n1, device)
        XtX.addmm_(Xb.T, Xb)
        XtY.addmm_(Xb.T, Yb)
        # .cpu() first: MPS has no float64, so the widening must happen on CPU.
        sum_x += Xb.sum(dim=0).cpu().double()
        sum_y += Yb.sum(dim=0).cpu().double()
        sum_yy += (Yb ** 2).sum(dim=0).cpu().double()
        del Xb, Yb

    XtX = XtX.cpu().double()
    XtY = XtY.cpu().double()
    mu_x = sum_x / N
    mu_y = sum_y / N
    # Centered moments from raw moments: Xc'Xc = X'X - N mu_x mu_x'
    XtX -= N * torch.outer(mu_x, mu_x)
    XtY -= N * torch.outer(mu_x, mu_y)
    XtX /= N
    XtY /= N
    XtX.diagonal().add_(lam)
    W = torch.linalg.solve(XtX, XtY)

    # In-sample R^2 from moments, using the unregularized centered moments.
    XtX.diagonal().sub_(lam)
    ss_tot = float((sum_yy - N * mu_y ** 2).sum())
    ss_res = float(
        (sum_yy - N * mu_y ** 2).sum()
        - 2.0 * N * (W * XtY).sum()
        + N * (W * (XtX @ W)).sum()
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return W.float(), mu_x.float(), mu_y.float(), r2


def rank_source_layers(src_keys, src_vals, tgt_k_layer, tgt_v_layer, idx,
                       lam=0.01, device="cpu"):
    """Rank source layers for one target layer by mean per-head K/V ridge R^2
    on a subsample. ``idx`` should stride across the whole corpus so the
    ranking sees every calibration document; the paper's ablation (S 4.3)
    calls this selection the largest single quality contributor."""
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


def fit_mapper(source, target, *, top_k=12, lam=0.01, calibration_texts=None,
               n_calib=500, seq_len=1024, stride=4, rank_subsample=16384,
               device=None, model_dtype=torch.float16, verbose=True):
    """Fit a Mapper for source -> target (HF model ids).

    Loads the two models sequentially (never both at once), captures
    calibration KV, ranks source layers per target layer, and fits the
    per-layer ridge maps. Defaults reproduce the paper's protocol:
    500 FineWeb-Edu sequences x 1024 tokens, stride 4, lambda=0.01.

    top_k=12 is the benchmark-selected value for Qwen3 1.7B->4B; sweep it
    for a new pair (the criterion that matters is task accuracy, not fit
    R^2 -- see the README).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = device or _default_device()
    cs = AutoConfig.from_pretrained(source)
    ct = AutoConfig.from_pretrained(target)
    if cs.num_key_value_heads != ct.num_key_value_heads or _head_dim(cs) != _head_dim(ct):
        raise ValueError(
            f"{source} and {target} are not a matched-KV pair "
            f"(KV heads {cs.num_key_value_heads} vs {ct.num_key_value_heads}, "
            f"head_dim {_head_dim(cs)} vs {_head_dim(ct)}); "
            "the closed-form mapper requires matched KV geometry")
    if top_k > cs.num_hidden_layers:
        raise ValueError(f"top_k={top_k} exceeds {cs.num_hidden_layers} source layers")

    tok = AutoTokenizer.from_pretrained(source)
    if calibration_texts is None:
        _log(verbose, f"streaming {n_calib} FineWeb-Edu calibration documents")
        calibration_texts = default_calibration_texts(n_calib)

    def capture(model_id, desc):
        _log(verbose, f"capturing calibration KV for {model_id} on {device}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=model_dtype).to(device).eval()
        t0 = time.time()
        keys, vals = collect_kv(model, tok, calibration_texts, seq_len=seq_len,
                                stride=stride, device=device, verbose=verbose,
                                desc=desc)
        _log(verbose, f"captured {keys[0].shape[0]} observations in {time.time()-t0:.0f}s")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return keys, vals

    src_keys, src_vals = capture(source, "source")
    tgt_keys, tgt_vals = capture(target, "target")

    n_obs = src_keys[0].shape[0]
    rank_idx = torch.linspace(0, n_obs - 1, min(rank_subsample, n_obs)).long()
    _log(verbose, f"ranking source layers (per-head K/V R^2, lam={lam})")
    rankings = []
    for tl in range(len(tgt_keys)):
        order, _ = rank_source_layers(src_keys, src_vals, tgt_keys[tl], tgt_vals[tl],
                                      rank_idx, lam=lam, device=device)
        rankings.append(order)

    layers, r2s = [], []
    for tl in range(len(tgt_keys)):
        chosen = rankings[tl][:top_k]
        W_k, mu_xk, mu_yk, r2k = ridge_streaming(src_keys, tgt_keys[tl], chosen,
                                                 lam=lam, device=device)
        W_v, mu_xv, mu_yv, r2v = ridge_streaming(src_vals, tgt_vals[tl], chosen,
                                                 lam=lam, device=device)
        layers.append({"source_layers": [int(s) for s in chosen],
                       "W_k": W_k, "mu_xk": mu_xk, "mu_yk": mu_yk,
                       "W_v": W_v, "mu_xv": mu_xv, "mu_yv": mu_yv})
        r2s.append({"k": r2k, "v": r2v})
        _log(verbose, f"  target layer {tl:2d}: sources={sorted(chosen)}  "
                      f"R^2 K={r2k:.3f} V={r2v:.3f}")

    meta = {
        "source": source, "target": target,
        "n_kv_heads": int(cs.num_key_value_heads), "head_dim": _head_dim(cs),
        "rope_theta_source": get_rope_theta(cs), "rope_theta_target": get_rope_theta(ct),
        "top_k": top_k, "lam": lam,
        "mean_r2_k": sum(x["k"] for x in r2s) / len(r2s),
        "mean_r2_v": sum(x["v"] for x in r2s) / len(r2s),
        "calibration": {"n_texts": len(calibration_texts), "seq_len": seq_len,
                        "token_stride": stride, "n_observations": int(n_obs)},
        "method": "arXiv:2608.03893 closed-form per-layer ridge, content-space keys",
    }
    return Mapper(layers, meta)
