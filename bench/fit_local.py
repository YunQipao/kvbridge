"""Fit a mapper on this machine (Apple Silicon included).

    python -m bench.fit_local --pair tiny --top-k 8 --out mappers/tiny-k8

Fitting is memory-bound, not compute-bound: the whole calibration KV for both
models is held in RAM at once. The estimator below prints that number before
loading anything, because on a 24 GB Mac the difference between a 10-minute
fit and a swap-death is the choice of --n-calib.

    capture bytes ~= n_calib * (seq_len / stride) * n_layers * n_kv_heads
                     * head_dim * 2 (fp16) * 2 (K and V)

Calibration data: `--texts FILE` splits a local text file into documents, so a
fit needs no network beyond the model weights. Without it, kvbridge streams
FineWeb-Edu (the paper's corpus), which needs the `datasets` package.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from kvbridge.fit import fit_mapper

from .pairs import PAIRS, get_pair


def _head_dim(cfg):
    hd = getattr(cfg, "head_dim", None)
    return int(hd) if hd else cfg.hidden_size // cfg.num_attention_heads


def capture_gb(model_id, n_calib, seq_len, stride):
    cfg = AutoConfig.from_pretrained(model_id)
    n_obs = n_calib * (seq_len // stride)
    per_obs = cfg.num_hidden_layers * cfg.num_key_value_heads * _head_dim(cfg) * 2 * 2
    return n_obs * per_obs / 1e9


def conditioning(source, n_calib, seq_len, stride, top_k):
    """Rows per free parameter in the per-layer ridge.

    Each target layer regresses H*d outputs on `top_k` source layers' worth of
    features, so the design is [n_obs, top_k * H * d]. When n_obs is not
    comfortably larger than that width the ridge interpolates: the reported
    in-sample R^2 goes to ~1.0 while the mapper transfers nothing. The
    published fit (n_calib=500, top_k=12) sits at ~10 rows per parameter.
    """
    cfg = AutoConfig.from_pretrained(source)
    n_obs = n_calib * (seq_len // stride)
    p = top_k * cfg.num_key_value_heads * _head_dim(cfg)
    return n_obs, p, n_obs / p


def local_texts(path, n_calib, min_chars=3000):
    """Split a text file into ~n_calib documents of at least min_chars."""
    text = Path(path).read_text()
    step = max(min_chars, len(text) // max(1, n_calib))
    docs = [text[i:i + step] for i in range(0, len(text), step)]
    docs = [d for d in docs if len(d) >= min_chars // 2][:n_calib]
    if not docs:
        raise SystemExit(f"{path} is too short for {n_calib} calibration documents")
    return docs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="tiny", choices=sorted(PAIRS))
    ap.add_argument("--top-k", type=int, default=None, help="default: the pair's top_k")
    ap.add_argument("--n-calib", type=int, default=None, help="default: the pair's n_calib")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--texts", default=None, help="local text file to calibrate on")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the memory confirmation")
    args = ap.parse_args(argv)

    pair = get_pair(args.pair)
    top_k = args.top_k or pair.top_k
    n_calib = args.n_calib or pair.n_calib
    out = Path(args.out or f"mappers/{pair.name}-k{top_k}")

    gb = sum(capture_gb(m, n_calib, args.seq_len, args.stride)
             for m in (pair.source, pair.target))
    n_obs, p, ratio = conditioning(pair.source, n_calib, args.seq_len,
                                   args.stride, top_k)
    print(f"{pair.source} -> {pair.target}   top_k={top_k}  n_calib={n_calib}")
    print(f"estimated calibration capture: {gb:.1f} GB RAM (plus model weights)")
    print(f"design: {n_obs} rows x {p} features = {ratio:.1f} rows/parameter")
    if ratio < 4 and not args.yes:
        raise SystemExit(
            f"only {ratio:.1f} rows per parameter -- the ridge will interpolate and "
            f"report a near-perfect R^2 that means nothing. Raise --n-calib to "
            f">= {int(4 * p / (args.seq_len // args.stride)) + 1}, lower --top-k, "
            f"or pass --yes if this is a plumbing test.")
    if gb > 14 and not args.yes:
        raise SystemExit(
            f"that is {gb:.1f} GB of activations; re-run with --yes if you have the "
            f"headroom, or lower --n-calib (halving it halves this number)")

    texts = local_texts(args.texts, n_calib) if args.texts else None
    mapper = fit_mapper(pair.source, pair.target, top_k=top_k, lam=args.lam,
                        calibration_texts=texts, n_calib=n_calib,
                        seq_len=args.seq_len, stride=args.stride,
                        device=args.device, model_dtype=torch.float16)
    mapper.meta["fit_host"] = "local"
    mapper.save(out)
    print(f"\nsaved {mapper}\n  -> {out}")
    print(f"  mean R^2  K={mapper.meta['mean_r2_k']:.3f}  V={mapper.meta['mean_r2_v']:.3f}")
    return out


if __name__ == "__main__":
    main()
