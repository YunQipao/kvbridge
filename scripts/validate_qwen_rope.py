#!/usr/bin/env python3
"""Validate RoPE strip/reapply against keys from real Qwen3 caches."""

import argparse
import gc
import sys

import torch

from kvbridge.cache_compat import cache_layer_kv
from kvbridge.rope import get_rope_theta, strip_rope, apply_rope
from kvbridge.model_utils import load_causal_lm, load_tokenizer


def validate(model_id, text, seq_len, device, local_files_only=False):
    tokenizer = load_tokenizer(model_id, local_files_only=local_files_only)
    model = load_causal_lm(model_id, device=device, dtype="float32",
                          local_files_only=local_files_only)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
    with torch.no_grad():
        cache = model(ids, use_cache=True).past_key_values
    key, _ = cache_layer_kv(cache, 0)
    positions = torch.arange(key.shape[2], device=key.device)
    original = key[0].float()
    restored = apply_rope(strip_rope(original, positions, get_rope_theta(model.config)),
                          positions, get_rope_theta(model.config))
    delta = (restored - original).abs()
    rel = (restored - original).norm() / original.norm().clamp_min(1e-12)
    print(f"{model_id} layer=0 max_abs_error={delta.max().item():.8g} "
          f"mean_abs_error={delta.mean().item():.8g} relative_l2_error={rel.item():.8g}")
    del cache, model, tokenizer
    gc.collect()
    return float(rel)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seq_len > 64:
        raise ValueError("P2 smoke sequence length must be <= 64")
    text = "CacheBridge validates rotary position handling in Qwen models."
    for model_id in (args.source, args.target):
        if validate(model_id, text, args.seq_len, args.device, args.local_files_only) > 1e-4:
            raise RuntimeError(f"RoPE round-trip error too large for {model_id}")


if __name__ == "__main__":
    sys.exit(main())
