#!/usr/bin/env python3
"""Capture one short DynamicCache from each real model and print geometry."""

import argparse
import gc
import sys

import torch

from kvbridge.cache_compat import cache_layer_kv, cache_num_layers
from kvbridge.geometry import check_kv_compatibility
from kvbridge.model_utils import load_causal_lm, load_config, load_tokenizer, model_input_device


def _device(model, requested):
    if requested != "auto":
        return torch.device(requested)
    return next(model.parameters()).device


def capture(model_id, text, seq_len, device, device_map, local_files_only=False):
    tokenizer = load_tokenizer(model_id, local_files_only=local_files_only)
    model = load_causal_lm(model_id, device=device, device_map=device_map,
                          dtype="float32", local_files_only=local_files_only)
    run_device = model_input_device(model)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=seq_len).input_ids.to(run_device)
    with torch.no_grad():
        output = model(ids, use_cache=True)
    cache = output.past_key_values
    key, value = cache_layer_kv(cache, 0)
    expected = (int(model.config.num_hidden_layers),
                int(model.config.num_key_value_heads),
                int(getattr(model.config, "head_dim",
                            model.config.hidden_size // model.config.num_attention_heads)))
    actual = (cache_num_layers(cache), key.shape[1], key.shape[3])
    if actual != expected:
        raise RuntimeError(f"{model_id}: cache geometry {actual} != config {expected}")
    print(model_id)
    print(f"cache_layers={actual[0]} K_shape={tuple(key.shape)} V_shape={tuple(value.shape)}")
    print(f"dtype={key.dtype} device={key.device} sequence_length={key.shape[2]} "
          f"kv_heads={key.shape[1]} head_dim={key.shape[3]}")
    peak = (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    del output, cache, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return peak


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--prompt", default="CacheBridge transfers key value cache between compatible language models.")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seq_len > 64:
        raise ValueError("P2 smoke sequence length must be <= 64")
    compatible, _, _ = check_kv_compatibility(
        load_config(args.source, local_files_only=args.local_files_only),
        load_config(args.target, local_files_only=args.local_files_only))
    if not compatible:
        raise RuntimeError("model KV geometry is incompatible")
    peaks = []
    for model_id in (args.source, args.target):
        peaks.append(capture(model_id, args.prompt, args.seq_len, args.device,
                             args.device_map, args.local_files_only))
    print(f"peak_gpu_memory_bytes={max(peaks)}")


if __name__ == "__main__":
    sys.exit(main())
