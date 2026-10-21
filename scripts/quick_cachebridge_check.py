#!/usr/bin/env python3
"""Offline real-model debug pipeline, reusing one load per model.

ENGINEERING ASSUMPTION: two local texts, 64 tokens, four boundaries and
one fixed source layer per target layer. This checks execution, NOT accuracy,
formal support selection, or paper-scale reproduction.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from kvbridge.cache_compat import cache_layer_kv
from kvbridge.cachebridge import cachebridge_metadata, make_log_causal_boundaries
from kvbridge.formal import (
    _capture_target_weights, capture_source_shard, check_tokenizer_alignment,
    fit_one_target_layer, load_calibration_texts, tokenize_records, write_manifest,
)
from kvbridge.geometry import check_kv_compatibility
from kvbridge.mapper import Mapper
from kvbridge.model_utils import (
    load_causal_lm, load_config, load_tokenizer, print_runtime_diagnostics, require_device,
)
from kvbridge.rope import get_rope_theta
from kvbridge.transfer import transfer


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/quick_debug_1.7b_to_4b")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    print("NON-PAPER-SCALE DEBUG RUN: no benchmark or semantic-accuracy claim", flush=True)
    print_runtime_diagnostics()
    require_device(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"debug output already exists: {out}; use a new --output")
    out.mkdir(parents=True, exist_ok=True)
    source_config = load_config(args.source, local_files_only=True)
    target_config = load_config(args.target, local_files_only=True)
    compatible, sg, tg = check_kv_compatibility(source_config, target_config)
    if not compatible:
        raise RuntimeError("models are not KV-compatible")
    source_tok = load_tokenizer(args.source, local_files_only=True)
    target_tok = load_tokenizer(args.target, local_files_only=True)
    dataset = Path(__file__).resolve().parents[1] / "tests/data/cachebridge_quick.txt"
    records = load_calibration_texts(dataset_file=dataset, n_calib=2,
                                    tokenizer=source_tok, min_tokens=64)
    rows = tokenize_records(records, source_tok, 64)
    target_rows = tokenize_records(records, target_tok, 64)
    check_tokenizer_alignment(rows, target_rows)
    boundaries = make_log_causal_boundaries(64, 4, 12)
    write_manifest(out, dataset=str(dataset), split="local_debug", seed=42,
                   records=records, rows=rows, seq_len=64, token_stride=4,
                   num_boundaries=4, boundary_start=12, n_observations=32,
                   token_alignment_verified=True)
    source_dir, target_dir = out / "traces/source", out / "traces/target"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    print("loading source once and capturing debug traces", flush=True)
    source = load_causal_lm(args.source, device=args.device, dtype="auto",
                            local_files_only=True)
    for i, row in enumerate(rows):
        shard = capture_source_shard(source, row.to(args.device)[None], 4)
        torch.save(shard, source_dir / f"source_{i:05d}.pt")
    context = "A laboratory stores its reference samples in cabinet seven."
    suffix = " Which cabinet contains the reference samples?"
    prefix_ids = source_tok(context, return_tensors="pt").input_ids.to(args.device)
    if not torch.equal(prefix_ids.cpu(), target_tok(context, return_tensors="pt").input_ids):
        raise RuntimeError("handoff token IDs are not aligned")
    source_cache = source(prefix_ids, use_cache=True).past_key_values
    del source
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print("loading target once and capturing real eager-attention sensitivities", flush=True)
    target = load_causal_lm(args.target, device=args.device, dtype="auto",
                            local_files_only=True, attn_implementation="eager")
    for i, row in enumerate(target_rows):
        shard = dict(zip(("keys", "values", "weights_k", "weights_v"),
                         _capture_target_weights(target, row.to(args.device)[None], 4,
                                                 boundaries)))
        shard["token_count"] = int(row.numel())
        torch.save(shard, target_dir / f"target_{i:05d}.pt")
    # Fixed debug support avoids the expensive real-model selection stage.
    selected = [[layer % sg["num_hidden_layers"]] for layer in range(tg["num_hidden_layers"])]
    heads, dim = tg["num_key_value_heads"], sg["head_dim"]
    layers = []
    for layer, support in enumerate(selected):
        layers.append(fit_one_target_layer(source_dir, target_dir, layer, support,
                                            heads, list(range(heads)), dim,
                                            stats_dtype=torch.float64))
        print(f"debug fit layer {layer + 1}/{len(selected)}", flush=True)
    meta = cachebridge_metadata(
        source_model=args.source, target_model=args.target,
        num_source_layers=sg["num_hidden_layers"], num_target_layers=tg["num_hidden_layers"],
        num_source_kv_heads=heads, num_target_kv_heads=heads, head_dim=dim,
        head_assignment=list(range(heads)), selected_source_layers=selected,
        rope_theta_source=get_rope_theta(source_config),
        rope_theta_target=get_rope_theta(target_config), num_boundaries=4)
    meta.update(debug_run=True, smoke_source_layers=1, top_k=1, n_calibration=2,
                calibration_length=64, actual_observations=32,
                fit_backend="reference", layer_selection_method="fixed_debug_support",
                calibration_dataset=str(dataset), boundary_values=boundaries)
    mapper = Mapper(layers, meta)
    mapper.save(out, dtype=torch.float32)
    reloaded = Mapper.load(out, device=args.device, dtype=torch.float32)
    for left, right in zip(mapper.layers, reloaded.layers):
        for key in ("W_k", "W_v", "mu_xk", "mu_xv", "mu_yk", "mu_yv"):
            if not torch.equal(left[key].to(device="cpu", dtype=torch.float32),
                               right[key].cpu()):
                raise RuntimeError(f"artifact round-trip mismatch: {key}")
    print("mapping held-out prefix and forwarding only target suffix", flush=True)
    mapped = transfer(source_cache, reloaded)
    prefix_n = prefix_ids.shape[1]
    for layer in range(tg["num_hidden_layers"]):
        for tensor in cache_layer_kv(mapped, layer):
            if tensor.shape != (1, heads, prefix_n, dim) or not torch.isfinite(tensor).all():
                raise RuntimeError(f"invalid mapped cache geometry/values at layer {layer}")
    suffix_ids = target_tok(suffix, add_special_tokens=False,
                            return_tensors="pt").input_ids.to(args.device)
    result = target(suffix_ids, past_key_values=mapped, use_cache=True)
    if not torch.isfinite(result.logits).all():
        raise RuntimeError("target continuation logits are not finite")
    if cache_layer_kv(result.past_key_values, 0)[0].shape[2] != prefix_n + suffix_ids.shape[1]:
        raise RuntimeError("target continuation cache length mismatch")
    summary = dict(status="DEBUG_PIPELINE_EXECUTED", debug_run=True,
                   paper_faithful_validation=False,
                   source_prefill_tokens=int(prefix_n), mapped_tokens=int(prefix_n),
                   target_prefix_prefill_tokens=0, target_suffix_tokens=int(suffix_ids.shape[1]),
                   elapsed_seconds=time.perf_counter() - started,
                   peak_gpu_memory_bytes=(torch.cuda.max_memory_allocated(args.device)
                                          if args.device.startswith("cuda") else 0))
    (out / "quick_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
