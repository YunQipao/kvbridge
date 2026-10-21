#!/usr/bin/env python3
"""Validate a CacheBridge mapper artifact, including optional model geometry."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from kvbridge.cachebridge import normalize_head_assignment
from kvbridge.geometry import check_kv_compatibility
from kvbridge.model_utils import load_config
from kvbridge.mapper import Mapper


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--paper-faithful", action="store_true")
    parser.add_argument("--paper-profile", choices=["qwen3"])
    args = parser.parse_args(argv)
    path = Path(args.mapper)
    mapper = Mapper.load(path, dtype=torch.float32)
    meta = mapper.meta
    if args.paper_profile == "qwen3":
        expected = {"calibration_length": 1024, "token_stride": 4,
                    "top_k": 8, "ridge_lambda": 0.01,
                    "num_boundaries": 32, "boundary_start_token": 12}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"qwen3 paper profile mismatch: {key}")
    required = ("method", "source", "target", "num_source_layers",
                "num_target_layers", "num_source_kv_heads", "num_target_kv_heads",
                "head_dim", "head_assignment", "selected_source_layers",
                "ridge_lambda", "weighting_method", "token_stride")
    missing = [key for key in required if key not in meta]
    if missing:
        raise RuntimeError(f"missing metadata: {missing}")
    if meta["method"] not in ("cachebridge", "head_local", "full_head",
                              "arXiv:2608.03893 closed-form per-layer ridge, content-space keys"):
        raise RuntimeError(f"unsupported mapper method: {meta['method']}")
    if meta["method"] not in ("cachebridge", "head_local"):
        if args.paper_faithful and meta["method"] == "full_head":
            if meta.get("weighting_method") != "uniform":
                raise RuntimeError("Full-Head paper validation requires uniform weighting")
            if args.paper_profile == "qwen3" and meta.get("method") != "full_head":
                raise RuntimeError("qwen3 Full-Head validation requires method=full_head")
        print("PASS")
        print(json.dumps({"method": meta["method"], "target_layers": len(mapper.layers)},
                         indent=2))
        return 0
    if args.paper_faithful:
        if (meta.get("debug_run") or meta.get("smoke_source_layers")
                or meta.get("layer_selection_method") == "fixed_debug_support"):
            raise RuntimeError("debug/smoke support is not paper-faithful")
        if meta.get("weighting_method") != "attn_repair":
            raise RuntimeError("paper-faithful validation requires weighting_method=attn_repair")
        if int(meta.get("n_calibration", 0) or 0) <= 0:
            raise RuntimeError("paper-faithful artifact has no calibration rows")
        if int(meta.get("calibration_length", 0) or 0) < 64:
            raise RuntimeError("paper-faithful artifact sequence length is too short")
        if meta.get("fit_backend") not in ("torch_bounded", "reference", "torch"):
            raise RuntimeError("unsupported formal fit backend")
        if "boundary_values" not in meta or not meta["boundary_values"]:
            raise RuntimeError("formal artifact is missing causal boundaries")
        if meta.get("weighting_method") == "uniform_smoke_override":
            raise RuntimeError("uniform smoke weighting is not paper-faithful")
        if args.paper_profile == "qwen3" and meta.get("method") != "cachebridge":
            raise RuntimeError("qwen3 CacheBridge validation requires method=cachebridge")
    if args.source and meta["source"] != args.source:
        raise RuntimeError("source model metadata mismatch")
    if args.target and meta["target"] != args.target:
        raise RuntimeError("target model metadata mismatch")
    blocks = meta.get("head_map") or normalize_head_assignment(
        meta["head_assignment"], meta["num_source_kv_heads"], meta["num_target_kv_heads"])
    if len(mapper.layers) != meta["num_target_layers"]:
        raise RuntimeError("target layer count mismatch")
    d = int(meta["head_dim"])
    h = int(meta["num_target_kv_heads"])
    for index, entry in enumerate(mapper.layers):
        selected = entry["source_layers"]
        if selected != meta["selected_source_layers"][index]:
            raise RuntimeError(f"layer {index}: selected source layer metadata mismatch")
        shape = (h, len(selected) * len(blocks[0]) * d, d)
        for key in ("W_k", "W_v"):
            if tuple(entry[key].shape) != shape:
                raise RuntimeError(f"layer {index}: {key} shape {tuple(entry[key].shape)} != {shape}")
        for key in ("W_k", "W_v", "mu_xk", "mu_yk", "mu_xv", "mu_yv"):
            if not torch.isfinite(entry[key]).all():
                raise RuntimeError(f"layer {index}: non-finite {key}")
        if args.paper_faithful and len(selected) != int(meta.get("top_k", len(selected))):
            raise RuntimeError(f"layer {index}: selected layer count is not top_k")
        if any(int(layer) < 0 or int(layer) >= int(meta["num_source_layers"])
               for layer in selected):
            raise RuntimeError(f"layer {index}: source layer out of range")
    if args.source and args.target:
        source_config = load_config(args.source, local_files_only=args.local_files_only)
        target_config = load_config(args.target, local_files_only=args.local_files_only)
        compatible, source_geo, target_geo = check_kv_compatibility(
            source_config, target_config)
        if not compatible:
            raise RuntimeError("artifact models are not KV-compatible")
        if source_geo["num_key_value_heads"] != int(meta["num_source_kv_heads"]):
            raise RuntimeError("source config KV-head count differs from artifact")
        if target_geo["num_key_value_heads"] != int(meta["num_target_kv_heads"]):
            raise RuntimeError("target config KV-head count differs from artifact")
    with tempfile.TemporaryDirectory() as temp:
        mapper.save(temp, dtype=torch.float32)
        reloaded = Mapper.load(temp, dtype=torch.float32)
        for left, right in zip(mapper.layers, reloaded.layers):
            for key in ("W_k", "W_v", "mu_xk", "mu_yk", "mu_xv", "mu_yv"):
                if not torch.equal(left[key], right[key]):
                    raise RuntimeError(f"save/load mismatch in {key}")
    print("PASS")
    print(json.dumps({"method": meta["method"], "source": meta["source"],
                      "target": meta["target"],
                      "target_layers": len(mapper.layers),
                      "target_kv_heads": h, "head_dim": d}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
