#!/usr/bin/env python3
"""Formal restartable CacheBridge calibration driver.

The torch_bounded backend uses staged trace shards and streaming sufficient
statistics. It is algorithmically equivalent to the published bounded-panel
construction; the authors' private fused CUDA kernel is not claimed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from kvbridge.cachebridge import cachebridge_metadata, make_log_causal_boundaries
from kvbridge.formal import (capture_stage, check_tokenizer_alignment,
                             fit_one_target_layer, fit_one_target_layer_full_head,
                             load_calibration_texts, load_completed_trace_manifest,
                             select_source_layers,
                             tokenize_records, write_manifest)
from kvbridge.geometry import check_kv_compatibility, build_head_map
from kvbridge.mapper import Mapper
from kvbridge.model_utils import load_config, load_tokenizer, print_runtime_diagnostics
from kvbridge.rope import get_rope_theta, rope_spec
from kvbridge.protocol import load_json, reuse_manifest_prefix
from kvbridge.distkv_compat import (pack_jsonl, resolve_attn_repair,
                                    select_layers_ridge_subsample, validate_packed_records)


def _marker(root, name):
    return Path(root) / f"{name}.done"


def _prepare_rows(args):
    if not hasattr(args, "calibration_pack_mode"):
        args.calibration_pack_mode = "records"
    source_tok = load_tokenizer(args.source, revision=args.source_revision,
                                local_files_only=args.local_files_only)
    if args.calibration_pack_mode == "distkv_concat":
        target_tok = load_tokenizer(args.target_tokenizer or args.target,
            revision=args.target_revision, local_files_only=args.local_files_only)
        if args.reuse_calibration_manifest:
            prior = load_json(args.reuse_calibration_manifest)
            if prior.get("calibration_pack_mode") != "distkv_concat":
                raise ValueError("packed calibration requires a packed parent manifest")
            records = reuse_manifest_prefix(prior, args.n_calib)
            if any(len(r["token_ids"]) != args.seq_len for r in records):
                raise ValueError("packed manifest sequence length mismatch")
            if prior.get("packing_tokenizers") != [args.source, args.target_tokenizer or args.target]:
                raise ValueError("packed manifest tokenizer references differ; repack to verify alignment")
            if prior.get("token_alignment_verified") is not True:
                raise ValueError("packed manifest lacks verified tokenizer alignment")
            validate_packed_records(records, source_tok, target_tok)
            args._parent_manifest = {
                "path": args.reuse_calibration_manifest,
                "hash": hashlib.sha256(Path(args.reuse_calibration_manifest).read_bytes()).hexdigest(),
                "record_count": len(prior["records"]), "prefix_length": args.n_calib}
            rows = [torch.tensor(r["token_ids"], dtype=torch.long) for r in records]
        else:
            if not args.dataset_file:
                raise ValueError("distkv_concat requires a local --dataset-file JSONL")
            records, rows = pack_jsonl(args.dataset_file, target_tok, source_tok,
                                      args.n_calib, args.seq_len)
        return records, rows, [row.clone() for row in rows]
    if args.reuse_calibration_manifest:
        manifest_path = Path(args.reuse_calibration_manifest)
        prior = load_json(manifest_path)
        parent_count = len(prior.get("records", []))
        records = reuse_manifest_prefix(prior, args.n_calib)
        if prior.get("calibration_pack_mode", "records") != "records":
            raise ValueError("records mode cannot reuse a packed calibration manifest")
        if any("text" not in r for r in records):
            raise ValueError("paired calibration manifest lacks reusable records")
        args._parent_manifest = {
            "path": str(manifest_path),
            "hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "record_count": parent_count,
            "prefix_length": args.n_calib,
        }
    else:
        records = load_calibration_texts(dataset=args.dataset,
            dataset_file=args.dataset_file, n_calib=args.n_calib, seed=args.seed,
            tokenizer=source_tok, min_tokens=args.seq_len)
    source_rows = tokenize_records(records, source_tok, args.seq_len)
    target_tok = load_tokenizer(args.target_tokenizer or args.target,
                                revision=args.target_revision,
                                local_files_only=args.local_files_only)
    target_rows = tokenize_records(records, target_tok, args.seq_len)
    check_tokenizer_alignment(source_rows, target_rows)
    return records, source_rows, target_rows


def fit(args):
    for name, default in (("calibration_pack_mode", "records"),
                          ("layer_selection_mode", "paper_ols"),
                          ("selection_subsample", 4096), ("attn_repair_mode", "auto"),
                          ("selection_seed", None), ("stats_dtype", "float64")):
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.selection_seed is None:
        args.selection_seed = args.seed
    stats_dtype = getattr(torch, args.stats_dtype)
    if args.stats_dtype == "float32" and args.fit_backend == "fused_equivalent":
        raise ValueError("float32 statistics require reference/torch/torch_bounded; fused_equivalent accumulates float64")
    paper_weights = resolve_attn_repair(args.method, args.attn_repair_mode)
    if args.attn_repair_mode == "paper" and (args.num_boundaries != 32 or args.boundary_start != 12):
        raise ValueError("explicit paper repair requires 32 boundaries and boundary-start=12")
    if args.smoke_uniform_weights and args.attn_repair_mode == "paper":
        raise ValueError("paper repair conflicts with smoke uniform override")
    if args.paper_profile:
        defaults = dict(n_calib=500, seq_len=1024, token_stride=4,
                        ridge_lambda=0.01, num_boundaries=32, boundary_start=12)
        if args.paper_profile == "qwen3":
            defaults.update(source="Qwen/Qwen3-14B", target="Qwen/Qwen3-32B", top_k=8)
        elif args.paper_profile == "ministral_3b_to_14b":
            defaults["top_k"] = 20
        elif args.paper_profile == "ministral_8b_to_14b":
            defaults["top_k"] = 12
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if args.fit_backend == "fused":
        raise NotImplementedError(
            "authors' original fused CUDA kernel is not public"
        )
    if args.method == "cachebridge" and args.smoke_uniform_weights and not args.smoke_source_layers:
        raise ValueError("--smoke-uniform-weights requires explicit --smoke-source-layers")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config_path = output / "calibration_config.json"
    previous = {}
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("stats_dtype", "float64") != args.stats_dtype:
            raise ValueError("resume stats_dtype changed; use a new output directory")
        if (args.layer_selection_mode == "distkv_ridge_subsample" and
                (previous.get("selection_seed") if previous.get("selection_seed") is not None
                 else previous.get("seed")) != args.selection_seed):
            raise ValueError("resume selection_seed changed; use a new output directory")
        calibration_keys = ("source", "target", "source_revision", "target_revision",
                            "dataset", "dataset_file", "n_calib", "seq_len", "seed",
                            "token_stride", "num_boundaries", "boundary_start", "dtype",
                            "head_map_mode", "target_capture_mode")
        changed = [key for key in calibration_keys
                   if previous.get(key) != getattr(args, key)]
        if previous.get("calibration_pack_mode", "records") != args.calibration_pack_mode:
            changed.append("calibration_pack_mode")
        if changed:
            raise ValueError(f"resume calibration configuration changed: {changed}")
    trace_root = Path(args.trace_dir or output / "traces")
    source_dir, target_dir = trace_root / "source", trace_root / "target"
    print_runtime_diagnostics()
    source_config = load_config(args.source, revision=args.source_revision,
                                local_files_only=args.local_files_only)
    target_config = load_config(args.target, revision=args.target_revision,
                                local_files_only=args.local_files_only)
    compatible, source_geo, target_geo = check_kv_compatibility(
        source_config, target_config, args.head_map_mode)
    if not compatible:
        raise RuntimeError("source and target KV geometry are incompatible")
    if args.top_k > source_geo["num_hidden_layers"]:
        raise ValueError("top_k exceeds source layer count")
    completed_traces = (args.resume
                        and _marker(trace_root, "source_traces").is_file()
                        and _marker(trace_root, "target_traces").is_file())
    if completed_traces:
        if not config_path.is_file():
            raise ValueError("cannot safely resume completed traces without calibration_config.json")
        manifest = load_completed_trace_manifest(
            output, trace_root, dataset=args.dataset or args.dataset_file,
            seed=args.seed, n_calib=args.n_calib, seq_len=args.seq_len,
            token_stride=args.token_stride, num_boundaries=args.num_boundaries,
            boundary_start=args.boundary_start)
        n_obs = manifest["actual_observations"]
        print("resuming completed source/target traces; dataset and tokenizers "
              "will NOT be reloaded or resampled", flush=True)
    else:
        records, source_rows, target_rows = _prepare_rows(args)
        n_obs = sum((int(row.numel()) + args.token_stride - 1) // args.token_stride
                    for row in source_rows)
        manifest = write_manifest(output, dataset=args.dataset or args.dataset_file,
            split="train", seed=args.seed, records=records, rows=source_rows,
            seq_len=args.seq_len, token_stride=args.token_stride,
            num_boundaries=args.num_boundaries, boundary_start=args.boundary_start,
            n_observations=n_obs, token_alignment_verified=True,
            parent_manifest=getattr(args, "_parent_manifest", None))
        manifest["calibration_pack_mode"] = args.calibration_pack_mode
        if args.calibration_pack_mode == "distkv_concat":
            manifest["records"] = records
            manifest["packing_tokenizers"] = [args.source, args.target_tokenizer or args.target]
        (output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))
    boundaries = make_log_causal_boundaries(args.seq_len, args.num_boundaries,
                                             args.boundary_start)
    if not boundaries:
        raise ValueError("no causal boundaries available")
    (output / "calibration_config.json").write_text(json.dumps(vars(args), indent=2))
    common = dict(device=args.device, device_map=args.device_map, dtype=args.dtype,
                  local_files_only=args.local_files_only, stride=args.token_stride,
                  target_capture_mode=args.target_capture_mode)
    t0 = time.time()
    if not args.resume or not _marker(trace_root, "source_traces").exists():
        capture_stage(args.source, source_rows, source_dir, source=True,
                      revision=args.source_revision, **common)
        _marker(trace_root, "source_traces").parent.mkdir(parents=True, exist_ok=True)
        _marker(trace_root, "source_traces").touch()
    if not args.resume or not _marker(trace_root, "target_traces").exists():
        capture_stage(args.target, target_rows, target_dir, source=False,
                      boundaries=boundaries, revision=args.target_revision, **common)
        _marker(trace_root, "target_traces").touch()

    head_map = build_head_map(source_geo["num_key_value_heads"],
                              target_geo["num_key_value_heads"], args.head_map_mode)
    if args.smoke_source_layers:
        k = args.smoke_source_layers
        selected = [[round(i * (source_geo["num_hidden_layers"] - 1) / max(k - 1, 1))
                     for i in range(k)] for _ in range(target_geo["num_hidden_layers"])]
        rankings = [{"target_layer": i, "ranked": row, "scores": [], "selected": row}
                    for i, row in enumerate(selected)]
    elif args.reuse_layer_selection:
        rankings = json.loads(Path(args.reuse_layer_selection).read_text())
        selected = [row["selected"] for row in rankings]
    elif args.resume and config_path.exists() and (output / "layer_selection.json").exists():
        if (previous.get("layer_selection_mode", "paper_ols") != args.layer_selection_mode
                or (args.layer_selection_mode == "distkv_ridge_subsample" and
                    (previous.get("selection_subsample", 4096) != args.selection_subsample
                     or previous.get("ridge_lambda") != args.ridge_lambda))):
            raise ValueError("resume layer-selection recipe changed")
        rankings = json.loads((output / "layer_selection.json").read_text())
        if (len(rankings) != target_geo["num_hidden_layers"]
                or any(row["target_layer"] != i or len(row["selected"]) != args.top_k
                       for i, row in enumerate(rankings))):
            raise ValueError("resume layer-selection checkpoint does not match geometry/top_k")
        selected = [row["selected"] for row in rankings]
        print("resuming completed source-layer selection", flush=True)
    elif args.layer_selection_mode == "distkv_ridge_subsample":
        rankings = select_layers_ridge_subsample(source_dir, target_dir,
            source_geo["num_hidden_layers"], target_geo["num_hidden_layers"],
            target_geo["num_key_value_heads"], args.top_k, head_map,
            args.ridge_lambda, args.selection_subsample, args.selection_seed,
            stats_dtype=stats_dtype)
        selected = [row["selected"] for row in rankings]
    else:
        rankings = select_source_layers(source_dir, target_dir,
            source_geo["num_hidden_layers"], target_geo["num_hidden_layers"],
            target_geo["num_key_value_heads"], args.top_k,
            panel_size=args.selection_panel_size,
            head_map=head_map if source_geo["num_key_value_heads"] != target_geo["num_key_value_heads"] else None)
        selected = [row["selected"] for row in rankings]
    selection_path = output / "layer_selection.json"
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rankings, indent=2))
    temporary.replace(selection_path)

    layers = []
    if source_geo["num_key_value_heads"] == target_geo["num_key_value_heads"] and args.head_assignment_shift:
        assignment = [(i + args.head_assignment_shift) % source_geo["num_key_value_heads"]
                      for i in range(target_geo["num_key_value_heads"])]
        head_map = [[x] for x in assignment]
    else:
        assignment = [block[0] for block in head_map]
    for layer, chosen in enumerate(selected):
        if args.method in ("cachebridge", "head_local"):
            entry = fit_one_target_layer(source_dir, target_dir, layer, chosen,
                target_geo["num_key_value_heads"], assignment, source_geo["head_dim"],
                ridge_lambda=args.ridge_lambda, stats_device=args.stats_device,
                stats_dtype=stats_dtype,
                uniform_weights=(
                    not paper_weights
                    or (args.method == "cachebridge" and args.smoke_uniform_weights)
                ),
                backend=args.fit_backend, chunk_size=args.fit_chunk_size,
                head_map=head_map)
        else:
            entry = fit_one_target_layer_full_head(source_dir, target_dir, layer, chosen,
                target_geo["num_key_value_heads"], source_geo["head_dim"],
                ridge_lambda=args.ridge_lambda, stats_device=args.stats_device,
                stats_dtype=stats_dtype, backend=args.fit_backend,
                chunk_size=args.fit_chunk_size,
                source_panel_heads=source_geo["num_key_value_heads"],
                uniform_weights=not paper_weights)
        layers.append(entry)
        panel_heads = (source_geo["num_key_value_heads"] if args.method == "full_head"
                       else len(head_map[0]))
        print(f"target layer {layer}: selected={chosen} input_dim="
              f"{len(chosen) * panel_heads * source_geo['head_dim']} output_dim={source_geo['head_dim']}", flush=True)

    weighting = ("uniform_smoke_override" if args.smoke_uniform_weights else
                 "attn_repair" if paper_weights else "uniform")
    meta = cachebridge_metadata(source_model=args.source, target_model=args.target,
        num_source_layers=source_geo["num_hidden_layers"],
        num_target_layers=target_geo["num_hidden_layers"],
        num_source_kv_heads=source_geo["num_key_value_heads"],
        num_target_kv_heads=target_geo["num_key_value_heads"],
        head_dim=source_geo["head_dim"], head_assignment=assignment,
        head_map=head_map, head_map_mode=args.head_map_mode,
        selected_source_layers=selected, rope_theta_source=get_rope_theta(source_config),
        rope_theta_target=get_rope_theta(target_config), ridge_lambda=args.ridge_lambda,
        weighting_method=weighting, token_stride=args.token_stride,
        num_boundaries=args.num_boundaries, boundary_start_token=args.boundary_start)
    meta.update({"top_k": args.smoke_source_layers or args.top_k,
        "rope_config_source": rope_spec(source_config),
        "rope_config_target": rope_spec(target_config),
        "source_geometry": source_geo, "target_geometry": target_geo,
        "head_feature_width": args.top_k * len(head_map[0]) * source_geo["head_dim"],
        "full_head_feature_width": args.top_k * source_geo["num_key_value_heads"] * source_geo["head_dim"],
        "fit_backend": "torch_bounded" if args.fit_backend in ("torch", "torch_bounded") else args.fit_backend,
        "stats_dtype": args.stats_dtype, "coefficient_dtype": "float32",
        "calibration_dataset": args.dataset or args.dataset_file,
        "n_calibration": args.n_calib, "calibration_length": args.seq_len,
        "actual_observations": n_obs, "source_revision": args.source_revision,
        "target_revision": args.target_revision, "boundary_values": boundaries,
        "device_map": args.device_map})
    meta["target_capture_mode"] = args.target_capture_mode
    meta.update(calibration_pack_mode=args.calibration_pack_mode,
                layer_selection_mode=args.layer_selection_mode,
                selection_subsample=args.selection_subsample,
                attn_repair_mode=args.attn_repair_mode, weighting=weighting,
                selection_seed=args.selection_seed)
    if args.reuse_layer_selection:
        meta["reused_layer_selection"] = args.reuse_layer_selection
        parent_config = Path(args.reuse_layer_selection).parent / "calibration_config.json"
        if parent_config.exists():
            meta["layer_selection_mode"] = load_json(parent_config).get("layer_selection_mode", "paper_ols")
    if args.method == "full_head":
        meta["method"] = "full_head"
    elif args.method == "head_local":
        meta["method"] = "head_local"
    mapper = Mapper(layers, meta)
    mapper.save(output, dtype=torch.float32)
    (output / "metadata.json").write_text(json.dumps(mapper.meta, indent=2))
    summary = {"elapsed_seconds": time.time() - t0, "actual_observations": n_obs,
               "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
               "fit_backend": meta["fit_backend"], "manifest": manifest}
    (output / "fit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"saved {output}\npeak_gpu_memory_bytes={summary['peak_gpu_memory_bytes']}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source"); p.add_argument("--target")
    p.add_argument("--target-tokenizer",
                   help="explicit tokenizer reference when target model snapshot omits tokenizer files")
    p.add_argument("--dataset"); p.add_argument("--dataset-file"); p.add_argument("--n-calib", type=int)
    p.add_argument("--seq-len", type=int); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--method", choices=["cachebridge", "head_local", "full_head"], required=True)
    p.add_argument("--ridge-lambda", type=float); p.add_argument("--token-stride", type=int)
    p.add_argument("--top-k", type=int); p.add_argument("--num-boundaries", type=int)
    p.add_argument("--selection-panel-size", type=int, default=4,
                   help="target layers per bounded OLS selection panel")
    p.add_argument("--boundary-start", type=int); p.add_argument("--device", default="cpu")
    p.add_argument("--device-map"); p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--stats-device", default="cpu"); p.add_argument(
        "--fit-backend",
        choices=["reference", "torch", "torch_bounded", "fused_equivalent", "fused"],
        default="torch_bounded")
    p.add_argument("--fit-chunk-size", type=int, default=1024)
    p.add_argument("--trace-dir"); p.add_argument("--output", required=True); p.add_argument("--resume", action="store_true")
    p.add_argument("--local-files-only", action="store_true"); p.add_argument("--source-revision"); p.add_argument("--target-revision"); p.add_argument("--log-dir")
    p.add_argument("--smoke-source-layers", type=int); p.add_argument("--smoke-uniform-weights", action="store_true")
    p.add_argument("--paper-profile", choices=["qwen3", "ministral_3b_to_14b",
                                                "ministral_8b_to_14b"])
    p.add_argument("--reuse-calibration-manifest")
    p.add_argument("--reuse-layer-selection")
    p.add_argument("--calibration-pack-mode", choices=["records", "distkv_concat"], default="records")
    p.add_argument("--layer-selection-mode", choices=["paper_ols", "distkv_ridge_subsample"], default="paper_ols")
    p.add_argument("--selection-subsample", type=int, default=4096)
    p.add_argument("--selection-seed", type=int, default=None, help="defaults to --seed")
    p.add_argument("--stats-dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--attn-repair-mode", choices=["auto", "paper", "off"], default="auto")
    p.add_argument("--head-assignment-shift", type=int, default=0)
    p.add_argument("--head-map-mode", choices=["auto", "matched", "block"],
                   default=os.environ.get("HEAD_MAP_MODE", "auto"))
    p.add_argument("--target-capture-mode", choices=["full", "boundary_chunked"],
                   default=os.environ.get("TARGET_CAPTURE_MODE", "full"))
    args = p.parse_args(argv)
    if args.paper_profile:
        defaults = dict(n_calib=500, seq_len=1024, token_stride=4,
                        ridge_lambda=0.01, num_boundaries=32, boundary_start=12)
        if args.paper_profile == "qwen3":
            defaults.update(source="Qwen/Qwen3-14B", target="Qwen/Qwen3-32B", top_k=8)
        elif args.paper_profile == "ministral_3b_to_14b":
            defaults["top_k"] = 20
        else:
            defaults["top_k"] = 12
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if not args.source or not args.target:
        raise ValueError("--source and --target are required; Ministral profile IDs are CLI slots")
    if args.n_calib is None or args.seq_len is None or args.token_stride is None:
        raise ValueError("calibration settings are required")
    if bool(args.dataset) == bool(args.dataset_file) and not args.reuse_calibration_manifest:
        raise ValueError("provide exactly one of --dataset and --dataset-file")
    fit(args)


if __name__ == "__main__":
    main()
