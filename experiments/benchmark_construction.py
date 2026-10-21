"""Benchmark formal mapper construction with the paper's timing scope."""
from __future__ import annotations

import argparse
import csv
import io
import json
import resource
import time
from pathlib import Path

import torch

from kvbridge.cachebridge import weighted_ridge_centered
from kvbridge.fused_fit import fit_weighted_panels
from kvbridge.formal import fit_one_target_layer, fit_one_target_layer_full_head

SCOPE = "weight_staging+sufficient_statistics+solve+metadata+serialization"
PROVENANCE = "engineering_equivalent"


def _synthetic(budget, backend):
    torch.manual_seed(0)
    x, y = torch.randn(budget, 64), torch.randn(budget, 16)
    weights = torch.rand(budget)
    if backend in ("generic", "reference"):
        return weighted_ridge_centered(x, y, weights)[0]
    return fit_weighted_panels(x, y, weights, backend=backend)


def _serialize_payload(layers, budget, method, backend):
    buffer = io.BytesIO()
    torch.save({"layers": layers, "n_calib": int(budget),
                "method": method, "backend": backend,
                "implementation_provenance": PROVENANCE}, buffer)
    return buffer.getvalue()


def run_formal(trace_dir, selection_path, budget, method, backend,
               *, ridge_lambda=0.01, chunk_size=1024):
    """Fit every target layer from the first ``budget`` staged sequences."""
    root = Path(trace_dir)
    source_dir, target_dir = root / "source", root / "target"
    source_files = sorted(source_dir.glob("source_*.pt"))
    target_files = sorted(target_dir.glob("target_*.pt"))
    if len(source_files) < budget or len(target_files) < budget:
        raise ValueError(f"trace-dir has fewer than budget={budget} paired shards")
    selection = json.loads(Path(selection_path).read_text())
    if not isinstance(selection, list) or not selection:
        raise ValueError("layer-selection must be a non-empty list")
    first = torch.load(target_files[0], map_location="cpu", weights_only=False)
    n_heads = int(first["keys"][0].shape[1])
    head_dim = int(first["keys"][0].shape[2])
    assignment = list(range(n_heads))
    layers = []
    for layer_index, row in enumerate(selection):
        selected = [int(x) for x in row["selected"]]
        debug = {}
        if method in ("cachebridge", "head_local"):
            entry = fit_one_target_layer(
                source_dir, target_dir, layer_index, selected, n_heads,
                assignment, head_dim, ridge_lambda=ridge_lambda,
                stats_dtype=torch.float64, uniform_weights=method == "head_local",
                backend=backend, chunk_size=chunk_size, max_sequences=budget,
                debug_stats=debug)
        elif method == "full_head":
            entry = fit_one_target_layer_full_head(
                source_dir, target_dir, layer_index, selected, n_heads,
                head_dim, ridge_lambda=ridge_lambda, stats_dtype=torch.float64,
                backend=backend, chunk_size=chunk_size, max_sequences=budget,
                debug_stats=debug)
        else:
            raise ValueError(f"unsupported method: {method}")
        entry["debug_stats"] = debug
        layers.append(entry)
    return layers


def _run_formal_once(args, budget):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    layers = run_formal(args.trace_dir, args.layer_selection, budget,
                        args.method, args.backend,
                        ridge_lambda=args.ridge_lambda,
                        chunk_size=args.chunk_size)
    payload = _serialize_payload(layers, budget, args.method, args.backend)
    return {
        "budget": budget,
        "method": args.method,
        "backend": args.backend,
        "construction_seconds": time.perf_counter() - start,
        "serialized_bytes": len(payload),
        "peak_gpu_memory": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else 0,
        "peak_cpu_memory": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "implementation_provenance": PROVENANCE,
        "scope": SCOPE,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["synthetic", "formal"], default="synthetic")
    p.add_argument("--trace-dir")
    p.add_argument("--layer-selection")
    p.add_argument("--method", choices=["cachebridge", "head_local", "full_head"],
                   default="cachebridge")
    p.add_argument("--budgets", default="50,100,200,500")
    p.add_argument("--backend", choices=["generic", "reference", "torch_bounded",
                                         "fused_equivalent"], default="torch_bounded")
    p.add_argument("--ridge-lambda", type=float, default=0.01)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--output", default="results/construction_benchmark.csv")
    args = p.parse_args(argv)
    budgets = [int(x) for x in args.budgets.split(",")]
    if any(x <= 0 for x in budgets):
        raise ValueError("budgets must be positive")
    rows = []
    for budget in budgets:
        if args.mode == "synthetic":
            start = time.perf_counter()
            W = _synthetic(budget, args.backend)
            payload = _serialize_payload({"W": W}, budget, args.method, args.backend)
            rows.append({
                "budget": budget, "method": args.method, "backend": args.backend,
                "construction_seconds": time.perf_counter() - start,
                "serialized_bytes": len(payload), "peak_gpu_memory": 0,
                "peak_cpu_memory": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "implementation_provenance": PROVENANCE, "scope": "synthetic",
            })
        else:
            if not args.trace_dir or not args.layer_selection:
                raise ValueError("formal mode requires --trace-dir and --layer-selection")
            rows.append(_run_formal_once(args, budget))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
