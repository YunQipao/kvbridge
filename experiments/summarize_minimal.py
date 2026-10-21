#!/usr/bin/env python3
"""Summarize the measured minimal CacheBridge validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_SOURCE = "Qwen/Qwen3-14B"
PAPER_TARGET = "Qwen/Qwen3-32B"
METHODS = ("target_native", "full_head", "head_local", "cachebridge")


def _read(path):
    return json.loads(Path(path).read_text())


def _mapper_size(path):
    return (Path(path) / "mapper.safetensors").stat().st_size


def _size_record(path):
    value = _mapper_size(path)
    return {"bytes": value, "MiB": value / (1024 ** 2), "GiB": value / (1024 ** 3)}


def build_summary(source, target, results_dir, artifacts, benchmark_path):
    results_dir = Path(results_dir)
    scores = {method: float(_read(results_dir / f"hellaswag_{method}.json")["accuracy"])
              for method in METHODS}
    native = scores["target_native"]
    retention = {
        f"{method}_retention": scores[method] / native if native else None
        for method in ("full_head", "head_local", "cachebridge")
    }
    delta = {
        "head_local_delta_vs_full_head": scores["head_local"] - scores["full_head"],
        "cachebridge_delta_vs_full_head": scores["cachebridge"] - scores["full_head"],
        "cachebridge_delta_vs_head_local": scores["cachebridge"] - scores["head_local"],
    }
    sizes = {method: _size_record(path) for method, path in artifacts.items()}
    benchmark = _read(benchmark_path)
    latency = {str(row["prefix"]): {
        "full_head_ms": row["full_head_ms"]["median"],
        "cachebridge_ms": row["cachebridge_ms"]["median"],
    } for row in benchmark if row.get("prefix") in (128, 512, 1024)}
    missing_prefixes = {"128", "512", "1024"} - set(latency)
    if missing_prefixes:
        raise ValueError(f"transfer benchmark missing required prefixes: {sorted(missing_prefixes)}")
    size_ratio = (_mapper_size(artifacts["full_head"])
                  / _mapper_size(artifacts["cachebridge"]))
    summary = {
        "source_model": source,
        "target_model": target,
        "paper_pair": source == PAPER_SOURCE and target == PAPER_TARGET,
        "reduced_scale": not (source == PAPER_SOURCE and target == PAPER_TARGET),
        "fused_fit_paper_kernel_reproduced": False,
        "task": "HellaSwag",
        "target_native_accuracy": native,
        "full_head_accuracy": scores["full_head"],
        "head_local_accuracy": scores["head_local"],
        "cachebridge_accuracy": scores["cachebridge"],
        "accuracy": {
            "target_native_accuracy": native,
            "full_head_accuracy": scores["full_head"],
            "head_local_accuracy": scores["head_local"],
            "cachebridge_accuracy": scores["cachebridge"],
        },
        "full_head_retention": retention["full_head_retention"],
        "head_local_retention": retention["head_local_retention"],
        "cachebridge_retention": retention["cachebridge_retention"],
        "head_local_delta_vs_full_head": delta["head_local_delta_vs_full_head"],
        "cachebridge_delta_vs_full_head": delta["cachebridge_delta_vs_full_head"],
        "cachebridge_delta_vs_head_local": delta["cachebridge_delta_vs_head_local"],
        "retention": retention,
        "delta": delta,
        "full_head_size": sizes["full_head"],
        "head_local_size": sizes["head_local"],
        "cachebridge_size": sizes["cachebridge"],
        "mapper_size": sizes,
        "full_head_size_over_cachebridge_size": size_ratio,
        "mapping_latency_ms_median": latency,
        "latency_scope": "mapper application only; excludes source prefill, transport, and decode",
        "efficiency": {
            "full_head_size_over_cachebridge_size": size_ratio,
            "latency_measurement": "current implementation measurement",
            "latency_scope": "mapper application only; excludes source prefill, transport, and decode",
        },
    }
    summary["benchmark_transfer"] = benchmark
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--results-dir", default="results/minimal")
    parser.add_argument("--full-head", default="artifacts/minimal_full_head")
    parser.add_argument("--head-local", default="artifacts/minimal_head_local")
    parser.add_argument("--cachebridge", default="artifacts/minimal_cachebridge")
    parser.add_argument("--benchmark", default="results/minimal/transfer_benchmark.json")
    parser.add_argument("--output", default="results/minimal/summary.json")
    args = parser.parse_args(argv)
    summary = build_summary(
        args.source, args.target, args.results_dir,
        {"full_head": args.full_head, "head_local": args.head_local,
         "cachebridge": args.cachebridge},
        args.benchmark,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
