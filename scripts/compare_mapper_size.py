#!/usr/bin/env python3
"""Compare serialized mapper payloads and theoretical support widths."""

import argparse
import json
from pathlib import Path
from safetensors.torch import load_file


def size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def coefficient_count(path):
    tensors = load_file(str(Path(path) / "mapper.safetensors"))
    return sum(int(t.numel()) for t in tensors.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-head", required=True)
    parser.add_argument("--cachebridge", required=True)
    parser.add_argument("--source-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--num-selected-layers", type=int)
    args = parser.parse_args()
    full_meta = json.loads((Path(args.full_head) / "mapper.json").read_text())
    cb_meta = json.loads((Path(args.cachebridge) / "mapper.json").read_text())
    source_heads = args.source_kv_heads or int(full_meta["num_source_kv_heads"])
    head_dim = args.head_dim or int(full_meta["head_dim"])
    selected_layers = args.num_selected_layers or len(full_meta["source_layers"][0])
    full_bytes = size(args.full_head)
    local_bytes = size(args.cachebridge)
    full_dim = selected_layers * source_heads * head_dim
    local_dim = selected_layers * head_dim
    full_coeff = coefficient_count(args.full_head)
    cb_coeff = coefficient_count(args.cachebridge)
    result = {
        "full_head_bytes": full_bytes,
        "cachebridge_bytes": local_bytes,
        "full_head_coefficient_count": full_coeff,
        "cachebridge_coefficient_count": cb_coeff,
        "full_head_GB": full_bytes / 1e9,
        "cachebridge_GB": local_bytes / 1e9,
        "actual_size_ratio": full_bytes / local_bytes if local_bytes else None,
        "full_head_input_dim": full_dim,
        "head_local_input_dim": local_dim,
        "theoretical_coefficient_ratio": full_dim / local_dim,
        "coefficient_ratio": full_coeff / cb_coeff if cb_coeff else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
