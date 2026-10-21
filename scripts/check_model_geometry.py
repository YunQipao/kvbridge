#!/usr/bin/env python3
"""Print and validate matched-KV model geometry without loading weights."""

import argparse
import json
import sys

from kvbridge.model_utils import load_config, ministral_load_kwargs

from kvbridge.geometry import check_kv_compatibility, geometry_dict, build_head_map


def _print_block(label, model, values):
    print(label)
    print(f"model: {model}")
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rope_theta",
                "rope_scaling", "rope_parameters", "dtype"):
        value = values[key]
        if not isinstance(value, (str, int, float, type(None))):
            value = json.dumps(value, sort_keys=True)
        print(f"{key}: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--head-map-mode", choices=["auto", "matched", "block"], default=None)
    args = parser.parse_args(argv)
    import os
    head_map_mode = args.head_map_mode or os.environ.get("HEAD_MAP_MODE", "auto")
    source_config = load_config(args.source, revision=args.source_revision,
                                local_files_only=args.local_files_only)
    _print_block("SOURCE", args.source, geometry_dict(source_config))
    try:
        target_config = load_config(args.target, revision=args.target_revision,
                                    local_files_only=args.local_files_only)
    except OSError as exc:
        print(f"TARGET config unavailable: {args.target}", file=sys.stderr)
        print("KV geometry compatible: UNKNOWN (target config unavailable)", file=sys.stderr)
        raise RuntimeError("cannot establish CacheBridge head correspondence") from exc
    compatible, source, target = check_kv_compatibility(source_config, target_config, head_map_mode)
    for model_id, config in ((args.source, source_config), (args.target, target_config)):
        if getattr(config, "model_type", None) == "mistral3":
            ministral_load_kwargs(model_id, config)
            print(f"verified local text-only checkpoint namespace: {model_id}")
    _print_block("TARGET", args.target, target)
    if compatible:
        head_map = build_head_map(source["num_key_value_heads"], target["num_key_value_heads"], head_map_mode)
        print("KV geometry compatible: YES")
        print(f"head_map_mode: {head_map_mode}")
        group_size = len(head_map[0])
        print(f"assignment_mode: {'grouped_contiguous' if group_size > 1 else 'identity'}")
        print(f"group_size: {group_size}")
        print(f"head_assignment: {json.dumps(head_map)}")
        print("head_map:")
        for head, block in enumerate(head_map):
            print(f"{head} <- {block}")
        return 0
    print("KV geometry compatible: NO")
    print("Requires identical head_dim; auto/block requires source KV heads >= target KV heads and divisibility; matched requires equal counts")
    return 2


if __name__ == "__main__":
    sys.exit(main())
