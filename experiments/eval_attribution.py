"""Held-out K/V attribution and NLL@4K evaluator.

The paper's exact stream IDs are unavailable. This driver creates a
deterministic stream manifest and labels it ENGINEERING-EQUIVALENT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from kvbridge.cache_compat import cache_layer_kv, cache_num_layers
from kvbridge.mapper import Mapper
from kvbridge.model_utils import load_causal_lm, load_tokenizer, model_input_device
from kvbridge.protocol import assert_mapper_method
from kvbridge.transfer import transfer


def r2_metric(predicted, target):
    predicted, target = predicted.float(), target.float()
    residual = (target - predicted).square().sum()
    centered = (target - target.mean()).square().sum()
    if centered == 0:
        return 1.0 if residual == 0 else 0.0
    return float(1.0 - residual / centered)


def mean_nll(logits, targets):
    return float(F.cross_entropy(logits.float(), targets.long(), reduction="mean"))


def build_teacher_forcing_inputs(prefix, continuation):
    """Return bridge token, teacher-forced inputs, and prediction targets."""
    if prefix.ndim != 2 or continuation.ndim != 2:
        raise ValueError("prefix and continuation must have shape [batch, tokens]")
    if prefix.shape[1] < 1 or continuation.shape[1] < 1:
        raise ValueError("prefix and continuation must be non-empty")
    return prefix[:, -1:], continuation[:, :-1], continuation


def teacher_forcing_accounting(prefix_length, continuation_length):
    if prefix_length < 1 or continuation_length < 1:
        raise ValueError("prefix and continuation must be non-empty")
    return {
        "mapped_prefix_tokens": int(prefix_length - 1),
        "bridge_tokens": 1,
        "teacher_forced_tokens": int(continuation_length - 1),
        "target_prefix_prefill_tokens": 0,
    }


def cache_tensor(cache, kind="key"):
    chunks = []
    for layer in range(cache_num_layers(cache)):
        key, value = cache_layer_kv(cache, layer)
        chunks.append((key if kind == "key" else value).float().reshape(-1))
    return torch.cat(chunks)


@torch.no_grad()
def evaluate_kv_r2(source, target, mapper, records, device):
    """Compute K/V R2 on a supplied held-out evaluation manifest."""
    values = {"K_R2": [], "V_R2": []}
    for record in records:
        source_ids = torch.tensor(record["source_tokens"], device=device).unsqueeze(0)
        target_ids = torch.tensor(record["target_tokens"], device=device).unsqueeze(0)
        prefix = int(record["prefix_length"])
        source_cache = source(source_ids[:, :prefix], use_cache=True).past_key_values
        mapped = transfer(source_cache, mapper)
        native = target(target_ids[:, :prefix], use_cache=True).past_key_values
        values["K_R2"].append(r2_metric(cache_tensor(mapped, "key"),
                                        cache_tensor(native, "key")))
        values["V_R2"].append(r2_metric(cache_tensor(mapped, "value"),
                                        cache_tensor(native, "value")))
    return {name: sum(items) / len(items) for name, items in values.items()}


def _read_streams(path, source_tokenizer, target_tokenizer, count, length,
                  horizon=128):
    records = []
    for index, text in enumerate(Path(path).read_text().splitlines()):
        if not text.strip():
            continue
        source_ids = source_tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=length).input_ids[0]
        target_ids = target_tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=length).input_ids[0]
        if source_ids.numel() >= length and torch.equal(source_ids, target_ids):
            prefix = length - horizon
            records.append({"index": index, "source_tokens": source_ids[:length].tolist(),
                            "target_tokens": target_ids[:length].tolist(),
                            "prefix_length": prefix})
        elif source_ids.numel() >= length:
            raise RuntimeError(
                f"stream {index} has different source/target token IDs; "
                "attribution requires exact tokenizer alignment")
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"needed {count} streams with {length} tokens, got {len(records)}")
    return records


@torch.no_grad()
def evaluate_mapping(source, target, mapper, records, device):
    nll = []
    accounting = {
        "mapped_prefix_tokens": 0,
        "bridge_tokens": 0,
        "teacher_forced_tokens": 0,
        "target_prefix_prefill_tokens": 0,
    }
    for record in records:
        source_ids = torch.tensor(record["source_tokens"], device=device).unsqueeze(0)
        target_ids = torch.tensor(record["target_tokens"], device=device).unsqueeze(0)
        prefix = int(record["prefix_length"])
        source_prefix = source_ids[:, :prefix]
        target_prefix = target_ids[:, :prefix]
        continuation = target_ids[:, prefix:]
        bridge_token, teacher_forced, targets = build_teacher_forcing_inputs(
            target_prefix, continuation)
        for key, value in teacher_forcing_accounting(
                prefix, continuation.shape[1]).items():
            accounting[key] += value
        cache_prefix = source_prefix[:, :-1]
        source_cache = source(cache_prefix, use_cache=True).past_key_values
        mapped = transfer(source_cache, mapper)
        native = target(cache_prefix, use_cache=True).past_key_values
        scorer_input = torch.cat((bridge_token, teacher_forced), dim=1)
        mapped_logits = target(scorer_input, past_key_values=mapped,
                               use_cache=False).logits
        native_logits = target(scorer_input, past_key_values=native,
                               use_cache=False).logits
        nll.append({"mapped": mean_nll(mapped_logits.reshape(-1, mapped_logits.shape[-1]),
                                       targets.reshape(-1)),
                    "native": mean_nll(native_logits.reshape(-1, native_logits.shape[-1]),
                                       targets.reshape(-1))})
    return {
        "NLL": {
            "mapped": sum(x["mapped"] for x in nll) / len(nll),
            "native": sum(x["native"] for x in nll) / len(nll),
        },
        "token_accounting": accounting,
    }


def evaluate_nll_long(source, target, mapper, records, device):
    """Compute NLL using the bridge-token teacher-forcing alignment."""
    result = evaluate_mapping(source, target, mapper, records, device)
    return {"NLL": result["NLL"], "token_accounting": result["token_accounting"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True); p.add_argument("--target", required=True)
    p.add_argument("--head-local", required=True); p.add_argument("--cachebridge", required=True)
    p.add_argument("--stream-file", required=True); p.add_argument("--streams", type=int, default=16)
    p.add_argument("--kv-manifest",
                   help="paired HellaSwag evaluation manifest used for K/V R2")
    p.add_argument("--stream-length", type=int, default=4096); p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/attribution.json")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    target_tok = load_tokenizer(args.target, local_files_only=args.local_files_only)
    source_tok = load_tokenizer(args.source, local_files_only=args.local_files_only)
    records = _read_streams(args.stream_file, source_tok, target_tok, args.streams,
                            args.stream_length, horizon=128)
    nll_manifest = {"records": [r["index"] for r in records], "streams": args.streams,
                "stream_length": args.stream_length, "prefix_length": args.stream_length - 128,
                "teacher_forced_horizon": 128,
                "provenance": "ENGINEERING-EQUIVALENT"}
    kv_manifest = (json.loads(Path(args.kv_manifest).read_text())
                   if args.kv_manifest else None)
    kv_records = (kv_manifest.get("records", []) if kv_manifest else [])
    if kv_records and not isinstance(kv_records[0], dict):
        raise ValueError(
            "--kv-manifest must contain tokenized records for K/V R2; "
            "indices alone cannot reconstruct model prefixes")
    target = load_causal_lm(args.target, device=args.device, dtype="auto",
                            local_files_only=args.local_files_only)
    source = load_causal_lm(args.source, device=args.device, dtype="auto",
                            local_files_only=args.local_files_only)
    device = model_input_device(target)
    outputs = {}
    for name, path in (("head_local", args.head_local), ("cachebridge", args.cachebridge)):
        mapper = Mapper.load(path, device=device, dtype=torch.float32)
        assert_mapper_method(mapper, name)
        nll_result = evaluate_nll_long(source, target, mapper, records, device)
        kv_result = (evaluate_kv_r2(source, target, mapper, kv_records, device)
                     if kv_records else {"K_R2": None, "V_R2": None})
        outputs[name] = {**kv_result, **nll_result}
    result = {"kv_r2_manifest": kv_manifest, "nll_manifest": nll_manifest,
              "provenance": "ENGINEERING-EQUIVALENT", "results": outputs}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
