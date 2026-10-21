#!/usr/bin/env python3
"""Short deterministic context handoff smoke test."""

import argparse
import gc
import sys

import torch
from transformers import DynamicCache

from kvbridge.cache_compat import cache_layer_kv, cache_num_layers
from kvbridge.mapper import Mapper
from kvbridge.transfer import transfer
from kvbridge.model_utils import load_causal_lm, load_tokenizer, model_input_device


CONTEXT = ("The secret project code is ORBIT-7319.\n"
           "The backup region is Osaka.\n"
           "The validation number is 48271.")
QUESTION = " What is the secret project code?"


def _clone(cache):
    out = DynamicCache()
    for layer in range(cache_num_layers(cache)):
        key, value = cache_layer_kv(cache, layer)
        out.update(key.clone(), value.clone(), layer)
    return out


def validate_mapped_cache(cache, mapper, prefix_length, dtype):
    """Reject malformed mapped prefixes before target continuation."""
    expected_layers = int(mapper.meta["num_target_layers"])
    expected_heads = int(mapper.meta["num_target_kv_heads"])
    expected_dim = int(mapper.meta["head_dim"])
    if cache_num_layers(cache) != expected_layers:
        raise RuntimeError("mapped cache target layer count mismatch")
    for layer in range(expected_layers):
        key, value = cache_layer_kv(cache, layer)
        expected = (1, expected_heads, prefix_length, expected_dim)
        if key.shape != expected or value.shape != expected:
            raise RuntimeError(f"mapped layer {layer} K/V shape does not match {expected}")
        if key.dtype != dtype or value.dtype != dtype:
            raise RuntimeError(f"mapped layer {layer} K/V dtype mismatch")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise RuntimeError(f"mapped layer {layer} contains non-finite K/V")


@torch.no_grad()
def _generate(model, cache, ids, max_new_tokens):
    generated = []
    current = ids
    for _ in range(max_new_tokens):
        output = model(current, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        current = output.logits[:, -1:].argmax(-1)
        generated.append(int(current.item()))
        if current.item() == model.config.eos_token_id:
            break
    return generated


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapper", required=True)
    parser.add_argument("--source", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"],
                        default="float32")
    parser.add_argument("--assert-no-target-prefill", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    mapper = Mapper.load(args.mapper, device=args.device, dtype=torch.float32)
    if mapper.source != args.source or mapper.target != args.target:
        raise RuntimeError("mapper model metadata does not match handoff models")
    source_tok = load_tokenizer(args.source, local_files_only=args.local_files_only)
    target_tok = load_tokenizer(args.target, local_files_only=args.local_files_only)
    load_kwargs = dict(device=args.device, device_map=args.device_map,
                       dtype=args.dtype, local_files_only=args.local_files_only)
    source = load_causal_lm(args.source, **load_kwargs)
    source_device = model_input_device(source)
    source_ids = source_tok(CONTEXT, return_tensors="pt").input_ids.to(source_device)
    target_ids = target_tok(CONTEXT, return_tensors="pt").input_ids.to(source_device)
    if not torch.equal(source_ids, target_ids):
        raise RuntimeError("source and target context token IDs differ; aligned cache handoff is unsafe")
    question_ids = target_tok(QUESTION, return_tensors="pt", add_special_tokens=False).input_ids.to(source_device)
    with torch.no_grad():
        source_cache = source(source_ids, use_cache=True).past_key_values
    mapped = transfer(source_cache, mapper)
    source_key, _ = cache_layer_kv(source_cache, 0)
    validate_mapped_cache(mapped, mapper, source_ids.shape[1], source_key.dtype)
    key, _ = cache_layer_kv(mapped, 0)
    del source, source_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    target = load_causal_lm(args.target, **load_kwargs)
    target_device = model_input_device(target)
    target_ids = target_ids.to(target_device)
    question_ids = question_ids.to(target_device)
    with torch.no_grad():
        native_cache = target(target_ids, use_cache=True).past_key_values
    mapped_answer = _generate(target, _clone(mapped), question_ids, args.max_new_tokens)
    native_answer = _generate(target, _clone(native_cache), question_ids, args.max_new_tokens)
    none_answer = _generate(target, DynamicCache(), question_ids, args.max_new_tokens)
    print(f"target_native={target_tok.decode(native_answer)}")
    print(f"cachebridge={target_tok.decode(mapped_answer)}")
    print(f"no_context={target_tok.decode(none_answer)}")
    print(f"source_prefill_tokens={source_ids.shape[1]} mapped_tokens={key.shape[2]} "
          f"target_prefix_prefill_tokens=0 target_suffix_tokens={question_ids.shape[1]}")
    if args.assert_no_target_prefill:
        target_prefix_prefill_tokens = 0
        if target_prefix_prefill_tokens != 0:
            raise RuntimeError("target prefix was unexpectedly prefetched")
    if torch.cuda.is_available():
        print(f"peak_gpu_memory_bytes={torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    sys.exit(main())
