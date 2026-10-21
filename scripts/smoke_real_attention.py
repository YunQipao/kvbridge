#!/usr/bin/env python3
"""Capture one real Qwen3 attention tensor with the eager backend."""

import argparse
import sys

import torch

from kvbridge.cachebridge import compute_gqa_sensitivities
from kvbridge.rope import get_rope_theta
from kvbridge.model_utils import load_causal_lm, load_tokenizer


def _as_bhtd(tensor, batch, heads, tokens, dim):
    if tensor.shape == (batch, tokens, heads * dim):
        return tensor.reshape(batch, tokens, heads, dim).transpose(1, 2)
    if tensor.shape == (batch, heads, tokens, dim):
        return tensor
    if tensor.shape == (batch, tokens, heads, dim):
        return tensor.transpose(1, 2)
    raise RuntimeError(f"cannot interpret attention tensor shape {tuple(tensor.shape)}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seq_len > 64:
        raise ValueError("P2 smoke sequence length must be <= 64")
    tokenizer = load_tokenizer(args.model, local_files_only=args.local_files_only)
    model = load_causal_lm(args.model, device=args.device, dtype="float32",
                          attn_implementation="eager", local_files_only=args.local_files_only)
    if args.layer < 0 or args.layer >= model.config.num_hidden_layers:
        raise ValueError("layer out of range")
    captured = {}
    attention = model.model.layers[args.layer].self_attn

    def capture_q_norm(_module, _inputs, output):
        captured["q"] = output.detach()

    def capture_v(_module, _inputs, output):
        captured["v"] = output.detach()

    q_norm = getattr(attention, "q_norm", None)
    if q_norm is None:
        raise RuntimeError("Qwen attention q_norm was not found")
    hooks = [q_norm.register_forward_hook(capture_q_norm),
             attention.v_proj.register_forward_hook(capture_v)]
    ids = tokenizer("CacheBridge real attention smoke input.", return_tensors="pt",
                    truncation=True, max_length=args.seq_len).input_ids.to(args.device)
    with torch.no_grad():
        output = model(ids, use_cache=True, output_attentions=True)
    for hook in hooks:
        hook.remove()
    attentions = output.attentions
    if attentions is None or attentions[args.layer] is None:
        raise RuntimeError("expected attention probabilities were not captured")
    probs = attentions[args.layer].float()
    B, Hq, Tq, Tk = probs.shape
    D = model.config.head_dim
    q = captured["q"].float()
    q = _as_bhtd(q, B, Hq, Tq, D)
    v = _as_bhtd(captured["v"].float(), B, model.config.num_key_value_heads, Tk, D)
    causal = torch.ones_like(probs, dtype=torch.bool).tril()
    valid = torch.ones(B, Tk, dtype=torch.bool, device=probs.device)
    wk, wv = compute_gqa_sensitivities(
        probs, q, v, valid_mask=valid, causal_mask=causal)
    # The public outputs are now aggregated over query heads by KV head.
    expanded_v = v.repeat_interleave(Hq // v.shape[1], dim=1)
    o = torch.matmul(probs, expanded_v)
    for name, value in (("attention", probs), ("q", q), ("v", v), ("o", o),
                        ("w_K", wk), ("w_V", wv)):
        print(f"{name}_shape={tuple(value.shape)} dtype={value.dtype} "
              f"min={value.min().item():.8g} max={value.max().item():.8g} "
              f"mean={value.mean().item():.8g} finite={bool(torch.isfinite(value).all())} "
              f"nonnegative={bool((value >= 0).all())}")
    if not torch.isfinite(wk).all() or not torch.isfinite(wv).all():
        raise RuntimeError("non-finite attention sensitivity")
    print(f"rope_theta={get_rope_theta(model.config)} attention_implementation=eager")
    return 0


if __name__ == "__main__":
    sys.exit(main())
