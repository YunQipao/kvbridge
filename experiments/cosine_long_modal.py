# Eval-only diagnostic (paper S 4.5): position-bucketed attention-output
# cosine at 8K/16K prefixes, to attribute the long-context retention decline.
#
#   - buckets FLAT across position   -> mechanism 2 (global fit deficit at
#     long range); recalibrating on longer sequences will NOT help
#   - cosine FALLING in late buckets -> mechanism 1 (positional
#     extrapolation past the 1024-token fit context); the fix is a refit
#     with longer or mixed-length calibration sequences
#
# Reuses the pipeline's cached fit (k=12 default), the 48x16512-token held-out
# blocks, and the chunked cosine implementation (memory-bounded per query
# head, so 16K fits an A100-40GB: ~26 GB peak beside both models).
# One container, ~10 min, ~$1. Results also saved to the volume as JSON.
#
# Usage (from the repo root):
#   modal run experiments/cosine_long_modal.py
#   modal run experiments/cosine_long_modal.py --k 24
#   POC_GPU=A100-80GB modal run experiments/cosine_long_modal.py

import os

import modal

import fit_eval_modal as pipeline

GPU_TYPE = os.environ.get("POC_GPU", "A100")   # 16K cosine needs ~26 GB

app = modal.App("kv-cosine-long",
                image=pipeline.image.add_local_python_source("fit_eval_modal"))
vol = modal.Volume.from_name("kvbridge-out", create_if_missing=True)


@app.function(gpu=GPU_TYPE, timeout=3600, memory=32768, volumes={"/out": vol})
def cosine_long(k: int = 12, source: str = "Qwen/Qwen3-1.7B",
                target: str = "Qwen/Qwen3-4B", prefixes: str = "8192,16384",
                n_blocks: int = 4, n_buckets: int = 8, n_calib: int = 500,
                seq_len: int = 1024, lam: float = 0.01, n_heldout: int = 48):
    import json

    import torch
    from transformers import AutoModelForCausalLM

    from fit_eval_modal import (CONTINUATION_LENGTH, attention_output_cosine,
                     fit_cache_path, get_rope_theta, heldout_path,
                     mapper_to_device, transfer_cache)

    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    prefix_list = [int(x) for x in prefixes.split(",")]

    fit_p = fit_cache_path(source, target, n_calib, seq_len, lam, k)
    assert os.path.exists(fit_p), f"missing fit cache {os.path.basename(fit_p)}"
    mapper = torch.load(fit_p, map_location="cpu", weights_only=False)["mapper"]

    held_p = heldout_path(source, n_calib, n_heldout,
                          max(prefix_list) + CONTINUATION_LENGTH)
    assert os.path.exists(held_p), f"missing held-out blocks {os.path.basename(held_p)}"
    blocks = torch.load(held_p, map_location="cpu", weights_only=False)

    print(f"loading {source} and {target} (GPU: {GPU_TYPE})", flush=True)
    m_src = AutoModelForCausalLM.from_pretrained(source, dtype=dtype).to(device).eval()
    m_tgt = AutoModelForCausalLM.from_pretrained(target, dtype=dtype).to(device).eval()
    cs = m_src.config
    H, d = cs.num_key_value_heads, cs.head_dim
    theta_s = get_rope_theta(cs)
    theta_t = get_rope_theta(m_tgt.config)
    mapper_gpu = mapper_to_device(mapper, device)

    results = {}
    for prefix in prefix_list:
        per_block = []
        for bi in range(n_blocks):
            pre = blocks[bi][:prefix].unsqueeze(0).to(device)
            pos = torch.arange(prefix, device=device)
            with torch.no_grad():
                src_cache = m_src(pre, use_cache=True).past_key_values
                real_cache = m_tgt(pre, use_cache=True).past_key_values
            mapped = transfer_cache(src_cache, mapper_gpu, pos, theta_s, theta_t,
                                    H, d, device, dtype)
            del src_cache
            torch.cuda.empty_cache()
            # attention_output_cosine only reads the caches (no mutation), so
            # no defensive deepcopies -- that is what keeps 16K in memory.
            overall, _, buckets = attention_output_cosine(
                m_tgt, pre, mapped, real_cache, pos, n_buckets=n_buckets)
            per_block.append({"overall": overall, "buckets": buckets})
            print(f"  prefix={prefix:6d} block={bi}: overall={overall:.4f} | "
                  + " ".join(f"{b:.3f}" for b in buckets), flush=True)
            del mapped, real_cache
            torch.cuda.empty_cache()
        n = len(per_block)
        results[prefix] = {
            "overall": sum(b["overall"] for b in per_block) / n,
            "buckets": [sum(b["buckets"][i] for b in per_block) / n
                        for i in range(n_buckets)],
            "per_block": per_block,
        }

    out_name = (f"/out/cosine_long_{source.split('/')[-1]}_to_"
                f"{target.split('/')[-1]}_k{k}_b{n_buckets}.json")
    with open(out_name, "w") as f:
        json.dump({"k": k, "n_blocks": n_blocks, "n_buckets": n_buckets,
                   "prefixes": prefix_list,
                   "results": {str(p): results[p] for p in prefix_list}}, f, indent=2)
    vol.commit()
    print(f"saved {os.path.basename(out_name)} to volume kvbridge-out", flush=True)
    return results


@app.local_entrypoint()
def main(k: int = 12, source: str = "Qwen/Qwen3-1.7B",
         target: str = "Qwen/Qwen3-4B", prefixes: str = "8192,16384",
         n_blocks: int = 4, n_buckets: int = 8):
    results = cosine_long.remote(k=k, source=source, target=target,
                                 prefixes=prefixes, n_blocks=n_blocks,
                                 n_buckets=n_buckets)
    print(f"\n== Position-bucketed attention-output cosine "
          f"(k={k}, {n_blocks} blocks, buckets early -> late) ==")
    for prefix in sorted(results):
        r = results[prefix]
        nb = len(r["buckets"])
        early = sum(r["buckets"][:nb // 2]) / (nb // 2)
        late = sum(r["buckets"][nb // 2:]) / (nb - nb // 2)
        print(f"  prefix={prefix:6d}: overall={r['overall']:.4f} | "
              + " ".join(f"{b:.3f}" for b in r["buckets"]))
        print(f"                 early-half {early:.4f} vs late-half {late:.4f}"
              f"  (delta {late - early:+.4f})")
    print("\nreading: FLAT buckets -> global long-range fit deficit "
          "(recalibration won't help);\n"
          "         FALLING late buckets -> positional extrapolation "
          "(refit with longer/mixed-length calibration)")
