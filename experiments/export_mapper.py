# Export a fitted mapper from the Modal volume into the kvbridge package format
# (mapper.safetensors + mapper.json) and optionally push it to the Hub.
#
#   modal run experiments/export_mapper.py --k 12
#   modal run experiments/export_mapper.py --k 8 --repo-id you/kvbridge-qwen3-1.7b-to-4b-k8
#   modal run experiments/export_mapper.py --k 12 --no-push        # write to the volume only
#
# Pushing needs a Modal secret named `huggingface` holding a write-scoped
# HF_TOKEN:  modal secret create huggingface HF_TOKEN=hf_...

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.51,<5",
        "safetensors>=0.4",
        "huggingface-hub>=0.23",
    )
    .add_local_python_source("kvbridge")
)

app = modal.App("kvbridge-export", image=image)
vol = modal.Volume.from_name("kvbridge-out", create_if_missing=True)

CARD = """---
license: mit
library_name: kvbridge
tags:
  - kv-cache
  - inference
  - qwen3
base_model:
  - {source}
  - {target}
---

# kvbridge {short_source} -> {short_target} (top_k={k})

A fitted KV cache mapper: it converts a `{short_source}` KV cache into one
`{short_target}` can generate from, so the prompt is prefilled once on the
small model instead of on the big one.

```python
from kvbridge import Mapper, transfer

mapper = Mapper.from_pretrained("{repo_id}")
small_cache = small(prompt_ids, use_cache=True).past_key_values
big_cache = transfer(small_cache, mapper)
big.generate(input_ids=prompt_ids, past_key_values=big_cache, max_new_tokens=128)
```

Code, results and fitting instructions: https://github.com/{gh}/kvbridge

Fit on {n_texts} FineWeb-Edu sequences x {seq_len} tokens, stride-{stride}
subsampled, ridge lambda={lam}. Mean in-sample R^2: K {r2_k:.3f}, V {r2_v:.3f}.
Weights are bfloat16.

Method: *Cross-Model KV Cache Transfer in LLM Families* (Heo et al., NVIDIA,
[arXiv:2608.03893](https://arxiv.org/abs/2608.03893)).
"""


@app.function(timeout=3600, memory=32768, volumes={"/out": vol},
              secrets=[modal.Secret.from_name("huggingface")])
def export(source: str, target: str, k: int, repo_id: str, push: bool,
           gh_user: str, n_calib: int = 500, seq_len: int = 1024,
           lam: float = 0.01, stride: int = 4):
    import os

    import torch
    from transformers import AutoConfig

    from kvbridge.mapper import Mapper
    from kvbridge.rope import get_rope_theta

    short_s, short_t = source.split("/")[-1], target.split("/")[-1]
    fit_path = (f"/out/fitcache_{short_s}_to_{short_t}"
                f"_n{n_calib}_L{seq_len}_s{stride}_lam{lam}_k{k}.pt")
    if not os.path.exists(fit_path):
        raise FileNotFoundError(f"no fit cache at {fit_path}; run the fit first")

    blob = torch.load(fit_path, map_location="cpu", weights_only=False)
    layers, summary = blob["mapper"]["layers"], blob["summary"]

    cs, ct = AutoConfig.from_pretrained(source), AutoConfig.from_pretrained(target)
    head_dim = getattr(cs, "head_dim", None) or cs.hidden_size // cs.num_attention_heads

    meta = {
        "source": source, "target": target,
        "n_kv_heads": int(cs.num_key_value_heads), "head_dim": int(head_dim),
        "rope_theta_source": get_rope_theta(cs), "rope_theta_target": get_rope_theta(ct),
        "top_k": k, "lam": lam,
        "mean_r2_k": summary["r2_k"], "mean_r2_v": summary["r2_v"],
        "calibration": {"n_texts": n_calib, "seq_len": seq_len,
                        "token_stride": stride},
        "method": "arXiv:2608.03893 closed-form per-layer ridge, content-space keys",
    }
    mapper = Mapper([{**e, "source_layers": [int(s) for s in e["source_layers"]]}
                     for e in layers], meta)

    out_dir = f"/out/export_{short_s}_to_{short_t}_k{k}"
    mapper.save(out_dir)
    vol.commit()

    size_mb = sum(os.path.getsize(os.path.join(out_dir, f))
                  for f in os.listdir(out_dir)) / 1e6
    print(f"exported {mapper} -> {out_dir} ({size_mb:.0f} MB)", flush=True)
    print(f"  mean R^2: K={summary['r2_k']:.3f} V={summary['r2_v']:.3f}", flush=True)

    if not push:
        return {"path": out_dir, "size_mb": size_mb}

    from huggingface_hub import HfApi
    card = CARD.format(source=source, target=target, short_source=short_s,
                       short_target=short_t, k=k, repo_id=repo_id, gh=gh_user,
                       n_texts=n_calib, seq_len=seq_len, stride=stride,
                       lam=lam, r2_k=summary["r2_k"], r2_v=summary["r2_v"])
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(card)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=out_dir, repo_type="model",
                      commit_message=f"kvbridge mapper {short_s} -> {short_t}, top_k={k}")
    url = f"https://huggingface.co/{repo_id}"
    print(f"pushed to {url}", flush=True)
    return {"path": out_dir, "size_mb": size_mb, "url": url}


@app.local_entrypoint()
def main(source: str = "Qwen/Qwen3-1.7B", target: str = "Qwen/Qwen3-4B",
         k: int = 12, repo_id: str = "", push: bool = True,
         gh_user: str = "Sidd03192"):
    if not repo_id:
        s, t = source.split("/")[-1].lower(), target.split("/")[-1].lower()
        suffix = "" if k == 12 else f"-k{k}"
        repo_id = f"Siddharth85/kvbridge-{s}-to-{t}{suffix}"
    print(export.remote(source, target, k, repo_id, push, gh_user))
