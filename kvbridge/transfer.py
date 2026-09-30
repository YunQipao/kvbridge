"""Runtime: map a source model's KV cache into the target's expected cache.

One matmul per target layer per cache type, run in the mapper's dtype
(bfloat16 by default) so it lands on tensor cores. Keys are mapped in
content space: strip source RoPE -> linear map -> re-apply target RoPE.
"""

import torch
from transformers import DynamicCache

from .cache_compat import cache_layer_kv, cache_num_layers
from .rope import apply_rope, strip_rope
from .cachebridge import validate_head_assignment


def transfer(cache, mapper, positions=None):
    """Map ``cache`` (a batch-1 DynamicCache from the source model) into a
    new DynamicCache in the target model's expected format.

    positions: 1-D tensor of the cached tokens' positions. Defaults to
    ``arange(T)``, which is correct for a cache that starts at the beginning
    of the sequence.

    The mapper is moved to the cache's device in place on first use
    (``Mapper.to`` is a no-op afterwards). Output dtype follows the source
    cache's dtype.
    """
    method = mapper.meta.get("method")
    if method not in (None, "full_head", "cachebridge",
                      "arXiv:2608.03893 closed-form per-layer ridge, content-space keys"):
        raise ValueError(f"unsupported mapper method: {method}")
    n_src = cache_num_layers(cache)
    k0, _ = cache_layer_kv(cache, 0)
    if k0.shape[0] != 1:
        raise ValueError(f"transfer supports batch size 1, got batch {k0.shape[0]}")
    device = k0.device
    out_dtype = k0.dtype
    T = k0.shape[2]

    max_needed = max(max(entry["source_layers"]) for entry in mapper.layers)
    if max_needed >= n_src:
        raise ValueError(
            f"source cache has {n_src} layers but the mapper selects source layer "
            f"{max_needed}; is this cache from {mapper.source}?")

    mapper.to(device=device)
    compute_dtype = mapper.dtype
    if positions is None:
        positions = torch.arange(T, device=device)
    if positions.shape[0] != T:
        raise ValueError(f"positions has length {positions.shape[0]} but cache has {T} tokens")

    theta_s, theta_t = mapper.rope_theta_source, mapper.rope_theta_target
    H, d = mapper.n_kv_heads, mapper.head_dim

    src_K, src_V = [], []
    for layer in range(n_src):
        k, v = cache_layer_kv(cache, layer)
        k, v = k[0].float(), v[0].float()                        # [H, T, d]
        kc = strip_rope(k, positions, theta_s)
        src_K.append(kc.transpose(0, 1).reshape(T, -1).to(compute_dtype))   # [T, H*d]
        src_V.append(v.transpose(0, 1).reshape(T, -1).to(compute_dtype))

    out = DynamicCache()
    is_cachebridge = mapper.meta.get("method") == "cachebridge"
    if is_cachebridge:
        n_target_heads = int(mapper.meta.get("num_target_kv_heads", mapper.n_kv_heads))
        assignment = validate_head_assignment(
            mapper.meta.get("head_assignment"),
            int(mapper.meta["num_source_kv_heads"]), n_target_heads)
        if H != int(mapper.meta["num_source_kv_heads"]) or d != mapper.head_dim:
            raise ValueError("source cache KV geometry does not match CacheBridge mapper")
    else:
        assignment = None
    for tl, entry in enumerate(mapper.layers):
        chosen = entry["source_layers"]
        if is_cachebridge:
            # CacheBridge stores one compact affine map per target KV head:
            # W_* [Ht, k*d, d], mu_x* [Ht, k*d], mu_y* [Ht, d].
            Yks, Yvs = [], []
            for h in range(n_target_heads):
                sh = assignment[h]
                Xk = torch.cat([src_K[sl].reshape(T, H, d)[:, sh, :] for sl in chosen], dim=-1)
                Xv = torch.cat([src_V[sl].reshape(T, H, d)[:, sh, :] for sl in chosen], dim=-1)
                Yks.append(torch.addmm(entry["mu_yk"][h],
                                       Xk - entry["mu_xk"][h], entry["W_k"][h]))
                Yvs.append(torch.addmm(entry["mu_yv"][h],
                                       Xv - entry["mu_xv"][h], entry["W_v"][h]))
            Kt = torch.stack(Yks, dim=1).float().transpose(0, 1)  # [H, T, d]
            Vt = torch.stack(Yvs, dim=1).float().transpose(0, 1)
        else:
            Xk = torch.cat([src_K[sl] for sl in chosen], dim=-1)
            # centered affine map: (x - mu_x) @ W + mu_y
            Yk = torch.addmm(entry["mu_yk"], Xk - entry["mu_xk"], entry["W_k"])
            Xv = torch.cat([src_V[sl] for sl in chosen], dim=-1)
            Yv = torch.addmm(entry["mu_yv"], Xv - entry["mu_xv"], entry["W_v"])
            Kt = Yk.float().reshape(T, H, d).transpose(0, 1)         # [H, T, d]
            Vt = Yv.float().reshape(T, H, d).transpose(0, 1)
        Kt = apply_rope(Kt, positions, theta_t)
        out.update(Kt.to(out_dtype).unsqueeze(0), Vt.to(out_dtype).unsqueeze(0), tl)
    return out
