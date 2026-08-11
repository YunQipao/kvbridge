"""Runtime: map a source model's KV cache into the target's expected cache.

One matmul per target layer per cache type, run in the mapper's dtype
(bfloat16 by default) so it lands on tensor cores. Keys are mapped in
content space: strip source RoPE -> linear map -> re-apply target RoPE.
"""

import torch
from transformers import DynamicCache

from .cache_compat import cache_layer_kv, cache_num_layers
from .rope import apply_rope, strip_rope


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
    for tl, entry in enumerate(mapper.layers):
        chosen = entry["source_layers"]
        Xk = torch.cat([src_K[sl] for sl in chosen], dim=-1)
        Yk = torch.addmm(entry["mu_yk"], Xk - entry["mu_xk"], entry["W_k"])
        Xv = torch.cat([src_V[sl] for sl in chosen], dim=-1)
        Yv = torch.addmm(entry["mu_yv"], Xv - entry["mu_xv"], entry["W_v"])
        Kt = Yk.float().reshape(T, H, d).transpose(0, 1)         # [H, T, d]
        Vt = Yv.float().reshape(T, H, d).transpose(0, 1)
        Kt = apply_rope(Kt, positions, theta_t)
        out.update(Kt.to(out_dtype).unsqueeze(0), Vt.to(out_dtype).unsqueeze(0), tl)
    return out
