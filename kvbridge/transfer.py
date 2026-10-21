"""Runtime: map a source model's KV cache into the target's expected cache.

One matmul per target layer per cache type, run in the mapper's dtype
(bfloat16 by default) so it lands on tensor cores. Keys are mapped in
content space: strip source RoPE -> linear map -> re-apply target RoPE.
"""

import torch
from transformers import DynamicCache

from .cache_compat import cache_layer_kv, cache_num_layers
from .rope import apply_rope, strip_rope
from .cachebridge import normalize_head_assignment, validate_head_map
from .mapper import TENSOR_KEYS


def transfer(cache, mapper, positions=None, *, layer_devices=None):
    """Map ``cache`` (a batch-1 DynamicCache from the source model) into a
    new DynamicCache in the target model's expected format.

    positions: 1-D tensor of the cached tokens' positions. Defaults to
    ``arange(T)``, which is correct for a cache that starts at the beginning
    of the sequence.

    Without layer_devices the legacy single-device path is unchanged.
    With layer_devices, keep the mapper on CPU and stage only the selected
    source blocks and one mapper layer at a time. The returned KV layers live
    on the corresponding target attention execution devices. No complete
    cache or mapper is gathered onto cuda:0. Output dtype follows the source.
    """
    method = mapper.meta.get("method")
    if method not in (None, "full_head", "cachebridge", "head_local",
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

    if layer_devices is not None and len(layer_devices) != len(mapper.layers):
        raise ValueError("layer_devices must specify one device per target mapper layer")
    if layer_devices is None:
        mapper.to(device=device)
    compute_dtype = mapper.dtype
    if positions is None:
        positions = torch.arange(T, device=device)
    if positions.shape[0] != T:
        raise ValueError(f"positions has length {positions.shape[0]} but cache has {T} tokens")

    theta_s = mapper.meta.get("rope_config_source", mapper.rope_theta_source)
    theta_t = mapper.meta.get("rope_config_target", mapper.rope_theta_target)
    for side, spec in (("source", theta_s), ("target", theta_t)):
        geometry = mapper.meta.get(f"{side}_geometry", {})
        params = geometry.get("rope_parameters") or {}
        if params.get("rope_type", params.get("type")) == "yarn" and not isinstance(spec, dict):
            raise ValueError(f"YaRN {side} mapper is missing native rotary metadata; refit required")
    H, d = (int(mapper.meta.get("num_source_kv_heads", mapper.n_kv_heads)),
            mapper.head_dim)

    src_K, src_V = [], []
    if layer_devices is None:
        for layer in range(n_src):
            k, v = cache_layer_kv(cache, layer)
            k, v = k[0].float(), v[0].float()                    # [H, T, d]
            kc = strip_rope(k, positions, theta_s)
            src_K.append(kc.transpose(0, 1).reshape(T, -1).to(compute_dtype))
            src_V.append(v.transpose(0, 1).reshape(T, -1).to(compute_dtype))

    out = DynamicCache()
    is_cachebridge = mapper.meta.get("method") in ("cachebridge", "head_local")
    if is_cachebridge:
        n_target_heads = int(mapper.meta.get("num_target_kv_heads", mapper.n_kv_heads))
        assignment = normalize_head_assignment(
            mapper.meta.get("head_assignment"),
            int(mapper.meta["num_source_kv_heads"]), n_target_heads)
        head_map = validate_head_map(mapper.meta.get("head_map"),
            int(mapper.meta["num_source_kv_heads"]), n_target_heads,
            mode=mapper.meta.get("head_map_mode", "matched"))
        if H != int(mapper.meta["num_source_kv_heads"]) or d != mapper.head_dim:
            raise ValueError("source cache KV geometry does not match CacheBridge mapper")
    else:
        assignment = None
        head_map = None
    for tl, entry in enumerate(mapper.layers):
        if layer_devices is not None:
            Kt, Vt = _transfer_staged_layer(cache, entry, positions,
                layer_devices[tl], compute_dtype, theta_s, theta_t,
                H, d, assignment, head_map)
        else:
            Kt, Vt = _apply_layer(entry, src_K, src_V, T, H, d, assignment, head_map)
            Kt = apply_rope(Kt, positions, theta_t)
        out.update(Kt.to(out_dtype).unsqueeze(0), Vt.to(out_dtype).unsqueeze(0), tl)
        del Kt, Vt
    return out


def _transfer_staged_layer(cache, entry, positions, device, dtype,
                           theta_s, theta_t, heads, dim, assignment, head_map):
    """One bounded staging scope; temporary coefficients/features die on return."""
    device = torch.device(device)
    local = {key: entry[key].to(device=device) for key in TENSOR_KEYS}
    local["source_layers"] = entry["source_layers"]
    src_K, src_V = {}, {}
    tokens = positions.shape[0]
    for layer in dict.fromkeys(entry["source_layers"]):
        key, value = cache_layer_kv(cache, layer)
        # Strip on the source layer's own GPU, then move just this selected
        # block to the current target layer. Keep the original cache untouched.
        content = strip_rope(key[0].float(), positions.to(key.device), theta_s)
        src_K[layer] = content.transpose(0, 1).reshape(tokens, -1).to(device=device, dtype=dtype)
        src_V[layer] = value[0].transpose(0, 1).reshape(tokens, -1).to(device=device, dtype=dtype)
        del content, key, value
    key, value = _apply_layer(local, src_K, src_V, tokens, heads, dim, assignment, head_map)
    return apply_rope(key, positions.to(device), theta_t), value


def _apply_layer(entry, src_K, src_V, T, H, d, assignment, head_map=None):
    """The existing affine mapper equations, shared by both placement paths."""
    chosen = entry["source_layers"]
    if assignment is not None:
        # CacheBridge stores one compact affine map per target KV head:
        # W_* [Ht, k*d, d], mu_x* [Ht, k*d], mu_y* [Ht, d].
        Yks, Yvs = [], []
        for h in range(len(assignment)):
            source_heads = head_map[h] if head_map is not None else [assignment[h]]
            Xk = torch.cat([src_K[sl].reshape(T, H, d)[:, source_heads, :].reshape(T, -1)
                            for sl in chosen], dim=-1)
            Xv = torch.cat([src_V[sl].reshape(T, H, d)[:, source_heads, :].reshape(T, -1)
                            for sl in chosen], dim=-1)
            Yks.append(torch.addmm(entry["mu_yk"][h],
                                   Xk - entry["mu_xk"][h], entry["W_k"][h]))
            Yvs.append(torch.addmm(entry["mu_yv"][h],
                                   Xv - entry["mu_xv"][h], entry["W_v"][h]))
        Kt = torch.stack(Yks, dim=1).float().transpose(0, 1)  # [H, T, d]
        Vt = torch.stack(Yvs, dim=1).float().transpose(0, 1)
    else:
        Xk = torch.cat([src_K[sl] for sl in chosen], dim=-1)
        Xv = torch.cat([src_V[sl] for sl in chosen], dim=-1)
        if entry["W_k"].ndim == 3:
            Yks = [torch.addmm(entry["mu_yk"][h],
                               Xk - entry["mu_xk"][h], entry["W_k"][h])
                   for h in range(entry["W_k"].shape[0])]
            Yvs = [torch.addmm(entry["mu_yv"][h],
                               Xv - entry["mu_xv"][h], entry["W_v"][h])
                   for h in range(entry["W_v"].shape[0])]
            Kt = torch.stack(Yks, dim=1).float().transpose(0, 1)
            Vt = torch.stack(Yvs, dim=1).float().transpose(0, 1)
        else:
            # Legacy format-1 Full-Head artifacts remain readable.
            Yk = torch.addmm(entry["mu_yk"], Xk - entry["mu_xk"], entry["W_k"])
            Yv = torch.addmm(entry["mu_yv"], Xv - entry["mu_xv"], entry["W_v"])
            Kt = Yk.float().reshape(T, H, d).transpose(0, 1)
            Vt = Yv.float().reshape(T, H, d).transpose(0, 1)
    return Kt, Vt
