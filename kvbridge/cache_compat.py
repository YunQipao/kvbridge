"""DynamicCache access across transformers versions.

The internals changed across 4.5x:
  <= ~4.53: ``cache.key_cache[l]`` / ``cache.value_cache[l]``
  >= ~4.54: ``cache.layers[l].keys`` / ``cache.layers[l].values``
"""


def cache_layer_kv(cache, layer_idx):
    """Return (keys, values) tensors [B, H, T, d] for one layer."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def cache_num_layers(cache):
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def cache_geometry(cache):
    """Return cache geometry without depending on a DynamicCache version."""
    n_layers = cache_num_layers(cache)
    if n_layers == 0:
        return {"num_layers": 0}
    key, value = cache_layer_kv(cache, 0)
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("cache tensors must have shape [B, H, T, D]")
    if key.shape != value.shape:
        raise ValueError("key/value cache shapes differ")
    return {
        "num_layers": n_layers,
        "batch": int(key.shape[0]),
        "num_kv_heads": int(key.shape[1]),
        "sequence_length": int(key.shape[2]),
        "head_dim": int(key.shape[3]),
        "dtype": str(key.dtype),
        "device": str(key.device),
    }


def validate_cache_geometry(cache, *, num_layers=None, batch=None,
                            num_kv_heads=None, sequence_length=None,
                            head_dim=None):
    geometry = cache_geometry(cache)
    expected = {
        "num_layers": num_layers,
        "batch": batch,
        "num_kv_heads": num_kv_heads,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
    }
    for key, value in expected.items():
        if value is not None and geometry[key] != int(value):
            raise ValueError(f"cache {key}={geometry[key]} != expected {value}")
    for layer in range(geometry["num_layers"]):
        key, value = cache_layer_kv(cache, layer)
        if not key.is_floating_point() or not value.is_floating_point():
            raise ValueError("cache K/V tensors must be floating point")
        if not key.isfinite().all() or not value.isfinite().all():
            raise ValueError(f"cache layer {layer} contains NaN or Inf")
    return geometry
