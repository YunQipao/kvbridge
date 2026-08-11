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
