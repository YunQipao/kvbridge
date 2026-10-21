"""Model geometry checks shared by smoke scripts and fitting."""

from pathlib import Path


def text_config(config):
    """Select language geometry, never the composite model's vision config."""
    if getattr(config, "model_type", None) in ("mistral3", "ministral3"):
        return getattr(config, "text_config", None) or config
    return config


def json_safe(value):
    """Convert config values into deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    # torch.dtype is intentionally handled without importing torch here.
    if value.__class__.__name__ == "dtype":
        return str(value)
    raise TypeError(f"unsupported non-JSON config value: {type(value)!r}")


def head_dim(config):
    config = text_config(config)
    value = getattr(config, "head_dim", None)
    return int(value) if value is not None else int(config.hidden_size // config.num_attention_heads)


def geometry_dict(config):
    config = text_config(config)
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    return {
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": head_dim(config),
        "rope_theta": json_safe(getattr(config, "rope_theta", None) or
                                (getattr(config, "rope_parameters", None) or {}).get("rope_theta")),
        "rope_scaling": json_safe(getattr(config, "rope_scaling", None)),
        "rope_parameters": json_safe(getattr(config, "rope_parameters", None)),
        "dtype": json_safe(dtype),
    }


def build_head_map(source_heads, target_heads, mode="auto"):
    source_heads, target_heads = int(source_heads), int(target_heads)
    if source_heads <= 0 or target_heads <= 0:
        raise ValueError("KV-head counts must be positive")
    if mode == "auto":
        mode = "matched" if source_heads == target_heads else "block"
    if mode not in ("matched", "block"):
        raise ValueError("head map mode must be 'auto', 'matched' or 'block'")
    if mode == "matched":
        if source_heads != target_heads:
            raise ValueError("matched head mapping requires equal KV-head counts")
        return [[h] for h in range(target_heads)]
    if source_heads < target_heads:
        raise ValueError("block head mapping requires source KV heads >= target KV heads")
    if source_heads % target_heads:
        raise ValueError("block head mapping requires evenly sized contiguous blocks")
    width = source_heads // target_heads
    return [list(range(h * width, (h + 1) * width)) for h in range(target_heads)]


def check_kv_compatibility(source_config, target_config, head_map_mode="auto"):
    source = geometry_dict(source_config)
    target = geometry_dict(target_config)
    compatible = (source["head_dim"] == target["head_dim"])
    try:
        head_map = build_head_map(source["num_key_value_heads"],
                                  target["num_key_value_heads"], head_map_mode)
    except ValueError:
        head_map = None
        compatible = False
    source["head_map"] = head_map
    target["head_map_mode"] = head_map_mode
    return compatible, source, target
