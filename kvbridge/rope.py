"""RoPE strip/apply helpers (Llama/Qwen half-rotation convention).

The KV cache stores position-rotated keys. kvbridge maps keys in "content
space" (rotation removed), so a mapper fit at one context length stays valid
at any position the target's RoPE supports (paper section 3.3). YaRN's native
amplitude scaling is included in both the forward transform and its inverse.
"""

import torch
from .geometry import text_config


def rope_spec(config):
    """Persist the complete Ministral text config for native YaRN reconstruction."""
    config = text_config(config)
    if getattr(config, "model_type", None) == "ministral3":
        return config.to_dict()
    return get_rope_theta(config)


def native_cos_sin(positions, head_dim, spec, device, dtype):
    if not isinstance(spec, dict):
        return build_cos_sin(positions, head_dim, spec, device, dtype)
    from transformers import AutoConfig
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding
    values = dict(spec)
    model_type = values.pop("model_type")
    if model_type != "ministral3":
        raise ValueError(f"unsupported native rotary config: {model_type}")
    config = AutoConfig.for_model(model_type, **values)
    if config.head_dim != head_dim:
        raise ValueError("rotary config head_dim does not match cache")
    rotary = Ministral3RotaryEmbedding(config).to(device)
    x = torch.empty(1, positions.numel(), head_dim, device=device, dtype=dtype)
    with torch.no_grad():
        cos, sin = rotary(x, positions.to(device)[None])
    return cos[0], sin[0]


def get_rope_theta(config) -> float:
    """Read rope_theta from a HF config across transformers versions
    (it moved under ``config.rope_parameters`` in >= 4.57)."""
    config = text_config(config)
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and params.get("rope_theta") is not None:
        return float(params["rope_theta"])
    raise AttributeError(f"could not locate rope_theta on {type(config).__name__}")


def build_cos_sin(positions, head_dim, rope_theta, device, dtype):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = positions.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    # angles cover d/2 frequencies; repeating them over both halves matches
    # the rotate_half convention transformers uses
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def strip_rope(k, positions, rope_theta):
    """Inverse RoPE. k: [H, T, d] post-rotation keys -> content space."""
    cos, sin = native_cos_sin(positions, k.shape[-1], rope_theta, k.device, k.dtype)
    result = k * cos[None] - rotate_half(k) * sin[None]
    # YaRN may scale both cos and sin: the inverse is not just the transpose.
    if isinstance(rope_theta, dict):
        result = result / (cos.square() + sin.square())[None]
    return result


def apply_rope(k, positions, rope_theta):
    """Forward RoPE. k: [H, T, d] content-space keys -> rotated."""
    cos, sin = native_cos_sin(positions, k.shape[-1], rope_theta, k.device, k.dtype)
    return k * cos[None] + rotate_half(k) * sin[None]
