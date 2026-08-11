"""RoPE strip/apply helpers (Llama/Qwen half-rotation convention).

The KV cache stores position-rotated keys. kvbridge maps keys in "content
space" (rotation removed), so a mapper fit at one context length stays valid
at any position the target's RoPE supports (paper section 3.3). RoPE rotations
are orthogonal, so stripping is exact up to float error.
"""

import torch


def get_rope_theta(config) -> float:
    """Read rope_theta from a HF config across transformers versions
    (it moved under ``config.rope_parameters`` in >= 4.57)."""
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
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def strip_rope(k, positions, rope_theta):
    """Inverse RoPE. k: [H, T, d] post-rotation keys -> content space."""
    cos, sin = build_cos_sin(positions, k.shape[-1], rope_theta, k.device, k.dtype)
    return k * cos[None] - rotate_half(k) * sin[None]


def apply_rope(k, positions, rope_theta):
    """Forward RoPE. k: [H, T, d] content-space keys -> rotated."""
    cos, sin = build_cos_sin(positions, k.shape[-1], rope_theta, k.device, k.dtype)
    return k * cos[None] + rotate_half(k) * sin[None]
