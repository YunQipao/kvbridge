"""RoPE strip/apply must be exact inverses, and must match the rotation the
target model's attention actually applies -- a convention mismatch here is
silent: the mapper still fits, and retrieval just quietly degrades.
"""

import torch

from kvbridge.rope import apply_rope, build_cos_sin, strip_rope

THETA = 1_000_000.0
H, T, D = 8, 64, 128


def _keys():
    torch.manual_seed(0)
    return torch.randn(H, T, D, dtype=torch.float32)


def test_strip_then_apply_is_identity():
    k = _keys()
    pos = torch.arange(T)
    back = apply_rope(strip_rope(k, pos, THETA), pos, THETA)
    assert torch.allclose(k, back, atol=1e-5)


def test_apply_then_strip_is_identity():
    k = _keys()
    pos = torch.arange(T)
    back = strip_rope(apply_rope(k, pos, THETA), pos, THETA)
    assert torch.allclose(k, back, atol=1e-5)


def test_rotation_preserves_norm():
    k = _keys()
    rotated = apply_rope(k, torch.arange(T), THETA)
    assert torch.allclose(k.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_position_zero_is_a_no_op():
    k = _keys()
    zeros = torch.zeros(T, dtype=torch.long)
    assert torch.allclose(apply_rope(k, zeros, THETA), k, atol=1e-6)


def test_offset_positions_shift_the_rotation():
    """A cache slice re-rotated at its true offset must equal the full-sequence
    rotation of those positions. This is what makes the `hybrid` strategy and
    any mid-sequence transfer legal."""
    k = _keys()
    full = apply_rope(k, torch.arange(T), THETA)
    offset = 16
    part = apply_rope(k[:, offset:], torch.arange(offset, T), THETA)
    assert torch.allclose(full[:, offset:], part, atol=1e-5)


def test_matches_transformers_convention():
    """Cross-check against the rotation transformers applies to keys."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    k = _keys()
    pos = torch.arange(T)
    cos, sin = build_cos_sin(pos, D, THETA, k.device, k.dtype)
    # transformers expects [B, H, T, d] and broadcasts cos/sin over [B, T, d].
    _, ref = apply_rotary_pos_emb(k.unsqueeze(0), k.unsqueeze(0),
                                  cos.unsqueeze(0), sin.unsqueeze(0))
    assert torch.allclose(apply_rope(k, pos, THETA), ref[0], atol=1e-5)
