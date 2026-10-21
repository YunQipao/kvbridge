import torch
from kvbridge.fused_fit import fit_reference_and_fused


def test_fused_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(32, 6)
    y = torch.randn(32, 3)
    w = torch.rand(32)
    ref, fused = fit_reference_and_fused(x, y, w, .01)
    assert torch.allclose(ref, fused, atol=1e-6, rtol=1e-6)

