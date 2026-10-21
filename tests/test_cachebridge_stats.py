import torch

from kvbridge.cachebridge import fit_head_local_streaming


def test_head_local_streaming_fit_uses_assigned_head_only():
    source = [torch.tensor([[[1.0, 2.0], [100.0, 200.0]]]),
              torch.tensor([[[3.0, 4.0], [300.0, 400.0]]])]
    target = [torch.tensor([[[4.0, 6.0], [40.0, 60.0]]])]
    layers = fit_head_local_streaming(
        [source], [target], [[0, 1]], head_assignment=[0, 1],
        num_target_layers=1, ridge_lambda=0.01)
    assert layers[0]["W_k"].shape == (2, 4, 2)
    assert layers[0]["mu_xk"].shape == (2, 4)
