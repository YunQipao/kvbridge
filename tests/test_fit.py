"""The ridge solvers.

`ridge_streaming` accumulates normal equations over row chunks so a design
matrix of ~10^5 rows x 12k columns never materializes. That optimization is
only safe if it agrees with the direct solve, and if its moment-based R^2
agrees with the residual-based one -- both are asserted here on data small
enough to compute the honest answer.
"""

import torch

from kvbridge.fit import r2_score, ridge_centered, ridge_streaming

H, D = 2, 8


def synthetic(n=512, n_src_layers=3, noise=0.05, seed=0):
    """Target features generated as a known linear map of the source layers."""
    torch.manual_seed(seed)
    src = [torch.randn(n, H, D) for _ in range(n_src_layers)]
    X = torch.cat([s.reshape(n, -1) for s in src], dim=-1)
    W = torch.randn(X.shape[1], H * D) / X.shape[1] ** 0.5
    Y = X @ W + 0.5 + noise * torch.randn(n, H * D)
    return [s.half() for s in src], Y.reshape(n, H, D).half(), W


def test_recovers_a_known_linear_map():
    src, tgt, _ = synthetic(n=4096, noise=0.01)
    _, _, _, r2 = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=1e-4)
    assert r2 > 0.99, r2


def test_ignores_unselected_source_layers():
    """Fitting on only layer 0 of a 3-layer generator must do clearly worse --
    a guard against `chosen` being silently ignored."""
    src, tgt, _ = synthetic(n=4096, noise=0.01)
    _, _, _, r2_all = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=1e-4)
    _, _, _, r2_one = ridge_streaming(src, tgt, chosen=[0], lam=1e-4)
    assert r2_one < r2_all - 0.2


def test_streaming_matches_the_direct_solve():
    src, tgt, _ = synthetic(n=1024)
    W_s, mu_x, mu_y, _ = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=0.01,
                                         chunk=97)  # non-divisor, to cross chunk edges
    X = torch.cat([s.float().reshape(s.shape[0], -1) for s in src], dim=-1)
    Y = tgt.float().reshape(tgt.shape[0], -1)
    W_d, mu_xd, mu_yd = ridge_centered(X, Y, lam=0.01)

    assert torch.allclose(mu_x, mu_xd, atol=1e-3)
    assert torch.allclose(mu_y, mu_yd, atol=1e-3)
    assert (W_s - W_d).abs().max() < 1e-2


def test_moment_based_r2_matches_residuals():
    src, tgt, _ = synthetic(n=1024)
    W, mu_x, mu_y, r2 = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=0.01)
    X = torch.cat([s.float().reshape(s.shape[0], -1) for s in src], dim=-1)
    Y = tgt.float().reshape(tgt.shape[0], -1)
    assert abs(r2 - r2_score(X, W, mu_x, mu_y, Y)) < 1e-3


def test_prediction_is_centered_correctly():
    """The bias lives in the means, so predictions must be unbiased on the
    fitting data -- an off-by-one in the centering shows up as a mean shift."""
    src, tgt, _ = synthetic(n=1024)
    W, mu_x, mu_y, _ = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=0.01)
    X = torch.cat([s.float().reshape(s.shape[0], -1) for s in src], dim=-1)
    Y = tgt.float().reshape(tgt.shape[0], -1)
    pred = (X - mu_x) @ W + mu_y
    assert (pred.mean(0) - Y.mean(0)).abs().max() < 1e-2


def test_regularization_shrinks_the_solution():
    src, tgt, _ = synthetic(n=256)
    W_small, *_ = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=1e-4)
    W_large, *_ = ridge_streaming(src, tgt, chosen=[0, 1, 2], lam=10.0)
    assert W_large.norm() < W_small.norm()
