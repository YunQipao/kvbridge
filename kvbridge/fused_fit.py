"""Construction backends for weighted centered ridge.

The fused-equivalent path follows the public two-pass/batched-GEMM algorithm.
It is not the authors' private CUDA kernel.
"""
from __future__ import annotations

import torch

from .cachebridge import WeightedRidgeStats, weighted_ridge_centered


def _panel_shape(X, Y, weights):
    if X.ndim == 2 and Y.ndim == 2:
        if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
            raise ValueError("weights must have shape [N]")
        return 1, X.shape[0], X.shape[1], Y.shape[1]
    if X.ndim != 3 or Y.ndim != 3 or X.shape[:2] != Y.shape[:2]:
        raise ValueError("panels must have [N,H,P] and [N,H,D]")
    if weights.ndim not in (1, 2) or weights.shape[0] != X.shape[0]:
        raise ValueError("weights must have shape [N] or [N,H]")
    if weights.ndim == 2 and weights.shape[1] != X.shape[1]:
        raise ValueError("per-head weights must have shape [N,H]")
    return X.shape[1], X.shape[0], X.shape[2], Y.shape[2]


def _head_weights(weights, head):
    return weights if weights.ndim == 1 else weights[:, head]


def _result_from_stats(stats, ridge_lambda):
    solved = [s.finalize(ridge_lambda) for s in stats]
    return tuple(torch.stack([x[i] for x in solved]) for i in range(4))


def fit_reference(X, Y, weights, ridge_lambda=0.01, chunk_size=None):
    """Materialized/direct correctness backend."""
    heads, _, _, _ = _panel_shape(X, Y, weights)
    if X.ndim == 2:
        return weighted_ridge_centered(X, Y, weights, ridge_lambda)
    return tuple(torch.stack([
        weighted_ridge_centered(X[:, h], Y[:, h], _head_weights(weights, h),
                                ridge_lambda)[i]
        for h in range(heads)
    ]) for i in range(4))


def fit_torch_bounded(X, Y, weights, ridge_lambda=0.01, chunk_size=1024):
    """Streaming sufficient-statistics backend with bounded observation chunks."""
    heads, n, p, d = _panel_shape(X, Y, weights)
    stats = [WeightedRidgeStats(p, d, dtype=torch.float64) for _ in range(heads)]
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        for h in range(heads):
            x = X[start:stop] if X.ndim == 2 else X[start:stop, h]
            y = Y[start:stop] if Y.ndim == 2 else Y[start:stop, h]
            stats[h].add_batch(x, y, _head_weights(weights, h)[start:stop])
    result = _result_from_stats(stats, ridge_lambda)
    if X.ndim == 2:
        return tuple(item[0] for item in result)
    return result


def fit_fused_equivalent(X, Y, weights, ridge_lambda=0.01, chunk_size=1024):
    """Batched weighted means and head-batched GEMM over bounded chunks."""
    original_rank = X.ndim
    heads, n, p, d = _panel_shape(X, Y, weights)
    if original_rank == 2:
        X, Y = X[:, None, :], Y[:, None, :]
    w = weights[:, None] if weights.ndim == 1 else weights
    X = X.to(dtype=torch.float64)
    Y = Y.to(dtype=torch.float64)
    w = w.to(dtype=torch.float64, device=X.device)
    total = w.sum(dim=0).clamp_min(torch.finfo(w.dtype).tiny)
    mu_x = (X * w[:, :, None]).sum(dim=0) / total[:, None]
    mu_y = (Y * w[:, :, None]).sum(dim=0) / total[:, None]
    A = torch.zeros(heads, p, p, dtype=torch.float64, device=X.device)
    B = torch.zeros(heads, p, d, dtype=torch.float64, device=X.device)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        xc = X[start:stop] - mu_x[None]
        yc = Y[start:stop] - mu_y[None]
        sqrt_w = w[start:stop, :, None].sqrt()
        xh = (xc * sqrt_w).permute(1, 0, 2).contiguous()
        yh = (yc * sqrt_w).permute(1, 0, 2).contiguous()
        A += torch.bmm(xh.transpose(1, 2), xh)
        B += torch.bmm(xh.transpose(1, 2), yh)
    eye = torch.eye(p, dtype=A.dtype, device=A.device)[None]
    W = torch.linalg.solve(A + ridge_lambda * eye, B)
    bias = mu_y - torch.bmm(mu_x[:, None, :], W).squeeze(1)
    if original_rank == 2:
        return W[0], bias[0], mu_x[0], mu_y[0]
    return W, bias, mu_x, mu_y


def fit_weighted_panels(X, Y, weights, ridge_lambda=0.01, *,
                        backend="fused_equivalent", chunk_size=1024):
    """Return only W while retaining the historical helper API."""
    fn = {
        "reference": fit_reference,
        "torch": fit_torch_bounded,
        "torch_bounded": fit_torch_bounded,
        "fused_equivalent": fit_fused_equivalent,
    }.get(backend)
    if fn is None:
        raise ValueError(f"unknown construction backend: {backend}")
    return fn(X, Y, weights, ridge_lambda, chunk_size=chunk_size)[0]


def fit_reference_and_fused(X, Y, weights, ridge_lambda=0.01, chunk_size=1024):
    return (fit_reference(X, Y, weights, ridge_lambda)[0],
            fit_fused_equivalent(X, Y, weights, ridge_lambda, chunk_size)[0])


def fused_scratch_shape(num_heads, chunk_size, feature_width, output_width):
    """Largest panel shape; it does not depend on total observation count."""
    return {
        "x_panel": (int(num_heads), int(chunk_size), int(feature_width)),
        "y_panel": (int(num_heads), int(chunk_size), int(output_width)),
    }
