import pytest
import torch

from kvbridge.cachebridge import (
    weighted_ridge_centered,
    weighted_ridge_streaming,
)
from kvbridge.fit import ridge_centered


def random_problem(seed=7, n=31, p=5, q=3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, p, generator=g)
    Y = torch.randn(n, q, generator=g)
    weights = torch.rand(n, generator=g) + 0.1
    return X, Y, weights


def test_streaming_matches_direct_weighted_ridge():
    X, Y, weights = random_problem()
    direct = weighted_ridge_centered(X, Y, weights, ridge_lambda=0.01)
    batches = [(X[:11], Y[:11], weights[:11]),
               (X[11:23], Y[11:23], weights[11:23]),
               (X[23:], Y[23:], weights[23:])]
    streamed = weighted_ridge_streaming(batches, X.shape[1], Y.shape[1], 0.01)
    for a, b in zip(streamed, direct):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-8)


def test_uniform_weights_match_the_existing_centered_solver_with_same_objective():
    X, Y, _ = random_problem(n=40)
    # The old solver normalizes its covariance by N.  Scale lambda in the new
    # unnormalized objective to represent the same regularization strength.
    W_new, bias_new, mx, my = weighted_ridge_centered(X, Y, torch.ones(40), 0.01 * 40)
    W_old, mx_old, my_old = ridge_centered(X, Y, lam=0.01)
    assert torch.allclose(W_new.float(), W_old, atol=1e-6, rtol=1e-5)
    assert torch.allclose(mx.float(), mx_old, atol=1e-6)
    assert torch.allclose(my.float(), my_old, atol=1e-6)
    assert torch.allclose(bias_new, my - mx @ W_new, atol=1e-10)


def test_zero_weight_rows_do_not_change_result():
    X, Y, weights = random_problem()
    base = weighted_ridge_centered(X, Y, weights, 0.2)
    padded = weighted_ridge_centered(
        torch.cat([X, torch.randn(4, X.shape[1])]),
        torch.cat([Y, torch.randn(4, Y.shape[1])]),
        torch.cat([weights, torch.zeros(4)]), 0.2)
    for a, b in zip(base, padded):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-8)


def test_invalid_weights_and_shapes_fail():
    X, Y, weights = random_problem(n=5)
    with pytest.raises(ValueError, match="non-negative"):
        weighted_ridge_centered(X, Y, weights.clone().fill_(-1))
    with pytest.raises(ValueError, match="positive"):
        weighted_ridge_centered(X, Y, torch.zeros(5))
    with pytest.raises(ValueError, match="same number"):
        weighted_ridge_centered(X, Y[:4], weights)
    with pytest.raises(ValueError, match="shape"):
        weighted_ridge_centered(X, Y, weights[:2])


def test_streaming_shape_mismatch_fails():
    X, Y, weights = random_problem(n=5)
    with pytest.raises(ValueError, match="dimensions"):
        weighted_ridge_streaming([(X, Y, weights)], 4, Y.shape[1])
