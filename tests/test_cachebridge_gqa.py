import torch

from kvbridge.cachebridge import compute_gqa_sensitivities


def test_gqa_sensitivity_aggregates_query_heads_by_kv_owner():
    probs = torch.tensor(
        [[[[1.0, 0.0], [0.5, 0.5]],
          [[1.0, 0.0], [0.25, 0.75]],
          [[0.0, 1.0], [0.5, 0.5]],
          [[0.0, 1.0], [0.75, 0.25]]]]
    )
    q = torch.ones(1, 4, 2, 2)
    v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]],
                       [[2.0, 0.0], [0.0, 2.0]]]])
    causal = torch.ones_like(probs, dtype=torch.bool)
    causal[:, :, 0, 1] = False
    wk, wv = compute_gqa_sensitivities(probs, q, v, causal_mask=causal)
    assert wk.shape == wv.shape == (1, 2, 2)
    assert torch.isfinite(wk).all() and torch.isfinite(wv).all()
    assert (wk >= 0).all() and (wv >= 0).all()
    assert not torch.allclose(wk[:, 0], wk[:, 1])
