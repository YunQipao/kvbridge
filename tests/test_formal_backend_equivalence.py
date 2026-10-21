import torch

from kvbridge.formal import fit_one_target_layer


def _write_traces(root, n=3):
    source = root / "source"
    target = root / "target"
    source.mkdir()
    target.mkdir()
    torch.manual_seed(7)
    for i in range(n):
        xk = [torch.randn(8, 2, 3), torch.randn(8, 2, 3)]
        xv = [torch.randn(8, 2, 3), torch.randn(8, 2, 3)]
        yk = [torch.randn(8, 2, 3), torch.randn(8, 2, 3)]
        yv = [torch.randn(8, 2, 3), torch.randn(8, 2, 3)]
        wk = [torch.rand(2, 8), torch.rand(2, 8)]
        wv = [torch.rand(2, 8), torch.rand(2, 8)]
        torch.save({"keys": xk, "values": xv}, source / f"source_{i:05d}.pt")
        torch.save({"keys": yk, "values": yv, "weights_k": wk,
                    "weights_v": wv}, target / f"target_{i:05d}.pt")


def test_formal_backends_match_on_staged_traces(tmp_path):
    _write_traces(tmp_path)
    kwargs = dict(source_dir=tmp_path / "source", target_dir=tmp_path / "target",
                  layer_index=0, selected=[0, 1], n_heads=2,
                  head_assignment=[0, 1], head_dim=3, ridge_lambda=0.01)
    reference = fit_one_target_layer(**kwargs, backend="reference")
    bounded = fit_one_target_layer(**kwargs, backend="torch_bounded")
    fused = fit_one_target_layer(**kwargs, backend="fused_equivalent", chunk_size=3)
    for key in ("W_k", "W_v", "mu_xk", "mu_xv", "mu_yk", "mu_yv"):
        assert torch.allclose(reference[key], bounded[key], atol=1e-5, rtol=1e-5)
        assert torch.allclose(reference[key], fused[key], atol=1e-5, rtol=1e-5)
