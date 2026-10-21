import torch
from kvbridge.formal import fit_one_target_layer_full_head


def test_target_heads_have_independent_maps(tmp_path):
    src = {"keys": [torch.randn(4, 2, 2)], "values": [torch.randn(4, 2, 2)]}
    # Head 0 and head 1 are deliberately different affine targets.
    x = src["keys"][0]
    y = torch.stack([x[:, 0] * 2, x[:, 1] * -3], dim=1)
    v = torch.stack([src["values"][0][:, 0] + 1, src["values"][0][:, 1] - 2], dim=1)
    (tmp_path / "src").mkdir(); (tmp_path / "tgt").mkdir()
    torch.save(src, tmp_path / "src/source_00000.pt")
    torch.save({"keys": [y], "values": [v]}, tmp_path / "tgt/target_00000.pt")
    got = fit_one_target_layer_full_head(tmp_path / "src", tmp_path / "tgt", 0, [0], 2, 2,
                                         ridge_lambda=0.0, stats_dtype=torch.float64)
    assert got["W_k"].shape == (2, 4, 2)
    assert not torch.allclose(got["W_k"][0], got["W_k"][1])

