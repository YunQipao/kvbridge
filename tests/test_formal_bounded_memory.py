import torch

from kvbridge.formal import fit_one_target_layer


def _traces(root, n):
    (root / "source").mkdir(parents=True)
    (root / "target").mkdir(parents=True)
    for i in range(n):
        source = {"keys": [torch.randn(9, 2, 3), torch.randn(9, 2, 3)],
                  "values": [torch.randn(9, 2, 3), torch.randn(9, 2, 3)]}
        target = {"keys": [torch.randn(9, 2, 3)], "values": [torch.randn(9, 2, 3)],
                  "weights_k": [torch.rand(2, 9)], "weights_v": [torch.rand(2, 9)]}
        torch.save(source, root / "source" / f"source_{i:05d}.pt")
        torch.save(target, root / "target" / f"target_{i:05d}.pt")


def test_fused_panel_shape_is_independent_of_sequence_count(tmp_path):
    small, large = tmp_path / "small", tmp_path / "large"
    _traces(small, 2)
    _traces(large, 8)

    def run(root):
        stats = {}
        fit_one_target_layer(root / "source", root / "target", 0, [0, 1], 2,
                             [0, 1], 3, backend="fused_equivalent",
                             chunk_size=4, debug_stats=stats)
        return stats

    assert run(small)["max_x_panel"] == run(large)["max_x_panel"]
    assert run(small)["max_y_panel"] == run(large)["max_y_panel"]
