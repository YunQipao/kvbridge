import torch

from kvbridge.formal import BOUNDARY_INDEXING_DESCRIPTION


def test_boundary_three_uses_prefix_positions_before_three():
    tokens = torch.arange(6)
    boundary = 3
    assert boundary == 3
    assert tokens[tokens < boundary].tolist() == [0, 1, 2]
    assert tokens[tokens >= boundary].tolist() == [3, 4, 5]
    assert "first-future-query index" in BOUNDARY_INDEXING_DESCRIPTION
    assert "eligible prefix positions are < c" in BOUNDARY_INDEXING_DESCRIPTION
