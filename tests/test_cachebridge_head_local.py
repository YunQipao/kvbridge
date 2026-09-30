import pytest
import torch

from kvbridge.cachebridge import (
    build_head_local_features,
    validate_head_assignment,
)


def test_identity_assignment_reads_only_matching_head_and_stable_layers():
    # Values encode layer/head so both selection and concatenation order are visible.
    source = torch.zeros(4, 2, 3, 2)
    for layer in range(4):
        for head in range(3):
            source[layer, :, head] = 10 * layer + head
    got = build_head_local_features(source, [2, 0], target_head=1,
                                    head_assignment=[0, 1, 2])
    assert got.shape == (2, 4)
    assert torch.equal(got, torch.tensor([[21., 21., 1., 1.], [21., 21., 1., 1.]]))


def test_other_source_heads_are_not_in_feature():
    source = torch.arange(1 * 1 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 2)
    got = build_head_local_features(source, [0], target_head=0,
                                    head_assignment=[2, 0, 1])
    assert torch.equal(got, source[0, :, 2, :])


def test_invalid_assignment_and_head_counts_fail():
    with pytest.raises(ValueError, match="invalid source KV head"):
        validate_head_assignment([3], n_source_heads=3, n_target_heads=1)
    with pytest.raises(ValueError, match="requires an explicit assignment"):
        validate_head_assignment(None, n_source_heads=2, n_target_heads=3)
    with pytest.raises(ValueError, match="expected 2"):
        validate_head_assignment([0], n_source_heads=2, n_target_heads=2)


def test_invalid_layer_or_rank_fails():
    source = torch.zeros(2, 1, 2, 4)
    with pytest.raises(ValueError, match="invalid layer"):
        build_head_local_features(source, [2], 0, [0, 1])
    with pytest.raises(ValueError, match="shape"):
        build_head_local_features(source[0], [0], 0, [0, 1])
