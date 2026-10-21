from types import SimpleNamespace

import pytest

from kvbridge.geometry import check_kv_compatibility, geometry_dict, head_dim


def config(**overrides):
    values = dict(num_hidden_layers=4, hidden_size=1024,
                  num_attention_heads=8, num_key_value_heads=4,
                  head_dim=128, rope_theta=10000.0, rope_scaling=None,
                  torch_dtype="bfloat16")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_geometry_and_identity_assignment():
    compatible, source, target = check_kv_compatibility(config(), config())
    assert compatible
    assert source["num_key_value_heads"] == target["num_key_value_heads"] == 4
    assert list(range(source["num_key_value_heads"])) == [0, 1, 2, 3]


@pytest.mark.parametrize("changes", [
    {"num_key_value_heads": 8},
    {"head_dim": 64},
])
def test_incompatible_geometry_is_rejected(changes):
    compatible, _, _ = check_kv_compatibility(config(), config(**changes))
    assert not compatible


def test_head_dim_fallback():
    assert head_dim(config(head_dim=None)) == 128
    assert geometry_dict(config())["rope_theta"] == 10000.0
