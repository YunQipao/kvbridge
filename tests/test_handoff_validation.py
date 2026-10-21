from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from scripts.test_handoff import validate_mapped_cache


def cache(layers=2, heads=3, tokens=5, dim=4, dtype=torch.float32):
    result = DynamicCache()
    for layer in range(layers):
        tensor = torch.zeros(1, heads, tokens, dim, dtype=dtype)
        result.update(tensor.clone(), tensor.clone(), layer)
    return result


@pytest.fixture
def mapper():
    return SimpleNamespace(meta={"num_target_layers": 2,
                                 "num_target_kv_heads": 3, "head_dim": 4})


def test_valid_mapped_cache(mapper):
    validate_mapped_cache(cache(), mapper, 5, torch.float32)


@pytest.mark.parametrize("kwargs", [
    {"layers": 1}, {"heads": 2}, {"tokens": 4}, {"dim": 8},
    {"dtype": torch.float64},
])
def test_rejects_incorrect_cache_geometry(mapper, kwargs):
    with pytest.raises(RuntimeError, match="mapped"):
        validate_mapped_cache(cache(**kwargs), mapper, 5, torch.float32)


def test_rejects_nonfinite_cache(mapper):
    result = cache()
    result.layers[1].values[0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_mapped_cache(result, mapper, 5, torch.float32)
