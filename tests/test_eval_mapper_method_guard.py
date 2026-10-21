import pytest
from kvbridge.protocol import assert_mapper_method


class M:
    def __init__(self, method):
        self.meta = {"method": method}


def test_cross_method_artifacts_are_rejected():
    with pytest.raises(ValueError):
        assert_mapper_method(M("cachebridge"), "full_head")
    with pytest.raises(ValueError):
        assert_mapper_method(M("full_head"), "cachebridge")
    with pytest.raises(ValueError):
        assert_mapper_method(M("full_head"), "head_local")
    with pytest.raises(ValueError):
        assert_mapper_method(M("head_local"), "cachebridge")
