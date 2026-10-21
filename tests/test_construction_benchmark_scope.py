from experiments.benchmark_construction import SCOPE


def test_formal_scope_contains_only_requested_phases():
    assert SCOPE == "weight_staging+sufficient_statistics+solve+metadata+serialization"

