from experiments.benchmark_construction import main


def test_scope_is_documented():
    assert "serialization" in "weight_staging+sufficient_statistics+solve+metadata+serialization"

