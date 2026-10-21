from experiments.benchmark_construction import PROVENANCE, SCOPE


def test_formal_construction_scope_and_provenance():
    assert "sufficient_statistics" in SCOPE
    assert "serialization" in SCOPE
    assert PROVENANCE == "engineering_equivalent"
