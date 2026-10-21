from kvbridge.protocol import (make_evaluation_manifest, systematic_indices,
                               validate_evaluation_manifest)


def test_manifest_is_reusable_and_deterministic():
    a = make_evaluation_manifest(17, 5, 123)
    b = make_evaluation_manifest(17, 5, 999)
    for field in ("indices", "population_size", "num_samples", "sampling"):
        assert a[field] == b[field]
    assert a["seed"] == 123
    assert b["seed"] == 999
    assert a["seed_affects_selection"] is False
    assert b["seed_affects_selection"] is False
    assert a["indices"] == systematic_indices(17, 5)


def test_all_evaluation_modes_can_consume_the_same_manifest():
    manifest = make_evaluation_manifest(23, 7, 123)
    for _mode in ("target_native", "full_head", "head_local", "cachebridge"):
        assert validate_evaluation_manifest(manifest, 23) == manifest["indices"]
