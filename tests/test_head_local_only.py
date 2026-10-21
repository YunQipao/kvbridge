from kvbridge.formal import fit_one_target_layer


def test_head_local_entry_point_exists():
    assert callable(fit_one_target_layer)

