from inspect import signature

from kvbridge.formal import fit_one_target_layer, fit_one_target_layer_full_head


def test_formal_fitters_expose_backend_dispatch():
    assert "backend" in signature(fit_one_target_layer).parameters
    assert "backend" in signature(fit_one_target_layer_full_head).parameters
    assert "max_sequences" in signature(fit_one_target_layer).parameters
    assert "debug_stats" in signature(fit_one_target_layer_full_head).parameters
