from kvbridge.cachebridge import make_log_causal_boundaries, make_observation_indices


def test_short_boundary_count_is_explicit_and_in_bounds():
    result = make_log_causal_boundaries(20, 32, 12)
    assert result == sorted(set(result))
    assert all(12 <= value < 20 for value in result)


def test_observation_stride_has_stable_phase_and_padding():
    mask = [True, True, False, True, True, False, True, True]
    result = make_observation_indices(8, stride=4, valid_mask=__import__("torch").tensor(mask))
    assert result.tolist() == [0, 4]
