from kvbridge.protocol import systematic_indices


def test_systematic_sampling_evenly_spaced():
    assert systematic_indices(10, 4) == [0, 2, 5, 7]

