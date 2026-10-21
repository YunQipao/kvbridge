from kvbridge.protocol import cyclic_assignment


def test_cyclic_assignments():
    assert cyclic_assignment(8, 0) == list(range(8))
    assert cyclic_assignment(8, 7)[0] == 7

