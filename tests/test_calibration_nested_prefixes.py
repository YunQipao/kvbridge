def test_nested_prefixes():
    stream = list(range(500))
    assert stream[:50] == stream[:100][:50]
    assert stream[:200] == stream[:500][:200]

