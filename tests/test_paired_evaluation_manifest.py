from kvbridge.protocol import make_evaluation_manifest, validate_evaluation_manifest


def test_all_modes_can_consume_one_manifest():
    manifest = make_evaluation_manifest(17, 5, 999, "systematic")
    consumed = [
        validate_evaluation_manifest(manifest, 17)
        for _ in ("target_native", "full_head", "head_local", "cachebridge")
    ]
    assert consumed[0] == consumed[1] == consumed[2] == consumed[3]
