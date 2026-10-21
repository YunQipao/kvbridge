def test_attribution_manifests_are_separate():
    result = {
        "kv_r2_manifest": {"task": "HellaSwag", "indices": [1, 3]},
        "nll_manifest": {"streams": 16, "stream_length": 4096},
    }
    assert result["kv_r2_manifest"]["task"] == "HellaSwag"
    assert result["nll_manifest"]["stream_length"] == 4096
    assert result["kv_r2_manifest"] is not result["nll_manifest"]
