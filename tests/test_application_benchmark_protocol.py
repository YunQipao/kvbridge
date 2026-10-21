from experiments.benchmark_transfer import _stats


def test_application_stats_have_required_quantiles():
    assert set(_stats([1., 2., 3.])) >= {"median", "mean", "std", "p25", "p75"}


def test_application_benchmark_cpu_with_mock_model(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace
    import torch
    import experiments.benchmark_transfer as benchmark

    calls = []

    class Model:
        config = SimpleNamespace(vocab_size=16)

        def __call__(self, ids, use_cache):
            assert ids.device == torch.device("cpu")
            calls.append("prefill")
            return SimpleNamespace(past_key_values="cache")

    monkeypatch.setattr(benchmark, "load_causal_lm", lambda *a, **k: Model())
    monkeypatch.setattr(benchmark, "model_input_device", lambda model: torch.device("cpu"))
    monkeypatch.setattr(benchmark.Mapper, "load", lambda *a, **k: "mapper")
    monkeypatch.setattr(benchmark, "transfer", lambda *a: calls.append("map"))
    output = tmp_path / "latency.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--source", "mock", "--target", "mock",
        "--cachebridge-mapper", "mock", "--device", "cpu", "--prefix-lengths", "4",
        "--warmup", "2", "--repeat", "3", "--output", str(output)])
    benchmark.main()
    assert calls == ["prefill"] + ["map"] * 5
    assert json.loads(output.read_text())[0]["cachebridge_ms"]["median"] >= 0
