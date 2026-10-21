import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from experiments.summarize_minimal import build_summary
from kvbridge.formal import fit_one_target_layer
from kvbridge.protocol import validate_evaluation_manifest


def test_each_task_requires_its_own_population_manifest():
    hellaswag = {"population_size": 100, "num_samples": 2, "indices": [0, 50]}
    arc = {"population_size": 50, "num_samples": 2, "indices": [0, 25]}
    assert validate_evaluation_manifest(hellaswag, 100) == [0, 50]
    with pytest.raises(ValueError, match="population size"):
        validate_evaluation_manifest(hellaswag, 50)
    assert validate_evaluation_manifest(arc, 50) == [0, 25]


def test_summary_retention_and_deltas(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    scores = {
        "target_native": 0.8,
        "full_head": 0.6,
        "head_local": 0.7,
        "cachebridge": 0.75,
    }
    for method, accuracy in scores.items():
        (results / f"hellaswag_{method}.json").write_text(
            json.dumps({"accuracy": accuracy}))
    artifacts = {}
    for method in ("full_head", "head_local", "cachebridge"):
        path = tmp_path / method
        path.mkdir()
        (path / "mapper.safetensors").write_bytes(b"x" * (10 if method == "cachebridge" else 20))
        artifacts[method] = str(path)
    benchmark = results / "transfer_benchmark.json"
    benchmark.write_text(json.dumps([
        {"prefix": n, "full_head_ms": {"median": 4.0},
         "cachebridge_ms": {"median": 2.0}}
        for n in (128, 512, 1024)
    ]))
    summary = build_summary("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", results,
                            artifacts, benchmark)
    assert summary["paper_pair"] is False
    assert summary["reduced_scale"] is True
    assert summary["retention"]["cachebridge_retention"] == pytest.approx(0.75 / 0.8)
    assert summary["delta"]["cachebridge_delta_vs_head_local"] == pytest.approx(0.05)
    assert summary["efficiency"]["full_head_size_over_cachebridge_size"] == 2
    assert summary["full_head_size_over_cachebridge_size"] == 2
    assert set(summary["mapping_latency_ms_median"]) == {"128", "512", "1024"}
    assert summary["mapping_latency_ms_median"]["512"] == {
        "full_head_ms": 4.0, "cachebridge_ms": 2.0}
    assert summary["latency_scope"] == (
        "mapper application only; excludes source prefill, transport, and decode")
    assert summary["fused_fit_paper_kernel_reproduced"] is False


def test_summary_rejects_incomplete_required_latency_prefixes(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for method in ("target_native", "full_head", "head_local", "cachebridge"):
        (results / f"hellaswag_{method}.json").write_text(json.dumps({"accuracy": 0.5}))
    artifacts = {}
    for method in ("full_head", "head_local", "cachebridge"):
        path = tmp_path / method
        path.mkdir()
        (path / "mapper.safetensors").write_bytes(b"artifact")
        artifacts[method] = str(path)
    benchmark = results / "transfer_benchmark.json"
    benchmark.write_text(json.dumps([{"prefix": 128,
        "full_head_ms": {"median": 1.0}, "cachebridge_ms": {"median": 0.5}}]))
    with pytest.raises(ValueError, match="missing required prefixes"):
        build_summary("source", "target", results, artifacts, benchmark)


def _write_trace_pair(root, weights):
    source = root / "source"
    target = root / "target"
    source.mkdir()
    target.mkdir()
    source_tensor = torch.tensor([[[0.0], [1.0]], [[1.0], [2.0]], [[3.0], [4.0]]])
    target_tensor = torch.tensor([[[0.0]], [[2.0]], [[0.0]]])
    torch.save({"keys": [source_tensor], "values": [source_tensor]},
               source / "source_00000.pt")
    torch.save({
        "keys": [target_tensor], "values": [target_tensor],
        "weights_k": [torch.tensor(weights).reshape(1, -1)],
        "weights_v": [torch.tensor(weights).reshape(1, -1)],
    }, target / "target_00000.pt")
    return source, target


def test_uniform_smoke_weights_control_fitting_weights(tmp_path):
    source, target = _write_trace_pair(tmp_path, [1.0, 0.01, 0.01])
    weighted = fit_one_target_layer(
        source, target, 0, [0], 1, [0], 1, ridge_lambda=0,
        stats_dtype=torch.float64, uniform_weights=False)
    uniform = fit_one_target_layer(
        source, target, 0, [0], 1, [0], 1, ridge_lambda=0,
        stats_dtype=torch.float64, uniform_weights=True)
    assert not torch.allclose(weighted["mu_xk"], uniform["mu_xk"])
