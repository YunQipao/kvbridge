from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


DRIVER = Path(__file__).resolve().parents[1] / "scripts/run_minimal_cachebridge_validation.sh"


def _run_driver(tmp_path, stage, overrides=None):
    # Replace only the driver subprocess's Python entry point: no model/data loads.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.jsonl"
    shim = bin_dir / "python3"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DRIVER_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('DRIVER_TEST_FAIL_MODULE') in sys.argv[1:]:\n"
        "    sys.exit(23)\n"
    )
    shim.chmod(0o755)
    env = dict(os.environ)
    for name in ("PAIR_PROFILE", "HEAD_MAP_MODE", "TARGET_CAPTURE_MODE", "SOURCE_MODEL", "TARGET_MODEL", "TARGET_TOKENIZER", "DATASET",
                 "DATASET_FILE", "DEVICE", "DEVICE_MAP", "LOCAL_FILES_ONLY",
                 "N_CALIB", "SEQ_LEN", "TOKEN_STRIDE", "TOP_K", "RIDGE_LAMBDA",
                 "NUM_BOUNDARIES", "BOUNDARY_START", "EVAL_SAMPLES", "WARMUP",
                 "REPEAT", "SEED", "CALIBRATION_OUT", "FULL_HEAD_OUT",
                 "HEAD_LOCAL_OUT", "RESULTS_DIR", "MANIFEST_DIR"):
        env.pop(name, None)
    env.update({"PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
                "DRIVER_TEST_LOG": str(log),
                "RESULTS_DIR": str(tmp_path / "results"),
                "MANIFEST_DIR": str(tmp_path / "manifests")})
    env.update(overrides or {})
    result = subprocess.run(["bash", str(DRIVER), stage], env=env,
                            capture_output=True, text=True, check=False)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def _arg(call, name):
    return call[call.index(name) + 1]


@pytest.mark.parametrize("top_k", [None, "7"])
def test_ministral_profile_isolated_paths_and_explicit_overrides(tmp_path, top_k):
    overrides = {"PAIR_PROFILE": "paper_ministral_3b_14b", "N_CALIB": "2"}
    if top_k is not None:
        overrides["TOP_K"] = top_k
    result, calls = _run_driver(tmp_path, "fit", overrides)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    for call in calls:
        assert _arg(call, "--source").endswith("/Ministral-3-3B-Base-2512")
        assert _arg(call, "--target").endswith("/Ministral-3-14B-Base-2512")
        assert _arg(call, "--target-tokenizer") == _arg(call, "--target")
        assert _arg(call, "--top-k") == (top_k or "20")
        assert _arg(call, "--n-calib") == "2"
        assert _arg(call, "--seq-len") == "1024"
        assert _arg(call, "--num-boundaries") == "32"
        assert _arg(call, "--boundary-start") == "12"
        assert "ministral_3b_14b_n2" in _arg(call, "--output")
        assert "--local-files-only" in call


def test_ministral_profile_preserves_model_overrides(tmp_path):
    result, calls = _run_driver(tmp_path, "preflight", {
        "PAIR_PROFILE": "paper_ministral_3b_14b",
        "SOURCE_MODEL": "/local/source", "TARGET_MODEL": "/local/target"})
    assert result.returncode == 0, result.stderr
    assert all(_arg(call, "--source") == "/local/source" for call in calls)
    assert all(_arg(call, "--target") == "/local/target" for call in calls)


@pytest.mark.parametrize("small_pair", [False, True])
def test_preflight_defaults_use_fineweb_and_never_downgrade_small_pair(tmp_path, small_pair):
    overrides = ({"SOURCE_MODEL": "Qwen/Qwen3-1.7B", "TARGET_MODEL": "Qwen/Qwen3-4B"}
                 if small_pair else {})
    result, calls = _run_driver(tmp_path, "preflight", overrides)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert _arg(calls[0], "--source") == overrides.get("SOURCE_MODEL", "Qwen/Qwen3-14B")
    assert _arg(calls[0], "--target") == overrides.get("TARGET_MODEL", "Qwen/Qwen3-32B")
    alignment = calls[1]
    assert _arg(alignment, "--dataset") == "HuggingFaceFW/fineweb-edu"
    assert "--dataset-file" not in alignment
    assert _arg(alignment, "--n-calib") == "50"
    assert _arg(alignment, "--seq-len") == "1024"


def test_preflight_uses_explicit_local_calibration_file(tmp_path):
    calibration = tmp_path / "calibration with spaces.jsonl"
    calibration.write_text('{"text": "local unit test fixture"}\n')
    result, calls = _run_driver(tmp_path, "preflight", {"DATASET_FILE": str(calibration)})
    assert result.returncode == 0, result.stderr
    assert _arg(calls[1], "--dataset-file") == str(calibration)
    assert "--dataset" not in calls[1]


@pytest.mark.parametrize("stage", ["preflight", "fit"])
def test_missing_calibration_file_fails_before_python(tmp_path, stage):
    missing = tmp_path / "missing.jsonl"
    result, calls = _run_driver(tmp_path, stage, {"DATASET_FILE": str(missing)})
    assert result.returncode != 0
    assert f"Calibration file does not exist: {missing}" in result.stderr
    assert calls == []


def test_small_pair_fit_preserves_minimal_profile_and_shared_inputs(tmp_path):
    result, calls = _run_driver(tmp_path, "fit", {
        "SOURCE_MODEL": "Qwen/Qwen3-1.7B", "TARGET_MODEL": "Qwen/Qwen3-4B",
        "TARGET_TOKENIZER": "Qwen/Qwen3-4B"})
    assert result.returncode == 0, result.stderr
    assert [_arg(call, "--method") for call in calls] == ["cachebridge", "full_head", "head_local"]
    for call in calls:
        for name, value in {"--n-calib": "50", "--seq-len": "1024", "--top-k": "8",
                            "--num-boundaries": "32", "--token-stride": "4",
                            "--ridge-lambda": "0.01", "--boundary-start": "12",
                            "--source": "Qwen/Qwen3-1.7B", "--target": "Qwen/Qwen3-4B",
                            "--target-tokenizer": "Qwen/Qwen3-4B",
                            "--dataset": "HuggingFaceFW/fineweb-edu"}.items():
            assert _arg(call, name) == value
        assert "--dataset-file" not in call
        assert "--smoke-uniform-weights" not in call
    assert [_arg(call, "--output") for call in calls] == [
        "artifacts/minimal_cachebridge", "artifacts/minimal_full_head", "artifacts/minimal_head_local"]
    assert "--reuse-calibration-manifest" not in calls[0]
    for call in calls[1:]:
        assert _arg(call, "--reuse-calibration-manifest") == "artifacts/minimal_cachebridge/calibration_manifest.json"
        assert _arg(call, "--reuse-layer-selection") == "artifacts/minimal_cachebridge/layer_selection.json"


def test_eval_defaults_share_200_example_manifest(tmp_path):
    result, calls = _run_driver(tmp_path, "eval")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    manifest = _arg(calls[0], "--evaluation-manifest-output")
    for call in calls:
        assert _arg(call, "--num-eval-samples") == "200"
    assert all(_arg(call, "--evaluation-manifest") == manifest for call in calls[1:])


@pytest.mark.parametrize("device_map", ["auto", "balanced_low_0", "none"])
def test_eval_passes_device_map_to_all_four_modes(tmp_path, device_map):
    result, calls = _run_driver(tmp_path, "eval", {
        "DEVICE_MAP": device_map, "DEVICE": "cpu", "EVAL_SAMPLES": "500"})
    assert result.returncode == 0, result.stderr
    assert [_arg(call, "--mode") for call in calls] == [
        "target_native", "full_head", "head_local", "cachebridge"]
    assert all(_arg(call, "--device-map") == device_map for call in calls)
    assert all(_arg(call, "--device") == "cpu" for call in calls)
    assert all(_arg(call, "--num-eval-samples") == "500" for call in calls)


def test_explicit_profile_overrides_are_preserved(tmp_path):
    result, calls = _run_driver(tmp_path, "fit", {
        "N_CALIB": "100", "SEQ_LEN": "512", "TOP_K": "4",
        "TOKEN_STRIDE": "2", "RIDGE_LAMBDA": "0.02",
        "NUM_BOUNDARIES": "16", "BOUNDARY_START": "8"})
    assert result.returncode == 0, result.stderr
    for call in calls:
        for name, value in {"--n-calib": "100", "--seq-len": "512", "--top-k": "4",
                            "--token-stride": "2", "--ridge-lambda": "0.02",
                            "--num-boundaries": "16", "--boundary-start": "8"}.items():
            assert _arg(call, name) == value


def test_fit_passes_head_map_and_capture_modes(tmp_path):
    result, calls = _run_driver(tmp_path, "fit", {
        "HEAD_MAP_MODE": "block", "TARGET_CAPTURE_MODE": "boundary_chunked"})
    assert result.returncode == 0, result.stderr
    assert all(_arg(call, "--head-map-mode") == "block" for call in calls)
    assert all(_arg(call, "--target-capture-mode") == "boundary_chunked" for call in calls)


def test_head_map_mode_and_capture_mode_are_explicit_driver_controls():
    script = DRIVER.read_text()
    assert 'HEAD_MAP_MODE="${HEAD_MAP_MODE:-auto}"' in script
    assert 'TARGET_CAPTURE_MODE="${TARGET_CAPTURE_MODE:-full}"' in script
    assert "--head-map-mode \"$HEAD_MAP_MODE\"" in script
    assert "--target-capture-mode \"$TARGET_CAPTURE_MODE\"" in script


def test_fit_failure_stops_stage_without_eval_or_summary(tmp_path):
    script = DRIVER.read_text()
    assert "set -euo pipefail" in script
    assert "all) preflight; fit; eval_stage; benchmark; summary ;;" in script
    result, calls = _run_driver(tmp_path, "all", {
        "DRIVER_TEST_FAIL_MODULE": "experiments.fit_cachebridge"})
    assert result.returncode == 23
    fits = [call for call in calls if "experiments.fit_cachebridge" in call]
    assert len(fits) == 1
    assert _arg(fits[0], "--method") == "cachebridge"
    assert not any("experiments.eval_paper_tasks" in call or
                   "experiments.summarize_minimal" in call or
                   "experiments/benchmark_transfer.py" in call for call in calls)


@pytest.mark.parametrize("device_map", ["auto", "balanced"])
def test_device_map_environment_reaches_all_fits(tmp_path, device_map):
    result, calls = _run_driver(tmp_path, "fit", {"DEVICE_MAP": device_map})
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert all(_arg(call, "--device-map") == device_map for call in calls)


def test_minimal_driver_exposes_required_stages_defaults_and_outputs():
    script = DRIVER.read_text()
    for stage in ("preflight", "fit", "eval", "benchmark", "summary", "all"):
        assert stage in script
    for value in ("N_CALIB", "SEQ_LEN", "TOKEN_STRIDE", "TOP_K",
                  "RIDGE_LAMBDA", "NUM_BOUNDARIES", "BOUNDARY_START",
                  "EVAL_SAMPLES", "artifacts/minimal_full_head",
                  "artifacts/minimal_head_local", "artifacts/minimal_cachebridge",
                  "MANIFEST_DIR"):
        assert value in script
    assert 'RESULTS_DIR="${RESULTS_DIR:-results/minimal}"' in script
    assert 'MANIFEST_DIR="${MANIFEST_DIR:-results/manifests}"' in script
    assert '"$MANIFEST_DIR/hellaswag.json"' in script
    assert '"$RESULTS_DIR/hellaswag_${mode}.json"' in script
    assert '"$RESULTS_DIR/hellaswag_target_native.json"' in script
    assert 'TARGET_TOKENIZER="${TARGET_TOKENIZER:-$SOURCE_MODEL}"' in script
    assert '--target-tokenizer "$TARGET_TOKENIZER"' in script
    assert "--reuse-calibration-manifest" in script
    assert "--reuse-layer-selection" in script
    assert 'fit_one cachebridge "$CALIBRATION_OUT"' in script
    assert 'fit_one full_head "$FULL_HEAD_OUT"' in script
    assert 'fit_one head_local "$HEAD_LOCAL_OUT"' in script
    assert '"$RIDGE_LAMBDA" --token-stride "$TOKEN_STRIDE"' in script


def test_sweep_creates_task_specific_manifests():
    script = (DRIVER.parent / "run_calibration_sweep.sh").read_text()
    assert "results/manifests/" in script
    assert 'task_slug="${task,,}"' in script
    assert '--evaluation-manifest "$manifest"' in script
