"""Manifest validation for dataset-free completed-trace resume."""

import json

import pytest

from kvbridge.formal import load_completed_trace_manifest


def _completed(tmp_path):
    traces = tmp_path / "traces"
    for stage in ("source", "target"):
        folder = traces / stage
        folder.mkdir(parents=True)
        (folder / f"{stage}_00000.pt").touch()
        (traces / f"{stage}_traces.done").touch()
    manifest = dict(dataset="local.txt", seed=42, n_calib=1, seq_len=64,
                    token_stride=4, num_boundaries=4, boundary_start_token=12,
                    token_lengths=[64], actual_observations=16,
                    indices=[0], sample_ids=[None], token_alignment_verified=True)
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    args = dict(dataset="local.txt", seed=42, n_calib=1, seq_len=64,
                token_stride=4, num_boundaries=4, boundary_start=12)
    return traces, manifest, args


def test_completed_resume_needs_no_tokenizer_or_dataset(tmp_path):
    traces, manifest, args = _completed(tmp_path)
    assert load_completed_trace_manifest(tmp_path, traces, **args) == manifest


def test_missing_shard_is_rejected(tmp_path):
    traces, _, args = _completed(tmp_path)
    (traces / "source" / "source_00000.pt").unlink()
    with pytest.raises(ValueError, match="source trace shards"):
        load_completed_trace_manifest(tmp_path, traces, **args)


@pytest.mark.parametrize("field,value", [("seed", 43), ("actual_observations", 99),
                                       ("token_alignment_verified", False)])
def test_invalid_manifest_is_rejected(tmp_path, field, value):
    traces, manifest, args = _completed(tmp_path)
    manifest[field] = value
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_completed_trace_manifest(tmp_path, traces, **args)


def test_legacy_resume_is_explicit_not_a_new_alignment_check(tmp_path, capsys):
    traces, manifest, args = _completed(tmp_path)
    del manifest["token_alignment_verified"]
    (tmp_path / "calibration_manifest.json").write_text(json.dumps(manifest))
    load_completed_trace_manifest(tmp_path, traces, **args)
    assert "NOT rechecking token IDs" in capsys.readouterr().out
