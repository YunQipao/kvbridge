"""Offline cache adapter tests: no models, GPU or network."""
import json

import datasets
import pytest

from kvbridge.offline_dataset import cached_dataset_splits


def prepare(root, name, *, expected=1):
    folder = root / "owner___toy" / name / "0.0.0" / "revision"
    folder.mkdir(parents=True)
    (folder / "dataset_info.json").write_text(json.dumps({
        "config_name": name, "dataset_name": "toy",
        "splits": {"test": {"num_examples": expected}}}))
    (folder / "toy-test.arrow").touch()


def forbid_hub(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline loading must never call the Hub")
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    monkeypatch.setattr(datasets, "get_dataset_config_names", forbidden)


def test_cache_configs_sorted_and_no_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    prepare(tmp_path, "zoology")
    prepare(tmp_path, "algebra")
    forbid_hub(monkeypatch)
    monkeypatch.setattr(datasets.Dataset, "from_file", lambda path: [path])
    result = list(cached_dataset_splits("owner/toy", "test"))
    assert [name for name, _ in result] == ["algebra", "zoology"]
    assert [name for name, _ in cached_dataset_splits("owner/toy", "test", "zoology")] == ["zoology"]


def test_missing_cache_reports_location(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    forbid_hub(monkeypatch)
    with pytest.raises(FileNotFoundError, match="No prepared dataset cache"):
        list(cached_dataset_splits("owner/toy", "test"))


def test_partial_cache_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    prepare(tmp_path, "algebra", expected=2)
    monkeypatch.setattr(datasets.Dataset, "from_file", lambda path: [path])
    with pytest.raises(RuntimeError, match="Incomplete cached split"):
        list(cached_dataset_splits("owner/toy", "test"))


def test_rows_offline_flag_avoids_config_discovery(monkeypatch):
    from experiments.eval_mmlu_redux import _rows
    import kvbridge.offline_dataset as offline
    forbid_hub(monkeypatch)
    calls = []
    def cached(dataset, split, config):
        calls.append((dataset, split, config))
        return [("algebra", [{"question": "Q", "choices": ["X", "Y"], "answer": 0}])]
    monkeypatch.setattr(offline, "cached_dataset_splits", cached)
    assert len(_rows("owner/toy", "test", None, local_files_only=True)) == 1
    assert calls == [("owner/toy", "test", None)]
