"""Offline tokenizer resolution without Hub downloads or real models."""

import pytest

from kvbridge import model_utils


def test_offline_hub_tokenizer_uses_snapshot_directory(tmp_path, monkeypatch):
    calls = []

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    def tokenizer(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs["local_files_only"] is True
        return "local tokenizer"

    monkeypatch.setattr(model_utils, "snapshot_download", snapshot)
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained", tokenizer)
    result = model_utils.load_tokenizer("Qwen/Qwen3-1.7B", revision="commit",
                                       local_files_only=True, cache_dir="/cache")
    assert result == "local tokenizer"
    assert calls == [("Qwen/Qwen3-1.7B",
                      dict(revision="commit", cache_dir="/cache", local_files_only=True))]


def test_local_tokenizer_does_not_call_snapshot(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("local path must not call Hub")

    monkeypatch.setattr(model_utils, "snapshot_download", forbidden)
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained",
                        lambda path, **kwargs: path)
    assert model_utils.load_tokenizer(tmp_path, local_files_only=True) == str(tmp_path)


def test_missing_snapshot_fails_without_online_retry(monkeypatch):
    calls = []

    def missing(*args, **kwargs):
        calls.append(kwargs)
        raise model_utils.LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(model_utils, "snapshot_download", missing)
    with pytest.raises(FileNotFoundError, match="no cached tokenizer snapshot"):
        model_utils.load_tokenizer("missing/model", local_files_only=True)
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True


def test_offline_environment_resolves_hub_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_utils, "snapshot_download",
                        lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(model_utils.AutoTokenizer, "from_pretrained",
                        lambda path, **kwargs: (path, kwargs["local_files_only"]))
    assert model_utils.load_tokenizer("Qwen/Qwen3-1.7B") == (str(tmp_path), True)


def test_missing_absolute_directory_is_not_a_hub_id(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        model_utils.local_tokenizer_reference(tmp_path / "missing")
