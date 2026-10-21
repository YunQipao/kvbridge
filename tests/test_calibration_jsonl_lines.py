"""Local JSONL parsing regressions; no models, network, or GPU required."""

import json

import pytest

from kvbridge.formal import load_calibration_texts


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029", "\u000b", "\u000c"])
def test_jsonl_preserves_unicode_separators(tmp_path, separator):
    path = tmp_path / "calibration.jsonl"
    texts = [f"before{separator}after", "second record"]
    path.write_text("\n".join(
        json.dumps({"id": i, "text": text}, ensure_ascii=False)
        for i, text in enumerate(texts)) + "\n", encoding="utf-8")

    records = load_calibration_texts(dataset_file=path, n_calib=2)

    assert [record["text"] for record in records] == texts
    assert [record["offset"] for record in records] == [0, 1]


def test_jsonl_reports_physical_line_for_invalid_json(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"text": "ok"}\n\n{"text": "unfinished\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:3, column"):
        load_calibration_texts(dataset_file=path, n_calib=2)


def test_jsonl_crlf_and_blank_lines_keep_offsets(tmp_path):
    path = tmp_path / "windows.jsonl"
    path.write_bytes(b'{"text":"first"}\r\n\r\n{"text":"last"}\r\n')
    records = load_calibration_texts(dataset_file=path, n_calib=2)
    assert [record["text"] for record in records] == ["first", "last"]
    assert [record["offset"] for record in records] == [0, 2]


@pytest.mark.parametrize("row", [{"text": None}, ["text"], {"other": "value"}])
def test_jsonl_requires_text_string(tmp_path, row):
    path = tmp_path / "invalid_schema.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected an object with a string text field"):
        load_calibration_texts(dataset_file=path, n_calib=1)
