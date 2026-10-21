"""CPU-only protocol tests; no models, network, or DistKV checkout required."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kvbridge.distkv_profiles import (PROMPT, choice_text, letter, prepare_samples,
                                     profile_rows, decode, continuation_mode)


@pytest.mark.parametrize("choices", [
    ["one", "two"], [{"text": "one"}, {"label": "B", "text": "two"}],
    {"label": ["A", "B"], "text": ["one", "two"]}, {"text": ["one", "two"]}])
def test_choices_match_reference(choices):
    assert choice_text(choices) == "A. one\nB. two"


def test_prompt_literal():
    expected = '''Accurately answer the following question:
Question?
Choices:
A. one
B. two
Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D/E".
- Do not include any explanations, additional text, or punctuation besides the answer.
The correct answer is'''
    assert PROMPT.format(question="Question?", choices="A. one\nB. two") == expected


@pytest.mark.parametrize("text,expected", [("The correct answer is C", "C"),
    ("(C)", "C"), ("The\ncorrect answer is (D)", "D"), ("0", "A"),
    ("4", "E"), ("5", None), ("A then the correct answer is C", "C"), ("none", None)])
def test_parser_reference(text, expected):
    assert letter(text) == expected


def dataset_rows():
    return [dict(question="Q", choices=["one", "two"], answer=0, correct_answer="B"),
            dict(question="excluded", choices=["one", "two"], answer=0,
                 error_type="NO CORRECT ANSWER"),
            dict(question="Q2", choices=["one", "two"], answer="B"),
            dict(question="bad label", choices=["one", "two"], answer="unknown")]


def test_raw_labels_and_original_indices():
    samples = prepare_samples(dataset_rows())
    assert [s["original_index"] for s in samples] == [0, 2]
    assert [s["answer"] for s in samples] == ["A", "B"]
    assert samples[0]["question"] == "Q"


def test_manifest_roundtrip_and_cross_profile(monkeypatch, tmp_path):
    import datasets
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: dataset_rows())
    rows, manifest = profile_rows(str(tmp_path), "distkv_canonical")
    saved = json.loads(json.dumps(manifest))
    exact, reused = profile_rows(str(tmp_path), "distkv_exact", saved)
    assert exact == rows and reused == manifest
    assert [s["original_index"] for s in manifest["samples"]] == [0, 2]
    assert manifest["answer_field"] == "answer"
    saved["samples"].reverse()
    with pytest.raises(ValueError, match="sample order/content"):
        profile_rows(str(tmp_path), "distkv_exact", saved)


def test_cached_groups_provenance_first_and_no_hub(monkeypatch):
    import datasets
    import kvbridge.offline_dataset as offline
    def forbidden(*args, **kwargs):
        raise AssertionError("cached profiles must not discover configs or load from Hub/disk")
    monkeypatch.setattr(datasets, "load_from_disk", forbidden)
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    monkeypatch.setattr(datasets, "get_dataset_config_names", forbidden)
    seen = []
    def cached(dataset, split):
        seen.append((dataset, split))
        return [("zoology", dataset_rows()), ("algebra", dataset_rows())]
    monkeypatch.setattr(offline, "cached_dataset_splits", cached)
    rows, manifest = profile_rows(None, "distkv_exact", count=3, sampling="first")
    assert manifest["dataset_loader"] == "hf_cached_configs"
    assert manifest["config_order"] == ["algebra", "zoology"]
    assert manifest["indices"] == [0, 1, 2]
    assert [s["sample_id"] for s in manifest["samples"]] == ["algebra:0", "algebra:2", "zoology:0"]
    assert [s["row_index"] for s in manifest["samples"]] == [0, 2, 0]
    assert [s["answer"] for s in manifest["samples"]] == ["A", "B", "A"]
    assert rows[0]["subject"] == "algebra"
    assert seen == [("edinburgh-dawg/mmlu-redux-2.0", "test")]
    canonical, reused = profile_rows(None, "distkv_canonical", json.loads(json.dumps(manifest)))
    assert canonical == rows and reused == manifest
    _, systematic = profile_rows(None, "distkv_exact", count=2, sampling="systematic")
    assert systematic["indices"] == [0, 2]


def test_disk_metadata_and_empty_path(monkeypatch, tmp_path):
    import datasets
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: dataset_rows())
    _, manifest = profile_rows(str(tmp_path), "distkv_exact", count=1, sampling="first")
    assert manifest["dataset_loader"] == "load_from_disk"
    assert manifest["samples"][0]["original_index"] == 0
    with pytest.raises(ValueError, match="empty"):
        profile_rows("", "distkv_exact")


def test_cached_reference_audit_uses_original_rule_functions(monkeypatch):
    import scripts.audit_distkv_protocol as audit
    reference = '''
BENCHMARK = {"local_path": "unused"}
def _choice_text(value):
    return "\\n".join(f"{chr(65+i)}. {text}" for i,text in enumerate(value))
def _letter(value):
    return chr(65+value) if isinstance(value,int) else (value if value in "ABCDE" else None)
def prepare_samples(limit):
    from datasets import load_from_disk
    rows = []
    for i,row in enumerate(load_from_disk(BENCHMARK["local_path"])):
        if str(row.get("error_type", "")).lower() == "no correct answer": continue
        answer = _letter(row["answer"])
        if answer: rows.append({"id":i,"prompt":row["question"],"answer":answer})
    return rows
'''
    monkeypatch.setattr(audit.subprocess, "check_output",
                        lambda args, **kw: reference if "show" in args else "commit")
    samples, commit = audit.reference_samples("unused", groups=[("algebra", dataset_rows())])
    assert [s["original_index"] for s in samples] == ["algebra:0", "algebra:2"]
    assert [s["answer"] for s in samples] == ["A", "B"]
    assert commit == "commit"


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[10, 11, 12, 13]]))

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "The correct answer is C"


class Model:
    generation_config = SimpleNamespace(eos_token_id=[99])

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, 67 if len(self.calls) == 1 else 99] = 1
        return SimpleNamespace(logits=logits, past_key_values="cache")


@pytest.mark.parametrize("mode,prefix", [("bridge_once", 3), ("distkv_duplicate_last", 4)])
def test_continuation_semantics(mode, prefix):
    model, seen = Model(), []
    def factory(ids):
        seen.append(ids.clone())
        return "cache"
    result = decode(model, Tokenizer(), "prompt", factory, "cpu", mode=mode)
    assert seen[0].tolist() == [[10, 11, 12, 13][:prefix]]
    assert model.calls[0][0].tolist() == [[13]]
    assert model.calls[0][1]["cache_position"].tolist() == [3]
    assert model.calls[1][1]["cache_position"].tolist() == [4]
    assert "attention_mask" not in model.calls[0][1]
    assert result == (2, "The correct answer is C", prefix, 2)
    # Native and mapped use exactly this same evaluator, not different token loops.


def test_current_defaults_unchanged():
    assert continuation_mode("current") == "bridge_once"
    assert continuation_mode("distkv_canonical") == "bridge_once"
    assert continuation_mode("distkv_exact") == "distkv_duplicate_last"
    with pytest.raises(ValueError):
        continuation_mode("current", "distkv_duplicate_last")
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "experiments/fit_cachebridge.py").read_text())
    defaults = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant):
            for kw in n.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[n.args[0].value] = kw.value.value
    assert defaults["--stats-dtype"] == "float64"
    assert defaults["--selection-seed"] is None
    assert defaults["--layer-selection-mode"] == "paper_ols"


def test_float32_seed_and_ridge_reference():
    from kvbridge.distkv_compat import selection_indices, ridge_probe_r2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        reference_indices = torch.randperm(5000)[:4096]
    assert torch.equal(selection_indices(5000, 4096, 123), reference_indices)
    generator = torch.Generator().manual_seed(3)
    x, y = torch.randn(30, 4, generator=generator), torch.randn(30, 2, generator=generator)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    gram = xc.T @ xc
    gram.diagonal().add_(0.01)
    w = torch.linalg.solve(gram, xc.T @ yc)
    bias = my - mx @ w
    expected = 1 - ((y - (x @ w + bias)) ** 2).sum() / ((y - my) ** 2).sum()
    actual = ridge_probe_r2(x, y, 0.01, torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    assert ridge_probe_r2(x, y, 0.01).dtype == torch.float64


def test_float32_selection_multishard_reference(tmp_path):
    from kvbridge.distkv_compat import select_layers_ridge_subsample
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); target.mkdir()
    g = torch.Generator().manual_seed(7)
    xs = [torch.randn(20, 2, 2, generator=g) for _ in range(2)]
    y = torch.randn(20, 1, 2, generator=g)
    for i, (start, end) in enumerate([(0, 9), (9, 20)]):
        torch.save({c: [x[start:end] for x in xs] for c in ("keys", "values")}, source / f"source_{i:05}.pt")
        torch.save({c: [y[start:end]] for c in ("keys", "values")}, target / f"target_{i:05}.pt")
    actual = select_layers_ridge_subsample(source, target, 2, 1, 1, 1,
                                          [[0, 1]], .01, 12, 123, torch.float32)[0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        idx = torch.randperm(20)[:12]
    expected = []
    for x in xs:
        x, yy = x[idx].reshape(12, -1).float(), y[idx, 0].float()
        mx, my = x.mean(0), yy.mean(0)
        xc, yc = x - mx, yy - my
        gram = xc.T @ xc
        gram.diagonal().add_(.01)
        w = torch.linalg.solve(gram, xc.T @ yc)
        pred = x @ w + (my - mx @ w)
        expected.append(float(1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean(0)) ** 2).sum()))
    assert actual["selection_indices"] == idx.tolist()
    assert actual["scores"] == pytest.approx(expected, abs=2e-6)
    assert actual["selected"] == [max(range(2), key=expected.__getitem__)]


@pytest.mark.parametrize("profile", ["current", "distkv_canonical", "distkv_exact"])
def test_all_modes_same_evaluator(monkeypatch, tmp_path, profile):
    import sys
    import datasets
    import experiments.eval_mmlu_redux as driver
    import kvbridge.distkv_profiles as profiles
    monkeypatch.setattr(datasets, "load_from_disk", lambda path: dataset_rows())
    monkeypatch.setattr(driver, "_rows", lambda *a, **kw: [
        dict(subject="toy", question="Q", choices=["one", "two"], answer=0)])
    monkeypatch.setattr(driver, "load_tokenizer", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(driver, "load_causal_lm", lambda *a, **kw: Model())
    monkeypatch.setattr(driver, "model_input_device", lambda model: "cpu")
    monkeypatch.setattr(driver, "assert_text_tokenizer_alignment", lambda *a: None)
    monkeypatch.setattr(driver, "transfer", lambda *a, **kw: "mapped")
    seen = []
    def decoder(model, tok, prompt, factory, device, max_new, mode="bridge_once"):
        n = 4 if mode == "distkv_duplicate_last" else 3
        factory(torch.ones(1, n, dtype=torch.long))
        seen.append((prompt, mode))
        return 0, "A", n, 1
    monkeypatch.setattr(profiles, "decode", decoder)
    monkeypatch.setattr(driver, "greedy_decode", decoder)
    manifest = tmp_path / "manifest.json"
    results = []
    for mode in ("target_native", "source_native", "full_head", "head_local", "cachebridge"):
        output = tmp_path / f"{mode}.json"
        argv = ["eval", "--source", "mock", "--target", "mock", "--mode", mode,
                "--device", "cpu", "--protocol", "distkv_greedy", "--eval-profile", profile,
                "--dataset-local-path", str(tmp_path), "--output", str(output)]
        argv += ["--evaluation-manifest-output" if mode == "target_native" else "--evaluation-manifest", str(manifest)]
        if mode not in ("target_native", "source_native"):
            argv += ["--mapper", "mock"]
            monkeypatch.setattr(driver.Mapper, "load", lambda *a, _mode=mode, **kw: SimpleNamespace(meta={"method": _mode}))
        monkeypatch.setattr(sys, "argv", argv)
        driver.main()
        result = json.loads(output.read_text())
        results.append(result["evaluation_manifest"])
        assert result["cache_continuation_mode"] == continuation_mode(profile)
        if mode not in ("target_native", "source_native"):
            assert result["target_prefix_prefill_tokens"] == 0
    assert all(m == results[0] for m in results)
    size = 1 if profile == "current" else 2
    assert all(seen[i:i + size] == seen[:size] for i in range(0, len(seen), size))
