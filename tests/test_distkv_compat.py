"""CPU-only recipe checks: no model loaders, dataset downloads, or network."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kvbridge.distkv_compat import (greedy_decode, greedy_prompt, pack_jsonl,
    parse_greedy_answer, resolve_attn_repair, ridge_probe_r2, selection_indices,
    select_layers_ridge_subsample, validate_packed_records)


class Tokenizer:
    eos_token_id = 99

    def __init__(self):
        self.specials = []

    def __call__(self, text, add_special_tokens, return_tensors=None):
        self.specials.append(add_special_tokens)
        ids = list(text.encode("utf-8"))
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

    def decode(self, ids, skip_special_tokens):
        return "The correct answer is B" if ids == [66, 99] else "unexpected"


def test_concat_exact_blocks_provenance_and_no_specials(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps({"text": x}) for x in
                              ["a" * 300, "b" * 300, "c" * 500, "d" * 600]), encoding="utf-8")
    target, source = Tokenizer(), Tokenizer()
    records, rows = pack_jsonl(path, target, source, 2, 512)
    assert len(rows) == 2 and all(len(row) == 512 for row in rows)
    assert rows[0].tolist() == [97] * 300 + [98] * 212
    assert rows[1].tolist() == [98] * 88 + [99] * 424
    assert records[0]["provenance"] == [
        {"line": 1, "id": None, "token_start": 0, "token_end": 300, "text": "a" * 300},
        {"line": 2, "id": None, "token_start": 0, "token_end": 212, "text": "b" * 300}]
    assert target.specials == source.specials == [False] * 3  # stop before row 4
    validate_packed_records(records, source, target)
    records[0]["token_ids"][0] = 0
    with pytest.raises(ValueError, match="differs from provenance"):
        validate_packed_records(records, source, target)


def test_concat_conversations_and_incomplete_tail(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"conversations": [{"value": "ab"}, {"value": "cd"}]}))
    _, rows = pack_jsonl(path, Tokenizer(), Tokenizer(), 1, 4)
    assert rows[0].tolist() == list(b"ab\nc")
    with pytest.raises(RuntimeError, match="only 1 complete.*discarded 1 trailing"):
        pack_jsonl(path, Tokenizer(), Tokenizer(), 2, 4)


def test_concat_rejects_tokenizer_mismatch(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text":"abcd"}\n')
    class Other(Tokenizer):
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=[1])
    with pytest.raises(RuntimeError, match="tokenizer IDs differ"):
        pack_jsonl(path, Tokenizer(), Other(), 1, 4)


@pytest.mark.parametrize("method", ["cachebridge", "head_local", "full_head"])
def test_repair_modes(method):
    assert resolve_attn_repair(method) == (method == "cachebridge")
    assert resolve_attn_repair(method, "auto") == (method == "cachebridge")
    assert resolve_attn_repair(method, "paper")
    assert not resolve_attn_repair(method, "off")


def test_seeded_subsample_default_4096():
    first = selection_indices(5000)
    assert len(first) == 4096
    assert torch.equal(first, selection_indices(5000, seed=42))
    assert not torch.equal(first, selection_indices(5000, seed=43))
    # Compatibility probes retain DistKV's randperm order even when all rows
    # are selected (sorting changes float32 reduction order).
    assert sorted(selection_indices(10).tolist()) == list(range(10))


def test_ridge_subsample_matches_manual_grouped_reference(tmp_path):
    gen = torch.Generator().manual_seed(8)
    src = [torch.randn(13, 4, 2, generator=gen, dtype=torch.float64) for _ in range(2)]
    tgt = [torch.randn(13, 2, 2, generator=gen, dtype=torch.float64)]
    sd, td = tmp_path / "source", tmp_path / "target"
    sd.mkdir(); td.mkdir()
    for index, (lo, hi) in enumerate([(0, 6), (6, 13)]):
        torch.save({k: [x[lo:hi] for x in src] for k in ("keys", "values")}, sd / f"source_{index:05d}.pt")
        torch.save({k: [x[lo:hi] for x in tgt] for k in ("keys", "values")}, td / f"target_{index:05d}.pt")
    actual = select_layers_ridge_subsample(sd, td, 2, 1, 2, 1,
                                         [[0, 1], [2, 3]], 0.01, 9, 42)
    indices = selection_indices(13, 9, 42)
    expected = []
    for candidate in range(2):
        values = []
        for head, block in enumerate([[0, 1], [2, 3]]):
            x = src[candidate][indices][:, block].flatten(1)
            y = tgt[0][indices, head]
            x, y = x - x.mean(0), y - y.mean(0)
            w = torch.linalg.solve(x.T @ x + .01 * torch.eye(4, dtype=x.dtype), x.T @ y)
            values.append(1 - ((y - x @ w) ** 2).sum() / (y ** 2).sum())
        expected.append(float(torch.stack(values).mean()))
    assert actual[0]["scores"] == pytest.approx(expected)
    assert actual[0]["selected"] == [max(range(2), key=expected.__getitem__)]
    assert actual == select_layers_ridge_subsample(sd, td, 2, 1, 2, 1,
                                                  [[0, 1], [2, 3]], 0.01, 9, 42)


def test_prompt_matches_specification():
    expected = '''Accurately answer the following question:
Which?
Choices:
A. first
B. second
Instructions:
- Carefully read the question and all options.
- Select the single most correct answer.
- Respond ONLY in the following format: "The correct answer is A/B/C/D/E".
- Do not include any explanations, additional text, or punctuation besides the answer.
The correct answer is'''
    assert greedy_prompt({"question": "Which?", "choices": ["first", "second"]}) == expected


@pytest.mark.parametrize("text,answer", [("The correct answer is C", 2),
    ("A then the correct answer is D", 3), (" E ", 4), ("answer b", 1), ("nothing", -1)])
def test_answer_parser(text, answer):
    assert parse_greedy_answer(text) == answer


def test_greedy_native_and_mapped_same_flow_and_zero_target_prefill():
    tokenizer = Tokenizer()
    class Model:
        def __init__(self):
            self.calls = []
        def __call__(self, token, **kwargs):
            self.calls.append((token.clone(), kwargs))
            logits = torch.zeros(1, 1, 100)
            logits[0, 0, 66 if len(self.calls) == 1 else 99] = 10
            return SimpleNamespace(logits=logits, past_key_values="cache")
    results = []
    for mapped in [False, True]:
        model, prefixes = Model(), []
        def factory(ids):
            prefixes.append(ids.clone())
            return "mapped" if mapped else "native"
        results.append(greedy_decode(model, tokenizer, "hello", factory, "cpu"))
        assert prefixes[0].shape == (1, 4)
        assert model.calls[0][0].tolist() == [[ord("o")]]  # only bridge on target
        assert model.calls[1][0].tolist() == [[66]]
        assert [call[1]["cache_position"].item() for call in model.calls] == [4, 5]
        assert [call[1]["attention_mask"].shape[1] for call in model.calls] == [5, 6]
        assert model.calls[0][1]["past_key_values"] == ("mapped" if mapped else "native")
    assert results[0] == results[1] == (1, "The correct answer is B", 4, 2)


def test_greedy_respects_max_new_tokens_and_model_eos():
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[66, 99])
        def __call__(self, token, **kwargs):
            logits = torch.zeros(1, 1, 100)
            logits[0, 0, 66] = 10
            return SimpleNamespace(logits=logits, past_key_values="cache")
    result = greedy_decode(Model(), Tokenizer(), "hello", lambda ids: "cache", "cpu", 64)
    assert result[3] == 1
    model = Model()
    model.generation_config = SimpleNamespace(eos_token_id=99)
    result = greedy_decode(model, Tokenizer(), "hello", lambda ids: "cache", "cpu", 3)
    assert result[3] == 3


def test_existing_cli_defaults_and_paper_selection_call_unchanged():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "experiments/fit_cachebridge.py").read_text())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        defaults[node.args[0].value] = keyword.value.value
    assert defaults["--calibration-pack-mode"] == "records"
    assert defaults["--layer-selection-mode"] == "paper_ols"
    assert defaults["--attn-repair-mode"] == "auto"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "select_source_layers"]
    assert len(calls) == 1
    assert {k.arg for k in calls[0].keywords} == {"panel_size", "head_map"}
    driver = (root / "scripts/run_distkv_compatible.sh").read_text()
    assert 'MANIFEST="$RESULT_DIR/manifest.json"' in driver
    assert 'MANIFEST_ARGS=(--evaluation-manifest "$MANIFEST")' in driver


def test_paper_ols_remains_unregularized_and_matches_manual_reference(tmp_path):
    from kvbridge.formal import select_source_layers
    generator = torch.Generator().manual_seed(15)
    x = [torch.randn(12, 1, 2, generator=generator, dtype=torch.float64) for _ in range(2)]
    y = [torch.randn(12, 1, 2, generator=generator, dtype=torch.float64)]
    sd, td = tmp_path / "source", tmp_path / "target"
    sd.mkdir(); td.mkdir()
    torch.save({k: x for k in ("keys", "values")}, sd / "source_00000.pt")
    torch.save({k: y for k in ("keys", "values")}, td / "target_00000.pt")
    actual = select_source_layers(sd, td, 2, 1, 1, 1)
    expected = []
    for candidate in x:
        xc = candidate[:, 0] - candidate[:, 0].mean(0)
        yc = y[0][:, 0] - y[0][:, 0].mean(0)
        w = torch.linalg.lstsq(xc, yc, driver="gelsd").solution
        expected.append(float(1 - (yc - xc @ w).square().sum() / yc.square().sum()))
    assert actual[0]["scores"] == pytest.approx(expected)
    assert actual == select_source_layers(sd, td, 2, 1, 1, 1, ridge_lambda=100)


def test_cli_opt_in_modes_and_defaults(monkeypatch, tmp_path):
    import experiments.fit_cachebridge as driver
    seen = []
    monkeypatch.setattr(driver, "fit", seen.append)
    common = ["--source", "mock-source", "--target", "mock-target", "--method", "cachebridge",
              "--n-calib", "300", "--seq-len", "512", "--token-stride", "4",
              "--dataset-file", str(tmp_path / "unused.jsonl"), "--output", str(tmp_path)]
    driver.main(common)
    assert seen[-1].calibration_pack_mode == "records"
    assert seen[-1].layer_selection_mode == "paper_ols"
    assert seen[-1].attn_repair_mode == "auto"
    driver.main(common + ["--calibration-pack-mode", "distkv_concat",
        "--layer-selection-mode", "distkv_ridge_subsample", "--attn-repair-mode", "off"])
    assert seen[-1].calibration_pack_mode == "distkv_concat"
    assert seen[-1].selection_subsample == 4096
    assert seen[-1].attn_repair_mode == "off"


def test_all_greedy_modes_share_manifest_and_mapped_accounting(monkeypatch, tmp_path):
    import sys
    import experiments.eval_mmlu_redux as driver

    rows = [{"subject": "toy", "question": str(i), "choices": ["yes", "no"], "answer": 1}
            for i in range(7)]
    monkeypatch.setattr(driver, "_rows", lambda *a, **k: rows)
    monkeypatch.setattr(driver, "load_tokenizer", lambda *a, **k: Tokenizer())
    class Model:
        def __call__(self, ids, **kwargs):
            return SimpleNamespace(past_key_values="cache")
    monkeypatch.setattr(driver, "load_causal_lm", lambda *a, **k: Model())
    monkeypatch.setattr(driver, "model_input_device", lambda model: "cpu")
    monkeypatch.setattr(driver, "assert_text_tokenizer_alignment", lambda *a: None)
    monkeypatch.setattr(driver, "transfer", lambda *a, **k: "mapped-cache")
    def decode(model, tokenizer, prompt, factory, device, max_new_tokens):
        factory(torch.tensor([[1, 2]], device="cpu"))
        return 1, "B", 2, 1
    monkeypatch.setattr(driver, "greedy_decode", decode)
    manifest = tmp_path / "manifest.json"
    seen = []
    for mode in ("target_native", "source_native", "full_head", "head_local", "cachebridge"):
        output = tmp_path / f"{mode}.json"
        args = ["eval", "--mode", mode, "--source", "mock", "--target", "mock",
                "--device", "cpu", "--protocol", "distkv_greedy", "--output", str(output)]
        if mode == "target_native":
            args += ["--num-eval-samples", "3", "--evaluation-manifest-output", str(manifest)]
        else:
            args += ["--evaluation-manifest", str(manifest)]
        if mode in ("full_head", "head_local", "cachebridge"):
            args += ["--mapper", "mock"]
            monkeypatch.setattr(driver.Mapper, "load", lambda *a, _mode=mode, **k:
                                SimpleNamespace(meta={"method": _mode}))
        monkeypatch.setattr(sys, "argv", args)
        driver.main()
        result = json.loads(output.read_text())
        seen.append(result["evaluation_manifest"])
        assert result["total"] == 3
        assert result["accuracy"] == 1
        if mode in ("full_head", "head_local", "cachebridge"):
            assert result["target_prefix_prefill_tokens"] == 0
            assert result["mapped_tokens"] == 6
    assert all(value == seen[0] for value in seen)
