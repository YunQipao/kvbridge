"""Numerical OLS and shard-I/O regression tests; no model downloads."""

import math

import pytest
import torch

from kvbridge.formal import select_source_layers


def _traces(tmp_path, singular=False):
    generator = torch.Generator().manual_seed(71)
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    combined = {"keys": [], "values": []}
    for component in combined:
        layers = [torch.randn(18, 2, 3, generator=generator, dtype=torch.float64)
                  for _ in range(3)]
        if singular:
            for layer in layers:
                layer[..., 2] = layer[..., 0]
        combined[component] = layers
    receiver = {
        component: [layers[2] * 1.7 + 2, layers[0] * -0.8 + 3]
        for component, layers in combined.items()
    }
    for index, (lo, hi) in enumerate(((0, 7), (7, 18))):
        torch.save({name: [x[lo:hi] for x in layers]
                    for name, layers in combined.items()},
                   source / f"source_{index:05d}.pt")
        torch.save({name: [x[lo:hi] for x in layers]
                    for name, layers in receiver.items()},
                   target / f"target_{index:05d}.pt")
    return source, target, combined, receiver


@pytest.mark.parametrize("singular", [False, True])
def test_panel_selection_matches_materialized_ols(tmp_path, singular):
    source, target, combined, receiver = _traces(tmp_path, singular)
    result = select_source_layers(source, target, 3, 2, 2, 1, panel_size=2)
    assert [row["selected"] for row in result] == [[2], [0]]
    for layer, row in enumerate(result):
        for source_layer, score in enumerate(row["scores"]):
            expected = 0
            for component in ("keys", "values"):
                for head in range(2):
                    x = combined[component][source_layer][:, head]
                    y = receiver[component][layer][:, head]
                    x = x - x.mean(0)
                    y = y - y.mean(0)
                    w = torch.linalg.lstsq(x, y, driver="gelsd").solution
                    expected += float(1 - (y - x @ w).square().sum() / y.square().sum())
            assert score == pytest.approx(expected / 4, abs=1e-9)


def test_trace_load_count_depends_on_panels_not_source_layers(tmp_path, monkeypatch):
    source, target, _, _ = _traces(tmp_path)
    original = torch.load
    calls = []

    def counted(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", counted)
    panel_size = 1
    select_source_layers(source, target, 3, 2, 2, 1, panel_size=panel_size)
    assert len(calls) == 2 * 2 * math.ceil(2 / panel_size)


def test_panel_size_does_not_change_scores(tmp_path):
    source, target, _, _ = _traces(tmp_path)
    one = select_source_layers(source, target, 3, 2, 2, 2, panel_size=1)
    two = select_source_layers(source, target, 3, 2, 2, 2, panel_size=2)
    for left, right in zip(one, two):
        assert left["selected"] == right["selected"]
        assert left["scores"] == pytest.approx(right["scores"], abs=1e-10)


def test_block_selection_matches_centered_ols_with_both_source_heads(tmp_path):
    generator = torch.Generator().manual_seed(912)
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    blocks = [[0, 1], [2, 3]]
    src = {name: [torch.randn(30, 4, 3, generator=generator, dtype=torch.float64)
                  for _ in range(3)] for name in ("keys", "values")}
    tgt = {name: [torch.stack([layers[chosen][:, block].sum(1) + 2
                              for block in blocks], dim=1) for chosen in [2, 0]]
           for name, layers in src.items()}
    torch.save(src, source / "source_00000.pt")
    torch.save(tgt, target / "target_00000.pt")
    result = select_source_layers(source, target, 3, 2, 2, 1, head_map=blocks)
    assert [row["selected"] for row in result] == [[2], [0]]
    for layer, row in enumerate(result):
        for source_layer, score in enumerate(row["scores"]):
            expected = 0.
            for name in ("keys", "values"):
                for head, block in enumerate(blocks):
                    x = src[name][source_layer][:, block].reshape(30, -1)
                    y = tgt[name][layer][:, head]
                    x, y = x - x.mean(0), y - y.mean(0)
                    w = torch.linalg.lstsq(x, y, driver="gelsd").solution
                    expected += float(1 - (y - x @ w).square().sum() / y.square().sum())
            assert score == pytest.approx(expected / 4, abs=1e-9)
