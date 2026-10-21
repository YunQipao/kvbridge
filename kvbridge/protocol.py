"""Deterministic experiment protocol helpers.

These helpers keep paired methods and evaluation baselines on identical
example/layer manifests.  They contain no model-loading code.
"""

from __future__ import annotations

import json
from pathlib import Path


def assert_mapper_method(mapper, mode):
    expected = {"full_head": "full_head", "cachebridge": "cachebridge",
                "head_local": "head_local"}.get(mode)
    if expected is None:
        return
    actual = mapper.meta.get("method")
    if actual != expected:
        raise ValueError(
            f"mapper method mismatch: mode={mode!r} requires {expected!r}, "
            f"artifact declares {actual!r}"
        )


def validate_mapped_evaluation_accounting(accounting):
    """Mapped evaluation must not prefill the target shared prefix."""
    if int(accounting.get("target_prefix_prefill_tokens", 0)) != 0:
        raise ValueError("mapped evaluation performed target prefix prefill")


def systematic_indices(n, m):
    n, m = int(n), int(m)
    if n < 0 or m < 0 or m > n:
        raise ValueError("require 0 <= m <= n")
    return [(i * n) // m for i in range(m)] if m else []


def make_evaluation_manifest(n, num_samples=None, seed=42, sampling="systematic"):
    n = int(n)
    m = n if num_samples is None else int(num_samples)
    if sampling == "systematic":
        indices = systematic_indices(n, m)
    elif sampling == "random":
        import random
        rng = random.Random(int(seed))
        indices = sorted(rng.sample(range(n), m))
    else:
        raise ValueError(f"unsupported sampling={sampling!r}")
    return {
        "population_size": n,
        "num_samples": m,
        "sampling": sampling,
        "seed": int(seed),
        "seed_affects_selection": sampling == "random",
        "indices": indices,
        "provenance": "ENGINEERING-EQUIVALENT" if sampling == "systematic" else "seeded",
    }


def validate_evaluation_manifest(manifest, population_size):
    """Validate a manifest before any evaluation mode consumes its indices."""
    population_size = int(population_size)
    indices = [int(index) for index in manifest.get("indices", [])]
    if manifest.get("population_size") != population_size:
        raise ValueError("evaluation manifest population size mismatch")
    if manifest.get("num_samples") != len(indices):
        raise ValueError("evaluation manifest sample count mismatch")
    if len(set(indices)) != len(indices) or any(
            index < 0 or index >= population_size for index in indices):
        raise ValueError("evaluation manifest contains invalid indices")
    return indices


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def load_json(path):
    return json.loads(Path(path).read_text())


def reuse_manifest_prefix(manifest, n_calib):
    records = list(manifest.get("records", []))
    n_calib = int(n_calib)
    if n_calib > len(records):
        raise ValueError(
            f"requested n_calib={n_calib} exceeds reusable manifest "
            f"record count={len(records)}")
    return records[:n_calib]


def cyclic_assignment(n_heads, shift):
    n_heads, shift = int(n_heads), int(shift)
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    return [(h + shift) % n_heads for h in range(n_heads)]
