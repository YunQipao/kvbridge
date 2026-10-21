import torch

from kvbridge.cachebridge import cachebridge_metadata
from kvbridge.mapper import Mapper, normalize_mapper_meta


def _layer(heads=2, width=2, dim=2):
    return {
        "source_layers": [0],
        "W_k": torch.zeros(heads, width, dim),
        "W_v": torch.zeros(heads, width, dim),
        "mu_xk": torch.zeros(heads, width),
        "mu_xv": torch.zeros(heads, width),
        "mu_yk": torch.zeros(heads, dim),
        "mu_yv": torch.zeros(heads, dim),
    }


def _cachebridge_meta():
    return cachebridge_metadata(
        source_model="s", target_model="t",
        num_source_layers=2, num_target_layers=1,
        num_source_kv_heads=2, num_target_kv_heads=2, head_dim=2,
        head_assignment=[0, 1], selected_source_layers=[[0]],
        rope_theta_source=1.0, rope_theta_target=1.0)


def test_new_cachebridge_artifact_uses_canonical_model_keys(tmp_path):
    meta = _cachebridge_meta()
    assert meta["source"] == "s"
    assert meta["target"] == "t"
    assert "source_model" not in meta
    assert "target_model" not in meta


def test_legacy_model_keys_normalize_and_round_trip_canonically(tmp_path):
    meta = _cachebridge_meta()
    legacy = dict(meta)
    legacy.pop("source")
    legacy.pop("target")
    legacy["source_model"] = "s"
    legacy["target_model"] = "t"
    mapper = Mapper([_layer()], legacy)
    assert mapper.source == "s"
    assert mapper.target == "t"
    mapper.save(tmp_path / "mapper", dtype=torch.float32)
    loaded = Mapper.load(tmp_path / "mapper")
    assert loaded.meta["source"] == "s"
    assert loaded.meta["target"] == "t"
    assert "source_model" not in loaded.meta
    assert "target_model" not in loaded.meta


def test_full_head_metadata_uses_uniform_weighting_and_canonical_keys():
    meta = _cachebridge_meta()
    meta.update(method="full_head", weighting_method="uniform")
    # Full-Head width is selected_layers * source_heads * head_dim.
    mapper = Mapper([_layer(width=4)], meta)
    assert mapper.meta["source"] == "s"
    assert mapper.meta["target"] == "t"
    assert mapper.meta["weighting_method"] == "uniform"


def test_normalization_does_not_override_canonical_values():
    normalized = normalize_mapper_meta({
        "source": "canonical-s", "target": "canonical-t",
        "source_model": "legacy-s", "target_model": "legacy-t"})
    assert normalized == {"source": "canonical-s", "target": "canonical-t"}
