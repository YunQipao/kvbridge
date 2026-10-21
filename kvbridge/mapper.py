"""The Mapper artifact: per-target-layer linear maps plus metadata.

On disk a mapper is a directory (or HF Hub repo) with two files:

- ``mapper.safetensors`` -- tensors named ``layer_{i:02d}.{W_k,mu_xk,mu_yk,
  W_v,mu_xv,mu_yv}``
- ``mapper.json`` -- metadata: model pair, KV geometry, RoPE thetas, the
  per-target-layer source-layer selection, and fit provenance

Weights are stored in bfloat16 by default: transfer runs its matmuls in
bfloat16 regardless, so storing anything wider only doubles the download.
Pass ``dtype=torch.float32`` to ``save`` to keep full fit precision.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TENSOR_KEYS = ("W_k", "mu_xk", "mu_yk", "W_v", "mu_xv", "mu_yv")
_REQUIRED_META = ("source", "target", "n_kv_heads", "head_dim",
                  "rope_theta_source", "rope_theta_target")


def normalize_mapper_meta(meta):
    """Return metadata with the canonical model-pair keys.

    Older artifacts used ``source_model``/``target_model``. Canonical fields
    take precedence when both are present; aliases are removed so re-saving
    a legacy artifact also emits only the canonical model-pair fields.
    """
    normalized = dict(meta)
    if "source" not in normalized and "source_model" in normalized:
        normalized["source"] = normalized["source_model"]
    if "target" not in normalized and "target_model" in normalized:
        normalized["target"] = normalized["target_model"]
    normalized.pop("source_model", None)
    normalized.pop("target_model", None)
    if (normalized.get("method") in ("cachebridge", "head_local")
            and all(key in normalized for key in
                    ("head_assignment", "num_source_kv_heads", "num_target_kv_heads"))):
        from .cachebridge import normalize_head_assignment
        blocks = normalize_head_assignment(normalized["head_assignment"],
            int(normalized["num_source_kv_heads"]), int(normalized["num_target_kv_heads"]))
        grouped = any(len(block) > 1 for block in blocks)
        if grouped and "head_map" in normalized and normalized["head_map"] != blocks:
            raise ValueError("head_assignment and head_map disagree")
        normalized.setdefault("head_map", blocks)
        normalized.setdefault("head_map_mode", "block" if grouped else "matched")
    return normalized


class Mapper:
    """Holds ``layers`` (list of dicts: ``source_layers`` + the six tensors
    per target layer) and ``meta`` (JSON-safe dict). ``to()`` follows
    nn.Module semantics: it moves tensors in place and returns self, so
    repeated calls are free."""

    FORMAT_VERSION = 1

    def __init__(self, layers, meta):
        meta = normalize_mapper_meta(meta)
        missing = [k for k in _REQUIRED_META if k not in meta]
        if missing:
            raise ValueError(f"mapper meta missing required keys: {missing}")
        if meta.get("method") in ("cachebridge", "head_local"):
            from .cachebridge import normalize_head_assignment, validate_head_map
            required = ("source", "target", "num_source_layers",
                        "num_target_layers", "num_source_kv_heads",
                        "num_target_kv_heads", "head_assignment",
                        "selected_source_layers", "ridge_lambda",
                        "weighting_method", "token_stride", "num_boundaries",
                        "boundary_start_token")
            absent = [key for key in required if key not in meta]
            if absent:
                raise ValueError(f"CacheBridge mapper meta missing required keys: {absent}")
            normalize_head_assignment(meta["head_assignment"],
                                     int(meta["num_source_kv_heads"]),
                                     int(meta["num_target_kv_heads"]))
            meta.setdefault("head_map", [[x] for x in meta["head_assignment"]])
            meta.setdefault("head_map_mode", "matched")
            validate_head_map(meta["head_map"], int(meta["num_source_kv_heads"]),
                              int(meta["num_target_kv_heads"]), mode=meta["head_map_mode"])
            block_widths = {len(block) for block in meta["head_map"]}
            if len(block_widths) != 1:
                raise ValueError("stacked mapper tensors require equal head-map block widths")
            block_width = next(iter(block_widths))
            if len(layers) != int(meta["num_target_layers"]):
                raise ValueError("CacheBridge target layer count mismatch")
            if len(meta["selected_source_layers"]) != len(layers):
                raise ValueError("CacheBridge selected source layer count mismatch")
            for index, entry in enumerate(layers):
                selected = meta["selected_source_layers"][index]
                if entry["source_layers"] != selected:
                    raise ValueError("CacheBridge selected source layers do not match tensors")
                p = len(selected) * block_width * int(meta["head_dim"])
                h = int(meta["num_target_kv_heads"])
                d = int(meta["head_dim"])
                for key, shape in (("W_k", (h, p, d)), ("W_v", (h, p, d)),
                                   ("mu_xk", (h, p)), ("mu_xv", (h, p)),
                                   ("mu_yk", (h, d)), ("mu_yv", (h, d))):
                    if tuple(entry[key].shape) != shape or not torch.isfinite(entry[key]).all():
                        raise ValueError(f"CacheBridge invalid shape or nonfinite {key}")
        elif meta.get("method") == "full_head":
            required = ("source", "target", "num_source_layers",
                        "num_target_layers", "num_source_kv_heads",
                        "num_target_kv_heads", "head_dim",
                        "selected_source_layers", "ridge_lambda",
                        "weighting_method")
            absent = [key for key in required if key not in meta]
            if absent:
                raise ValueError(f"Full-Head mapper meta missing required keys: {absent}")
            if meta.get("weighting_method") != "uniform":
                raise ValueError("Full-Head mapper must declare weighting_method=uniform")
            h, d = int(meta["num_target_kv_heads"]), int(meta["head_dim"])
            for index, entry in enumerate(layers):
                selected = meta["selected_source_layers"][index]
                if entry["source_layers"] != selected:
                    raise ValueError("Full-Head selected source layers do not match tensors")
                p = len(selected) * int(meta["num_source_kv_heads"]) * d
                for key, shape in (("W_k", (h, p, d)), ("W_v", (h, p, d)),
                                   ("mu_xk", (h, p)), ("mu_xv", (h, p)),
                                   ("mu_yk", (h, d)), ("mu_yv", (h, d))):
                    if tuple(entry[key].shape) != shape or not torch.isfinite(entry[key]).all():
                        raise ValueError(f"Full-Head invalid shape or nonfinite {key}")
        self.layers = layers
        self.meta = dict(meta)

    # ---- convenience accessors ----
    @property
    def source(self):
        return self.meta["source"]

    @property
    def target(self):
        return self.meta["target"]

    @property
    def n_kv_heads(self):
        return int(self.meta["n_kv_heads"])

    @property
    def head_dim(self):
        return int(self.meta["head_dim"])

    @property
    def rope_theta_source(self):
        return float(self.meta["rope_theta_source"])

    @property
    def rope_theta_target(self):
        return float(self.meta["rope_theta_target"])

    @property
    def top_k(self):
        return self.meta.get("top_k")

    @property
    def device(self):
        return self.layers[0]["W_k"].device

    @property
    def dtype(self):
        return self.layers[0]["W_k"].dtype

    def __repr__(self):
        return (f"Mapper({self.source} -> {self.target}, "
                f"{len(self.layers)} target layers, top_k={self.top_k}, "
                f"{self.dtype} on {self.device})")

    def to(self, device=None, dtype=None):
        for entry in self.layers:
            for key in TENSOR_KEYS:
                entry[key] = entry[key].to(device=device, dtype=dtype)
        return self

    # ---- persistence ----
    def save(self, path, dtype=torch.bfloat16):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for i, entry in enumerate(self.layers):
            for key in TENSOR_KEYS:
                tensors[f"layer_{i:02d}.{key}"] = (
                    entry[key].detach().to("cpu", dtype).contiguous())
        save_file(tensors, str(path / "mapper.safetensors"))
        format_version = 2 if self.meta.get("method") in ("cachebridge", "head_local") else self.FORMAT_VERSION
        meta = normalize_mapper_meta(self.meta)
        meta = {**meta,
                "format_version": format_version,
                "n_target_layers": len(self.layers),
                "source_layers": [
                    [int(s) for s in entry["source_layers"]] for entry in self.layers],
                "weights_dtype": str(dtype).removeprefix("torch.")}
        with open(path / "mapper.json", "w") as f:
            json.dump(meta, f, indent=2)
        return path

    @classmethod
    def _from_files(cls, json_path, safetensors_path, device="cpu", dtype=None):
        with open(json_path) as f:
            meta = normalize_mapper_meta(json.load(f))
        version = meta.get("format_version")
        if version not in (cls.FORMAT_VERSION, 2) or (
            version == 2 and meta.get("method") not in ("cachebridge", "head_local")):
            raise ValueError(f"unsupported mapper format_version={version} "
                             f"(this kvbridge reads versions {cls.FORMAT_VERSION} and 2)")
        tensors = load_file(str(safetensors_path), device=str(device))
        layers = []
        for i in range(meta["n_target_layers"]):
            entry = {"source_layers": list(meta["source_layers"][i])}
            for key in TENSOR_KEYS:
                t = tensors[f"layer_{i:02d}.{key}"]
                entry[key] = t.to(dtype) if dtype is not None else t
            layers.append(entry)
        meta = {k: v for k, v in meta.items() if k != "source_layers"}
        return cls(layers, meta)

    @classmethod
    def load(cls, path, device="cpu", dtype=None):
        path = Path(path)
        return cls._from_files(path / "mapper.json", path / "mapper.safetensors",
                               device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, repo_or_path, device="cpu", dtype=None,
                        revision=None, token=None):
        """Load from a local directory or a HuggingFace Hub repo id."""
        p = Path(repo_or_path)
        if p.is_dir():
            return cls.load(p, device=device, dtype=dtype)
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(repo_or_path, "mapper.json",
                                    revision=revision, token=token)
        st_path = hf_hub_download(repo_or_path, "mapper.safetensors",
                                  revision=revision, token=token)
        return cls._from_files(json_path, st_path, device=device, dtype=dtype)
