"""Paper-facing CacheBridge calibration helpers.

The implementation is deliberately split into bounded, restartable stages:

1. tokenize a deterministic calibration manifest;
2. write one source/target trace shard per sequence;
3. rank source layers with centered single-layer probes;
4. fit one target layer at a time with weighted sufficient statistics.

The paper describes the fused panel algorithm but does not publish its kernel.
The ``torch_bounded`` path below is an algorithmically equivalent reference
implementation; it is not the authors' CUDA kernel.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from .cache_compat import cache_layer_kv
from .cachebridge import (
    WeightedRidgeStats,
    build_head_local_features,
    compute_gqa_sensitivities,
    make_log_causal_boundaries,
    repair_weights,
    validate_head_map,
)
from .geometry import geometry_dict
from .geometry import text_config
from .model_utils import load_causal_lm, load_tokenizer, model_input_device
from .rope import rope_spec, strip_rope


def capture_native_ministral_query(attention, callback):
    """Intercept the native eager score entry, including YaRN and query scale.

    Scoped to this module instance and restored by the returned handle. This
    calibration helper is synchronous; concurrent forwards are unsupported.
    """
    import inspect
    from unittest.mock import patch
    module = inspect.getmodule(type(attention))
    if module is None or not hasattr(module, "eager_attention_forward"):
        raise RuntimeError("Ministral native eager attention entry is unavailable")
    original = module.eager_attention_forward

    def eager(instance, query, *args, **kwargs):
        if instance is attention:
            callback(query)
        return original(instance, query, *args, **kwargs)

    scoped = patch.object(module, "eager_attention_forward", eager)
    scoped.start()

    class Handle:
        def remove(self):
            scoped.stop()
    return Handle()

BOUNDARY_INDEXING_DESCRIPTION = (
    "boundary c is the zero-based first-future-query index; "
    "eligible prefix positions are < c"
)


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    raise TypeError(f"manifest value is not JSON-safe: {type(value)!r}")


def load_calibration_texts(*, dataset=None, dataset_file=None, n_calib=500,
                           seed=42, min_tokens=32, tokenizer=None):
    """Return deterministic text records and provenance.

    Local files accept one paragraph per blank-line block or JSONL records with
    a ``text`` field. FineWeb-Edu uses a shuffled streaming split. The exact
    paper row IDs are not public, so the selected stream offsets are recorded.
    """
    if bool(dataset) == bool(dataset_file):
        raise ValueError("provide exactly one of dataset or dataset_file")
    records = []
    if dataset_file:
        path = Path(dataset_file)
        if path.suffix.lower() == ".jsonl":
            # Iterate physical lines. str.splitlines() also splits Unicode
            # separators (e.g. U+2028) that are legal inside JSON strings.
            with path.open(encoding="utf-8") as handle:
                for offset, line in enumerate(handle):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid calibration JSONL at {path}:{offset + 1}, "
                            f"column {exc.colno}: {exc.msg}"
                        ) from exc
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise ValueError(
                            f"Invalid calibration JSONL at {path}:{offset + 1}: "
                            "expected an object with a string text field")
                    records.append({"offset": offset, "id": row.get("id"),
                                    "text": row["text"]})
        else:
            raw = path.read_text(encoding="utf-8")
            for offset, text in enumerate(raw.split("\n\n")):
                if text.strip():
                    records.append({"offset": offset, "id": None,
                                    "text": text.strip()})
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for FineWeb-Edu calibration") from exc
        name = "sample-10BT" if dataset == "HuggingFaceFW/fineweb-edu" else None
        kwargs = {"split": "train", "streaming": True}
        if name:
            kwargs["name"] = name
        stream = load_dataset(dataset, **kwargs).shuffle(seed=seed, buffer_size=10_000)
        iterator = iter(stream)
        try:
            for offset, row in enumerate(iterator):
                text = row.get("text", "")
                if text:
                    records.append({"offset": offset, "id": row.get("id"),
                                    "text": text})
                if len(records) >= n_calib * 4:
                    break
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        # Release the streaming iterator before interpreter finalization. Some
        # datasets/fsspec versions retain a background HTTP worker otherwise.
        for obj in (stream, getattr(stream, "_ex_iterable", None)):
            for method in ("close", "shutdown"):
                callback = getattr(obj, method, None)
                if callback is not None:
                    try:
                        callback()
                    except Exception:
                        pass
        del iterator, stream
        gc.collect()
    if tokenizer is not None:
        kept = []
        for record in records:
            ids = tokenizer(record["text"], add_special_tokens=True,
                            truncation=True, max_length=10_000).input_ids
            if len(ids) >= min_tokens:
                record["token_length_before_truncation"] = len(ids)
                kept.append(record)
            if len(kept) >= n_calib:
                break
        records = kept
    if len(records) < n_calib:
        raise RuntimeError(f"only found {len(records)} usable calibration records; "
                           f"needed {n_calib}")
    return records[:n_calib]


def tokenize_records(records, tokenizer, seq_len):
    rows = []
    for index, record in enumerate(records):
        encoded = tokenizer(record["text"], return_tensors="pt",
                            truncation=True, max_length=seq_len)
        ids = encoded.input_ids[0]
        if ids.numel() < 2:
            raise ValueError(f"calibration record {index} has fewer than 2 tokens")
        rows.append(ids)
        record["token_length"] = int(ids.numel())
        record["truncated"] = bool(record.get("token_length_before_truncation", len(ids))
                                    > len(ids))
    return rows


def check_tokenizer_alignment(source_rows, target_rows):
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target calibration row counts differ")
    for index, (source, target) in enumerate(zip(source_rows, target_rows)):
        if not torch.equal(source, target):
            raise RuntimeError(
                f"source and target tokenizer IDs differ at calibration row {index}; "
                "cross-model observation alignment is unsafe")


def write_manifest(output, *, dataset, split, seed, records, rows, seq_len,
                   token_stride, num_boundaries, boundary_start, n_observations,
                   token_alignment_verified=False, parent_manifest=None):
    manifest = {
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "indices": [r.get("offset") for r in records],
        "sample_ids": [r.get("id") for r in records],
        "token_lengths": [int(x.numel()) for x in rows],
        "truncated": [bool(r.get("truncated", False)) for r in records],
        "n_calib": len(rows),
        "seq_len": int(seq_len),
        "token_stride": int(token_stride),
        "num_boundaries": int(num_boundaries),
        "boundary_start_token": int(boundary_start),
        "actual_observations": int(n_observations),
        "boundary_indexing": BOUNDARY_INDEXING_DESCRIPTION,
        "token_alignment_verified": bool(token_alignment_verified),
        # Keeping the selected records makes a paired run independent of a
        # second streaming shuffle.  This is intentionally provenance data,
        # not a claim that the paper's unreleased row IDs are known.
        "records": [{"offset": r.get("offset"), "id": r.get("id"),
                     "text": r.get("text")} for r in records],
    }
    if parent_manifest:
        manifest["parent_manifest"] = parent_manifest["path"]
        manifest["parent_manifest_hash"] = parent_manifest["hash"]
        manifest["parent_record_count"] = int(parent_manifest["record_count"])
        manifest["prefix_length"] = int(parent_manifest["prefix_length"])
    path = Path(output) / "calibration_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(manifest), indent=2))
    return manifest


def load_completed_trace_manifest(output, trace_root, *, dataset, seed,
                                  n_calib, seq_len, token_stride,
                                  num_boundaries, boundary_start):
    """Reuse completed traces without resampling or contacting the dataset.

    ENGINEERING ASSUMPTION: legacy completion markers attest that capture
    finished after the driver's alignment check. They are not checksums.
    Legacy manifests cannot independently re-prove token alignment.
    """
    output, trace_root = Path(output), Path(trace_root)
    for stage in ("source_traces", "target_traces"):
        if not (trace_root / f"{stage}.done").is_file():
            raise ValueError(f"incomplete trace stage: {stage}")
    manifest = json.loads((output / "calibration_manifest.json").read_text())
    expected = dict(dataset=dataset, seed=seed, n_calib=n_calib, seq_len=seq_len,
                    token_stride=token_stride, num_boundaries=num_boundaries,
                    boundary_start_token=boundary_start)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"resume manifest mismatch: {key}")
    lengths = manifest.get("token_lengths", [])
    if (len(lengths) != n_calib or token_stride <= 0
            or any(not isinstance(n, int) or not 2 <= n <= seq_len for n in lengths)):
        raise ValueError("invalid resume token lengths or stride")
    for field in ("indices", "sample_ids"):
        if len(manifest.get(field, [])) != n_calib:
            raise ValueError(f"invalid resume manifest field: {field}")
    observations = sum((n + token_stride - 1) // token_stride for n in lengths)
    if manifest.get("actual_observations") != observations:
        raise ValueError("resume observation count mismatch")
    for stage in ("source", "target"):
        expected_names = {f"{stage}_{i:05d}.pt" for i in range(n_calib)}
        actual_names = {p.name for p in (trace_root / stage).glob(f"{stage}_*.pt")}
        if actual_names != expected_names:
            raise ValueError(f"incomplete or unexpected {stage} trace shards")
    if manifest.get("token_alignment_verified") is False:
        raise ValueError("resume manifest records failed token alignment")
    if "token_alignment_verified" not in manifest:
        print("WARNING: legacy manifest has no token-alignment attestation; "
              "reusing completed capture provenance, NOT rechecking token IDs.",
              flush=True)
    return manifest


def _to_bhtd(tensor, batch, heads, tokens, dim):
    if tensor.shape == (batch, tokens, heads, dim):
        return tensor.transpose(1, 2)
    if tensor.shape == (batch, tokens, heads * dim):
        return tensor.reshape(batch, tokens, heads, dim).transpose(1, 2)
    if tensor.shape == (batch, heads, tokens, dim):
        return tensor
    raise ValueError(f"cannot interpret tensor shape {tuple(tensor.shape)}")


def _capture_target_weights(model, ids, token_stride, boundaries, capture_mode="boundary_chunked"):
    """Capture target KV and per-layer/head K/V weights for one sequence.

    Full capture slices boundary rows from a single native eager pass.
    Boundary-chunked capture uses SDPA between boundaries and one eager query
    at each boundary. Both keep complete causal context and use the unchanged
    sensitivity equations.
    """
    if capture_mode not in ("full", "boundary_chunked"):
        raise ValueError("target capture mode must be 'full' or 'boundary_chunked'")
    captured = {}
    handles = []
    model_config = text_config(model.config)
    if (getattr(model_config, "model_type", None) != "ministral3"
            and getattr(model_config, "rope_scaling", None) not in (None, {})):
        raise RuntimeError("target RoPE scaling is not handled by the simple rope_theta reference")
    T = ids.shape[1]
    boundaries = list(boundaries)
    if not boundaries or boundaries != sorted(set(boundaries)) or boundaries[0] < 0:
        raise ValueError("capture boundaries must be nonempty, sorted, unique and nonnegative")
    setter = getattr(model, "set_attn_implementation", None)
    if setter is None:
        raise RuntimeError("boundary capture requires dynamic attention implementation switching")
    original_implementation = model.config._attn_implementation
    n_layers = model_config.num_hidden_layers
    hq = model_config.num_attention_heads
    hkv = model_config.num_key_value_heads
    d = getattr(model_config, "head_dim", model_config.hidden_size // hq)
    assignment = [u // (hq // hkv) for u in range(hq)]
    sample_count = (T + token_stride - 1) // token_stride
    all_wk = [torch.zeros(hkv, sample_count) for _ in range(n_layers)]
    all_wv = [torch.zeros(hkv, sample_count) for _ in range(n_layers)]
    capture_query = False
    # Validate before installing any hooks so unsupported architectures do not
    # leave partially installed capture state.
    for layer in model.model.layers:
        attention = layer.self_attn
        q_norm = getattr(attention, "q_norm", None)
        captured.setdefault("q", []).append(None)
        layer_index = len(captured["q"]) - 1
        def save_query(_module, _inputs, out, i=layer_index):
            if capture_query:
                captured["q"][i] = out.detach()
        if q_norm is not None:
            handles.append(q_norm.register_forward_hook(save_query))
        else:
            if getattr(model_config, "model_type", None) == "ministral3":
                def save_native(query, i=layer_index):
                    if capture_query:
                        captured["q"][i] = query.detach()
                handles.append(capture_native_ministral_query(attention, save_native))
            else:
                # Qwen2/Qwen2.5 has no q_norm. The q_proj output has the same
                # norm as the native positional query because RoPE is
                # orthogonal; reshape it to the attention query layout.
                def save_projected(_module, _inputs, out, i=layer_index):
                    if capture_query:
                        captured["q"][i] = out.view(out.shape[0], out.shape[1], hq, d).transpose(1, 2).detach()
                handles.append(attention.q_proj.register_forward_hook(save_projected))

    cache = None
    processed = 0

    def forward_chunk(start, end, boundary_step):
        nonlocal cache, capture_query
        implementation = "eager" if boundary_step else "sdpa"
        setter(implementation)
        if model.config._attn_implementation != implementation:
            raise RuntimeError(f"target did not switch attention implementation to {implementation}")
        capture_query = boundary_step
        positions = torch.arange(start, end, device=ids.device)
        with torch.no_grad():
            result = model(ids[:, start:end], past_key_values=cache,
                           cache_position=positions, position_ids=positions[None],
                           use_cache=True, output_attentions=boundary_step,
                           return_dict=True, logits_to_keep=1)
        capture_query = False
        cache = result.past_key_values
        return result

    try:
        full_attentions = None
        full_cache = None
        if capture_mode == "full":
            # One native full-sequence eager pass. Boundary rows are sliced
            # from this complete causal tensor below; no 256-token context
            # is treated as an independent sequence.
            output = forward_chunk(0, T, True)
            full_attentions = list(output.attentions) if output.attentions is not None else None
            output.attentions = None
            full_cache = cache
            processed = T
            del output
        for boundary in boundaries:
            if boundary >= T:
                continue
            if capture_mode == "full":
                if (full_attentions is None or len(full_attentions) != n_layers
                        or any(a is None or tuple(a.shape) != (1, hq, T, T)
                               for a in full_attentions)):
                    raise RuntimeError("full target attention probabilities were not captured")
                attentions = [a[:, :, boundary:boundary + 1, :boundary + 1]
                              for a in full_attentions]
                boundary_cache = full_cache
            else:
                if processed < boundary:
                    output = forward_chunk(processed, boundary, False)
                    del output
                # Boundary c is the first future query; native attention includes
                # token c itself, but fitting eligibility remains strictly [0, c).
                output = forward_chunk(boundary, boundary + 1, True)
                processed = boundary + 1
                attentions = list(output.attentions) if output.attentions is not None else None
                output.attentions = None
                boundary_cache = cache
                del output
            if attentions is None or len(attentions) != n_layers or any(x is None for x in attentions):
                raise RuntimeError("target boundary attention probabilities were not captured")
            for layer in range(n_layers):
                layer_attention = attentions[layer]
                attentions[layer] = None
                if tuple(layer_attention.shape) != (1, hq, 1, boundary + 1):
                    raise RuntimeError("boundary attention must contain exactly one native query row")
                if captured["q"][layer] is None:
                    raise RuntimeError("target boundary query was not captured")
                q_raw = captured["q"][layer]
                if capture_mode == "full":
                    # q_norm emits [B,T,H,D]; q_proj/native hooks emit
                    # [B,H,T,D]. Normalize before slicing the token axis.
                    if getattr(model.model.layers[layer].self_attn, "q_norm", None) is not None:
                        q = q_raw.transpose(1, 2)
                    else:
                        q = q_raw
                    if tuple(q.shape) != (1, hq, T, d):
                        raise RuntimeError("full target query geometry mismatch")
                    q = q[:, :, boundary:boundary + 1, :]
                else:
                    q = _to_bhtd(q_raw, 1, hq, 1, d)
                if capture_mode != "full":
                    captured["q"][layer] = None
                probs = layer_attention.float()
                q = q.to(probs.device).float()
                _, native_v = cache_layer_kv(boundary_cache, layer)
                if capture_mode == "full":
                    native_v = native_v[:, :, :boundary + 1, :]
                if tuple(native_v.shape) != (1, hkv, boundary + 1, d):
                    raise RuntimeError("boundary value cache does not match its native attention row")
                v = native_v.to(probs.device).float()
                expanded_v = v[:, torch.as_tensor(assignment, device=probs.device)]
                key_positions = torch.arange(boundary + 1, device=probs.device)
                causal = (key_positions <= boundary)[None, None, None].expand_as(probs)
                query_o = torch.matmul(probs.masked_fill(~causal, 0), expanded_v)
                wk, wv = compute_gqa_sensitivities(
                    probs, q, v, query_o, query_to_kv=assignment,
                    valid_mask=torch.ones(1, boundary + 1, device=probs.device, dtype=torch.bool),
                    causal_mask=causal)
                keep = torch.arange(0, boundary + 1, token_stride, device=probs.device)
                eligible = keep < boundary
                for weights, accumulator in ((wk, all_wk[layer]), (wv, all_wv[layer])):
                    if tuple(weights.shape) != (1, hkv, boundary + 1):
                        raise RuntimeError("unexpected boundary sensitivity shape")
                    # Same equal-boundary-mass reduction as before; unseen
                    # future sample positions contribute zero to the full row.
                    sampled = weights.index_select(-1, keep)[0].masked_fill(~eligible[None], 0)
                    mass = sampled.sum(-1, keepdim=True)
                    unit = torch.where(mass > 0, sampled / mass.clamp_min(1e-30),
                                       torch.zeros_like(sampled))
                    accumulator[:, :keep.numel()] += unit.cpu()
                del layer_attention, probs, q, native_v, v, expanded_v, key_positions
                del causal, query_o, wk, wv, keep, eligible, weights, accumulator, sampled, mass, unit
            del attentions
        if processed < T:
            output = forward_chunk(processed, T, False)
            del output
    finally:
        capture_query = False
        for handle in reversed(handles):
            handle.remove()
        captured.clear()
        setter(original_implementation)
    keys, values = [], []
    for layer in range(n_layers):
        key, value = cache_layer_kv(cache, layer)
        if key.shape[2] != T or value.shape[2] != T:
            raise RuntimeError("incremental target cache did not cover the full sequence")
        theta = rope_spec(model_config)
        positions = torch.arange(T, device=key.device)
        keep = torch.arange(0, T, token_stride, device=key.device)
        canonical_key = strip_rope(key[0].float(), positions, theta)
        keys.append(canonical_key[:, keep].transpose(0, 1).cpu())
        values.append(value[0].float()[:, keep.to(value.device)].transpose(0, 1).cpu())
        all_wk[layer] /= len(boundaries)
        all_wv[layer] /= len(boundaries)
    return keys, values, all_wk, all_wv


def capture_source_shard(model, ids, stride):
    with torch.no_grad():
        output = model(ids, use_cache=True, return_dict=True)
    T = ids.shape[1]
    model_config = text_config(model.config)
    theta = rope_spec(model_config)
    keys, values = [], []
    for layer in range(model_config.num_hidden_layers):
        key, value = cache_layer_kv(output.past_key_values, layer)
        # With device_map=auto, each layer's cache can live on a different
        # GPU from the input embeddings. Construct indices beside that cache.
        positions = torch.arange(T, device=key.device)
        keep = torch.arange(0, T, stride, device=key.device)
        keys.append(strip_rope(key[0].float(), positions, theta)[:, keep]
                    .transpose(0, 1).cpu())
        values.append(value[0].float()[:, keep.to(value.device)].transpose(0, 1).cpu())
    return {"keys": keys, "values": values, "token_count": T}


def capture_stage(model_id, rows, trace_dir, *, source, device="cuda",
                  device_map=None, dtype="bfloat16", local_files_only=False,
                  stride=4, boundaries=None, revision=None,
                  target_capture_mode="full"):
    load_kwargs = {}
    if not source:
        load_kwargs["attn_implementation"] = "sdpa"
    model = load_causal_lm(model_id, revision=revision, device=device, device_map=device_map,
                            dtype=dtype, local_files_only=local_files_only,
                            **load_kwargs)
    model_device = model_input_device(model)
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        ids = row.to(model_device).unsqueeze(0)
        if source:
            shard = capture_source_shard(model, ids, stride)
        else:
            shard = dict(zip(("keys", "values", "weights_k", "weights_v"),
                             _capture_target_weights(model, ids, stride, boundaries,
                                                     capture_mode=target_capture_mode)))
            shard["token_count"] = int(ids.shape[1])
        torch.save(shard, trace_dir / f"{'source' if source else 'target'}_{index:05d}.pt")
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return peak


def _probe_r2(X, Y, ridge_lambda=0.0):
    X = X.double()
    Y = Y.double()
    mx, my = X.mean(0), Y.mean(0)
    xc, yc = X - mx, Y - my
    A = xc.T @ xc / X.shape[0]
    if ridge_lambda:
        A.diagonal().add_(ridge_lambda)
    W = torch.linalg.solve(A, xc.T @ yc / X.shape[0])
    residual = yc - xc @ W
    total = yc.square().sum().clamp_min(1e-18)
    return float(1 - residual.square().sum() / total)


def select_source_layers(source_dir, target_dir, n_source_layers, n_target_layers,
                         n_heads, top_k, ridge_lambda=0.0, panel_size=4, head_map=None):
    """Centered, unregularized OLS probes with bounded target-layer panels.

    ENGINEERING ASSUMPTION: panel size controls statistics memory, not the
    mathematical probe. Each trace pair is loaded once per panel, for both
    K/V and all source layers/heads. ``ridge_lambda`` remains accepted for
    caller compatibility; source-layer ranking uses OLS, not ridge.
    """
    files = sorted(Path(target_dir).glob("target_*.pt"))
    if not files:
        raise ValueError("no target trace shards")
    if not 1 <= top_k <= n_source_layers or panel_size <= 0:
        raise ValueError("invalid top_k or layer-selection panel_size")
    scores = []
    for start in range(0, n_target_layers, panel_size):
        stop = min(start + panel_size, n_target_layers)
        count = stop - start
        buckets = {}
        n = 0
        for shard_index, target_path in enumerate(files):
            index = target_path.stem.split("_")[-1]
            src = torch.load(Path(source_dir) / f"source_{index}.pt",
                             map_location="cpu", weights_only=False)
            tgt = torch.load(target_path, map_location="cpu", weights_only=False)
            for component in ("keys", "values"):
                # X: [source, head, token, dim], Y: [target panel, head, token, dim].
                X = torch.stack(src[component]).permute(0, 2, 1, 3).double()
                Y = torch.stack(tgt[component][start:stop]).permute(0, 2, 1, 3).double()
                if head_map is not None:
                    from .cachebridge import validate_head_map
                    blocks = validate_head_map(
                        head_map, X.shape[1], n_heads,
                        mode="block" if any(len(b) > 1 for b in head_map) else "matched")
                    if len({len(b) for b in blocks}) != 1:
                        raise ValueError("layer-selection requires equal head-map block widths")
                    # [Ls,Ht,N,block*D], preserving layer/head feature order.
                    X = torch.stack([X[:, block].permute(0, 2, 1, 3).flatten(2)
                                     for block in blocks], dim=1)
                if (X.shape[0] != n_source_layers or X.shape[1] != n_heads
                        or Y.shape[:3] != (count, n_heads, X.shape[2])
                        or (head_map is None and Y.shape[-1] != X.shape[-1])):
                    raise ValueError("layer-selection trace geometry mismatch")
                if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
                    raise ValueError("non-finite layer-selection trace")
                batch = {
                    "x": X.sum(2), "y": Y.sum(2),
                    "xx": X.transpose(-1, -2) @ X,
                    "xy": torch.einsum("shni,thnj->tshij", X, Y),
                    "yy": Y.square().sum((2, 3)),
                }
                if component not in buckets:
                    buckets[component] = batch
                else:
                    for name, value in batch.items():
                        buckets[component][name].add_(value)
                if component == "keys":
                    tokens = X.shape[2]
                elif X.shape[2] != tokens:
                    raise ValueError("K/V observation counts differ")
            n += tokens
            print(f"selection panel {start}:{stop} shard {shard_index + 1}/{len(files)}",
                  flush=True)
            del src, tgt, X, Y, batch
        panel_scores = torch.zeros(count, n_source_layers, dtype=torch.float64)
        for stats in buckets.values():
            xx = stats["xx"] - stats["x"][..., :, None] * stats["x"][..., None, :] / n
            xy = stats["xy"] - stats["x"][None, ..., :, None] * stats["y"][:, None, ..., None, :] / n
            yy = stats["yy"] - stats["y"].square().sum(-1) / n
            for source_layer in range(n_source_layers):
                # Batch over heads, solving all panel targets together. gelsd
                # handles rank-deficient OLS without inventing ridge regularization.
                rhs = xy[:, source_layer].permute(1, 2, 0, 3).reshape(n_heads, xx.shape[-1], -1)
                W = torch.linalg.lstsq(xx[source_layer], rhs, driver="gelsd").solution
                W = W.reshape(n_heads, xx.shape[-1], count, -1).permute(2, 0, 1, 3)
                cross = xy[:, source_layer]
                residual = yy - 2 * (W * cross).sum((-2, -1))
                residual += (W * (xx[source_layer][None] @ W)).sum((-2, -1))
                r2 = 1 - residual / yy.clamp_min(1e-18)
                panel_scores[:, source_layer] += r2.mean(-1) / 2
        if not torch.isfinite(panel_scores).all():
            raise RuntimeError("non-finite OLS layer-selection scores")
        for offset, layer_scores in enumerate(panel_scores.tolist()):
            order = sorted(range(n_source_layers), key=lambda x: (-layer_scores[x], x))
            scores.append({"target_layer": start + offset, "ranked": order,
                           "scores": layer_scores, "selected": order[:top_k]})
        print(f"selection completed target layers {start}:{stop}", flush=True)
    return scores


def _fused_trace_fit(source_dir, target_dir, layer_index, selected, n_heads,
                     head_assignment, source_panel_heads, head_dim,
                     ridge_lambda, uniform_weights, chunk_size=1024,
                     max_sequences=None, debug_stats=None,
                     output_dtype=torch.float32):
    """Bounded two-pass trace fit using head-batched GEMM."""
    files = sorted(Path(target_dir).glob("target_*.pt"))
    if max_sequences is not None:
        files = files[:int(max_sequences)]
    p = len(selected) * source_panel_heads * head_dim
    raw = {"k": [[] for _ in range(n_heads)], "v": [[] for _ in range(n_heads)]}
    lengths = []
    for target_path in files:
        tgt = torch.load(target_path, map_location="cpu", weights_only=False)
        length = int(tgt["keys"][layer_index].shape[0])
        lengths.append(length)
        for head in range(n_heads):
            raw["k"][head].append(tgt["weights_k"][layer_index][head].double())
            raw["v"][head].append(tgt["weights_v"][layer_index][head].double())
    weights = {
        name: [torch.ones(sum(lengths)) if uniform_weights else
               repair_weights(torch.cat(parts), feature_width=p)
               for parts in heads]
        for name, heads in raw.items()
    }
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    total = {name: torch.zeros(n_heads, dtype=torch.float64)
             for name in ("k", "v")}
    sum_x = {"k": torch.zeros(n_heads, p, dtype=torch.float64),
             "v": torch.zeros(n_heads, p, dtype=torch.float64)}
    sum_y = {"k": torch.zeros(n_heads, head_dim, dtype=torch.float64),
             "v": torch.zeros(n_heads, head_dim, dtype=torch.float64)}

    def panels(src, tgt, component, start, stop):
        xs, ys = [], []
        for head in range(n_heads):
            source_head = head_assignment[head]
            if source_panel_heads == 1:
                source_index = source_head[0] if isinstance(source_head, (list, tuple)) else source_head
                x = torch.cat([src[component][layer][start:stop, source_index, :]
                               for layer in selected], dim=-1)
            else:
                block = head_assignment[head]
                if isinstance(block, int):
                    block = list(range(source_panel_heads))
                x = torch.cat([src[component][layer][start:stop, block, :].reshape(
                    stop - start, -1) for layer in selected], dim=-1)
            y = tgt[component][layer_index][start:stop, head, :]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    for pass_index in range(2):
        A = {name: torch.zeros(n_heads, p, p, dtype=torch.float64)
             for name in ("k", "v")} if pass_index else None
        B = {name: torch.zeros(n_heads, p, head_dim, dtype=torch.float64)
             for name in ("k", "v")} if pass_index else None
        for batch_index, target_path in enumerate(files):
            index = target_path.stem.split("_")[-1]
            src = torch.load(Path(source_dir) / f"source_{index}.pt",
                             map_location="cpu", weights_only=False)
            tgt = torch.load(target_path, map_location="cpu", weights_only=False)
            lo, hi = offsets[batch_index], offsets[batch_index + 1]
            for start in range(0, hi - lo, chunk_size):
                left, right = start, min(hi - lo, start + chunk_size)
                for name, component in (("k", "keys"), ("v", "values")):
                    x, y = panels(src, tgt, component, left, right)
                    w = torch.stack([weights[name][h][lo + left:lo + right]
                                     for h in range(n_heads)]).double()
                    if pass_index == 0:
                        total[name] += w.sum(dim=1)
                        sum_x[name] += (x.double() * w[:, :, None]).sum(dim=1)
                        sum_y[name] += (y.double() * w[:, :, None]).sum(dim=1)
                    else:
                        mux = sum_x[name] / total[name][:, None].clamp_min(1e-30)
                        muy = sum_y[name] / total[name][:, None].clamp_min(1e-30)
                        xh = ((x.double() - mux[:, None, :]) *
                              w[:, :, None].sqrt())
                        yh = ((y.double() - muy[:, None, :]) *
                              w[:, :, None].sqrt())
                        A[name] += torch.bmm(xh.transpose(1, 2), xh)
                        B[name] += torch.bmm(xh.transpose(1, 2), yh)
                    if debug_stats is not None:
                        debug_stats["max_x_panel"] = max(
                            debug_stats.get("max_x_panel", (0, 0, 0)),
                            tuple(x.shape))
                        debug_stats["max_y_panel"] = max(
                            debug_stats.get("max_y_panel", (0, 0, 0)),
                            tuple(y.shape))
        if pass_index == 1:
            out = {"source_layers": [int(x) for x in selected]}
            for name in ("k", "v"):
                mux = sum_x[name] / total[name][:, None].clamp_min(1e-30)
                muy = sum_y[name] / total[name][:, None].clamp_min(1e-30)
                eye = torch.eye(p, dtype=torch.float64)[None]
                W = torch.linalg.solve(A[name] + ridge_lambda * eye, B[name])
                out[f"W_{name}"] = W.to(output_dtype)
                out[f"mu_x{name}"] = mux.to(output_dtype)
                out[f"mu_y{name}"] = muy.to(output_dtype)
            return out


def fit_one_target_layer(source_dir, target_dir, layer_index, selected, n_heads,
                         head_assignment, head_dim, ridge_lambda=0.01,
                         stats_device="cpu", stats_dtype=torch.float32,
                         uniform_weights=False, backend="torch_bounded",
                         chunk_size=1024, max_sequences=None,
                         debug_stats=None, head_map=None):
    if any(isinstance(head, (list, tuple)) for head in head_assignment):
        from .cachebridge import normalize_head_assignment
        source_heads = max(head for block in head_assignment for head in block) + 1
        blocks = normalize_head_assignment(head_assignment, source_heads, n_heads)
        if head_map is not None and head_map != blocks:
            raise ValueError("head_assignment and head_map disagree")
        head_map = blocks
        head_assignment = [block[0] for block in blocks]
    if backend == "fused_equivalent":
        return _fused_trace_fit(source_dir, target_dir, layer_index, selected,
                                n_heads, head_map if head_map is not None else head_assignment,
                                1 if head_map is None else len(head_map[0]), head_dim,
                                ridge_lambda, uniform_weights, chunk_size,
                                max_sequences, debug_stats, stats_dtype)
    if backend not in ("reference", "torch_bounded", "torch"):
        raise ValueError(f"unsupported formal fit backend: {backend}")
    source_heads_per_target = len(head_map[0]) if head_map is not None else 1
    p = len(selected) * source_heads_per_target * head_dim
    stats_k = [WeightedRidgeStats(p, head_dim, dtype=stats_dtype,
                                  device=stats_device) for _ in range(n_heads)]
    stats_v = [WeightedRidgeStats(p, head_dim, dtype=stats_dtype,
                                  device=stats_device) for _ in range(n_heads)]
    files = sorted(Path(target_dir).glob("target_*.pt"))
    if max_sequences is not None:
        files = files[:int(max_sequences)]
    # Repair must use the selected feature width and the complete calibration
    # observation set, so gather only the small per-head sensitivity vectors
    # for this layer before the second streaming pass over regression tensors.
    raw_weights = {"k": [[] for _ in range(n_heads)],
                   "v": [[] for _ in range(n_heads)]}
    lengths = []
    for target_path in files:
        tgt = torch.load(target_path, map_location="cpu", weights_only=False)
        lengths.append(int(tgt["keys"][layer_index].shape[0]))
        for head in range(n_heads):
            raw_weights["k"][head].append(tgt["weights_k"][layer_index][head].double())
            raw_weights["v"][head].append(tgt["weights_v"][layer_index][head].double())
    repaired = {
        name: [torch.ones(sum(lengths)) if uniform_weights else
               repair_weights(torch.cat(chunks), feature_width=p)
               for chunks in heads]
        for name, heads in raw_weights.items()
    }
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    for batch_index, target_path in enumerate(files):
        index = target_path.stem.split("_")[-1]
        src = torch.load(Path(source_dir) / f"source_{index}.pt",
                         map_location="cpu", weights_only=False)
        tgt = torch.load(target_path, map_location="cpu", weights_only=False)
        for head, source_head in enumerate(head_assignment):
            source_block = head_map[head] if head_map is not None else [source_head]
            Xk = torch.cat([src["keys"][layer][:, source_block, :].reshape(src["keys"][layer].shape[0], -1)
                            for layer in selected], dim=-1)
            Xv = torch.cat([src["values"][layer][:, source_block, :].reshape(src["values"][layer].shape[0], -1)
                            for layer in selected], dim=-1)
            lo, hi = offsets[batch_index:batch_index + 2]
            wk = repaired["k"][head][lo:hi]
            wv = repaired["v"][head][lo:hi]
            stats_k[head].add_batch(Xk, tgt["keys"][layer_index][:, head, :], wk)
            stats_v[head].add_batch(Xv, tgt["values"][layer_index][:, head, :], wv)
    out = {"source_layers": [int(x) for x in selected]}
    for name, buckets in (("k", stats_k), ("v", stats_v)):
        solved = [bucket.finalize(ridge_lambda) for bucket in buckets]
        out[f"W_{name}"] = torch.stack([x[0] for x in solved])
        out[f"mu_x{name}"] = torch.stack([x[2] for x in solved])
        out[f"mu_y{name}"] = torch.stack([x[3] for x in solved])
    return out


def fit_one_target_layer_full_head(source_dir, target_dir, layer_index, selected,
                                   n_heads, head_dim, ridge_lambda=0.01,
                                   stats_device="cpu", stats_dtype=torch.float32,
                                   backend="torch_bounded", chunk_size=1024,
                                   max_sequences=None, debug_stats=None,
                                   source_panel_heads=None, uniform_weights=True):
    """Reference Full-Head fit over the same staged traces and weights."""
    # Eq. 7: every target head gets an independent affine map.  The input
    # contains every source KV head, while the output is one target head.
    if backend == "fused":
        raise NotImplementedError(
            "authors' original fused CUDA kernel is not public"
        )
    if backend == "fused_equivalent":
        return _fused_trace_fit(
            source_dir, target_dir, layer_index, selected, n_heads,
            list(range(source_panel_heads or n_heads)), source_panel_heads or n_heads, head_dim, ridge_lambda, uniform_weights,
            chunk_size, max_sequences, debug_stats, stats_dtype)
    if backend not in ("reference", "torch_bounded", "torch"):
        raise ValueError(f"unsupported formal fit backend: {backend}")
    source_panel_heads = int(source_panel_heads or n_heads)
    p = len(selected) * source_panel_heads * head_dim
    stats_k = [WeightedRidgeStats(p, head_dim, dtype=stats_dtype,
                                  device=stats_device) for _ in range(n_heads)]
    stats_v = [WeightedRidgeStats(p, head_dim, dtype=stats_dtype,
                                  device=stats_device) for _ in range(n_heads)]
    files = sorted(Path(target_dir).glob("target_*.pt"))
    if max_sequences is not None:
        files = files[:int(max_sequences)]
    repaired = None
    if not uniform_weights:
        raw = {name: [[] for _ in range(n_heads)] for name in ("k", "v")}
        for path in files:
            trace = torch.load(path, map_location="cpu", weights_only=False)
            for name in raw:
                for head in range(n_heads):
                    raw[name][head].append(trace[f"weights_{name}"][layer_index][head].double())
        repaired = {name: [repair_weights(torch.cat(parts), feature_width=p)
                           for parts in heads] for name, heads in raw.items()}
    offset = 0
    for target_path in files:
        index = target_path.stem.split("_")[-1]
        src = torch.load(Path(source_dir) / f"source_{index}.pt",
                         map_location="cpu", weights_only=False)
        tgt = torch.load(target_path, map_location="cpu", weights_only=False)
        Xk = torch.cat([src["keys"][layer][:, :source_panel_heads].reshape(-1, source_panel_heads * head_dim)
                        for layer in selected], dim=-1)
        Xv = torch.cat([src["values"][layer][:, :source_panel_heads].reshape(-1, source_panel_heads * head_dim)
                        for layer in selected], dim=-1)
        weights = torch.ones(Xk.shape[0])
        for head in range(n_heads):
            wk = weights if repaired is None else repaired["k"][head][offset:offset + len(Xk)]
            wv = weights if repaired is None else repaired["v"][head][offset:offset + len(Xv)]
            stats_k[head].add_batch(Xk, tgt["keys"][layer_index][:, head, :], wk)
            stats_v[head].add_batch(Xv, tgt["values"][layer_index][:, head, :], wv)
        offset += len(Xk)
    out = {"source_layers": [int(x) for x in selected]}
    for name, buckets in (("k", stats_k), ("v", stats_v)):
        solved = [bucket.finalize(ridge_lambda) for bucket in buckets]
        out[f"W_{name}"] = torch.stack([x[0] for x in solved])
        out[f"mu_x{name}"] = torch.stack([x[2] for x in solved])
        out[f"mu_y{name}"] = torch.stack([x[3] for x in solved])
    return out
