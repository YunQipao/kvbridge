"""Reference CacheBridge primitives.

This module contains the tensor-level pieces of CacheBridge without model
loading or calibration-dataset code.  Tensor layouts are explicit:

* source KV traces: ``[source_layer, observation, kv_head, head_dim]``
* attention probabilities: ``[batch, query_head, query_token, key_token]``
* queries/values/outputs: ``[batch, query_head, token, feature]``

The implementation intentionally favors a readable reference path.  Fused
CUDA construction is a later optimization and is not hidden behind this API.
"""

from dataclasses import dataclass
import math

import torch


def validate_head_assignment(assignment, n_source_heads, n_target_heads):
    """Validate and return an explicit target-head -> source-head mapping."""
    if n_source_heads <= 0 or n_target_heads <= 0:
        raise ValueError("source and target KV-head counts must be positive")
    if assignment is None:
        if n_source_heads != n_target_heads:
            raise ValueError(
                "head-local CacheBridge requires an explicit assignment when "
                "source and target KV-head counts differ")
        assignment = list(range(n_target_heads))
    assignment = [int(x) for x in assignment]
    if len(assignment) != n_target_heads:
        raise ValueError(
            f"head_assignment has {len(assignment)} entries, expected "
            f"{n_target_heads}")
    if any(x < 0 or x >= n_source_heads for x in assignment):
        raise ValueError("head_assignment contains an invalid source KV head")
    return assignment


def validate_head_map(head_map, n_source_heads, n_target_heads, *, mode="matched"):
    """Validate target-KV-head -> ordered source-KV-head blocks."""
    if head_map is None:
        from .geometry import build_head_map
        head_map = build_head_map(n_source_heads, n_target_heads, mode)
    if len(head_map) != int(n_target_heads) or not all(head_map):
        raise ValueError("head_map must contain one non-empty list per target head")
    normalized = []
    for block in head_map:
        block = [int(x) for x in block]
        if any(x < 0 or x >= int(n_source_heads) for x in block):
            raise ValueError("head_map contains an invalid source KV head")
        normalized.append(block)
    if mode == "matched" and any(len(block) != 1 for block in normalized):
        raise ValueError("matched mode requires one source head per target head")
    if mode == "block":
        flattened = [head for block in normalized for head in block]
        if flattened != list(range(int(n_source_heads))):
            raise ValueError("block mode requires a contiguous, non-overlapping source-head partition")
    return normalized


def normalize_head_assignment(assignment, n_source_heads, n_target_heads):
    """Read legacy flat assignments or contiguous grouped assignments as blocks."""
    if assignment is not None and any(isinstance(x, (list, tuple)) for x in assignment):
        if not all(isinstance(x, (list, tuple)) for x in assignment):
            raise ValueError("head_assignment cannot mix scalar heads and groups")
        blocks = validate_head_map(assignment, n_source_heads, n_target_heads,
                                   mode="block")
        if len({len(block) for block in blocks}) != 1:
            raise ValueError("grouped assignment requires equal group sizes")
        return blocks
    return [[head] for head in validate_head_assignment(
        assignment, n_source_heads, n_target_heads)]


def build_head_local_features(source_kv, selected_source_layers,
                              target_head, head_assignment=None,
                              n_target_heads=None, head_map=None):
    """Gather one target head's head-local feature block.

    ``source_kv`` must have shape ``[Ls, N, Hs, D]``.  The returned tensor has
    shape ``[N, len(selected_source_layers) * block_width * D]``. Features
    concatenate layers in supplied order, then heads in their mapped order.
    """
    if source_kv.ndim != 4:
        raise ValueError("source_kv must have shape [layers, observations, heads, dim]")
    n_layers, _, n_source_heads, _ = source_kv.shape
    selected = [int(x) for x in selected_source_layers]
    if not selected:
        raise ValueError("selected_source_layers must not be empty")
    if any(x < 0 or x >= n_layers for x in selected):
        raise ValueError("selected_source_layers contains an invalid layer")
    if n_target_heads is None:
        n_target_heads = (len(head_map) if head_map is not None else
                          len(head_assignment) if head_assignment is not None else n_source_heads)
    if head_map is None:
        blocks = normalize_head_assignment(head_assignment, n_source_heads, int(n_target_heads))
    else:
        blocks = validate_head_map(
            head_map, n_source_heads, int(n_target_heads),
            mode="block" if any(len(block) > 1 for block in head_map) else "matched")
    target_head = int(target_head)
    if target_head < 0 or target_head >= len(blocks):
        raise ValueError("target_head is out of range")
    source_heads = blocks[target_head]
    return torch.cat([source_kv[layer, :, source_heads, :].reshape(source_kv.shape[1], -1)
                      for layer in selected], dim=-1)


def _validate_regression_inputs(X, Y, weights):
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be rank-2 matrices")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of observations")
    if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
        raise ValueError("weights must have shape [number of observations]")
    if not torch.is_floating_point(weights):
        weights = weights.float()
    if not torch.isfinite(weights).all():
        raise ValueError("weights must be finite")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative")
    if X.shape[0] == 0:
        raise ValueError("at least one observation is required")
    return weights


@dataclass
class WeightedRidgeStats:
    """Streaming weighted sufficient statistics for centered ridge."""

    n_features: int
    n_outputs: int
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"

    def __post_init__(self):
        if self.n_features <= 0 or self.n_outputs <= 0:
            raise ValueError("feature and output dimensions must be positive")
        self.sum_w = torch.zeros((), dtype=self.dtype, device=self.device)
        self.sum_wx = torch.zeros(self.n_features, dtype=self.dtype, device=self.device)
        self.sum_wy = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.sum_wxx = torch.zeros((self.n_features, self.n_features),
                                    dtype=self.dtype, device=self.device)
        self.sum_wxy = torch.zeros((self.n_features, self.n_outputs),
                                    dtype=self.dtype, device=self.device)

    def add_batch(self, X, Y, weights):
        weights = _validate_regression_inputs(X, Y, weights)
        if X.shape[1] != self.n_features or Y.shape[1] != self.n_outputs:
            raise ValueError("batch dimensions do not match the statistics object")
        X = X.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)
        weights = weights.to(device=self.device, dtype=self.dtype)
        self.sum_w += weights.sum()
        self.sum_wx += (X * weights[:, None]).sum(dim=0)
        self.sum_wy += (Y * weights[:, None]).sum(dim=0)
        self.sum_wxx += X.T @ (X * weights[:, None])
        self.sum_wxy += X.T @ (Y * weights[:, None])
        return self

    def finalize(self, ridge_lambda=0.01):
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.sum_w <= 0:
            raise ValueError("total weight must be positive")
        mu_x = self.sum_wx / self.sum_w
        mu_y = self.sum_wy / self.sum_w
        cov_xx = self.sum_wxx - torch.outer(self.sum_wx, self.sum_wx) / self.sum_w
        cov_xy = self.sum_wxy - torch.outer(self.sum_wx, self.sum_wy) / self.sum_w
        system = cov_xx + ridge_lambda * torch.eye(
            self.n_features, dtype=self.dtype, device=self.device)
        W = torch.linalg.solve(system, cov_xy)
        bias = mu_y - mu_x @ W
        return W, bias, mu_x, mu_y


def weighted_ridge_centered(X, Y, weights, ridge_lambda=0.01):
    """Reference weighted centered affine ridge solve.

    The sufficient statistics use the unnormalized objective from paper
    Eq. (8): ``sum_i w_i ||residual_i||^2 + lambda ||W||^2``.  The affine bias
    is recovered from weighted means and is therefore not regularized.
    """
    weights = _validate_regression_inputs(X, Y, weights).double().cpu()
    if ridge_lambda < 0 or weights.sum() <= 0:
        raise ValueError("ridge_lambda must be non-negative and total weight positive")
    X, Y = X.double().cpu(), Y.double().cpu()
    mu_x = (weights[:, None] * X).sum(0) / weights.sum()
    mu_y = (weights[:, None] * Y).sum(0) / weights.sum()
    centered_x = (X - mu_x) * weights.sqrt()[:, None]
    centered_y = (Y - mu_y) * weights.sqrt()[:, None]
    system = centered_x.T @ centered_x
    system.diagonal().add_(ridge_lambda)
    W = torch.linalg.solve(system, centered_x.T @ centered_y)
    return W, mu_y - mu_x @ W, mu_x, mu_y


def weighted_ridge_streaming(batches, n_features, n_outputs, ridge_lambda=0.01):
    """Fit from an iterable of ``(X, Y, weights)`` batches only."""
    stats = WeightedRidgeStats(n_features, n_outputs, device="cpu")
    for X, Y, weights in batches:
        stats.add_batch(X, Y, weights)
    return stats.finalize(ridge_lambda)


def _attention_mask(attention_probs, valid_mask=None, causal_mask=None):
    mask = torch.ones_like(attention_probs, dtype=torch.bool)
    if causal_mask is not None:
        if causal_mask.shape != attention_probs.shape:
            raise ValueError("causal_mask must match attention_probs shape")
        mask &= causal_mask.to(device=mask.device, dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.ndim == 2:
            if valid_mask.shape != (attention_probs.shape[0], attention_probs.shape[3]):
                raise ValueError("valid_mask must have shape [B, key_tokens]")
            mask &= valid_mask[:, None, None, :].to(device=mask.device, dtype=torch.bool)
        elif valid_mask.shape == attention_probs.shape:
            mask &= valid_mask.to(device=mask.device, dtype=torch.bool)
        else:
            raise ValueError("valid_mask must have shape [B, key_tokens] or attention shape")
    return mask


def _validate_attention_inputs(attention_probs, q, v, attention_output):
    if attention_probs.ndim != 4:
        raise ValueError("attention_probs must have shape [B, H, Tq, Tk]")
    if q.ndim != 4 or v.ndim != 4 or attention_output.ndim != 4:
        raise ValueError("q, v, and attention_output must have shape [B, H, T, D]")
    B, H, Tq, Tk = attention_probs.shape
    if q.shape[:3] != (B, H, Tq) or attention_output.shape[:3] != (B, H, Tq):
        raise ValueError("q and attention_output do not match attention dimensions")
    if v.shape[:3] != (B, H, Tk) or q.shape[-1] != v.shape[-1]:
        raise ValueError("v does not match attention key dimensions")


def compute_value_sensitivity(attention_probs, q, v, attention_output,
                              *, valid_mask=None, causal_mask=None):
    """Compute Eq. (15), returning ``[B, H, Tk]`` token sensitivities."""
    if causal_mask is None:
        raise ValueError("causal_mask is required to exclude invisible positions")
    _validate_attention_inputs(attention_probs, q, v, attention_output)
    mask = _attention_mask(attention_probs, valid_mask, causal_mask)
    return (attention_probs.masked_fill(~mask, 0).square()).sum(dim=2)


def compute_key_sensitivity(attention_probs, q, v, attention_output,
                            *, valid_mask=None, causal_mask=None):
    """Compute Eq. (16), returning ``[B, H, Tk]`` token sensitivities."""
    if causal_mask is None:
        raise ValueError("causal_mask is required to exclude invisible positions")
    _validate_attention_inputs(attention_probs, q, v, attention_output)
    mask = _attention_mask(attention_probs, valid_mask, causal_mask)
    probs = attention_probs.masked_fill(~mask, 0)
    d = q.shape[-1]
    q_factor = q.square().sum(dim=-1).div(d)[:, :, :, None]
    value_delta = v[:, :, None, :, :] - attention_output[:, :, :, None, :]
    distance = value_delta.square().sum(dim=-1)
    return (probs.square() * q_factor * distance).sum(dim=2)


def compute_gqa_sensitivities(attention_probs, q, v, attention_output=None,
                              *, query_to_kv=None, valid_mask=None,
                              causal_mask=None):
    """Compute Eq. (15)-(16) for grouped-query attention.

    Layouts are explicit: probabilities ``[B,Hq,Tq,Tk]``, query ``[B,Hq,Tq,D]``,
    values ``[B,Hkv,Tk,D]``.  ``query_to_kv[u]`` selects the KV head owned by
    query head ``u``.  If omitted, an integral contiguous GQA grouping is used.
    The returned tensors are ``[B,Hkv,Tk]`` and sum the query-head
    sensitivities belonging to each KV head, exactly as in the paper.
    """
    if attention_probs.ndim != 4 or q.ndim != 4 or v.ndim != 4:
        raise ValueError("GQA tensors must have layouts [B,H,T,D] and [B,H,Tq,Tk]")
    B, hq, tq, tk = attention_probs.shape
    if q.shape[:3] != (B, hq, tq) or q.shape[-1] != v.shape[-1]:
        raise ValueError("query and probability dimensions do not match")
    if v.shape[:3] != (B, v.shape[1], tk):
        raise ValueError("value tensor does not match attention key length")
    hkv = v.shape[1]
    if query_to_kv is None:
        if hq % hkv:
            raise ValueError("query heads must be divisible by KV heads")
        group = hq // hkv
        query_to_kv = [u // group for u in range(hq)]
    query_to_kv = [int(x) for x in query_to_kv]
    if len(query_to_kv) != hq or any(x < 0 or x >= hkv for x in query_to_kv):
        raise ValueError("invalid query_to_kv assignment")
    if attention_output is None:
        expanded_v = v[:, torch.as_tensor(query_to_kv, device=v.device)]
        mask = _attention_mask(attention_probs, valid_mask, causal_mask)
        attention_output = torch.matmul(attention_probs.masked_fill(~mask, 0), expanded_v)
    if attention_output.shape != (B, hq, tq, v.shape[-1]):
        raise ValueError("attention_output must have shape [B,Hq,Tq,D]")
    if causal_mask is None:
        raise ValueError("causal_mask is required to exclude invisible positions")
    mask = _attention_mask(attention_probs, valid_mask, causal_mask)
    probs = attention_probs.masked_fill(~mask, 0)
    q_factor = q.square().sum(dim=-1).div(q.shape[-1])
    expanded_v = v[:, torch.as_tensor(query_to_kv, device=v.device)]
    wk_per_query = compute_key_sensitivity(
        probs, q, expanded_v, attention_output,
        valid_mask=None, causal_mask=torch.ones_like(mask, dtype=torch.bool))
    wv_per_query = compute_value_sensitivity(
        probs, q, expanded_v, attention_output,
        valid_mask=None, causal_mask=torch.ones_like(mask, dtype=torch.bool))
    # The helper above uses the already-masked probabilities. Its output is
    # intentionally retained as a transparent reference implementation.
    wk = torch.zeros(B, hkv, tk, dtype=wk_per_query.dtype, device=wk_per_query.device)
    wv = torch.zeros_like(wk)
    for u, h in enumerate(query_to_kv):
        wk[:, h] += wk_per_query[:, u]
        wv[:, h] += wv_per_query[:, u]
    return wk, wv


def fit_head_local_streaming(source_batches, target_batches, selected_source_layers,
                              *, head_assignment, num_target_layers,
                              ridge_lambda=0.01, weights_k=None, weights_v=None,
                              stats_device="cpu"):
    """Fit compact K/V maps from bounded batches.

    Each batch is ``(source_layers, target_layers)`` where source/target are
    lists of ``[N,H,D]`` tensors.  Only the current batch is materialized;
    each target-layer/head/component owns a streaming sufficient-statistics
    accumulator.  This is the reference ``fit-backend=torch`` path.
    """
    if not source_batches:
        raise ValueError("source_batches must not be empty")
    first_source = source_batches[0]
    first_target = target_batches[0]
    hs = first_source[0].shape[1]
    ht = first_target[0].shape[1]
    d = first_source[0].shape[2]
    assignment = validate_head_assignment(head_assignment, hs, ht)
    selected = [[int(x) for x in row] for row in selected_source_layers]
    if len(selected) != num_target_layers:
        raise ValueError("selected_source_layers must match target layer count")
    stats = {}
    for layer in range(num_target_layers):
        p = len(selected[layer]) * d
        stats[layer] = [
            {"k": WeightedRidgeStats(p, d, device=stats_device),
             "v": WeightedRidgeStats(p, d, device=stats_device)}
            for _ in range(ht)
        ]
    for batch_index, (src, tgt) in enumerate(zip(source_batches, target_batches)):
        n = src[0].shape[0]
        wk = weights_k[batch_index] if weights_k is not None else torch.ones(n)
        wv = weights_v[batch_index] if weights_v is not None else torch.ones(n)
        for layer in range(num_target_layers):
            for h in range(ht):
                sh = assignment[h]
                xk = torch.cat([src[sl][:, sh, :] for sl in selected[layer]], dim=-1)
                xv = xk
                stats[layer][h]["k"].add_batch(xk, tgt[layer][:, h, :], wk)
                stats[layer][h]["v"].add_batch(xv, tgt[layer][:, h, :], wv)
    fitted = []
    for layer in range(num_target_layers):
        entries = {"source_layers": selected[layer]}
        for name, suffix in (("k", "k"), ("v", "v")):
            solved = [stats[layer][h][name].finalize(ridge_lambda)
                      for h in range(ht)]
            entries[f"W_{suffix}"] = torch.stack([item[0] for item in solved])
            entries[f"mu_x{suffix}"] = torch.stack([item[2] for item in solved])
            entries[f"mu_y{suffix}"] = torch.stack([item[3] for item in solved])
        fitted.append(entries)
    return fitted


def repair_weights(raw_sensitivity, feature_width, valid_mask=None):
    """Eq. (17)-(19): per-row mean-one weights with a Kish-ESS floor.

    The last dimension indexes observations; all leading dimensions are
    independent target layer/head/component groups. Invalid observations
    receive zero weight and do not count toward n or the effective sample size.
    """
    if raw_sensitivity.ndim < 1 or feature_width <= 0:
        raise ValueError("sensitivities need observations and positive feature width")
    if not torch.isfinite(raw_sensitivity).all() or (raw_sensitivity < 0).any():
        raise ValueError("sensitivities must be finite and non-negative")
    if valid_mask is None:
        mask = torch.ones_like(raw_sensitivity, dtype=torch.bool)
    else:
        mask = torch.broadcast_to(valid_mask.bool(), raw_sensitivity.shape)
    n = mask.sum(dim=-1, keepdim=True)
    if (n == 0).any():
        raise ValueError("no valid observations")
    raw = raw_sensitivity.double() * mask
    total = raw.sum(dim=-1, keepdim=True)
    # An all-zero attention trace supplies no preference: use uniform weights.
    normalized = torch.where(total > 0, raw * n / total.clamp_min(1e-300),
                             mask.double())
    cv2 = ((normalized - mask.double()).square() * mask).sum(dim=-1, keepdim=True) / n
    tau = n.clamp(max=2 * feature_width)
    alpha = torch.where(cv2 > 0,
                        ((n / tau - 1) / cv2.clamp_min(1e-300)).clamp(min=0).sqrt().clamp(max=1),
                        torch.ones_like(cv2))
    return ((1 + alpha * (normalized - 1)) * mask).to(raw_sensitivity.dtype)


def aggregate_boundary_sensitivities(raw_sensitivities, feature_width,
                                     valid_mask=None):
    """Implement the paper's per-boundary normalization before Eq.17.

    ``raw_sensitivities`` has layout ``[..., num_boundaries, observations]``.
    Every boundary is normalized to unit mass over eligible observations,
    boundaries are averaged with equal mass, and the ESS repair is applied to
    the resulting observation weights.  This helper is model-agnostic; model
    instrumentation must construct one raw sensitivity vector per boundary.
    """
    if raw_sensitivities.ndim < 2:
        raise ValueError("expected [..., boundaries, observations]")
    if not torch.isfinite(raw_sensitivities).all() or (raw_sensitivities < 0).any():
        raise ValueError("sensitivities must be finite and non-negative")
    if valid_mask is None:
        mask = torch.ones_like(raw_sensitivities, dtype=torch.bool)
    else:
        mask = torch.broadcast_to(valid_mask.bool(), raw_sensitivities.shape)
    mass = (raw_sensitivities * mask).sum(dim=-1, keepdim=True)
    normalized = torch.where(
        mass > 0,
        raw_sensitivities / mass.clamp_min(torch.finfo(raw_sensitivities.dtype).tiny),
        torch.zeros_like(raw_sensitivities),
    )
    normalized = normalized * mask
    combined = normalized.mean(dim=-2)
    combined = combined / combined.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(combined.dtype).tiny)
    return repair_weights(combined, feature_width, valid_mask=mask.any(dim=-2))


def make_log_causal_boundaries(seq_len, num_boundaries=32, start_token=12):
    """Return integer causal boundaries from ``start_token`` to ``seq_len-1``.

    The paper specifies logarithmic sampling but does not state collision
    handling or inclusive indexing.  We use inclusive token indices and fill
    rounded log-grid collisions with the smallest missing indices, preserving
    monotonicity.  Short sequences can therefore return fewer than the
    requested number of unique boundaries; this is explicit rather than
    silently duplicating a boundary.
    """
    seq_len, num_boundaries, start_token = map(int, (seq_len, num_boundaries, start_token))
    if seq_len <= 0 or num_boundaries <= 0:
        raise ValueError("seq_len and num_boundaries must be positive")
    lo, hi = max(0, start_token), seq_len - 1
    if lo > hi:
        return []
    count = min(num_boundaries, hi - lo + 1)
    values = torch.logspace(math.log10(max(1, lo)), math.log10(max(1, hi)),
                            steps=count).round().long().tolist()
    values = sorted(set(max(lo, min(hi, x)) for x in values))
    if len(values) < count:
        values = sorted(set(values).union(range(lo, hi + 1)))[:count]
    return values


def make_observation_indices(seq_len, stride=4, valid_mask=None, offset=0):
    """Return deterministic token indices sampled at ``offset, offset+stride,...``.

    Offset zero is an engineering choice because the paper specifies stride 4
    but does not specify the per-sequence phase; callers can pass another
    offset for phase balancing.  ``valid_mask`` is a one-dimensional boolean
    mask and excludes padding positions.
    """
    seq_len, stride, offset = map(int, (seq_len, stride, offset))
    if seq_len < 0 or stride <= 0 or offset < 0:
        raise ValueError("invalid sequence length, stride, or offset")
    indices = torch.arange(offset, seq_len, stride, dtype=torch.long)
    if valid_mask is not None:
        if valid_mask.ndim != 1 or valid_mask.shape[0] != seq_len:
            raise ValueError("valid_mask must have shape [seq_len]")
        indices = indices[valid_mask[indices].bool()]
    return indices


def cachebridge_metadata(*, source_model, target_model, num_source_layers,
                         num_target_layers, num_source_kv_heads,
                         num_target_kv_heads, head_dim, head_assignment,
                         selected_source_layers, rope_theta_source,
                         rope_theta_target, ridge_lambda=0.01,
                         weighting_method="attn_repair", token_stride=4,
                         num_boundaries=32, boundary_start_token=12,
                         head_map=None, head_map_mode="auto"):
    """Build the minimal serializable CacheBridge metadata schema."""
    blocks = normalize_head_assignment(
        head_assignment, int(num_source_kv_heads), int(num_target_kv_heads))
    from .geometry import build_head_map
    if head_map_mode == "auto":
        head_map_mode = "matched" if num_source_kv_heads == num_target_kv_heads else "block"
    if head_map is None:
        head_map = (blocks if head_map_mode == "matched"
                    else build_head_map(num_source_kv_heads, num_target_kv_heads,
                                        head_map_mode))
    head_map = validate_head_map(head_map, num_source_kv_heads,
                                 num_target_kv_heads, mode=head_map_mode)
    return {
        "method": "cachebridge",
        "format_version": 2,
        "source": source_model,
        "target": target_model,
        "num_source_layers": int(num_source_layers),
        "num_target_layers": int(num_target_layers),
        "num_source_kv_heads": int(num_source_kv_heads),
        "num_target_kv_heads": int(num_target_kv_heads),
        "n_kv_heads": int(num_target_kv_heads),
        "head_dim": int(head_dim),
        "head_assignment": head_map if any(len(b) > 1 for b in head_map) else [b[0] for b in head_map],
        "head_map": head_map,
        "head_map_mode": head_map_mode,
        "assignment_mode": "grouped_contiguous" if any(len(b) > 1 for b in head_map) else "matched",
        "group_size": len(head_map[0]),
        "selected_source_layers": [[int(x) for x in row]
                                    for row in selected_source_layers],
        "ridge_lambda": float(ridge_lambda),
        "lam": float(ridge_lambda),
        "weighting_method": weighting_method,
        "token_stride": int(token_stride),
        "num_boundaries": int(num_boundaries),
        "boundary_start_token": int(boundary_start_token),
        "tau_rule": "min(n, 2p)",
        "calibration_dataset": None,
        "n_calibration": None,
        "calibration_length": None,
        "source_revision": None,
        "target_revision": None,
        "rope_theta_source": float(rope_theta_source),
        "rope_theta_target": float(rope_theta_target),
    }
