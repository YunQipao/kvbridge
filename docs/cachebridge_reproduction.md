# CacheBridge Reproduction Notes

This P1 implementation adds only the synthetic, tensor-level reference
components. It does not load models or datasets.

## Paper mapping

- `build_head_local_features` implements Eq. (11): one architecture-indexed
  source KV head per target KV head, while retaining selected source layers.
- `compute_value_sensitivity` implements Eq. (15), `sum_u a_ui^2`.
- `compute_key_sensitivity` implements Eq. (16),
  `sum_u a_ui^2 * (||q_u||^2 / d) * ||v_i-o_u||^2`.
- `repair_weights` implements Eq. (17)-(19), mean-one normalization and
  the Kish effective-sample-size floor.
- `WeightedRidgeStats` and `weighted_ridge_centered` implement the weighted
  centered affine ridge objective in Eq. (8)-(10), with an unregularized bias.
- `make_log_causal_boundaries` uses the paper's 32 logarithmic boundaries from
  token 12 through token 1023 when `seq_len=1024`.

## Explicit assumptions

The paper does not specify the inclusive/exclusive boundary convention or
what to do when rounded logarithmic boundaries collide. This implementation
uses inclusive token indices, removes duplicates, and fills rounded collisions
with the smallest missing indices. For short sequences it returns fewer unique
boundaries rather than silently repeating observations.

The paper specifies stride 4 but not the sequence phase. The observation
helper defaults to offset 0 (`0, 4, 8, ...`); callers can provide another
offset for phase balancing.

The reference attention functions require explicit tensor layouts and accept
an optional causal mask. They do not infer or reshape model-specific layouts.

## Compatibility

Existing format-1 Full-Head mapper artifacts remain readable. CacheBridge
metadata uses format 2 and is selected explicitly by `method=cachebridge`.
No Full-Head code path is silently used for CacheBridge tensors.
