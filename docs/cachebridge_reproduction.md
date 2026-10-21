# CacheBridge Reproduction Audit

This repository keeps the original kvbridge Full-Head mapper and adds a
CacheBridge path. This document distinguishes implementation from runtime
validation; no accuracy, latency, memory, or paper number is inferred from
synthetic tests or source review.

## PAPER-FAITHFUL

The implementation follows the public equations for Head-Local mapping,
target-layer-specific centered single-layer OLS selection, content-space
RoPE handling, GQA-aware Attn-Repair (Eq. 15/16), per-boundary unit-mass
normalization, equal boundary aggregation, ESS repair with
`tau=min(n,2p)`, and weighted centered affine ridge (Eq. 8-10).

Matched Qwen3 KV geometry uses the explicit architecture assignment
`head_assignment=[0,...,Hkv-1]`; this is derived from config, not model name.
The formal fitter records calibration rows, token IDs are required to match
exactly between tokenizers, and source/target traces are staged separately.
If a local target snapshot omits tokenizer files, pass an explicit
`--target-tokenizer` reference. The minimal Qwen3 driver defaults this to the
source tokenizer because the matched Qwen3 pair shares the vocabulary; this
is an explicit compatibility assumption, not a general tokenizer fallback.

## ENGINEERING-EQUIVALENT conventions

**ENGINEERING-EQUIVALENT:** a boundary `c` denotes the zero-based first future
query and eligible sampled prefix positions satisfy `position < c`. The query
used is `c`, not `c+1`. The paper does not fully specify its Python-level
index convention; the manifest records this exact contract.
Rounded logarithmic collisions are removed and short sequences return the
available unique boundaries.

## Formal calibration

`experiments/fit_cachebridge.py` supports FineWeb-Edu or a local text/JSONL
file, deterministic seed selection, exact tokenizer alignment, 32 boundaries,
stride 4, top-k source selection, target eager-attention capture, per-layer
and per-head K/V weights, staged `.pt` trace shards, streaming sufficient
statistics, resume markers, and mapper artifacts. `torch_bounded` is a
bounded-panel algorithmic equivalent: it does not concatenate the full
calibration corpus into one `X` or `Y` matrix.

The formal target capture uses Transformers `attn_implementation="eager"` as
a correctness reference and explicitly aggregates query heads into their GQA
KV heads. It is not a performance implementation.

## Artifact

Artifacts include `mapper.safetensors`, `mapper.json`, `metadata.json`,
`calibration_manifest.json`, `calibration_config.json`, `layer_selection.json`,
and `fit_summary.json`. Formal validation uses:

```bash
python3 scripts/validate_mapper.py --mapper artifacts/run \
  --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B --paper-faithful
```

The validator rejects uniform smoke weighting, missing boundaries, missing
calibration, wrong target-layer selection, shape errors, and non-finite maps.
Legacy Full-Head format-1 artifacts remain loadable.

## ENGINEERING-EQUIVALENT implementations

**ENGINEERING-EQUIVALENT:** trace shards plus CPU float64 sufficient
statistics implement the published bounded construction at algorithm level.
They are intentionally slower and simpler than a fused implementation.

**NOT EXACTLY REPRODUCIBLE:** the authors' fused CUDA kernel source and
its exact low-level scheduling are not public. `--fit-backend fused` fails
explicitly rather than fabricating equivalence or speed.

`kvbridge.fused_fit` exposes `reference`, `torch_bounded`, and
`fused_equivalent`. The last one uses bounded chunks, weighted centering, and
head-batched GEMM. It is an engineering-equivalent implementation, not the
authors' original CUDA kernel and not a basis for claiming Figure 7.

The block-PCA control is not enabled: the paper does not specify enough
implementation detail to recover it unambiguously, so it is not guessed.

The public `fused_equivalent` backend uses bounded panels and head-batched
`torch.bmm`. It is an engineering-equivalent implementation. The authors'
original fused CUDA kernel, launch scheduling, and exact sample IDs are not
reproducible from the public paper.

The MMLU-Redux script implements a conditional-likelihood mapped-cache path
for the common dataset schema and fails on unknown schema/few-shot protocol;
the exact authors' example IDs and any unreleased evaluation harness changes
are not reproducible from the paper alone. The transfer benchmark measures
CacheBridge application with CUDA events when CUDA is available; it does not
claim paper latency or include upstream prefill in that metric.

## Manual validation

No runtime results are claimed by static code work. Run the small formal path:

```bash
bash scripts/run_manual_validation.sh
```

The default is Qwen3-1.7B to Qwen3-4B, 50 rows, sequence length 1024,
FineWeb-Edu, Attn-Repair, and `torch_bounded`. Override `N_CALIB` or
`SEQ_LEN` only when necessary and record such runs as reduced-scale checks.

For the requested paper-scale command:

```bash
bash scripts/run_full_reproduction.sh --dry-run
bash scripts/run_full_reproduction.sh
```

This requires local/offline availability of both models, FineWeb-Edu, a
working CUDA/Transformers installation, and sufficient memory. No default
script silently falls back from CUDA or from mapped cache to target-native
prefix prefill.

## ADDITIONAL EVALUATION

MMLU-Redux is separate from the four paper task families. Its default
protocol is the installed lm-evaluation-harness task definition; `--protocol
simple` is retained only as a debug fallback.

## Status labels

- **IMPLEMENTED:** formal code paths, legacy Full-Head compatibility, artifact
  validation, mapped transfer, conditional-likelihood evaluator interface,
  and application timing interface.
- **STATICALLY_REVIEWED:** source imports, CLI paths, `compileall`, and
  whitespace checks may be reviewed in the final audit; these are not runtime
  model validation.
- **REQUIRES_MANUAL_TEST:** real Qwen3 calibration, MMLU results, GPU memory,
  latency, and semantic handoff.
- **NOT_IMPLEMENTED:** the authors' original fused CUDA kernel and exact
  unreleased benchmark sample/protocol details.

## Paired protocol

Formal reproduction now writes independent artifacts:

```text
artifacts/qwen3_14b_to_32b_fullhead_500
artifacts/qwen3_14b_to_32b_cachebridge_500
```

The Full-Head artifact declares `method=full_head` and
`weighting_method=uniform`; CacheBridge declares `method=cachebridge` and
`weighting_method=attn_repair`. Evaluation rejects an artifact whose method
does not match its mode. The two fits reuse the CacheBridge calibration
manifest and layer-selection manifest, so they do not independently shuffle
or select rows/layers.

Systematic evaluation sampling uses `floor(i*N/m)`. The paper does not publish
the exact example IDs or this exact index convention, so generated evaluation
manifests are marked **ENGINEERING-EQUIVALENT**.

Paper-task modes consume one explicit per-task evaluation manifest. Attribution
K/V R2 accepts a tokenized HellaSwag evaluation manifest, while NLL@4K uses a
separate deterministic 16-stream manifest. NLL pre-fills `prefix[:-1]`,
processes one bridge token, and teacher-forces the continuation; mapped runs
record zero target shared-prefix prefill tokens.

Full-Head Eq. 7 is implemented as independent per-target-head affine maps:
`[H_target, k*H_source*d_source, d_target]`. Legacy flattened format-1
artifacts remain readable.

The public repository still does not contain the authors' original fused
CUDA/Triton kernel. `kvbridge.fused_fit` is a mathematical reference/fallback
and must not be used to claim Figure 7 or author-kernel timings.

## Layer-selection I/O And Resume

**ENGINEERING-EQUIVALENT:** centered OLS ranking accumulates float64 sums,
source Gram matrices, target squared norms, and source/target cross-products.
It solves unregularized probes with an SVD least-squares solver, including
rank-deficient inputs. The final mapper ridge parameter does not regularize
these selection probes.

**ENGINEERING ASSUMPTION:** `--selection-panel-size 4` bounds target-layer
statistics. Each source/target shard pair is loaded once per panel, jointly
for K/V and every matched head. Statistics memory depends on layer/head/feature
geometry and panel size, not total observation count. With 36 target layers
and 50 sequences, the default requires 900 shard loads instead of 201,600
in the previous nested implementation. This is a load-count calculation,
not a measured speedup. Progress reports identify the panel and shard.

`--resume` reuses completed trace stages and a completed
`layer_selection.json`. That file is published via an atomic rename.
Changes to recorded calibration settings are rejected before trace reuse.
There is no per-panel selection checkpoint or per-layer mapper-fit checkpoint;
an interrupted selection or fit repeats that stage. Keep existing traces
when resuming rather than deleting the artifact directory. A completion
marker is not a checksum or a guarantee of trace numerical correctness.

**REQUIRES_MANUAL_TEST:** panel/materialized OLS equivalence, singular probes,
panel-size invariance, shard-load regression, actual wall time and memory.
The added tests are in `tests/test_formal_layer_selection.py`.

## Lightweight Offline Validation

Use the short debug pipeline before expensive calibration:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_timed_validation.sh quick-units
CUDA_VISIBLE_DEVICES=0 bash scripts/run_timed_validation.sh quick-model
```

`quick` runs both and is now the default stage. The real-model check uses a
single Python process, cached models only, two local texts, length 64, stride 4,
four boundaries and one fixed source layer per target layer. It captures real
target eager-attention sensitivities, applies the existing ESS repair, fits
head-local weighted ridge, saves/reloads the mapper, and forwards only a
held-out context's suffix through the target using mapped KV. Non-finite KV or
logits and unexpected geometry/cache lengths fail explicitly.

**ENGINEERING ASSUMPTION:** fixed debug support replaces layer selection
only in this debug entry point. With 32 observations and width 128, the paper's
ESS floor is n; repair can yield uniform weights by the actual ESS rule.
This is not a weighting override. The check does not validate useful
semantic transfer, paper-scale attention weighting, benchmark scoring, or
layer-selection quality. Debug artifacts are separate from formal traces
and `--paper-faithful` rejects their metadata.

**ENGINEERING-EQUIVALENT:** offline tokenizer loading resolves a cached Hub
snapshot directory with `snapshot_download(local_files_only=True)` before
passing it to AutoTokenizer. This avoids the Transformers 4.57.6 Mistral
regex metadata lookup for a remote model ID without changing tokenization
rules or suppressing a failed alignment check.

If both trace stages are complete, formal `--resume` uses the stored
manifest without contacting FineWeb or tokenizers. It validates recorded
settings, shard names/counts and observation counts. A legacy manifest without
an alignment attestation emits a warning: trusting prior capture provenance
is not a new token-ID check and does not checksum or independently verify
the trace content. Fresh runs record the alignment check in their manifest.

Formal continuation:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_timed_validation.sh fit
CUDA_VISIBLE_DEVICES=0 bash scripts/run_timed_validation.sh artifact
CUDA_VISIBLE_DEVICES=0 LIMIT=20 bash scripts/run_timed_validation.sh eval
```

`artifact` validates and inspects the formal mapper and runs handoff without
loading MMLU. `eval` still needs the actual MMLU-Redux dataset. A fresh
FineWeb calibration or `preflight` alignment still needs dataset access;
`--local-files-only` concerns models, not the streaming dataset. For a local
calibration file, set `DATASET_FILE=/path/to/text.jsonl` and use a new `OUT`
rather than changing the recorded dataset for existing traces.

**REQUIRES_MANUAL_TEST:** the new offline loading/resume tests, debug pipeline
and modified loader integrations. No runtime success or timing is claimed.

## Minimal CacheBridge Validation

The smallest quality/efficiency loop is intentionally limited to HellaSwag:

```bash
bash scripts/run_minimal_cachebridge_validation.sh all
```

It fits and evaluates `target_native`, `full_head`, `head_local`, and
`cachebridge` with the same calibration rows, selected source layers, ridge
lambda, token stride, and evaluation manifest. The default is 50 calibration
sequences, length 1024, stride 4, 32 boundaries, top-k 8, lambda 0.01, and
200 HellaSwag examples. Each method has its own mapper artifact, while the
three fits reuse `artifacts/minimal_cachebridge/calibration_manifest.json`
and `layer_selection.json`.

The stage commands are `preflight`, `fit`, `eval`, `benchmark`, `summary`, and
`all`. Override `SOURCE_MODEL`, `TARGET_MODEL`, `N_CALIB`, or `EVAL_SAMPLES`
through the environment. Evaluation manifests are task-specific; a manifest
generated for HellaSwag must never be consumed by ARC-Challenge, WinoGrande,
or MMLU because their population sizes differ.

The summary records retention as `mapped_accuracy / target_native_accuracy`,
keeps Full-Head as the baseline, reports mapper file sizes for all three
methods, and labels transfer timing as a current implementation measurement.
Reduced model pairs are marked `paper_pair=false` and `reduced_scale=true`.
The original fused CUDA kernel is not reproduced; the summary therefore
always records `fused_fit_paper_kernel_reproduced=false`.

### Smoke Troubleshooting

The harness adapter chooses an available test split, then validation. Merely
having a `test_docs()` method does not imply that the task has test documents;
HellaSwag uses validation. An explicitly requested unavailable split fails
instead of silently selecting another population.

If fitting prints `saved` and then aborts with `PyGILState_Release` during
interpreter finalization, the fit stage has still failed: its nonzero exit
prevents the remaining mapper fits from running. Calibration now explicitly
closes the streaming iterator on early exit and errors. This does not establish
the cause of a native shutdown crash or guarantee that it fixes one. To isolate
streaming, export the saved calibration manifest's records to local JSONL and
set `DATASET_FILE` for a fresh smoke run. Keep the original artifacts for
comparison, and require all stages to exit successfully before accepting the
experiment as complete. Do not suppress the abort or force a successful exit.

## Qwen2.5 14B to 7B grouped KV heads

The geometry preflight, formal fitter and minimal driver now default to
`HEAD_MAP_MODE=auto`: equal KV-head counts retain identity mapping; a larger,
evenly divisible source count uses contiguous groups. `matched` explicitly
requires equal counts, while `block` explicitly requests the grouped path.
Different head dimensions and source-head expansion (4 to 8) remain unsupported.

For Qwen2.5-14B to Qwen2.5-7B, the groups are
`[[0,1],[2,3],[4,5],[6,7]]`. With top-k 8 and head dimension 128, Head-Local and
CacheBridge use 2048 input features per target KV head; Full-Head still uses all
8 source heads and 8192 features. Features are ordered by selected source layer,
then source head within the group, then channel. Target-head sensitivity,
weighted ridge, ESS repair and RoPE strip/reapply are unchanged. New grouped
artifacts store nested `head_assignment` as well as `head_map`; legacy flat
assignments and previous block artifacts remain readable.

MMLU-Redux uses `python -m experiments.eval_mmlu_redux`, not the original MMLU
task in `eval_paper_tasks`. It accepts `--device-map balanced_low_0` in native,
Full-Head, Head-Local and CacheBridge modes. Dispatched models are never moved
as whole models; source prefill uses the source embedding device and mapped
cache layers are staged to their target execution devices. Mapped scoring does
not prefill the shared prefix on the target. Custom `--output` directories also
contain the automatically generated manifest and MMLU-Redux CSV summary.
`--local-files-only` applies to models/tokenizers, not datasets or harness
dependencies; prepare those separately or use existing offline caches.

## Ministral Compatibility Audit

The formal target-weight capture currently requires `attention.q_norm` and a
simple `rope_theta` RoPE path. Configurations with non-empty RoPE scaling, or
attention modules without `q_norm`, fail explicitly with:
`Ministral reproduction requires additional architecture validation`.
This preserves the validated Qwen path rather than guessing model-specific
module or RoPE behavior. Ministral reproduction requires additional
architecture validation.
