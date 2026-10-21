# Opt-in DistKV evaluation profiles

The default `--eval-profile current` preserves historical row filtering,
corrected labels, parser and bridge-once decoding. Existing results/artifacts
are not modified. The existing queue script is intentionally not migrated.

`distkv_canonical` and `distkv_exact` support two offline dataset loaders:

* Without `--dataset-local-path`: `cached_dataset_splits(--dataset, --split)`
  reads all prepared configs using HF_DATASETS_CACHE. Configs are sorted by name;
  per-config row order is preserved. No Hub configuration discovery is attempted.
* Optional `--dataset-local-path`: reads a Hugging Face `save_to_disk` dataset
  using `load_from_disk`. A Hub snapshot or prepared Arrow cache root is NOT
  such a directory. An explicitly empty path is rejected (omit the option).

Neither loader downloads data. Both apply identical raw-label/filter/prompt/parser
rules. Manifest/result metadata record dataset_loader=hf_cached_configs or
load_from_disk. Cached samples additionally record config_name, row_index and
sample_id="config:row_index"; original_index uses that same unique ID. Different
loaders cannot silently reuse a manifest. Same cached samples in both evaluators
allow an internal comparison, but do not certify an external dataset's version
or original row order.

Both profiles copy reference raw-label/filter/prompt/choice/parser behavior.
Corrected labels are not substituted. Greedy generation uses model EOS IDs,
64 new tokens by default and no chat template. All five model/mapper modes use
one shared decoder. Profiles never choose or refit a mapper.

* `distkv_canonical`: prefill N-1 tokens, feed token N once (standard bridge).
* `distkv_exact`: prefill N tokens, feed the final token again with
  cache_position=N-1, exactly as the reference. This can behave differently
  across Transformers versions. No hidden correction is made. Like the
  reference loop, when no EOS is generated it performs a final forward after
  the last retained token; that extra prediction is discarded.
* `--cache-continuation-mode` explicitly overrides the compatibility profile
  default; it is recorded in result metadata. Current rejects duplicate-last.

New manifests record ordered sample contents, original row indices, population
hash, dataset source and raw answer policy. Canonical/exact can share a manifest:
its profile/mode describe its creator; result metadata describes each actual
evaluation. Current-profile manifests cannot be reused here. Use NEW result
directories for each evaluator profile, including separate summary CSVs.

## CPU sample audit before model loading

Supply the actual save_to_disk dataset path, not a guessed location:

```bash
export DISTKV_DATASET=/absolute/path/to/mmlu-redux-2.0-test
python scripts/audit_distkv_protocol.py \
  --distkv-repo /media/sda1/iic_data/luotao/DistKV \
  --dataset-local-path "$DISTKV_DATASET" \
  --manifest results/distkv_canonical_smoke500_v1/manifest.json \
  --create-manifest --num-eval-samples 500
```

For full evaluation, use a new manifest path and omit `--num-eval-samples`.
`--create-manifest` never overwrites an existing file. The script extracts only
reference constants/sample functions from local Git origin/c2c-cachebridge,
never switches branches, loads models or imports DistKV's runtime. It prints
both population/selected hashes and fails on mismatch. A matching sample audit
certifies samples/prompts/labels, NOT numerical mapper or decode equivalence.

No export or download is needed when using the existing 57-config cache:

```bash
export HF_DATASETS_CACHE=/media/sda1/iic_data/.cache/huggingface/datasets
python scripts/audit_distkv_protocol.py \
  --distkv-repo /media/sda1/iic_data/luotao/DistKV \
  --dataset edinburgh-dawg/mmlu-redux-2.0 --split test \
  --manifest results/distkv_exact_cached_first500_v1/manifest.json \
  --create-manifest --eval-profile distkv_exact \
  --num-eval-samples 500 --sampling first
```

`first` takes the first N parseable samples AFTER DistKV filtering. `systematic`
retains its existing evenly spaced selection. Both compatibility profiles allow
first; current rejects it. Audit applies extracted reference functions to the
same in-memory config groups without changing the datasets module or DistKV.
MATCH=true in cached mode certifies reference sample processing on those groups,
not equivalence to an unavailable external load_from_disk file.

## Calibration numerical switches

Historical defaults remain stats=float64, selection seed inherited from --seed,
paper_ols, auto repair. An explicit DistKV-style numerical recipe adds:

```
--layer-selection-mode distkv_ridge_subsample --selection-subsample 4096
--selection-seed 123 --stats-dtype float32 --attn-repair-mode off
```

The subsample follows manual_seed/randperm without sorting and is restored to
that order after loading trace shards. Probes and bounded fitting use the
requested precision. No torch_bounded solver or paper selection mathematics
was changed. Streaming sufficient-statistics accumulation is not bitwise
identical to DistKV's dense centered float32 GEMM. Model RNG history, hardware,
precision of stored/de-RoPE traces and conversation text preprocessing can
also affect numerical parity. In particular existing distkv_concat uses newline
joining for conversation-only records; reference uses spaces. Text-field JSONL
records avoid that particular discrepancy. Fused-equivalent float32 statistics
are rejected rather than silently accumulating float64.

Old artifacts remain unchanged; new numerical settings require new artifact
directories (resume rejects changed seed/precision). `--selection-seed` changes
only DistKV subsampling; it cannot affect paper_ols.

## Evaluation driver

By default the driver uses cached configs. Optionally set DISTKV_DATASET to a
verified save_to_disk directory. SAMPLING defaults to systematic; first is opt-in:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_distkv_profile_eval.sh distkv_canonical smoke500
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_distkv_profile_eval.sh distkv_canonical full
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_distkv_profile_eval.sh distkv_exact smoke500
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_distkv_profile_eval.sh distkv_exact full
SAMPLING=first CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_distkv_profile_eval.sh distkv_exact smoke500
```

Each invocation audits samples, then runs native/paper-mapper/recipe-mapper on
the same manifest, sequentially. No fit or GPU idle detection is performed.
It refuses to overwrite evaluation JSONs; choose a new RESULT_DIR when rerunning.
The default RECIPE_MAPPER is the existing float64 artifact: changing evaluator
does NOT turn it into a float32-refitted mapper. To compare a new numerical
recipe, fit to a new artifact then set RECIPE_MAPPER explicitly.

Keep the three evaluation JSONs as primary results: the historical evaluator's
summary CSV keys rows by mode, so its two cachebridge rows overwrite one another
in that CSV (not in their separately named JSONs). This historical behavior was
not changed as part of adding opt-in evaluation profiles.
