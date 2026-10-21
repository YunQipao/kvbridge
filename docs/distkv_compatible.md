# Opt-in DistKV-compatible experiments

Existing paper drivers/defaults are unchanged. The dedicated driver uses local
Qwen2.5-14B-Instruct -> Qwen2.5-7B-Instruct, OpenHermes 300 x 512,
stride 4, top-k 12, lambda 0.01 and contiguous grouped heads
`[[0,1],[2,3],[4,5],[6,7]]`. Models load with local-files-only. Dataset
evaluation follows the existing datasets loader; set HF_HUB_OFFLINE=1 and
HF_DATASETS_OFFLINE=1 when all dataset configs are cached and offline use is
required. No packages or global configuration are changed by the driver.

## Compatibility scope

The initial implementation followed the user-specified recipe. The reference
scripts have now been located in the local DistKV `origin/c2c-cachebridge` Git
ref (not its checked-out main branch). New opt-in evaluation profiles, sample
auditing and explicit numerical settings are documented in
[distkv_eval_profiles.md](distkv_eval_profiles.md). Legacy drivers still use the
historical `current` evaluator; they are not silently migrated. Shared settings
alone do not certify exact accuracy parity: data labels, continuation semantics,
conversation separators and floating-point accumulation matter.

## Independent switches

* `--calibration-pack-mode records` (default): original per-record tokenization.
* `distkv_concat`: target-tokenizer IDs, add_special_tokens=False, no inserted
  delimiter between records, contiguous exact blocks; final incomplete block
  discarded. Insufficient complete blocks cause an explicit error. Text comes
  from `text`, or newline-joined conversation `value` fields. Both tokenizers
  must produce identical per-row IDs. Block IDs and original physical JSONL
  line, ID, text, and token slices are saved for provenance and paired reuse.
* `--layer-selection-mode paper_ols` (default): original unmodified centered OLS.
* `distkv_ridge_subsample`: one seed-controlled global position subset (default
  4096) shared by every K/V/head/layer, grouped centered ridge probes and held-in
  R2 ranking. Lambda only changes this opt-in selection path. Subsampled source
  and target activations are retained on CPU; full trace sequences are not loaded
  onto a GPU. This can still consume substantial host memory.
* `--attn-repair-mode auto` (default): original method-based weighting.
* `paper`: paper sensitivity/ESS with 32 boundaries and boundary start 12,
  including Head-Local/Full-Head if explicitly requested.
* `off`: uniform weights even with method=cachebridge. Metadata explicitly
  records this as uniform, not paper weighting. Existing torch_bounded solver,
  target capture and RoPE mathematics are unchanged.

Resume checks packing/selection settings; use a new output directory when
changing the experimental recipe. Packed manifest reuse rechecks token IDs
against saved original text with the current tokenizers. Reused layer selection
is recorded separately, including the original mode when config is available.

## Evaluation

`--protocol distkv_greedy --max-new-tokens 64` uses the supplied literal prompt
and argmax decoding without temperature, through identical native/mapped bridge
token continuation code. No chat template is inserted. EOS or max-new-tokens
ends decoding. Answers prefer `the correct answer is X`, then standalone A-E
(case insensitive); unparsed/invalid option predictions count as wrong. Redux
correct_answer annotations supersede original answers when present; rows without
a valid question/options/answer are excluded. Full evaluation uses all valid test
rows (typically around 5700), not a hardcoded count. Generated answers are saved.
Mapped target shared-prefix prefill remains zero. Native evaluates the same full
prompt with its own prefix prefill. Source-native uses the source model with the
same prompt/parser; tokenizer alignment is checked for every greedy mode.

New greedy manifests include a population SHA256 to prevent pairing different
row populations. Do not reuse old lm_eval manifests: conditional likelihood and
greedy generation are different protocols. Run native and mapped greedy on the
new shared manifest. Smoke and full manifests live in separate directories.

## Commands (from repository root)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4
# Paper mapper on packed DistKV calibration configuration:
bash scripts/run_distkv_compatible.sh paper fit
# DistKV-style subsample/uniform recipe (independent artifact):
bash scripts/run_distkv_compatible.sh distkv fit

# Shared 500-question population for native and both recipes:
bash scripts/run_distkv_compatible.sh paper native500
bash scripts/run_distkv_compatible.sh paper eval500
bash scripts/run_distkv_compatible.sh distkv eval500

# All valid test questions, separate shared manifest:
bash scripts/run_distkv_compatible.sh paper nativefull
bash scripts/run_distkv_compatible.sh paper evalfull
bash scripts/run_distkv_compatible.sh distkv evalfull
```

Source/target paths, JSONL, mapper directory, results root and device map can be
overridden with SOURCE_MODEL, TARGET_MODEL, DATASET_FILE, MAPPER_DIR, RESULTS_ROOT,
DEVICE_MAP. Fit/eval failures stop the script immediately. No fit is triggered
by an eval stage. Run stages sequentially when creating the initial manifest.

Tests are CPU mocks/toys; they have not been run by Codex:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m pytest -q tests/test_distkv_compat.py tests/test_formal_layer_selection.py \
  tests/test_formal_backend_equivalence.py tests/test_mmlu_redux_protocol.py \
  tests/test_mmlu_redux_fewshot_no_leakage.py tests/test_grouped_head_support.py

CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
```
