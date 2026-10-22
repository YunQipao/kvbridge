#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Paper CacheBridge method + DistKV experimental configuration
#
# Mapper method:
#   Paper OLS layer selection
#   grouped Head-Local (8 -> 4)
#   Paper Attn-Repair
#   ESS shrinkage
#   weighted centered ridge
#   RoPE-aware mapping
#
# Experimental configuration:
#   Qwen2.5-14B-Instruct -> Qwen2.5-7B-Instruct
#   OpenHermes, 300 x 512
#   stride=4, top-k=12, lambda=0.01
#   MMLU-Redux
#   DistKV exact evaluation protocol
# ============================================================

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------
# User-configurable settings
# ---------------------------
GPUS="${GPUS:-0,1,2,3}"
DEVICE_MAP="${DEVICE_MAP:-balanced_low_0}"

SOURCE_MODEL="${SOURCE_MODEL:-$HOME/luotao/models/Qwen2.5-14B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/luotao/models/Qwen2.5-7B-Instruct}"
CALIB_DATA="${CALIB_DATA:-$HOME/luotao/openhermes_calibration.jsonl}"

HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/sda1/iic_data/.cache/huggingface/datasets}"
DISTKV_REPO="${DISTKV_REPO:-/media/sda1/iic_data/luotao/DistKV}"

MAPPER_DIR="${MAPPER_DIR:-artifacts/qwen25_14bi_to_7bi_distkvcfg_papercb_n300_len512_k12}"

# Default is the formal full evaluation.
# To do a quick 500-question smoke:
#   SCOPE=smoke500 bash scripts/run_paper_cachebridge_distkvcfg.sh
SCOPE="${SCOPE:-full}"

case "$SCOPE" in
  full)
    RESULT_DIR="${RESULT_DIR:-results/qwen25_14bi_to_7bi_distkvcfg_papercb_full}"
    SAMPLE_ARGS=()
    ;;
  smoke500)
    RESULT_DIR="${RESULT_DIR:-results/qwen25_14bi_to_7bi_distkvcfg_papercb_500}"
    SAMPLE_ARGS=(--num-eval-samples 500)
    ;;
  *)
    echo "ERROR: SCOPE must be full or smoke500" >&2
    exit 2
    ;;
esac

MANIFEST="$RESULT_DIR/manifest.json"

# ---------------------------
# Offline / CUDA environment
# ---------------------------
export CUDA_VISIBLE_DEVICES="$GPUS"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "Paper CacheBridge + DistKV configuration"
echo "============================================================"
echo "repo             : $PWD"
echo "git branch       : $(git branch --show-current 2>/dev/null || true)"
echo "git commit       : $(git rev-parse HEAD 2>/dev/null || true)"
echo "GPUs             : $CUDA_VISIBLE_DEVICES"
echo "source           : $SOURCE_MODEL"
echo "target           : $TARGET_MODEL"
echo "calibration      : $CALIB_DATA"
echo "HF dataset cache : $HF_DATASETS_CACHE"
echo "DistKV repo      : $DISTKV_REPO"
echo "mapper           : $MAPPER_DIR"
echo "scope            : $SCOPE"
echo "results          : $RESULT_DIR"
echo "============================================================"

# ---------------------------
# Preconditions
# ---------------------------
[[ -d "$SOURCE_MODEL" ]] || {
    echo "ERROR: missing source model: $SOURCE_MODEL" >&2
    exit 2
}

[[ -d "$TARGET_MODEL" ]] || {
    echo "ERROR: missing target model: $TARGET_MODEL" >&2
    exit 2
}

[[ -f "$CALIB_DATA" ]] || {
    echo "ERROR: missing OpenHermes calibration JSONL: $CALIB_DATA" >&2
    exit 2
}

[[ -d "$DISTKV_REPO/.git" ]] || {
    echo "ERROR: missing DistKV git repository: $DISTKV_REPO" >&2
    exit 2
}

# Protect against accidentally using the old main implementation.
BRANCH="$(git branch --show-current 2>/dev/null || true)"
if [[ "$BRANCH" != "cachebridge-repro" ]]; then
    echo "ERROR: current branch is '$BRANCH'; expected cachebridge-repro." >&2
    echo "Do not auto-switch in this script. Check the repository first." >&2
    exit 2
fi

# ---------------------------
# Step 1: model geometry
# ---------------------------
echo
echo "================ Step 1/5: geometry check ================"

python scripts/check_model_geometry.py \
  --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" \
  --head-map-mode auto \
  --local-files-only

# ---------------------------
# Step 2: create + audit exact DistKV evaluation population
# ---------------------------
echo
echo "================ Step 2/5: DistKV protocol audit ================"

if [[ -e "$MANIFEST" ]]; then
    echo "ERROR: manifest already exists: $MANIFEST" >&2
    echo "Use a fresh RESULT_DIR to avoid silently reusing an old population." >&2
    exit 2
fi

python scripts/audit_distkv_protocol.py \
  --distkv-repo "$DISTKV_REPO" \
  --dataset edinburgh-dawg/mmlu-redux-2.0 \
  --split test \
  --manifest "$MANIFEST" \
  --create-manifest \
  --eval-profile distkv_exact \
  --sampling systematic \
  --sampling-seed 42 \
  "${SAMPLE_ARGS[@]}"

echo
echo "Audit passed."
echo "The run will use exactly the audited DistKV-compatible population."

# ---------------------------
# Step 3: fit Paper CacheBridge mapper
# ---------------------------
echo
echo "================ Step 3/5: fit Paper CacheBridge ================"

if [[ -s "$MAPPER_DIR/mapper.safetensors" ]]; then
    echo "Existing completed mapper found:"
    echo "  $MAPPER_DIR/mapper.safetensors"
    echo "Skipping fit."
else
    python -u -m experiments.fit_cachebridge \
      --source "$SOURCE_MODEL" \
      --target "$TARGET_MODEL" \
      --target-tokenizer "$TARGET_MODEL" \
      --dataset-file "$CALIB_DATA" \
      --n-calib 300 \
      --seq-len 512 \
      --seed 42 \
      --method cachebridge \
      --head-map-mode auto \
      --calibration-pack-mode distkv_concat \
      --layer-selection-mode paper_ols \
      --attn-repair-mode paper \
      --target-capture-mode boundary_chunked \
      --top-k 12 \
      --token-stride 4 \
      --ridge-lambda 0.01 \
      --num-boundaries 32 \
      --boundary-start 12 \
      --stats-dtype float64 \
      --device cuda \
      --device-map "$DEVICE_MAP" \
      --dtype bfloat16 \
      --stats-device cpu \
      --fit-backend torch_bounded \
      --output "$MAPPER_DIR" \
      --local-files-only
fi

[[ -s "$MAPPER_DIR/mapper.safetensors" ]] || {
    echo "ERROR: mapper was not produced successfully." >&2
    exit 2
}

# ---------------------------
# Common exact-DistKV evaluation arguments
# ---------------------------
COMMON_EVAL=(
  --source "$SOURCE_MODEL"
  --target "$TARGET_MODEL"

  --dataset edinburgh-dawg/mmlu-redux-2.0
  --split test

  --eval-profile distkv_exact
  --protocol distkv_greedy
  --max-new-tokens 64

  --evaluation-manifest "$MANIFEST"

  --device cuda
  --device-map "$DEVICE_MAP"
  --local-files-only
)

# ---------------------------
# Step 4: Native target baseline
# ---------------------------
echo
echo "================ Step 4/5: Native 7B baseline ================"

python -u -m experiments.eval_mmlu_redux \
  "${COMMON_EVAL[@]}" \
  --mode target_native \
  --output "$RESULT_DIR/target_native.json"

# ---------------------------
# Step 5: Paper CacheBridge
# ---------------------------
echo
echo "================ Step 5/5: Paper CacheBridge ================"

python -u -m experiments.eval_mmlu_redux \
  "${COMMON_EVAL[@]}" \
  --mode cachebridge \
  --mapper "$MAPPER_DIR" \
  --output "$RESULT_DIR/paper_cachebridge.json"

# ---------------------------
# Final summary
# ---------------------------
echo
echo "================ Final summary ================"

python - "$RESULT_DIR/target_native.json" \
         "$RESULT_DIR/paper_cachebridge.json" <<'PY'
import json
import sys

native = json.load(open(sys.argv[1]))
cb = json.load(open(sys.argv[2]))

na = float(native["accuracy"])
ca = float(cb["accuracy"])

print(f"Target Native:")
print(f"  correct  = {native['correct']}/{native['total']}")
print(f"  accuracy = {na * 100:.2f}%")

print()
print(f"Paper CacheBridge:")
print(f"  correct  = {cb['correct']}/{cb['total']}")
print(f"  accuracy = {ca * 100:.2f}%")

print()
if na:
    print(f"Target retention = {ca / na * 100:.2f}%")
    print(f"Absolute delta   = {(ca - na) * 100:+.2f} pp")

print()
print("Protocol:")
print(f"  eval_profile            = {cb.get('eval_profile')}")
print(f"  cache_continuation_mode = {cb.get('cache_continuation_mode')}")
print(f"  protocol                = {cb.get('protocol')}")
print(f"  mapped_tokens           = {cb.get('mapped_tokens')}")
print(f"  target_prefix_prefill   = {cb.get('target_prefix_prefill_tokens')}")
PY

echo
echo "============================================================"
echo "DONE"
echo "Mapper:"
echo "  $MAPPER_DIR"
echo "Results:"
echo "  $RESULT_DIR/target_native.json"
echo "  $RESULT_DIR/paper_cachebridge.json"
echo "Manifest:"
echo "  $MANIFEST"
echo "============================================================"
