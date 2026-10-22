#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Qwen2.5-14B-Instruct -> Qwen2.5-7B-Instruct
#
# Sequential experiments:
#
#   1. Evaluate paper-style CacheBridge on full MMLU-Redux
#   2. Fit DistKV-compatible mapper
#   3. Evaluate DistKV-compatible mapper on the SAME manifest
#
# Important:
#   - Both evaluations use exactly the same MMLU-Redux manifest.
#   - GPU visibility is local to this script only.
# ============================================================


# ------------------------------------------------------------
# 0. Working directory
# ------------------------------------------------------------

REPO="$HOME/luotao/cachebridge-repro"
cd "$REPO"


# ------------------------------------------------------------
# 1. GPU / common environment
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_MODEL="$HOME/luotao/models/Qwen2.5-14B-Instruct"
TARGET_MODEL="$HOME/luotao/models/Qwen2.5-7B-Instruct"
TARGET_TOKENIZER="$TARGET_MODEL"

CALIB_DATA="$HOME/luotao/openhermes_calibration.jsonl"

RESULT_DIR="results/qwen25_14bi_to_7bi_distkv_full5700"
MANIFEST="$RESULT_DIR/manifest.json"

PAPER_MAPPER="artifacts/qwen25_14bi_to_7bi_openhermes_n300_len512_k12_paper_cb"
DISTKV_MAPPER="artifacts/qwen25_14bi_to_7bi_openhermes_n300_len512_k12_distkv_recipe"

LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"


# ------------------------------------------------------------
# 2. Basic checks
# ------------------------------------------------------------

echo "============================================================"
echo "Qwen2.5 CacheBridge / DistKV comparison"
echo "============================================================"
echo "Repository : $REPO"
echo "Source     : $SOURCE_MODEL"
echo "Target     : $TARGET_MODEL"
echo "Calibration: $CALIB_DATA"
echo "Manifest   : $MANIFEST"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "============================================================"


if [ ! -d "$SOURCE_MODEL" ]; then
    echo "ERROR: source model not found:"
    echo "  $SOURCE_MODEL"
    exit 1
fi

if [ ! -d "$TARGET_MODEL" ]; then
    echo "ERROR: target model not found:"
    echo "  $TARGET_MODEL"
    exit 1
fi

if [ ! -f "$CALIB_DATA" ]; then
    echo "ERROR: calibration file not found:"
    echo "  $CALIB_DATA"
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: MMLU-Redux manifest not found:"
    echo "  $MANIFEST"
    echo
    echo "You said the manifest-generation step has already been completed."
    exit 1
fi

if [ ! -d "$PAPER_MAPPER" ]; then
    echo "ERROR: paper CacheBridge mapper not found:"
    echo "  $PAPER_MAPPER"
    exit 1
fi


echo
echo "GPU status before experiments:"
nvidia-smi
echo


# ============================================================
# Experiment 1
# Paper-style CacheBridge + DistKV evaluation configuration
# ============================================================

echo "============================================================"
echo "[1/3] Evaluating Paper-style CacheBridge"
echo "============================================================"
echo "Mapper:"
echo "  $PAPER_MAPPER"
echo
echo "Evaluation:"
echo "  MMLU-Redux full shared manifest"
echo "  protocol = distkv_greedy"
echo "============================================================"

python -u -m experiments.eval_mmlu_redux \
  --mode cachebridge \
  --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" \
  --mapper "$PAPER_MAPPER" \
  --dataset edinburgh-dawg/mmlu-redux-2.0 \
  --split test \
  --protocol distkv_greedy \
  --max-new-tokens 64 \
  --evaluation-manifest "$MANIFEST" \
  --device cuda \
  --device-map balanced_low_0 \
  --output "$RESULT_DIR/paper_cachebridge.json" \
  --local-files-only \
  2>&1 | tee "$LOG_DIR/01_paper_cachebridge_eval.log"

echo
echo "[1/3] Paper-style CacheBridge evaluation completed."
echo


# ============================================================
# Experiment 2
# Fit DistKV-compatible closed-form recipe
# ============================================================

echo "============================================================"
echo "[2/3] Fitting DistKV-compatible CacheBridge recipe"
echo "============================================================"
echo "Configuration:"
echo "  calibration         = OpenHermes"
echo "  n_calib             = 300"
echo "  seq_len             = 512"
echo "  calibration packing = distkv_concat"
echo "  layer selection     = distkv_ridge_subsample"
echo "  selection samples   = 4096"
echo "  Attn-Repair         = OFF"
echo "  grouped head map    = auto (8 -> 4)"
echo "  top-k               = 12"
echo "  token stride        = 4"
echo "  ridge lambda        = 0.01"
echo "============================================================"

python -u -m experiments.fit_cachebridge \
  --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --dataset-file "$CALIB_DATA" \
  --n-calib 300 \
  --seq-len 512 \
  --seed 123 \
  --method cachebridge \
  --head-map-mode auto \
  --calibration-pack-mode distkv_concat \
  --layer-selection-mode distkv_ridge_subsample \
  --selection-subsample 4096 \
  --attn-repair-mode off \
  --target-capture-mode boundary_chunked \
  --top-k 12 \
  --token-stride 4 \
  --ridge-lambda 0.01 \
  --num-boundaries 32 \
  --boundary-start 12 \
  --device cuda \
  --device-map balanced_low_0 \
  --dtype bfloat16 \
  --stats-device cpu \
  --fit-backend torch_bounded \
  --output "$DISTKV_MAPPER" \
  --local-files-only \
  2>&1 | tee "$LOG_DIR/02_distkv_recipe_fit.log"

echo
echo "[2/3] DistKV-compatible mapper fitting completed."
echo


# Verify mapper was actually produced before evaluation.
if [ ! -d "$DISTKV_MAPPER" ]; then
    echo "ERROR: DistKV mapper directory was not produced:"
    echo "  $DISTKV_MAPPER"
    exit 1
fi


# ============================================================
# Experiment 3
# Evaluate DistKV-compatible recipe
# SAME MMLU-Redux manifest as Experiment 1
# ============================================================

echo "============================================================"
echo "[3/3] Evaluating DistKV-compatible recipe"
echo "============================================================"
echo "Mapper:"
echo "  $DISTKV_MAPPER"
echo
echo "Using SAME manifest:"
echo "  $MANIFEST"
echo "============================================================"

python -u -m experiments.eval_mmlu_redux \
  --mode cachebridge \
  --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" \
  --mapper "$DISTKV_MAPPER" \
  --dataset edinburgh-dawg/mmlu-redux-2.0 \
  --split test \
  --protocol distkv_greedy \
  --max-new-tokens 64 \
  --evaluation-manifest "$MANIFEST" \
  --device cuda \
  --device-map balanced_low_0 \
  --output "$RESULT_DIR/distkv_recipe_ours.json" \
  --local-files-only \
  2>&1 | tee "$LOG_DIR/03_distkv_recipe_eval.log"

echo
echo "[3/3] DistKV-compatible evaluation completed."
echo


# ============================================================
# Final summary
# ============================================================

echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================"

echo
echo "Paper CacheBridge result:"
echo "  $RESULT_DIR/paper_cachebridge.json"

echo
echo "DistKV-recipe result:"
echo "  $RESULT_DIR/distkv_recipe_ours.json"

echo
echo "Shared evaluation manifest:"
echo "  $MANIFEST"

echo
echo "Logs:"
echo "  $LOG_DIR/01_paper_cachebridge_eval.log"
echo "  $LOG_DIR/02_distkv_recipe_fit.log"
echo "  $LOG_DIR/03_distkv_recipe_eval.log"

echo
echo "============================================================"
