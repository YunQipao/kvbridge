#!/usr/bin/env bash
# Opt-in experiment. Does not source or alter paper-faithful default drivers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RECIPE="${1:-paper}"
STAGE="${2:-fit}"
case "$RECIPE" in
  paper) SELECTION=paper_ols; REPAIR=paper ;;
  distkv) SELECTION=distkv_ridge_subsample; REPAIR=off ;;
  *) echo 'Recipe must be paper or distkv' >&2; exit 2 ;;
esac
SOURCE_MODEL="${SOURCE_MODEL:-$HOME/luotao/models/Qwen2.5-14B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/luotao/models/Qwen2.5-7B-Instruct}"
DATASET_FILE="${DATASET_FILE:-$HOME/luotao/openhermes_calibration.jsonl}"
DEVICE_MAP="${DEVICE_MAP:-balanced_low_0}"
MAPPER_DIR="${MAPPER_DIR:-artifacts/distkv_compatible_qwen25_14bi_7bi_${RECIPE}_n300_len512_k12}"
RESULTS_ROOT="${RESULTS_ROOT:-results/distkv_compatible_qwen25_14bi_7bi}"

case "$STAGE" in
  fit)
    [[ -f "$DATASET_FILE" ]] || { echo "Missing JSONL: $DATASET_FILE" >&2; exit 2; }
    python -m experiments.fit_cachebridge \
      --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --target-tokenizer "$TARGET_MODEL" \
      --dataset-file "$DATASET_FILE" --n-calib 300 --seq-len 512 --seed 42 \
      --method cachebridge --head-map-mode auto \
      --calibration-pack-mode distkv_concat --layer-selection-mode "$SELECTION" \
      --selection-subsample 4096 --attn-repair-mode "$REPAIR" \
      --target-capture-mode boundary_chunked --top-k 12 --token-stride 4 \
      --ridge-lambda 0.01 --num-boundaries 32 --boundary-start 12 \
      --device cuda --device-map "$DEVICE_MAP" --dtype bfloat16 \
      --stats-device cpu --fit-backend torch_bounded --local-files-only \
      --output "$MAPPER_DIR"
    ;;
  eval500|evalfull|native500|nativefull|source500|sourcefull)
    SAMPLE_ARGS=()
    if [[ "$STAGE" == *500 ]]; then
      SAMPLE_ARGS=(--num-eval-samples 500); POPULATION=500
    else
      POPULATION=all
    fi
    # Both recipes and native baselines share one population manifest.
    RESULT_DIR="$RESULTS_ROOT/greedy_$POPULATION"
    MANIFEST="$RESULT_DIR/manifest.json"
    MANIFEST_ARGS=(--evaluation-manifest-output "$MANIFEST")
    [[ ! -f "$MANIFEST" ]] || MANIFEST_ARGS=(--evaluation-manifest "$MANIFEST")
    MODE=cachebridge
    MAPPER_ARGS=(--mapper "$MAPPER_DIR")
    RESULT_NAME="cachebridge_${RECIPE}"
    if [[ "$STAGE" == native* ]]; then
      MODE=target_native; MAPPER_ARGS=(); RESULT_NAME=target_native
    elif [[ "$STAGE" == source* ]]; then
      MODE=source_native; MAPPER_ARGS=(); RESULT_NAME=source_native
    fi
    if [[ "$MODE" == cachebridge && ! -f "$MAPPER_DIR/mapper.safetensors" ]]; then
      echo "Missing mapper: $MAPPER_DIR; run fit first" >&2; exit 2
    fi
    python -m experiments.eval_mmlu_redux \
      --mode "$MODE" --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
      "${MAPPER_ARGS[@]}" --dataset edinburgh-dawg/mmlu-redux-2.0 --split test \
      --protocol distkv_greedy --fewshot 0 --max-new-tokens 64 \
      "${SAMPLE_ARGS[@]}" --sampling systematic --sampling-seed 42 \
      "${MANIFEST_ARGS[@]}" --device cuda --device-map "$DEVICE_MAP" \
      --local-files-only --output "$RESULT_DIR/${RESULT_NAME}.json"
    ;;
  *) echo 'Stage: fit|eval500|evalfull|native500|nativefull|source500|sourcefull' >&2; exit 2 ;;
esac
