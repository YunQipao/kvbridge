#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PAIR_PROFILE="${PAIR_PROFILE:-}"
if [[ "$PAIR_PROFILE" == paper_ministral_3b_14b ]]; then
  SOURCE_MODEL="${SOURCE_MODEL:-$HOME/luotao/models/Ministral-3-3B-Base-2512}"
  TARGET_MODEL="${TARGET_MODEL:-$HOME/luotao/models/Ministral-3-14B-Base-2512}"
  TOP_K="${TOP_K:-20}"
  CALIBRATION_OUT="${CALIBRATION_OUT:-artifacts/ministral_3b_14b_n${N_CALIB:-50}_cachebridge}"
  FULL_HEAD_OUT="${FULL_HEAD_OUT:-artifacts/ministral_3b_14b_n${N_CALIB:-50}_full_head}"
  HEAD_LOCAL_OUT="${HEAD_LOCAL_OUT:-artifacts/ministral_3b_14b_n${N_CALIB:-50}_head_local}"
  RESULTS_DIR="${RESULTS_DIR:-results/ministral_3b_14b_n${N_CALIB:-50}}"
  MANIFEST_DIR="${MANIFEST_DIR:-results/manifests_ministral_3b_14b_n${N_CALIB:-50}}"
elif [[ -n "$PAIR_PROFILE" ]]; then
  printf 'Unknown PAIR_PROFILE: %s\n' "$PAIR_PROFILE" >&2
  exit 2
fi
SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-14B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-32B}"
if [[ "$PAIR_PROFILE" == paper_ministral_3b_14b ]]; then
  TARGET_TOKENIZER="${TARGET_TOKENIZER:-$TARGET_MODEL}"
fi
TARGET_TOKENIZER="${TARGET_TOKENIZER:-$SOURCE_MODEL}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
DATASET_FILE="${DATASET_FILE:-}"
if [[ -n "$DATASET_FILE" && ! -f "$DATASET_FILE" ]]; then
  printf 'Calibration file does not exist: %s\n' "$DATASET_FILE" >&2
  exit 2
fi
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
HEAD_MAP_MODE="${HEAD_MAP_MODE:-auto}"
TARGET_CAPTURE_MODE="${TARGET_CAPTURE_MODE:-full}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
N_CALIB="${N_CALIB:-50}"
SEQ_LEN="${SEQ_LEN:-1024}"
TOKEN_STRIDE="${TOKEN_STRIDE:-4}"
TOP_K="${TOP_K:-8}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-0.01}"
NUM_BOUNDARIES="${NUM_BOUNDARIES:-32}"
BOUNDARY_START="${BOUNDARY_START:-12}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"
SEED="${SEED:-42}"
CALIBRATION_OUT="${CALIBRATION_OUT:-artifacts/minimal_cachebridge}"
FULL_HEAD_OUT="${FULL_HEAD_OUT:-artifacts/minimal_full_head}"
HEAD_LOCAL_OUT="${HEAD_LOCAL_OUT:-artifacts/minimal_head_local}"
RESULTS_DIR="${RESULTS_DIR:-results/minimal}"
MANIFEST_DIR="${MANIFEST_DIR:-results/manifests}"
STAGE="${1:-all}"
if (( $# > 1 )); then
  printf 'Unexpected extra arguments. Set device mapping with DEVICE_MAP=balanced bash %s %s\n' "$0" "$STAGE" >&2
  exit 2
fi

MODEL_ARGS=()
[[ "$LOCAL_FILES_ONLY" == 1 ]] && MODEL_ARGS+=(--local-files-only)
DATA_ARGS=(--dataset "$DATASET")
# Model/tokenizer offline loading does not select a local calibration file.
[[ -n "$DATASET_FILE" ]] && DATA_ARGS=(--dataset-file "$DATASET_FILE")

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

fit_one() {
  local method="$1"
  local output="$2"
  local extra=()
  [[ "$method" != cachebridge ]] && extra+=(--reuse-calibration-manifest "$CALIBRATION_OUT/calibration_manifest.json"
                                           --reuse-layer-selection "$CALIBRATION_OUT/layer_selection.json")
  run python3 -m experiments.fit_cachebridge \
    --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --target-tokenizer "$TARGET_TOKENIZER" "${DATA_ARGS[@]}" \
    --n-calib "$N_CALIB" --seq-len "$SEQ_LEN" --seed "$SEED" --method "$method" \
    --ridge-lambda "$RIDGE_LAMBDA" --token-stride "$TOKEN_STRIDE" --top-k "$TOP_K" \
    --num-boundaries "$NUM_BOUNDARIES" --boundary-start "$BOUNDARY_START" \
    --head-map-mode "$HEAD_MAP_MODE" --target-capture-mode "$TARGET_CAPTURE_MODE" \
    --device "$DEVICE" --device-map "$DEVICE_MAP" --dtype bfloat16 \
    --stats-device cpu --fit-backend torch_bounded --output "$output" \
    "${extra[@]}" "${MODEL_ARGS[@]}"
}

preflight() {
  run python3 scripts/check_model_geometry.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --head-map-mode "$HEAD_MAP_MODE" \
    "${MODEL_ARGS[@]}"
  run python3 scripts/check_tokenizer_alignment.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --target-tokenizer "$TARGET_TOKENIZER" \
    "${DATA_ARGS[@]}" --n-calib "$N_CALIB" --seq-len "$SEQ_LEN" --seed "$SEED" \
    "${MODEL_ARGS[@]}"
}

fit() {
  mkdir -p "$RESULTS_DIR" "$MANIFEST_DIR"
  fit_one cachebridge "$CALIBRATION_OUT"
  fit_one full_head "$FULL_HEAD_OUT"
  fit_one head_local "$HEAD_LOCAL_OUT"
}

eval_task() {
  local mode="$1"
  local mapper_path="$2"
  local mapper=()
  mapper=(--mapper "$mapper_path")
  run python3 -m experiments.eval_paper_tasks \
    --task HellaSwag --mode "$mode" --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --device "$DEVICE" --device-map "$DEVICE_MAP" \
    --target-tokenizer "$TARGET_TOKENIZER" \
    "${mapper[@]}" --num-eval-samples "$EVAL_SAMPLES" --sampling systematic \
    --sampling-seed "$SEED" --evaluation-manifest "$MANIFEST_DIR/hellaswag.json" \
    --output "$RESULTS_DIR/hellaswag_${mode}.json" "${MODEL_ARGS[@]}"
}

eval_stage() {
  mkdir -p "$RESULTS_DIR" "$MANIFEST_DIR"
  run python3 -m experiments.eval_paper_tasks \
    --task HellaSwag --mode target_native --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --device "$DEVICE" --device-map "$DEVICE_MAP" \
    --target-tokenizer "$TARGET_TOKENIZER" \
    --num-eval-samples "$EVAL_SAMPLES" --sampling systematic --sampling-seed "$SEED" \
    --evaluation-manifest-output "$MANIFEST_DIR/hellaswag.json" \
    --output "$RESULTS_DIR/hellaswag_target_native.json" "${MODEL_ARGS[@]}"
  eval_task full_head "$FULL_HEAD_OUT"
  eval_task head_local "$HEAD_LOCAL_OUT"
  eval_task cachebridge "$CALIBRATION_OUT"
}

benchmark() {
  run python3 experiments/benchmark_transfer.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --full-head-mapper "$FULL_HEAD_OUT" --cachebridge-mapper "$CALIBRATION_OUT" \
    --prefix-lengths 128,512,1024 --warmup "$WARMUP" --repeat "$REPEAT" \
    --device "$DEVICE" --output "$RESULTS_DIR/transfer_benchmark.json" "${MODEL_ARGS[@]}"
}

summary() {
  run python3 -m experiments.summarize_minimal --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --results-dir "$RESULTS_DIR" --full-head "$FULL_HEAD_OUT" --head-local "$HEAD_LOCAL_OUT" \
    --cachebridge "$CALIBRATION_OUT" --benchmark "$RESULTS_DIR/transfer_benchmark.json" \
    --output "$RESULTS_DIR/summary.json"
}

case "$STAGE" in
  preflight) preflight ;;
  fit) fit ;;
  eval) eval_stage ;;
  benchmark) benchmark ;;
  summary) summary ;;
  all) preflight; fit; eval_stage; benchmark; summary ;;
  *) printf 'Usage: %s {preflight|fit|eval|benchmark|summary|all}\n' "$0" >&2; exit 2 ;;
esac
