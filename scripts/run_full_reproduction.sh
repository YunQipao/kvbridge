#!/usr/bin/env bash
set -euo pipefail

SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-14B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-32B}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
FULL_HEAD_OUT="${FULL_HEAD_OUT:-artifacts/qwen3_14b_to_32b_fullhead_500}"
CACHEBRIDGE_OUT="${CACHEBRIDGE_OUT:-artifacts/qwen3_14b_to_32b_cachebridge_500}"
HEAD_LOCAL_OUT="${HEAD_LOCAL_OUT:-artifacts/qwen3_14b_to_32b_headlocal_500}"
SWEEP_ROOT="${SWEEP_ROOT:-artifacts/qwen3_calibration_sweep}"
DRY_RUN=0; SKIP_FIT=0; SKIP_PAPER_TASKS=0; SKIP_SWEEP=0
SKIP_ATTRIBUTION=0; SKIP_CONSTRUCTION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;; --skip-fit) SKIP_FIT=1 ;;
    --skip-paper-tasks) SKIP_PAPER_TASKS=1 ;; --skip-sweep) SKIP_SWEEP=1 ;;
    --skip-attribution) SKIP_ATTRIBUTION=1 ;; --skip-construction) SKIP_CONSTRUCTION=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  (( DRY_RUN )) || "$@"
}

if (( ! SKIP_FIT )); then
  run python3 scripts/check_model_geometry.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --local-files-only
  run python3 scripts/check_tokenizer_alignment.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --dataset "$DATASET" --n-calib 500 --seq-len 1024 --local-files-only
  run python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --dataset "$DATASET" --n-calib 500 --seq-len 1024 --seed 42 --method cachebridge \
    --ridge-lambda 0.01 --token-stride 4 --top-k 8 --num-boundaries 32 --boundary-start 12 \
    --device-map auto --dtype bfloat16 --stats-device cpu --fit-backend torch_bounded \
    --output "$CACHEBRIDGE_OUT" --resume
  run python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --reuse-calibration-manifest "$CACHEBRIDGE_OUT/calibration_manifest.json" \
    --reuse-layer-selection "$CACHEBRIDGE_OUT/layer_selection.json" --n-calib 500 --seq-len 1024 \
    --seed 42 --method full_head --ridge-lambda 0.01 --token-stride 4 --top-k 8 \
    --num-boundaries 32 --boundary-start 12 --device-map auto --dtype bfloat16 \
    --stats-device cpu --fit-backend torch_bounded --output "$FULL_HEAD_OUT"
  run python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --reuse-calibration-manifest "$CACHEBRIDGE_OUT/calibration_manifest.json" \
    --reuse-layer-selection "$CACHEBRIDGE_OUT/layer_selection.json" --n-calib 500 --seq-len 1024 \
    --seed 42 --method head_local --ridge-lambda 0.01 --token-stride 4 --top-k 8 \
    --num-boundaries 32 --boundary-start 12 --device-map auto --dtype bfloat16 \
    --stats-device cpu --fit-backend torch_bounded --output "$HEAD_LOCAL_OUT"
  run python3 scripts/validate_mapper.py --mapper "$FULL_HEAD_OUT" --source "$SOURCE_MODEL" \
    --target "$TARGET_MODEL" --local-files-only --paper-faithful
  run python3 scripts/validate_mapper.py --mapper "$CACHEBRIDGE_OUT" --source "$SOURCE_MODEL" \
    --target "$TARGET_MODEL" --local-files-only --paper-faithful
fi

run python3 scripts/compare_mapper_size.py --full-head "$FULL_HEAD_OUT" \
  --cachebridge "$CACHEBRIDGE_OUT"

if (( ! SKIP_PAPER_TASKS )); then
  for task in HellaSwag ARC-Challenge WinoGrande MMLU; do
    manifest="results/${task}_evaluation_manifest.json"
    run python3 experiments/eval_paper_tasks.py --task "$task" --mode target_native \
      --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --sampling systematic \
      --sampling-seed 42 --evaluation-manifest-output "$manifest" \
      --output "results/${task}_target_native.json"
    run python3 experiments/eval_paper_tasks.py --task "$task" --mode full_head \
      --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --mapper "$FULL_HEAD_OUT" \
      --evaluation-manifest "$manifest" --output "results/${task}_full_head.json"
    run python3 experiments/eval_paper_tasks.py --task "$task" --mode head_local \
      --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --mapper "$HEAD_LOCAL_OUT" \
      --evaluation-manifest "$manifest" --output "results/${task}_head_local.json"
    run python3 experiments/eval_paper_tasks.py --task "$task" --mode cachebridge \
      --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --mapper "$CACHEBRIDGE_OUT" \
      --evaluation-manifest "$manifest" --output "results/${task}_cachebridge.json"
  done
  MMLU_REDUX_MANIFEST="results/mmlu_redux_evaluation_manifest.json"
  run python3 -m experiments.eval_mmlu_redux --mode target_native --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --evaluation-manifest-output "$MMLU_REDUX_MANIFEST"
  run python3 -m experiments.eval_mmlu_redux --mode full_head --source "$SOURCE_MODEL" \
    --target "$TARGET_MODEL" --mapper "$FULL_HEAD_OUT" --evaluation-manifest "$MMLU_REDUX_MANIFEST"
  run python3 -m experiments.eval_mmlu_redux --mode head_local --source "$SOURCE_MODEL" \
    --target "$TARGET_MODEL" --mapper "$HEAD_LOCAL_OUT" --evaluation-manifest "$MMLU_REDUX_MANIFEST"
  run python3 -m experiments.eval_mmlu_redux --mode cachebridge --source "$SOURCE_MODEL" \
    --target "$TARGET_MODEL" --mapper "$CACHEBRIDGE_OUT" --evaluation-manifest "$MMLU_REDUX_MANIFEST"
fi

run python3 experiments/benchmark_transfer.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
  --full-head-mapper "$FULL_HEAD_OUT" --cachebridge-mapper "$CACHEBRIDGE_OUT" \
  --prefix-lengths 128,512,1024

if (( ! SKIP_SWEEP )); then
  run env SOURCE_MODEL="$SOURCE_MODEL" TARGET_MODEL="$TARGET_MODEL" DATASET="$DATASET" ROOT="$SWEEP_ROOT" \
    bash scripts/run_calibration_sweep.sh
fi
if (( ! SKIP_CONSTRUCTION )); then
  for method in full_head cachebridge; do
    for backend in reference torch_bounded fused_equivalent; do
      output="results/construction_${method}_${backend}.csv"
      run python3 experiments/benchmark_construction.py --mode formal \
        --trace-dir "$CACHEBRIDGE_OUT/traces" \
        --layer-selection "$CACHEBRIDGE_OUT/layer_selection.json" \
        --method "$method" --budgets 50,100,200,500 --backend "$backend" \
        --output "$output"
    done
  done
fi
if (( ! SKIP_ATTRIBUTION )); then
  if [[ -z "${ATTRIBUTION_STREAM_FILE:-}" ]]; then
    echo "ATTRIBUTION_STREAM_FILE is required unless --skip-attribution is used" >&2
    exit 2
  fi
  run python3 experiments/eval_attribution.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --head-local "$HEAD_LOCAL_OUT" --cachebridge "$CACHEBRIDGE_OUT" \
    --stream-file "$ATTRIBUTION_STREAM_FILE" --local-files-only
fi
