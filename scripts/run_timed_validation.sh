#!/usr/bin/env bash
# Manual-only validation. Does not download models in pytest or delete traces.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-1.7B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
OUT="${OUT:-artifacts/qwen3_1.7b_to_4b_formal}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
DATASET_FILE="${DATASET_FILE:-}"
N_CALIB="${N_CALIB:-50}"
SEQ_LEN="${SEQ_LEN:-1024}"
LOG_DIR="${LOG_DIR:-logs/manual_$(date +%Y%m%d_%H%M%S)}"
STAGE="${1:-quick}"
mkdir -p "$LOG_DIR"
DATA_ARGS=(--dataset "$DATASET")
if [[ -n "$DATASET_FILE" ]]; then
    DATA_ARGS=(--dataset-file "$DATASET_FILE")
fi
MODEL_ARGS=()
if [[ "${LOCAL_FILES_ONLY:-1}" == 1 ]]; then
    MODEL_ARGS=(--local-files-only)
fi

run() {
    local name="$1"
    shift
    printf '\n[%s] %s\n' "$(date -Iseconds)" "$name"
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    local started=$SECONDS
    if "$@" 2>&1 | tee "$LOG_DIR/$name.log"; then
        printf '%s,%s,0\n' "$name" "$((SECONDS - started))" >> "$LOG_DIR/timing.csv"
    else
        local status=$?
        printf '%s,%s,%s\n' "$name" "$((SECONDS - started))" "$status" >> "$LOG_DIR/timing.csv"
        printf 'FAILED: %s; inspect %s/%s.log\n' "$name" "$LOG_DIR" "$name" >&2
        return "$status"
    fi
}

quick_units() {
    run quick_units python -m pytest -q tests/test_offline_loading.py \
        tests/test_offline_resume.py tests/test_formal_layer_selection.py \
        tests/test_cachebridge_artifact_validation.py
}

quick_model() {
    # A separate artifact and one source/target load, not the formal traces.
    run quick_model python -m scripts.quick_cachebridge_check \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --device cuda \
        --output "${QUICK_OUT:-artifacts/quick_debug_$(date +%Y%m%d_%H%M%S)}"
}

preflight() {
    run environment python -c \
        'from kvbridge.model_utils import print_runtime_diagnostics, require_device; print_runtime_diagnostics(); require_device("cuda")'
    run compileall python -m compileall -q kvbridge experiments scripts tests
    run whitespace git diff --check
    run pytest python -m pytest -q
    run geometry python scripts/check_model_geometry.py \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --local-files-only
    run alignment python scripts/check_tokenizer_alignment.py \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" "${DATA_ARGS[@]}" \
        --n-calib "$N_CALIB" --seq-len "$SEQ_LEN" --seed 42 --local-files-only
    run real_kv python scripts/smoke_real_kv.py \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
        --device cuda --device-map auto "${MODEL_ARGS[@]}"
    run rope python scripts/validate_qwen_rope.py \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --device cuda --seq-len 64 \
        "${MODEL_ARGS[@]}"
    run attention python scripts/smoke_real_attention.py \
        --model "$TARGET_MODEL" --device cuda --seq-len 32 "${MODEL_ARGS[@]}"
}

fit() {
    run fit python -m experiments.fit_cachebridge \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" "${DATA_ARGS[@]}" \
        --n-calib "$N_CALIB" --seq-len "$SEQ_LEN" --seed 42 \
        --method cachebridge --ridge-lambda 0.01 --token-stride 4 --top-k 8 \
        --num-boundaries 32 --boundary-start 12 --device cuda --dtype bfloat16 \
        --stats-device cpu --fit-backend torch_bounded --selection-panel-size 4 \
        --output "$OUT" --resume "${MODEL_ARGS[@]}"
}

artifact_checks() {
    if [[ ! -f "$OUT/mapper.json" || ! -f "$OUT/mapper.safetensors" ]]; then
        printf 'Mapper incomplete: %s; finish fit first.\n' "$OUT" >&2
        return 1
    fi
    run mapper python scripts/validate_mapper.py \
        --mapper "$OUT" --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
        --paper-faithful --local-files-only
    run selection python scripts/inspect_layer_selection.py --mapper "$OUT"
    run handoff python scripts/test_handoff.py \
        --mapper "$OUT" --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
        --device cuda --device-map auto --assert-no-target-prefill \
        --dtype bfloat16 --max-new-tokens 32 "${MODEL_ARGS[@]}"
}

postfit() {
    artifact_checks
    run target_20 python -m experiments.eval_mmlu_redux \
        --mode target_native --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
        --device cuda --fewshot 0 --limit 20 "${MODEL_ARGS[@]}" \
        --output results/mmlu_redux_target_native_20.json
    run cachebridge_20 python -m experiments.eval_mmlu_redux \
        --mode cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
        --mapper "$OUT" --device cuda --fewshot 0 --limit 20 "${MODEL_ARGS[@]}" \
        --output results/mmlu_redux_cachebridge_20.json
}

evaluation() {
    # Both modes get the same limit. Preserve paired results before using a
    # different limit: the existing summary CSV keys rows only by mode.
    local limit="${LIMIT:-100}"
    local limit_args=()
    if [[ "$limit" != full ]]; then
        limit_args=(--limit "$limit")
    fi
    for mode in target_native cachebridge; do
        local mapper_args=()
        if [[ "$mode" == cachebridge ]]; then
            mapper_args=(--mapper "$OUT")
        fi
        run "${mode}_${limit}" python -m experiments.eval_mmlu_redux \
            --mode "$mode" --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
            "${mapper_args[@]}" --device cuda --fewshot 0 "${limit_args[@]}" "${MODEL_ARGS[@]}" \
            --output "results/mmlu_redux_${mode}_${limit}.json"
    done
    cp results/mmlu_redux_summary.csv "results/mmlu_redux_summary_${limit}.csv"
}

benchmark() {
    # Start with bounded lengths. Extend PREFIX_LENGTHS only after checking VRAM.
    run mapping_benchmark python -m experiments.benchmark_transfer \
        --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --mapper "$OUT" \
        --device cuda --prefix-lengths "${PREFIX_LENGTHS:-512,1024,2048,4096}" \
        --warmup 3 --repeat 10 --output results/transfer_benchmark.json
}

case "$STAGE" in
    quick) quick_units; quick_model ;;
    quick-units) quick_units ;;
    quick-model) quick_model ;;
    all) preflight; fit; postfit ;;
    preflight) preflight ;;
    fit) fit ;;
    postfit) postfit ;;
    artifact) artifact_checks ;;
    eval) evaluation ;;
    benchmark) benchmark ;;
    *) printf 'Usage: bash %s {quick|quick-units|quick-model|all|preflight|fit|artifact|postfit|eval|benchmark}\n' "$0" >&2; exit 2 ;;
esac
printf '\nTiming records: %s/timing.csv (stage,wall_seconds,exit_code)\n' "$LOG_DIR"
