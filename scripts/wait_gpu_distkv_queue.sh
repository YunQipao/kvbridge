#!/usr/bin/env bash
# Run explicitly with bash. This script never kills another GPU user's task.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_CHECKS="${IDLE_CHECKS:-3}"
ROUND_SECONDS="${ROUND_SECONDS:-1800}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-500}"
MAX_IDLE_UTIL="${MAX_IDLE_UTIL:-5}"
RESULTS_DIR="${RESULTS_DIR:-results/qwen25_14bi_to_7bi_distkv_full5700}"
PAPER_MAPPER="artifacts/qwen25_14bi_to_7bi_openhermes_n300_len512_k12_paper_cb"
RECIPE_MAPPER="${RECIPE_MAPPER:-artifacts/qwen25_14bi_to_7bi_openhermes_n300_len512_k12_distkv_recipe}"
SOURCE_MODEL="${SOURCE_MODEL:-$HOME/luotao/models/Qwen2.5-14B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/luotao/models/Qwen2.5-7B-Instruct}"
DATASET_FILE="${DATASET_FILE:-$HOME/luotao/openhermes_calibration.jsonl}"
command -v nvidia-smi >/dev/null
command -v flock >/dev/null
command -v "$PYTHON_BIN" >/dev/null
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ && "$IDLE_CHECKS" =~ ^[1-9][0-9]*$ && "$ROUND_SECONDS" =~ ^[1-9][0-9]*$ ]] || exit 2
exec 9> .distkv_gpu_queue.lock
flock -n 9 || { echo 'Another copy of this queue is running.' >&2; exit 2; }
for path in "$SOURCE_MODEL" "$TARGET_MODEL" "$DATASET_FILE" "$PAPER_MAPPER/mapper.safetensors"; do
    [[ -e "$path" ]] || { echo "Missing: $path" >&2; exit 2; }
done
is_complete_result() {
    local path="$1" mode="$2"
    [[ -s "$path" ]] || return 1
    "$PYTHON_BIN" - "$path" "$mode" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1]))
    if value.get("mode") != sys.argv[2]: raise ValueError("mode mismatch")
    total = int(value.get("total", 0))
    if total <= 0 or int(value.get("correct", -1)) < 0: raise ValueError("missing totals")
except Exception as exc:
    print(f"invalid or incomplete result: {sys.argv[1]} ({exc})", file=sys.stderr)
    raise SystemExit(1)
PY
}

assert_or_absent() {
    local path="$1" mode="$2"
    [[ ! -e "$path" ]] && return 0
    is_complete_result "$path" "$mode" || {
        echo "Refusing to overwrite incomplete/incompatible result: $path" >&2
        exit 2
    }
    echo "Will reuse completed result: $path"
}

assert_or_absent "$RESULTS_DIR/target_native.json" target_native
assert_or_absent "$RESULTS_DIR/paper_cachebridge.json" cachebridge
assert_or_absent "$RESULTS_DIR/distkv_recipe_ours.json" cachebridge
if [[ -e "$RECIPE_MAPPER" ]]; then
    [[ -s "$RECIPE_MAPPER/mapper.safetensors" && -s "$RECIPE_MAPPER/metadata.json" && -s "$RECIPE_MAPPER/fit_summary.json" ]] || {
        echo "Refusing to use incomplete mapper: $RECIPE_MAPPER" >&2; exit 2;
    }
    echo "Will reuse completed mapper: $RECIPE_MAPPER"
fi

gpu_idle() {
    "$PYTHON_BIN" - "$MAX_IDLE_MEMORY_MIB" "$MAX_IDLE_UTIL" <<'PY'
import csv, subprocess, sys
def query(args):
    result = subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'], text=True)
    return list(csv.reader(result.splitlines(), skipinitialspace=True))
try:
    gpus = query(['--query-gpu=index,uuid,memory.used,utilization.gpu'])
    chosen = {int(row[0]): row for row in gpus if int(row[0]) in (0, 1, 2, 3)}
    if len(chosen) != 4:
        raise RuntimeError('GPU 0,1,2,3 not all visible')
    uuids = {row[1] for row in chosen.values()}
    apps = query(['--query-compute-apps=gpu_uuid,pid'])
    busy = any(row[0] in uuids for row in apps)
    for index, row in sorted(chosen.items()):
        print(f'GPU {index}: {row[2]} MiB, utilization {row[3]}%', flush=True)
        busy |= int(row[2]) > int(sys.argv[1]) or int(row[3]) > int(sys.argv[2])
    sys.exit(1 if busy else 0)
except Exception as exc:
    print(f'GPU check failed; refusing to launch: {exc}', file=sys.stderr)
    sys.exit(2)
PY
}

wait_idle() {
    local consecutive status round_started remaining
    while true; do
        round_started=$SECONDS
        consecutive=0
        echo "Starting idle confirmation round ($IDLE_CHECKS checks, ${POLL_SECONDS}s apart)."
        date
        while (( consecutive < IDLE_CHECKS )); do
            if gpu_idle; then
                consecutive=$((consecutive + 1))
                echo "Idle checks: $consecutive/$IDLE_CHECKS"
            else
                status=$?
                (( status == 1 )) || return "$status"
                echo 'GPUs occupied; abandoning this confirmation round.'
                break
            fi
            if (( consecutive < IDLE_CHECKS )); then sleep "$POLL_SECONDS"; fi
        done
        if (( consecutive == IDLE_CHECKS )); then break; fi
        # Round starts are 30 minutes apart, not 30 minutes after the last check.
        remaining=$((ROUND_SECONDS - (SECONDS - round_started)))
        if (( remaining > 0 )); then
            echo "Next round in ${remaining}s (round interval: ${ROUND_SECONDS}s)."
            sleep "$remaining"
        fi
    done
    # UUIDs ensure these are exactly the physical cards shown as 0-3 by nvidia-smi.
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
        awk -F', *' '$1 ~ /^[0-3]$/ {uuid[$1]=$2} END {printf "%s,%s,%s,%s",uuid[0],uuid[1],uuid[2],uuid[3]}')
    export CUDA_VISIBLE_DEVICES
}

run_stage() {
    echo "Waiting for GPUs before stage: $1"
    shift
    wait_idle
    echo "Starting: $*"
    date
    "$@"
    echo 'Stage completed.'
    date
}

run_eval_if_needed() {
    local output="$1" mode="$2"; shift 2
    if is_complete_result "$output" "$mode"; then
        echo "Skipping completed evaluation: $output"
        return 0
    fi
    run_stage "$mode evaluation" "$@"
}

COMMON=(--source "$SOURCE_MODEL" --target "$TARGET_MODEL" --device cuda --device-map balanced_low_0 --local-files-only)
EVAL=(--dataset edinburgh-dawg/mmlu-redux-2.0 --split test --protocol distkv_greedy --max-new-tokens 64)
MANIFEST="$RESULTS_DIR/manifest.json"
MANIFEST_ARGS=(--evaluation-manifest-output "$MANIFEST")
[[ ! -e "$MANIFEST" ]] || MANIFEST_ARGS=(--evaluation-manifest "$MANIFEST")
if [[ -e "$RESULTS_DIR/target_native.json" || -e "$RESULTS_DIR/paper_cachebridge.json" || -e "$RESULTS_DIR/distkv_recipe_ours.json" ]]; then
    [[ -s "$MANIFEST" ]] || {
        echo "Existing result requires the shared manifest: $MANIFEST" >&2
        exit 2
    }
fi

run_eval_if_needed "$RESULTS_DIR/target_native.json" target_native "$PYTHON_BIN" -u -m experiments.eval_mmlu_redux \
    "${COMMON[@]}" "${EVAL[@]}" --mode target_native \
    "${MANIFEST_ARGS[@]}" --output "$RESULTS_DIR/target_native.json"

run_eval_if_needed "$RESULTS_DIR/paper_cachebridge.json" cachebridge "$PYTHON_BIN" -u -m experiments.eval_mmlu_redux \
    "${COMMON[@]}" "${EVAL[@]}" --mode cachebridge --mapper "$PAPER_MAPPER" \
    --evaluation-manifest "$MANIFEST" --output "$RESULTS_DIR/paper_cachebridge.json"

if [[ -s "$RECIPE_MAPPER/mapper.safetensors" && -s "$RECIPE_MAPPER/metadata.json" && -s "$RECIPE_MAPPER/fit_summary.json" ]]; then
    echo "Skipping completed fit: $RECIPE_MAPPER"
else
    run_stage fit_distkv_recipe "$PYTHON_BIN" -u -m experiments.fit_cachebridge \
        "${COMMON[@]}" --target-tokenizer "$TARGET_MODEL" --dataset-file "$DATASET_FILE" \
        --n-calib 300 --seq-len 512 --seed 123 --method cachebridge --head-map-mode auto \
        --calibration-pack-mode distkv_concat --layer-selection-mode distkv_ridge_subsample \
        --selection-subsample 4096 --attn-repair-mode off --target-capture-mode boundary_chunked \
        --top-k 12 --token-stride 4 --ridge-lambda 0.01 --num-boundaries 32 --boundary-start 12 \
        --dtype bfloat16 --stats-device cpu --fit-backend torch_bounded --output "$RECIPE_MAPPER"
fi

run_eval_if_needed "$RESULTS_DIR/distkv_recipe_ours.json" cachebridge "$PYTHON_BIN" -u -m experiments.eval_mmlu_redux \
    "${COMMON[@]}" "${EVAL[@]}" --mode cachebridge --mapper "$RECIPE_MAPPER" \
    --evaluation-manifest "$MANIFEST" --output "$RESULTS_DIR/distkv_recipe_ours.json"
echo 'All four stages completed.'
