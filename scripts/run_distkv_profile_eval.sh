#!/usr/bin/env bash
# Separate opt-in driver. Does not change historical queues or result folders.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROFILE="${1:-distkv_canonical}"
SCOPE="${2:-smoke500}"
case "$PROFILE" in distkv_canonical|distkv_exact) ;; *) echo 'Profile: distkv_canonical|distkv_exact' >&2; exit 2;; esac
SAMPLE_ARGS=()
case "$SCOPE" in smoke500) SAMPLE_ARGS=(--num-eval-samples 500);; full) ;; *) echo 'Scope: smoke500|full' >&2; exit 2;; esac
DATA_ARGS=(--dataset edinburgh-dawg/mmlu-redux-2.0 --split test)
if [[ -n "${DISTKV_DATASET:-}" ]]; then
    DATA_ARGS+=(--dataset-local-path "$DISTKV_DATASET")
fi
SAMPLING="${SAMPLING:-systematic}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DISTKV_REPO="${DISTKV_REPO:-/media/sda1/iic_data/luotao/DistKV}"
RESULT_DIR="${RESULT_DIR:-results/${PROFILE}_${SCOPE}_v1}"
SOURCE_MODEL="${SOURCE_MODEL:-$HOME/luotao/models/Qwen2.5-14B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-$HOME/luotao/models/Qwen2.5-7B-Instruct}"
PAPER_MAPPER="${PAPER_MAPPER:-artifacts/qwen25_14bi_to_7bi_openhermes_n300_len512_k12_paper_cb}"
RECIPE_MAPPER="${RECIPE_MAPPER:-artifacts/qwen25_14bi_to_7bi_openhermes_n300_len512_k12_distkv_recipe}"
for mapper in "$PAPER_MAPPER" "$RECIPE_MAPPER"; do
    [[ -s "$mapper/mapper.safetensors" ]] || { echo "Missing mapper: $mapper" >&2; exit 2; }
done
for name in target_native paper_cachebridge distkv_recipe_ours; do
    [[ ! -e "$RESULT_DIR/$name.json" ]] || { echo "Result exists; choose a new RESULT_DIR: $RESULT_DIR/$name.json" >&2; exit 2; }
done
"$PYTHON_BIN" scripts/audit_distkv_protocol.py --distkv-repo "$DISTKV_REPO" \
    "${DATA_ARGS[@]}" --manifest "$RESULT_DIR/manifest.json" \
    --create-manifest --eval-profile "$PROFILE" --sampling "$SAMPLING" "${SAMPLE_ARGS[@]}"
# A pre-existing smoke manifest must not silently turn a full run into 500 rows.
"$PYTHON_BIN" - "$RESULT_DIR/manifest.json" "$SCOPE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
expected = min(500, m['population_size']) if sys.argv[2] == 'smoke500' else m['population_size']
if m['num_samples'] != expected:
    raise SystemExit('Manifest sample count does not match scope; choose a new RESULT_DIR')
PY
COMMON=(--source "$SOURCE_MODEL" --target "$TARGET_MODEL" --eval-profile "$PROFILE"
    "${DATA_ARGS[@]}" --protocol distkv_greedy --max-new-tokens 64
    --evaluation-manifest "$RESULT_DIR/manifest.json" --device cuda
    --device-map balanced_low_0 --local-files-only)
"$PYTHON_BIN" -u -m experiments.eval_mmlu_redux "${COMMON[@]}" --mode target_native \
    --output "$RESULT_DIR/target_native.json"
"$PYTHON_BIN" -u -m experiments.eval_mmlu_redux "${COMMON[@]}" --mode cachebridge \
    --mapper "$PAPER_MAPPER" --output "$RESULT_DIR/paper_cachebridge.json"
"$PYTHON_BIN" -u -m experiments.eval_mmlu_redux "${COMMON[@]}" --mode cachebridge \
    --mapper "$RECIPE_MAPPER" --output "$RESULT_DIR/distkv_recipe_ours.json"
