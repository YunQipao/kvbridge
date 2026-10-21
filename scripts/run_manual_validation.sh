#!/usr/bin/env bash
set -euo pipefail

SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-1.7B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DEVICE="${DEVICE:-cuda}"
N_CALIB="${N_CALIB:-50}"
SEQ_LEN="${SEQ_LEN:-1024}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
OUT="${OUT:-artifacts/qwen3_1.7b_to_4b_formal}"

python3 --version
python3 -m pytest -q
python3 scripts/check_model_geometry.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL"
python3 scripts/check_tokenizer_alignment.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
  --dataset "$DATASET" --n-calib "$N_CALIB" --seq-len "$SEQ_LEN"
python3 scripts/smoke_real_kv.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --device "$DEVICE"
python3 scripts/validate_qwen_rope.py --source "$SOURCE_MODEL" --target "$TARGET_MODEL" --device "$DEVICE"
python3 scripts/smoke_real_attention.py --model "$TARGET_MODEL" --device "$DEVICE"
python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
  --dataset "$DATASET" --n-calib "$N_CALIB" --seq-len "$SEQ_LEN" --seed 42 \
  --method cachebridge --ridge-lambda 0.01 --token-stride 4 --top-k 8 \
  --num-boundaries 32 --boundary-start 12 --device "$DEVICE" \
  --dtype bfloat16 --stats-device cpu --fit-backend torch_bounded --output "$OUT"
python3 scripts/validate_mapper.py --mapper "$OUT" --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" --paper-faithful
python3 scripts/inspect_layer_selection.py --mapper "$OUT"
python3 scripts/test_handoff.py --mapper "$OUT" --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" --device "$DEVICE"
python3 -m experiments.eval_mmlu_redux --mode target_native --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" --limit 20
python3 -m experiments.eval_mmlu_redux --mode cachebridge --source "$SOURCE_MODEL" \
  --target "$TARGET_MODEL" --mapper "$OUT" --limit 20
