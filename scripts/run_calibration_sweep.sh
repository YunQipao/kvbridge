#!/usr/bin/env bash
set -euo pipefail

SOURCE_MODEL="${SOURCE_MODEL:-Qwen/Qwen3-14B}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-32B}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
ROOT="${ROOT:-artifacts/qwen3_calibration_sweep}"
mkdir -p "$ROOT" results

python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
  --dataset "$DATASET" --n-calib 500 --seq-len 1024 --seed 42 --method cachebridge \
  --ridge-lambda 0.01 --token-stride 4 --top-k 8 --num-boundaries 32 \
  --boundary-start 12 --output "$ROOT/cachebridge_500"

for n in 50 100 200 500; do
  python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --reuse-calibration-manifest "$ROOT/cachebridge_500/calibration_manifest.json" \
    --reuse-layer-selection "$ROOT/cachebridge_500/layer_selection.json" \
    --n-calib "$n" --seq-len 1024 --seed 42 --method cachebridge \
    --ridge-lambda 0.01 --token-stride 4 --top-k 8 --num-boundaries 32 \
    --boundary-start 12 --output "$ROOT/cachebridge_$n"
  python3 -m experiments.fit_cachebridge --source "$SOURCE_MODEL" --target "$TARGET_MODEL" \
    --reuse-calibration-manifest "$ROOT/cachebridge_500/calibration_manifest.json" \
    --reuse-layer-selection "$ROOT/cachebridge_500/layer_selection.json" \
    --n-calib "$n" --seq-len 1024 --seed 42 --method full_head \
    --ridge-lambda 0.01 --token-stride 4 --top-k 8 --num-boundaries 32 \
    --boundary-start 12 --output "$ROOT/full_head_$n"
done

for n in 50 100 200 500; do
  for task in HellaSwag ARC-Challenge WinoGrande MMLU; do
    task_slug="${task,,}"
    task_slug="${task_slug//-/_}"
    manifest="results/manifests/${task_slug}_${n}.json"
    mkdir -p "$(dirname "$manifest")"
    native="results/sweep_${n}_${task}_target_native.json"
    common_args=(--task "$task" --source "$SOURCE_MODEL" --target "$TARGET_MODEL"
      --sampling systematic --sampling-seed 42)
    python3 experiments/eval_paper_tasks.py --mode target_native "${common_args[@]}" \
      --evaluation-manifest-output "$manifest" --output "$native"
    python3 experiments/eval_paper_tasks.py --mode full_head "${common_args[@]}" \
      --evaluation-manifest "$manifest" \
      --mapper "$ROOT/full_head_${n}" --output "results/sweep_${n}_${task}_full_head.json"
    python3 experiments/eval_paper_tasks.py --mode cachebridge "${common_args[@]}" \
      --evaluation-manifest "$manifest" \
      --mapper "$ROOT/cachebridge_${n}" --output "results/sweep_${n}_${task}_cachebridge.json"
  done
done

python3 - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
tasks = ["HellaSwag", "ARC-Challenge", "WinoGrande", "MMLU"]
with Path("results/calibration_sweep.csv").open("w", newline="") as stream:
    fields = ["method", "n_calib", *tasks, "mean_target_retention"]
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for n in (50, 100, 200, 500):
        native = {t: json.loads(Path(f"results/sweep_{n}_{t}_target_native.json").read_text())["accuracy"] for t in tasks}
        for method in ("full_head", "cachebridge"):
            values = [json.loads(Path(f"results/sweep_{n}_{t}_{method}.json").read_text())["accuracy"] for t in tasks]
            writer.writerow({"method": method, "n_calib": n, **dict(zip(tasks, values)),
                             "mean_target_retention": sum(v / native[t] for v, t in zip(values, tasks) if native[t]) / 4})
PY
