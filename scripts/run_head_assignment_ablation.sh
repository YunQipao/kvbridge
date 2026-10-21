#!/usr/bin/env bash
set -euo pipefail
for shift in 0 1 2 3 4 5 6 7; do
  python3 -m experiments.fit_cachebridge "$@" --method cachebridge \
    --head-assignment-shift "$shift" --output "artifacts/head_assignment_shift_$shift"
done
