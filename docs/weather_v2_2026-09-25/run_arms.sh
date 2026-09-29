#!/usr/bin/env bash
# Runs the weather-v2 arms of plan.md one after another (four cores, each fit uses all of them).
# Usage: bash docs/weather_v2_2026-09-25/run_arms.sh [arm ...]   (default: base noise5 B noise7 C)
set -euo pipefail
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then arms=(base noise5 B noise7 C); fi
mkdir -p outputs/wx2_logs
for arm in "${arms[@]}"; do
  echo "=== $(date -u +%FT%TZ) start configs/wx2_${arm}.yaml"
  .venv/bin/python -m prc.pipeline --config "configs/wx2_${arm}.yaml" --skip-predict 2>&1 | tee "outputs/wx2_logs/wx2_${arm}.log"
  echo "=== $(date -u +%FT%TZ) done configs/wx2_${arm}.yaml"
done
