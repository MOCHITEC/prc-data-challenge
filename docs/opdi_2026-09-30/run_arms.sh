#!/usr/bin/env bash
# The OPDI arms of plan.md §5, one after another on the main_flseq recipe (four cores, ~12 GB each: never two at once).
set -euo pipefail
mkdir -p outputs/opdi_logs
for arm in noise2 opdi stress; do
  echo "=== $(date -u +%FT%TZ) start configs/opdi_${arm}.yaml"
  .venv/bin/python -m prc.pipeline --config "configs/opdi_${arm}.yaml" --work-dir data/work_flseq --skip-predict \
    > "outputs/opdi_logs/opdi_${arm}.log" 2>&1
  grep -E "^fold |^CV " "outputs/opdi_logs/opdi_${arm}.log"
  echo "=== $(date -u +%FT%TZ) done configs/opdi_${arm}.yaml"
done
