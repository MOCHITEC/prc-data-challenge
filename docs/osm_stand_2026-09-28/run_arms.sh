#!/usr/bin/env bash
# Runs the OSM-distance arms of plan.md one after another. Usage: bash docs/osm_stand_2026-09-28/run_arms.sh [arm ...]
set -euo pipefail
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then arms=(noise1 dist base_u noise1_u dist_u); fi
mkdir -p outputs/osm_logs
for arm in "${arms[@]}"; do
  echo "=== $(date -u +%FT%TZ) start configs/osm_${arm}.yaml"
  .venv/bin/python -m prc.pipeline --config "configs/osm_${arm}.yaml" --skip-predict 2>&1 | tee "outputs/osm_logs/osm_${arm}.log"
  echo "=== $(date -u +%FT%TZ) done configs/osm_${arm}.yaml"
done
