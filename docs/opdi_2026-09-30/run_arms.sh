#!/usr/bin/env bash
# The OPDI arms of plan.md, one after another, after any running pipeline has finished (four cores, ~10 GB each).
set -euo pipefail
while pgrep -f "prc.pipeline" >/dev/null; do sleep 30; done
mkdir -p outputs/opdi_logs
for arm in noise2 opdi stress; do
  echo "=== $(date -u +%FT%TZ) start configs/opdi_${arm}.yaml"
  .venv/bin/python -m prc.pipeline --config "configs/opdi_${arm}.yaml" --skip-predict 2>&1 | tee "outputs/opdi_logs/opdi_${arm}.log" | grep -E "^fold |^CV |Traceback|Error" || true
  echo "=== $(date -u +%FT%TZ) done configs/opdi_${arm}.yaml"
done
