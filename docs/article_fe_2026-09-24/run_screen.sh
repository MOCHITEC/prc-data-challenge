#!/usr/bin/env bash
# Stage 1 of docs/article_fe_2026-09-24/plan.md: every arm on the forward fold, one after another (the box has
# four cores and each fit uses all of them, so running arms side by side would only make both slower and
# nondeterministic in wall clock). base goes first so a failure to reproduce production is seen before the
# candidates are spent. Usage: bash docs/article_fe_2026-09-24/run_screen.sh [fold] [arm ...]
set -euo pipefail
fold="${1:-forward}"
shift || true
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then arms=(base noise2 noise6 freq agg cyc meta); fi
log_dir="${AFE_LOG_DIR:-outputs/afe_logs}"
mkdir -p "$log_dir"
for arm in "${arms[@]}"; do
  cfg="configs/afe_${fold}_${arm}.yaml"
  echo "=== $(date -u +%FT%TZ) start $cfg"
  .venv/bin/python -m prc.pipeline --config "$cfg" --skip-predict 2>&1 | tee "$log_dir/afe_${fold}_${arm}.log"
  echo "=== $(date -u +%FT%TZ) done $cfg"
done
