#!/bin/bash
# S8: the remaining NbrNet runs, one at a time, each under run_guarded_nn.sh (13,000 MB guard), in the
# pre-registered order: jan + jul at seed 1 (ctx, noctx), then jan, jul, forward at seed 2 (ctx, noctx).
# A run whose predictions already exist is skipped, so the driver can be restarted after a container reset
# (nbr_train.py itself resumes from its local or committed checkpoint). After each run its predictions, run
# record and guard status are committed and pushed (no training process is alive at that moment).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 2
EXP=experiments/nbrnet_2026-10-15
TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01U4r4yp4haN4XVEVVfmBd5H'
RUNS="jan:1:ctx jan:1:noctx jul:1:ctx jul:1:noctx jan:2:ctx jan:2:noctx jul:2:ctx jul:2:noctx forward:2:ctx forward:2:noctx"
for r in $RUNS; do
  IFS=: read -r fold seed mode <<< "$r"
  run="${fold}_s${seed}_${mode}"
  if [ -f "$EXP/preds/$run.parquet" ] && [ -f "$EXP/runs/$run.json" ]; then echo "skip $run (done)"; continue; fi
  echo "$(date -u +%FT%TZ) start $run"
  "$EXP/run_guarded_nn.sh" "s8_$run" "$EXP/nbrnet/nbr_train.py" --fold "$fold" --seed "$seed" --mode "$mode"
  if [ ! -f "$EXP/preds/$run.parquet" ]; then echo "$(date -u +%FT%TZ) FAILED $run"; exit 1; fi
  cp "logs/nbrnet/s8_$run.status" "$EXP/runs/$run.guard_status.txt"
  for i in 1 2 3 4 5; do
    git add -f "$EXP/preds/$run.parquet" "$EXP/runs/$run.json" "$EXP/runs/$run.guard_status.txt" && \
    git commit -q -m "NbrNet S8: $run predictions and run record$TRAILER" -- "$EXP/preds/$run.parquet" "$EXP/runs/$run.json" "$EXP/runs/$run.guard_status.txt" && \
    git push -q origin HEAD && break
    sleep $((10 * i))
  done
  echo "$(date -u +%FT%TZ) done $run $(grep -o '"best_epoch": [0-9]*' "$EXP/runs/$run.json") $(tail -1 "logs/nbrnet/s8_$run.status")"
done
echo "$(date -u +%FT%TZ) ALL DONE"
