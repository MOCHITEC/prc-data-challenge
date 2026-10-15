#!/bin/bash
# NbrNet variant of scripts/run_guarded.sh.
# usage: experiments/nbrnet_2026-10-15/run_guarded_nn.sh <tag> <python args...>
#   e.g. ... run_guarded_nn.sh a2o_cv -m prc.pipeline --config configs/...yaml --skip-predict --fold-cache DIR
#        ... run_guarded_nn.sh s7_ctx experiments/nbrnet_2026-10-15/nbrnet/nbr_train.py --fold forward --seed 1 --mode ctx
# Runs from the repository root this script lives in (not a hard-coded checkout). Logs go to logs/nbrnet/
# (git-ignored): <tag>.log (stdout+stderr), <tag>.mem (10 s samples: own-cgroup total_rss MB), <tag>.status.
# Gate: waits (30 s polls) while another prc.pipeline / nbr_train / nbr_tokens / nbr_query run is alive, so one heavy job runs at a time.
# Guard: kills the process group when the OWN cgroup's total_rss (the cgroup carrying the ~13,680 MB limit)
# exceeds GUARD_MB (default 13,000 MB).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 2
TAG=$1; shift
L=${RUN_LOG_DIR:-logs/nbrnet}
mkdir -p "$L"
GUARD_MB=${GUARD_MB:-13000}
CG=/sys/fs/cgroup/memory$(grep '^[0-9]*:memory:' /proc/self/cgroup | cut -d: -f3)
rss() { awk '$1=="total_rss"{print int($2/1048576)}' "$1/memory.stat"; }
echo "gate $(date -u +%FT%TZ)" > "$L/$TAG.status"
while true; do
  np=$(ps -eo args | grep -v run_guarded_nn | grep -c '[p]rc.pipeline\|[n]br_train.py\|[n]br_tokens.py\|[n]br_query.py')
  if [ "$np" -eq 0 ]; then break; fi
  echo "$(date -u +%T) waiting other_heavy=$np rss=$(rss $CG)" >> "$L/$TAG.status"; sleep 30
done
start=$(date +%s)
echo "start $(date -u +%FT%TZ) args: $*" >> "$L/$TAG.status"
POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-1} setsid .venv/bin/python "$@" > "$L/$TAG.log" 2>&1 &
PID=$!
peak=0; killed=0
while kill -0 $PID 2>/dev/null; do
  r=$(rss $CG)
  [ "$r" -gt "$peak" ] && peak=$r
  echo "$(date -u +%T) $r" >> "$L/$TAG.mem"
  if [ "$r" -gt "$GUARD_MB" ]; then
    kill -9 -- -$PID 2>/dev/null; killed=1; echo "KILLED at total_rss $r MB" >> "$L/$TAG.status"; break
  fi
  sleep 10
done
wait $PID; rc=$?
end=$(date +%s)
echo "end $(date -u +%FT%TZ) rc=$rc killed=$killed wall_s=$((end-start)) peak_total_rss_MB=$peak" >> "$L/$TAG.status"
echo "DONE $TAG rc=$rc killed=$killed wall_s=$((end-start)) peak=$peak"
