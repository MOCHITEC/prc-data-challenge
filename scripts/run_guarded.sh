#!/bin/bash
# usage: scripts/run_guarded.sh <tag> <pipeline args...>   (logs/cbretune/ is git-ignored). Gated: waits (60 s polls) until no
# adsb_learn_ process runs and total_rss (own and parent cgroup) < 2,500 MB; kills when the OWN cgroup total_rss (the one carrying the 13,680 MB limit) exceeds 13,000 MB; the all-sessions sum is logged, not acted on.
cd /home/user/prc-data-challenge
TAG=$1; shift
L=${RUN_LOG_DIR:-logs/cbretune}
CG=/sys/fs/cgroup/memory$(grep '^[0-9]*:memory:' /proc/self/cgroup | cut -d: -f3)
PARENT=$(dirname $CG)
rss() { awk '$1=="total_rss"{print int($2/1048576)}' $1/memory.stat; }
# all sibling session cgroups (a container "restart" can leave an earlier session's processes in an old one)
allrss() { t=0; for d in /sys/fs/cgroup/memory/process_api/*/; do v=$(rss $d); t=$((t+v)); done; echo $t; }
echo "gate $(date -u +%FT%TZ)" > $L/$TAG.status
while true; do
  n=$(ps -eo args | grep -c '[a]dsb_learn_'); r=$(rss $CG); rp=$(allrss)
  np=$(ps -eo args | grep -c '[p]rc.pipeline\|[r]etune_cb.py')
  if [ "$n" -eq 0 ] && [ "$np" -eq 0 ] && [ "$r" -lt 2500 ] && [ "$rp" -lt 2500 ]; then break; fi
  echo "$(date -u +%T) waiting adsb=$n rss=$r parent=$rp" >> $L/$TAG.status; sleep 60
done
start=$(date +%s)
echo "start $(date -u +%FT%TZ) args: $*" >> $L/$TAG.status
setsid .venv/bin/python -m prc.pipeline "$@" > $L/$TAG.log 2>&1 &
PID=$!
peak=0; peakp=0; killed=0
while kill -0 $PID 2>/dev/null; do
  r=$(rss $CG); rp=$(allrss)
  [ "$r" -gt "$peak" ] && peak=$r
  [ "$rp" -gt "$peakp" ] && peakp=$rp
  echo "$(date -u +%T) $r $rp" >> $L/$TAG.mem
  if [ "$r" -gt 13000 ]; then  # own cgroup only (the one with the 13,680 MB limit); user-approved 2026-10-06
    kill -9 -- -$PID 2>/dev/null; killed=1; echo "KILLED at total_rss $r MB (parent $rp MB)" >> $L/$TAG.status; break
  fi
  sleep 10
done
wait $PID; rc=$?
end=$(date +%s)
echo "end $(date -u +%FT%TZ) rc=$rc killed=$killed wall_s=$((end-start)) peak_total_rss_MB=$peak parent_peak_MB=$peakp" >> $L/$TAG.status
echo "DONE $TAG rc=$rc killed=$killed wall_s=$((end-start)) peak=$peak parent_peak=$peakp"
