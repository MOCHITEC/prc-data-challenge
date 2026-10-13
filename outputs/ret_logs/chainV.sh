#!/usr/bin/env bash
cd /home/user/prc-data-challenge
SC=/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/fj
L=outputs/ret_logs/chainV.log
until grep -q "ALLDONE" outputs/ret_logs/chainF.log 2>/dev/null; do sleep 60; done
export PRC_FEBJUN_FOLDS=1
for c in main_flseq_osm_nolirf_l2_opt main_flseq_osm_nolirf_l2_opt_ucmix main_flseq_osm_nolirf_l2_opt_nmb main_flseq_osm_nolirf_l2_opt_cmix main_flseq_osm_nolirf_l2_opt_cmix_ltfm main_flseq_osm_nolirf_l2_opt_l6 main_flseq_osm_nolirf_l2_pa_opt_l6; do
  grep -q "rc=0 ${c}_fj\$" $L 2>/dev/null && continue
  echo "=== $(date -u +%FT%TZ) start ${c}_fj" >> $L
  .venv/bin/python -m prc.pipeline --config $SC/${c}_fj.yaml --skip-predict --fold-cache outputs/foldcache > outputs/ret_logs/V_${c}.log 2>&1
  rc=$?; echo "=== $(date -u +%FT%TZ) rc=$rc ${c}_fj" >> $L; grep -E "^CV" outputs/ret_logs/V_${c}.log | tail -1 >> $L
done
echo "=== ALLDONE" >> $L
