#!/usr/bin/env bash
# Final-phase member predictions on the 4-month final frame. Predict only; never --submit.
cd /home/user/prc-data-challenge
L=outputs/ret_logs/chainF.log
for c in main_flseq_osm_nolirf_l2_opt_l6_cmix main_flseq_osm_nolirf_l2_opt_l6_cmix_ltfm main_flseq_osm_nolirf_l2_pa_opt_l6; do
  [ -s outputs/final_pb/$c.parquet ] && continue
  echo "=== $(date -u +%FT%TZ) start $c" >> $L
  .venv/bin/python -m prc.pipeline --config configs/$c.yaml --skip-cv --raw-dir data/raw_f --work-dir data/work_final --out-dir outputs/final_pb > outputs/ret_logs/F_$c.log 2>&1
  rc=$?; echo "=== $(date -u +%FT%TZ) rc=$rc $c" >> $L
done
echo "=== ALLDONE" >> $L
