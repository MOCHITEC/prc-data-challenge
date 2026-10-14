import polars as pl, glob
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
fs=sorted(glob.glob('/home/user/prc-data-challenge/data/raw/training_2025-*.parquet'))
d=pl.concat([pl.read_parquet(f) for f in fs]).filter(pl.col('PHASE_mvt')=='DEP')
def sec(a,b): return (pl.col(a)-pl.col(b)).dt.total_seconds()
d=d.with_columns(
  y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64),
  matched=pl.col('AOBT_3_flt').is_not_null(),
  ms=sec('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'),
  mlobt=sec('MVT_TIME_UTC_mvt','LOBT_flt'),
  prox=sec('MVT_TIME_UTC_mvt','AOBT_3_flt'),
  month=pl.col('MVT_TIME_UTC_mvt').dt.month(),
)
r=pl.read_parquet(f'{S}/adsb/learn/cv_months12_rows_a2.parquet').select('MVT_ID_mvt','b','final_12','P12')
d=d.join(r,on='MVT_ID_mvt',how='left')
d.write_parquet(f'{S}/ideas/ht/d25.parquet')
print(d.shape, d['b'].null_count())
