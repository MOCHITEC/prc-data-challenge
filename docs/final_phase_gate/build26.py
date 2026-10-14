import polars as pl
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
d=pl.read_parquet('/home/user/prc-data-challenge/data/raw/final_ranking.parquet').filter(pl.col('PHASE_mvt')=='DEP')
def sec(a,b): return (pl.col(a)-pl.col(b)).dt.total_seconds()
d=d.with_columns(matched=pl.col('AOBT_3_flt').is_not_null(), ms=sec('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'),
  mlobt=sec('MVT_TIME_UTC_mvt','LOBT_flt'), prox=sec('MVT_TIME_UTC_mvt','AOBT_3_flt'), month=pl.col('MVT_TIME_UTC_mvt').dt.month())
rec=pl.read_parquet(f'{S}/ideas/final_recommended.parquet').rename({'TAXITIME_SEC_mvt':'rec'})
v4=pl.read_parquet(f'{S}/ideas/final_v4_cb5_mlp5.parquet').rename({'TAXITIME_SEC_mvt':'v4'})
pre=pl.read_parquet(f'{S}/ideas/pre_adsb_base_blend5_final.parquet')
print(pre.schema)
pre=pre.rename({c:'pre' for c in pre.columns if c!='MVT_ID_mvt'})
d=d.join(rec,on='MVT_ID_mvt',how='left').join(v4,on='MVT_ID_mvt',how='left').join(pre,on='MVT_ID_mvt',how='left')
print(d.height, d['rec'].null_count())
d.drop([c for c in d.columns if c.endswith('_flt') and c not in ('AOBT_3_flt','LOBT_flt','AIRCRAFT_OPERATOR_flt')]).write_parquet(f'{S}/ideas/ht/d26.parquet')
