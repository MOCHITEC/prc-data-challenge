# LFPG / LTFM / EHAM / LSZH: which time is MVT_ID closest to in id-order? (compliance-sensitive measurement)
import polars as pl, numpy as np
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP').with_columns(rank=pl.col('file')=='ranking.parquet')
df=df.with_columns(tm=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s'), ts=pl.col('SCHED_TIME_UTC_mvt').dt.epoch('s'), tb=pl.col('BLOCK_TIME_UTC_mvt').dt.epoch('s'),
                   tl=pl.col('LOBT_flt').dt.epoch('s'), ta=pl.col('AOBT_3_flt').dt.epoch('s'))
df=df.sort('ADEP_mvt','MVT_ID_mvt')
d=df.with_columns([(pl.col(c)-pl.col(c).shift(1)).over('ADEP_mvt').abs().alias('d_'+c) for c in ['tm','ts','tb','tl','ta']]+
                  [pl.col('MVT_ID_mvt').diff().over('ADEP_mvt').alias('gid')])
d=d.filter(pl.col('gid')<=50)
print('median |delta| between id-adjacent DEP rows (s)')
print(d.group_by('ADEP_mvt','rank').agg(pl.len(),*[pl.col('d_'+c).median().alias(c) for c in ['tm','ts','tb','tl','ta']],
      *[(pl.col('d_'+c)<=60).mean().round(3).alias(c+'_le60') for c in ['tm','ts','tb']]).sort('ADEP_mvt','rank'))
# LFPG: rank correlation of id with each time within runs of length >= 100 (training)
lf=df.filter(pl.col('ADEP_mvt')=='LFPG')
lf=lf.with_columns(run=(pl.col('MVT_ID_mvt').diff().fill_null(1e9)>50).cast(pl.Int32).cum_sum())
print(lf.group_by('rank').agg(pl.corr('MVT_ID_mvt','tm',method='spearman').alias('rho_mvt'),pl.corr('MVT_ID_mvt','tb',method='spearman').alias('rho_blk'),pl.corr('MVT_ID_mvt','ts',method='spearman').alias('rho_sch')))
print(lf.filter(~pl.col('rank')).select('MVT_ID_mvt','FLIGHT_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt').slice(5000,40))
