# Does file row position carry information about BLOCK beyond MVT? d = own MVT - median MVT of +-3 file neighbours (same file).
import polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet',columns=['file','rowpos','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt','FLIGHT_ID_mvt'])
df=df.sort('file','rowpos').with_columns(AP=pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('ADEP_mvt')).otherwise(pl.col('ADES_mvt')),
   tm=pl.col('MVT_TIME_UTC_mvt').dt.epoch('s').cast(pl.Float64))
nb=[pl.col('tm').shift(k).over('file') for k in [-3,-2,-1,1,2,3]]
df=df.with_columns(nbmed=pl.concat_list(nb).list.median(), tb=pl.col('BLOCK_TIME_UTC_mvt').dt.epoch('s'), ts=pl.col('SCHED_TIME_UTC_mvt').dt.epoch('s'))
df=df.with_columns(d=pl.col('tm')-pl.col('nbmed'), dB=pl.col('tb')-pl.col('nbmed'), y=pl.col('TAXITIME_SEC_mvt'))
print('|d| quantiles (s) by train/rank, phase:')
print(df.group_by(pl.col('file')=='ranking.parquet','PHASE_mvt').agg(pl.col('d').abs().quantile(0.5).alias('p50'),pl.col('d').abs().quantile(0.9).alias('p90'),pl.col('d').abs().quantile(0.99).alias('p99')))
print('Spearman(d, y) per airport/phase/set (y = taxi label; ARR labels visible in ranking)')
print(df.filter(pl.col('y').is_not_null()).group_by('AP','PHASE_mvt',pl.col('file')=='ranking.parquet').agg(pl.len(),pl.corr('d','y',method='spearman').round(4).alias('rho_d_y')).sort('PHASE_mvt','AP','file'))
r=df.filter(pl.col('PHASE_mvt')=='DEP',pl.col('y')>72000)
print(r.select('AP','d','dB','y'))
