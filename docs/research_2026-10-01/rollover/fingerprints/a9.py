import polars as pl, numpy as np
# Global (all airports) check: within same SCHED (and ADEP), is MVT_ID order tied to BLOCK, MVT or FLIGHT string?
f='/home/user/prc-data-challenge/data/raw/training_2025-07-01_2025-08-01.parquet'
d=pl.scan_parquet(f).select('MVT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_mvt').collect()
d=d.sort('MVT_ID_mvt')
print(d.select('MVT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','SCHED_TIME_UTC_mvt','FLIGHT_mvt').head(12))
dep=d.filter(pl.col('PHASE_mvt')=='DEP')
g=dep.with_columns(r_id=pl.col('MVT_ID_mvt').rank().over(['ADEP_mvt','SCHED_TIME_UTC_mvt']),
                   r_b=pl.col('BLOCK_TIME_UTC_mvt').rank().over(['ADEP_mvt','SCHED_TIME_UTC_mvt']),
                   r_m=pl.col('MVT_TIME_UTC_mvt').rank().over(['ADEP_mvt','SCHED_TIME_UTC_mvt']),
                   r_f=pl.col('FLIGHT_mvt').rank().over(['ADEP_mvt','SCHED_TIME_UTC_mvt']),
                   n=pl.len().over(['ADEP_mvt','SCHED_TIME_UTC_mvt'])).filter(pl.col('n')>=2)
for c in ['r_b','r_m','r_f']:
    x=g.select('r_id',c).drop_nulls().to_numpy(); print(c, 'agreement corr', np.corrcoef(x[:,0],x[:,1])[0,1], len(x))
