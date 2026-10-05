import polars as pl, numpy as np
R='/home/user/prc-data-challenge/data/raw/'
tr=pl.concat([pl.read_parquet(R+f'training_2025-{m}-01_2025-{n}-01.parquet',columns=['ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt']) for m,n in [('01','02'),('07','08')]]).filter(pl.col('PHASE_mvt')=='DEP')
tr=tr.with_columns(pl.col('BLOCK_TIME_UTC_mvt').dt.date().alias('date'),((pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds()/60).alias('dly'),((pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds()/60).alias('mvt_s'))
d=tr.group_by('ADEP_mvt','date').agg(pl.len().alias('n'),pl.col('dly').clip(0).mean().alias('dly_pos'),pl.col('dly').mean().alias('dly_mean'),(pl.col('TAXITIME_SEC_mvt')/60).mean().alias('taxi'),pl.col('mvt_s').mean().alias('mvt_s'))
a=pl.read_parquet('allp.parquet').select('APT_ICAO','date',pl.col('FLT_DEP_IFR_2').cast(pl.Float64).alias('n2'),pl.col('DLY_ALL_PRE_2').cast(pl.Float64).alias('allpre'))
j=d.join(a,left_on=['ADEP_mvt','date'],right_on=['APT_ICAO','date']).with_columns((pl.col('allpre')/pl.col('n2')).alias('pru_pf'))
print(j.select(pl.corr('pru_pf','dly_pos').alias('c1'),pl.corr('pru_pf','dly_mean').alias('c2'),(pl.col('n2')/pl.col('n')).mean().alias('cov')))
j=j.with_columns((pl.col('mvt_s')-pl.col('pru_pf')).alias('taxi_hat'))
for a_ in sorted(j['ADEP_mvt'].unique()):
  s=j.filter(pl.col('ADEP_mvt')==a_)
  print(a_, 'corr(pru_pf,mean pos delay)',round(np.corrcoef(s['pru_pf'],s['dly_pos'])[0,1],3),'ratio',round((s['pru_pf']/s['dly_pos']).median(),2), 'corr(taxi_hat, daily taxi)', round(np.corrcoef(s['taxi_hat'],s['taxi'])[0,1],2))
