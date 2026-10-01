from common import *
for ph,ref in [('ARR','ARVT_3_flt'),('DEP','AOBT_3_flt')]:
  for name in ['train_ext.parquet','rank_ext.parquet']:
    a=pl.read_parquet(name).filter(pl.col('PHASE_mvt')==ph).with_columns(g=(pl.col('MVT_TIME_UTC_mvt')-pl.col(ref)).dt.total_seconds())
    x=a.filter(pl.col('g').abs()>40000)
    print(ph,name,x.height)
    for r in x.select('ADEP_mvt','ADES_mvt','FLIGHT_mvt',pl.col('SCHED_TIME_UTC_mvt').dt.strftime('%m-%d %H:%M'),pl.col('MVT_TIME_UTC_mvt').dt.strftime('%m-%d %H:%M'),pl.col(ref).dt.strftime('%m-%d %H:%M'),pl.col('BLOCK_TIME_UTC_mvt').dt.strftime('%m-%d %H:%M'),'TAXITIME_SEC_mvt','g').iter_rows(): print(*r)
