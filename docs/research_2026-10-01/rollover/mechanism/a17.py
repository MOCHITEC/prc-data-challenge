from common import *
r=pl.read_parquet('rank_ext.parquet')
x=r.filter(pl.col('FLIGHT_mvt').is_in(['ITY340','ITY341']))
print(x.select('PHASE_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','AOBT_3_flt','EOBT_1_flt','IOBT_flt','ARVT_3_flt','AIRCRAFT_TYPE_mvt','STAND_mvt').sort('SCHED_TIME_UTC_mvt').filter(pl.col('SCHED_TIME_UTC_mvt').dt.date().is_between(pl.date(2026,7,14),pl.date(2026,7,19))))
