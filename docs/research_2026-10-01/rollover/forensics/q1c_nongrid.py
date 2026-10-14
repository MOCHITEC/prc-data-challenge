# Non-grid airports: unmatched rows whose MVT lies on the 13-value grid -- a separate reporting regime?
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP')
df=df.with_columns(train=pl.col('file')!='ranking.parquet', y=pl.col('TAXITIME_SEC_mvt'),
  ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  unm=pl.col('FLIGHT_ID_mvt').is_null(),
  mo=((pl.col('MVT_TIME_UTC_mvt').dt.second()+30)%60-30),
  bo=((pl.col('BLOCK_TIME_UTC_mvt').dt.second()+30)%60-30),
  actnull=pl.col('AIRCRAFT_TYPE_mvt').is_null())
df=df.with_columns(mgrid=pl.col('mo').abs()<=6, bgrid=pl.col('bo').abs()<=6, anti=(pl.col('mo')+pl.col('bo')).is_in([-1,0]),
   sf=(pl.col('bs').abs()<=6), roll=pl.col('y')>72000)
ng=['EDDF','EHAM','LEBL','LEMD','LSZH']
s=df.filter(pl.col('ADEP_mvt').is_in(ng), pl.col('unm'))
print('non-grid airports, unmatched rows: by (train, mgrid)')
print(s.group_by('ADEP_mvt','train','mgrid').agg(pl.len(), pl.col('bgrid').mean().round(3), pl.col('anti').mean().round(3), pl.col('sf').mean().round(3),
   (pl.col('y')-pl.col('ms')).abs().le(60).mean().round(3).alias('y_eq_ms'), pl.col('roll').sum(), pl.col('actnull').mean().round(3),
   pl.col('ms').median().alias('ms_med'), pl.col('y').median().alias('y_med'), (pl.col('ms').abs()>7200).mean().round(3).alias('ms_gt2h')).sort('ADEP_mvt','train','mgrid'))
# matched rows on grid at non-grid airports?
m=df.filter(pl.col('ADEP_mvt').is_in(ng), ~pl.col('unm'), pl.col('train'))
print('matched, grid MVT & anti-symmetric share', m.group_by('ADEP_mvt').agg(pl.len(), (pl.col('mgrid')&pl.col('bgrid')&pl.col('anti')).mean().round(4).alias('grid_anti'),
  (pl.col('mgrid')&pl.col('bgrid')).mean().round(4).alias('both_grid')).sort('ADEP_mvt'))
# EHAM/LSZH unmatched with grid: show anti among grid-both
print(s.filter(pl.col('train'),pl.col('mgrid'),pl.col('bgrid')).group_by('ADEP_mvt').agg(pl.len(),pl.col('anti').mean()))
