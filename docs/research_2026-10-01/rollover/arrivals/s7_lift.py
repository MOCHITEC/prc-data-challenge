import polars as pl, numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
df=pl.read_parquet('all.parquet').filter(pl.col('src')=='train')
TZ={'LIRF':'Europe/Rome','LSZH':'Europe/Zurich','EDDM':'Europe/Berlin','LEBL':'Europe/Madrid','LEMD':'Europe/Madrid','EDDF':'Europe/Berlin','LFPG':'Europe/Paris','EGLL':'Europe/London','EHAM':'Europe/Amsterdam','LTFM':'Europe/Istanbul'}
parts=[]
for ap,tz in TZ.items():
  parts.append(df.filter(pl.col('APT')==ap).with_columns(day=pl.col('SCHED_TIME_UTC_mvt').dt.convert_time_zone(tz).dt.date()))
df=pl.concat(parts).with_columns(y=pl.col('TAXITIME_SEC_mvt'),matched=pl.col('FLIGHT_ID_mvt').is_not_null(),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds().abs())
a=df.filter(pl.col('PHASE_mvt')=='ARR'); d=df.filter(pl.col('PHASE_mvt')=='DEP')
A=a.group_by('APT','day').agg(
  an_unm=(~pl.col('matched')).sum(),
  an_big=(pl.col('y')>3600).sum(),
  an_neg=(pl.col('y')<0).sum(),
  an_fill=((pl.col('bs')<=6)&(pl.col('y')>1800)).sum(),
  an_1d=(pl.col('y').abs()>43200).sum())
D=d.group_by('APT','day').agg(roll=(pl.col('y')>72000).sum(), n=pl.len())
X=D.join(A,on=['APT','day'],how='left').fill_null(0)
# widen to +-1 day window for ARR anomalies
X=X.sort('APT','day')
for c in ['an_unm','an_big','an_neg','an_fill','an_1d']:
  X=X.with_columns(pl.col(c).rolling_sum(3,center=True,min_samples=1).over('APT').alias(c+'_w'))
X.write_parquet('daylift.parquet')
L=X.filter(pl.col('APT')=='LIRF')
print('LIRF days',L.height,'roll days',(L['roll']>0).sum())
base=(L['roll']>0).mean()
for c in ['an_unm','an_big','an_neg','an_fill','an_1d']:
  for w in ['',  '_w']:
    for k in [1,2,3]:
      s=L.filter(pl.col(c+w)>=k)
      if s.height==0: continue
      print(f'{c+w:10s}>={k}: days {s.height:4d} rolldays {(s["roll"]>0).sum():3d} P={((s["roll"]>0).mean()):.3f} base={base:.3f} lift={((s["roll"]>0).mean()/base):.2f}')
# all airports pooled
print('ALL: roll days', (X['roll']>0).sum(), 'of', X.height)
for c in ['an_unm_w','an_big_w','an_1d_w']:
  s=X.filter(pl.col(c)>=1); print(c, s.height, (s['roll']>0).sum())
print(L.filter(pl.col('roll')>0).select('day','roll','n','an_unm','an_big','an_neg','an_fill','an_1d','an_unm_w','an_big_w'))
print(L.select([pl.col(c).mean() for c in ['an_unm','an_big','an_neg','an_fill','an_1d']]))
