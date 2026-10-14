# (a) SCHED minute pattern by class at LIRF; (b) counts of rows sharing the rollover fingerprint in ranking;
# (c) does "MVT on 13-value grid" add information for unmatched rows at the non-grid airports? small LGBM, leave-one-month-out.
import polars as pl, numpy as np, lightgbm as lgb
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
OUT='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/forensics/'
df=pl.read_parquet(OUT+'all.parquet').filter(pl.col('PHASE_mvt')=='DEP')
df=df.with_columns(train=pl.col('file')!='ranking.parquet', y=pl.col('TAXITIME_SEC_mvt'),
  ms=(pl.col('MVT_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  bs=(pl.col('BLOCK_TIME_UTC_mvt')-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds(),
  unm=pl.col('FLIGHT_ID_mvt').is_null(), mo=((pl.col('MVT_TIME_UTC_mvt').dt.second()+30)%60-30),
  smin5=(pl.col('SCHED_TIME_UTC_mvt').dt.minute()%5==0), month=pl.col('MVT_TIME_UTC_mvt').dt.month(), hr=pl.col('MVT_TIME_UTC_mvt').dt.hour())
df=df.with_columns(mgrid=pl.col('mo').abs()<=6, cls=pl.when(~pl.col('train')).then(pl.lit('RANK')).when(pl.col('y')>72000).then(pl.lit('R'))
  .when(pl.col('unm') & (pl.col('bs').abs()<=6)).then(pl.lit('SF')).when(pl.col('unm')).then(pl.lit('UN')).otherwise(pl.lit('M')))
print('(a) LIRF: SCHED minute %5==0 share by class (unmatched only for RANK)')
print(df.filter(pl.col('ADEP_mvt')=='LIRF', pl.col('unm')|pl.col('cls').is_in(['M','R'])).group_by('cls',pl.col('unm')).agg(pl.len(),pl.col('smin5').mean().round(3)).sort('cls'))
print('(b) ranking DEP rows sharing the rollover-row timestamp fingerprint (LIRF, unmatched, MVT on grid):',
  df.filter(~pl.col('train'),pl.col('ADEP_mvt')=='LIRF',pl.col('unm'),pl.col('mgrid')).height,
  '; of which ms>=12h:', df.filter(~pl.col('train'),pl.col('ADEP_mvt')=='LIRF',pl.col('unm'),pl.col('ms')>=43200).height,
  '; training same fingerprint:', df.filter(pl.col('train'),pl.col('ADEP_mvt')=='LIRF',pl.col('unm')).height, 'with', df.filter(pl.col('train'),pl.col('ADEP_mvt')=='LIRF',pl.col('unm'),pl.col('y')>72000).height,'rollovers')
# (c)
ng=['EDDF','EHAM','LEBL','LEMD','LSZH']
u=df.filter(pl.col('ADEP_mvt').is_in(ng),pl.col('unm'))
codes={a:i for i,a in enumerate(ng)}
def X(t,with_grid):
    cols=[t['ADEP_mvt'].replace_strict(codes).to_numpy(), np.clip(t['ms'].to_numpy(),-7200,90000), t['hr'].to_numpy()]
    if with_grid: cols.append(t['mgrid'].cast(pl.Int8).to_numpy())
    return np.column_stack(cols).astype(float)
tr=u.filter(pl.col('train'))
res={}
for wg in [False,True]:
    pred=np.zeros(tr.height); yy=np.clip(tr['y'].to_numpy().astype(float),0,7200)  # clip to avoid domination by outliers
    for m in range(1,13):
        te=(tr['month']==m).to_numpy()
        mdl=lgb.LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=15,min_child_samples=30,verbose=-1,categorical_feature=[0])
        mdl.fit(X(tr.filter(~pl.Series(te)),wg),yy[~te]); pred[te]=mdl.predict(X(tr.filter(pl.Series(te)),wg))
    res[wg]=pred
yy=np.clip(tr['y'].to_numpy().astype(float),0,7200)
for a in ng+['ALL']:
    msk=np.ones(tr.height,bool) if a=='ALL' else (tr['ADEP_mvt']==a).to_numpy()
    r0=np.sqrt(np.mean((res[False][msk]-yy[msk])**2)); r1=np.sqrt(np.mean((res[True][msk]-yy[msk])**2))
    print(f'(c) {a}: n={msk.sum()} LOMO RMSE (y clipped 0..7200) without grid flag {r0:.1f}, with {r1:.1f}')
print('ranking unmatched rows at non-grid airports with MVT on grid:', u.filter(~pl.col('train'),pl.col('mgrid')).group_by('ADEP_mvt').len().sort('ADEP_mvt'))
