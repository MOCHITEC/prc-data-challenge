import polars as pl, numpy as np
day=pl.read_parquet('daymean.parquet'); f=pl.read_parquet('pru_daily.parquet')
j=day.join(f,left_on=['ADEP_mvt','date'],right_on=['APT_ICAO','date'],how='left').with_columns(pl.col('date').dt.month().alias('m')).fill_null(0).fill_nan(0)
for cols in [['arr_dly_pf'],['reg_share'],['late_share'],['arr_dly_pf','reg_share','late_share']]:
  jj=j.with_columns([((pl.col(c)-pl.col(c).mean().over('ADEP_mvt','m'))/pl.col(c).std().over('ADEP_mvt','m').clip(1e-9)).alias(c) for c in cols]).with_columns((pl.col('mu')-pl.col('mu').mean().over('ADEP_mvt','m')).alias('y'))
  b=n_=0;base=new=0
  for tm,te in [(1,7),(7,1)]:
    A=jj.filter(pl.col('m')==tm);B=jj.filter(pl.col('m')==te)
    X=A.select(cols).to_numpy();w=np.linalg.lstsq(X,A['y'].to_numpy(),rcond=None)[0]
    p=B.select(cols).to_numpy()@w; y=B['y'].to_numpy(); n=B['n'].to_numpy()
    base+=(n*y**2).sum(); new+=(n*(y-p)**2).sum(); n_+=n.sum(); print(cols,tm,'->',te,'w',np.round(w,1))
  print(cols,'within-month daymean MSE',round(base/n_),'->',round(new/n_),' row rmse 274.0 ->',round((274**2-(base-new)/n_)**.5,2))
