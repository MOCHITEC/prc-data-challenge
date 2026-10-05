import polars as pl, numpy as np
day=pl.read_parquet('daymean.parquet'); f=pl.read_parquet('pru_daily.parquet')
j=day.join(f,left_on=['ADEP_mvt','date'],right_on=['APT_ICAO','date'],how='left').with_columns(pl.col('date').dt.month().alias('m')).fill_null(0).fill_nan(0)
cols=['arr_dly_pf','reg_share','late_share']
tot_base=0;tot_new=0;N=0
for a in sorted(j['ADEP_mvt'].unique()):
  s=j.filter(pl.col('ADEP_mvt')==a)
  for tm,te in [(1,7),(7,1)]:
    A=s.filter(pl.col('m')==tm); B=s.filter(pl.col('m')==te)
    X=np.c_[np.ones(len(A)),A.select(cols).to_numpy()]; y=A['mu'].to_numpy()
    w=np.linalg.lstsq(X.T@X+np.diag([0]+[1e-3]*len(cols))*len(A),X.T@y,rcond=None)[0]
    Xb=np.c_[np.ones(len(B)),B.select(cols).to_numpy()]; p=Xb@w - w[0] + B['mu'].mean()*0  # only slope part, keep intercept 0
    p=Xb[:,1:]@w[1:]; p=p-p.mean()
    yb=B['mu'].to_numpy(); n=B['n'].to_numpy()
    tot_base+=(n*yb**2).sum(); tot_new+=(n*(yb-p)**2).sum(); N+=n.sum()
print('daily-mean SSE share: base',tot_base/N,'after',tot_new/N)
# translate to row RMSE (genuine rmse 274)
mse=274.0**2
print('row RMSE est', (mse - (tot_base-tot_new)/N)**.5)
