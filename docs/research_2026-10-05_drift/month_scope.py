import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score
R='/home/user/prc-data-challenge/data/'
cols=['MVT_ID_mvt','ADEP_mvt','STAND_mvt','RUNWAY_mvt','FLIGHT_mvt','FLIGHT_ID_mvt','MVT_TIME_UTC_mvt','fl_nm_rate_other','stand_rwy_share']
tr=pl.read_parquet(R+'work/feat_train.parquet',columns=cols); rk=pl.read_parquet(R+'work/feat_rank.parquet',columns=cols)
def month_scope(d):
    d=d.with_columns(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y%m').alias('_ym'),pl.col('FLIGHT_ID_mvt').is_not_null().cast(pl.Float64).alias('_m'))
    g=['_ym','ADEP_mvt','FLIGHT_mvt']; n=pl.len().over(g)-1
    d=d.with_columns(pl.when(pl.col('FLIGHT_mvt').is_not_null()&(n>0)).then((pl.col('_m').sum().over(g)-pl.col('_m'))/n).otherwise(None).alias('fl_nm_rate_other_m'),
                     (pl.len().over(['_ym','ADEP_mvt','STAND_mvt','RUNWAY_mvt'])/pl.len().over(['_ym','ADEP_mvt','STAND_mvt'])).alias('stand_rwy_share_m'))
    return d
tr=month_scope(tr).filter(pl.col('MVT_TIME_UTC_mvt').dt.month().is_in([1,7])); rk=month_scope(rk)
a=tr.sample(150000,seed=0); b=rk.sample(150000,seed=1); y=np.r_[np.zeros(a.height),np.ones(b.height)]; f=np.random.default_rng(0).integers(0,2,len(y))
P=dict(objective='binary',num_leaves=15,min_data_in_leaf=200,learning_rate=0.1,verbose=-1,num_threads=1)
for c in ['fl_nm_rate_other','fl_nm_rate_other_m','stand_rwy_share','stand_rwy_share_m']:
    x=np.r_[a[c].cast(pl.Float64).to_numpy(),b[c].cast(pl.Float64).to_numpy()].reshape(-1,1); p=np.empty(len(y))
    for k in (0,1): p[f==k]=lgb.train(P,lgb.Dataset(x[f!=k],y[f!=k]),60).predict(x[f==k])
    print(c, round(roc_auc_score(y,p),3))
