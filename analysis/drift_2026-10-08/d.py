import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt']
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols)
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
tr=pd.read_parquet('data/raw/training_2025-01-01_2025-02-01.parquet',columns=cols)
def prep(d,ycol):
    t=pd.to_datetime(d.MVT_TIME_UTC_mvt); d['day']=t.dt.day; d['m']=t.dt.month
    d['res']=(d[ycol]-(t-pd.to_datetime(d.AOBT_3_flt)).dt.total_seconds()).clip(-1800,1800)
    return d
dep=prep(r[r.PHASE_mvt=='DEP'].merge(sub,on='MVT_ID_mvt'),'pred')
arr=r[(r.PHASE_mvt=='ARR')].copy(); arr['day']=pd.to_datetime(arr.MVT_TIME_UTC_mvt).dt.day; arr['m']=pd.to_datetime(arr.MVT_TIME_UTC_mvt).dt.month
t5=prep(tr[tr.PHASE_mvt=='DEP'].copy(),'TAXITIME_SEC_mvt')
a='LTFM'
x=pd.DataFrame({'pred_res26':dep[(dep.ADEP_mvt==a)&(dep.m==1)].groupby('day').res.mean(),
 'pred26':dep[(dep.ADEP_mvt==a)&(dep.m==1)].groupby('day').pred.mean(),
 'arr26':arr[(arr.ADES_mvt==a)&(arr.m==1)].groupby('day').TAXITIME_SEC_mvt.mean(),
 'lab_res25':t5[t5.ADEP_mvt==a].groupby('day').res.mean(),
 'y25':t5[t5.ADEP_mvt==a].groupby('day').TAXITIME_SEC_mvt.mean()})
print(x.round(0).to_string())
