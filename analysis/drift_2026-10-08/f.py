import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))]); tr=tr[tr.PHASE_mvt=='DEP']
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols); r=r[r.PHASE_mvt=='DEP']
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(sub,on='MVT_ID_mvt')
def f(d,y):
    t=pd.to_datetime(d.MVT_TIME_UTC_mvt); d['m']=t.dt.month; d['day']=t.dt.day
    d['res']=(d[y]-(t-pd.to_datetime(d.AOBT_3_flt)).dt.total_seconds()).clip(-1200,1200); return d
tr=f(tr,'TAXITIME_SEC_mvt'); r=f(r,'pred')
print(tr.groupby(['ADEP_mvt','m']).res.mean().unstack().round(0).to_string())
print(r.groupby(['ADEP_mvt','m']).res.mean().unstack().round(0).to_string())
x=r[(r.ADEP_mvt=='LFPG')&(r.m==7)].groupby('day').res.mean().round(0); print(x.to_dict())
x=tr[(tr.ADEP_mvt=='LFPG')&(tr.m==7)].groupby('day').res.mean().round(0); print(x.to_dict())
