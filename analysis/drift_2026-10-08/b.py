import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt','LOBT_flt','EOBT_1_flt']
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols); r=r[r.PHASE_mvt=='DEP']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))]); tr=tr[tr.PHASE_mvt=='DEP']
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(sub,on='MVT_ID_mvt'); r['y']=r.pred; tr['y']=tr.TAXITIME_SEC_mvt
for d in (r,tr):
    t=pd.to_datetime(d.MVT_TIME_UTC_mvt); d['m']=t.dt.month
    d['proxy']=(t-pd.to_datetime(d.AOBT_3_flt)).dt.total_seconds()
    d['mat']=d.proxy.notna()
    d['res']=d.y-d.proxy
    d['lobt']=(t-pd.to_datetime(d.LOBT_flt)).dt.total_seconds()
tr=tr[tr.m.isin([1,7])]
g=lambda d: d.groupby(['ADEP_mvt','m']).agg(match=('mat','mean'),proxy=('proxy','median'),res=('res','median'),res_mean=('res',lambda s:s.clip(-1800,1800).mean()),lobt=('lobt','median'),y=('y','median'))
a=g(tr); b=g(r)
o=a.join(b,lsuffix='25',rsuffix='26')
print(o.round(2).to_string())
