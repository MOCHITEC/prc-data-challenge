import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','STAND_mvt','RUNWAY_mvt','TAXITIME_SEC_mvt']
tr=pd.read_parquet('data/raw/training_2025-07-01_2025-08-01.parquet',columns=cols); tr=tr[(tr.PHASE_mvt=='DEP')&(tr.ADEP_mvt=='LFPG')]
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols); r=r[(r.PHASE_mvt=='DEP')&(r.ADEP_mvt=='LFPG')]
r=r[pd.to_datetime(r.MVT_TIME_UTC_mvt).dt.month==7]
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(sub,on='MVT_ID_mvt')
for d,y in ((tr,'TAXITIME_SEC_mvt'),(r,'pred')):
    t=pd.to_datetime(d.MVT_TIME_UTC_mvt); d['proxy']=(t-pd.to_datetime(d.AOBT_3_flt)).dt.total_seconds()
    d['res']=(d[y]-d.proxy).clip(-1200,1200); d['st']=d.STAND_mvt.astype(str).str[:1]; d['h']=t.dt.hour
for k in ['RUNWAY_mvt','st']:
    a=tr.groupby(k).agg(n25=('res','size'),res25=('res','mean'),proxy25=('proxy','mean'),y25=('TAXITIME_SEC_mvt','mean'))
    b=r.groupby(k).agg(n26=('res','size'),res26=('res','mean'),proxy26=('proxy','mean'),pred26=('pred','mean'))
    print(a.join(b,how='outer').round(0).to_string())
print('proxy', tr.proxy.describe().round(0).to_dict(), r.proxy.describe().round(0).to_dict())
