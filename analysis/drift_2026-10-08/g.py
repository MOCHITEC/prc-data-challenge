import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','STAND_mvt','RUNWAY_mvt','AIRCRAFT_TYPE_mvt','LOBT_flt','EOBT_1_flt']
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols); r=r[r.PHASE_mvt=='DEP']
t=pd.to_datetime(r.MVT_TIME_UTC_mvt); r['m']=t.dt.month; r['proxy']=(t-pd.to_datetime(r.AOBT_3_flt)).dt.total_seconds()
files={'v47':'/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet','v40':'/home/user/prc-data-challenge/outputs/v40_v39_7m_pos_12m.parquet','v34':'/home/user/prc-data-challenge/outputs/v34_v33_posjoin.parquet'}
for f in glob.glob('/home/user/pb-work/outputs/*.parquet')+glob.glob('/home/user/pb-work/outputs/s3/*.parquet'): files[f.split('outputs/')[1]]=f
for f in sorted(glob.glob('/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/v3*_theirs.parquet')): files[f.split('/')[-1]]=f
rows=[]
for k,f in files.items():
    s=pd.read_parquet(f)
    if 'TAXITIME_SEC_mvt' not in s: print('skip',k,s.columns[:5].tolist()); continue
    d=r.merge(s[['MVT_ID_mvt','TAXITIME_SEC_mvt']],on='MVT_ID_mvt')
    d['res']=(d.TAXITIME_SEC_mvt-d.proxy).clip(-1200,1200)
    g=d.groupby(['ADEP_mvt','m']).res.mean()
    rows.append(g.rename(k))
print(pd.concat(rows,axis=1).round(0).T.to_string())
