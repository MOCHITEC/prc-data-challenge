import pandas as pd, numpy as np, glob
cols=['PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','AOBT_3_flt']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))])
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols); d=pd.concat([tr,r]); d=d[d.PHASE_mvt=='DEP']
t={c:pd.to_datetime(d[c]) for c in cols[2:]}
d['ym']=t['MVT_TIME_UTC_mvt'].dt.strftime('%y%m')
pairs={'eobt_sched':('EOBT_1_flt','SCHED_TIME_UTC_mvt'),'eobt_lobt':('EOBT_1_flt','LOBT_flt'),'iobt_lobt':('IOBT_flt','LOBT_flt'),'lobt_sched':('LOBT_flt','SCHED_TIME_UTC_mvt')}
for k,(a,b) in pairs.items():
    x=(t[a]-t[b]).dt.total_seconds(); d[k+'_eq']=(x==0); d[k+'_med']=x
pd.set_option('display.width',300)
for k in pairs:
    g=d.groupby(['ADEP_mvt','ym'])[k+'_eq'].mean().unstack().round(2)
    print(k,'share equal'); print(g.to_string())
