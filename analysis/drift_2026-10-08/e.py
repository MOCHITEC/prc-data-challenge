import pandas as pd, numpy as np, glob
cols=['PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))]); tr=tr[tr.PHASE_mvt=='DEP']
t=pd.to_datetime(tr.MVT_TIME_UTC_mvt); tr['m']=t.dt.month
b=pd.to_datetime(tr.BLOCK_TIME_UTC_mvt); a=pd.to_datetime(tr.AOBT_3_flt)
tr['d']=(a-b).dt.total_seconds()  # = res
tr['bs']=b.dt.second; tr['as']=a.dt.second; tr['ms']=t.dt.second
for ap in ['LFPG','LIRF','LEBL','EGLL']:
    x=tr[tr.ADEP_mvt==ap]
    print(ap)
    print(x.groupby('m').d.apply(lambda s: s.clip(-600,600).round(-1).value_counts(normalize=True).head(4).round(3).to_dict()).to_string())
    print(x.groupby('m')[['bs','as','ms']].agg(lambda s:(s==0).mean()).round(2).T.to_string())
