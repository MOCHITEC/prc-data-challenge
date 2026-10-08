import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt']
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))]); tr=tr[tr.PHASE_mvt=='DEP']
t=pd.to_datetime(tr.MVT_TIME_UTC_mvt); tr['m']=t.dt.month
tr['res']=tr.TAXITIME_SEC_mvt-(t-pd.to_datetime(tr.AOBT_3_flt)).dt.total_seconds()
print(tr.groupby(['ADEP_mvt','m']).res.median().unstack().round(0).to_string())
print(tr.groupby(['ADEP_mvt','m']).TAXITIME_SEC_mvt.median().unstack().round(0).to_string())
