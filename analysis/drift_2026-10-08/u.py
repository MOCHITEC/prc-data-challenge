import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','EOBT_1_flt','AOBT_3_flt','TAXITIME_SEC_mvt']
d=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))]); d=d[(d.PHASE_mvt=='DEP')&d.AOBT_3_flt.notna()]
T=lambda c: pd.to_datetime(d[c])
d['lme']=(T('LOBT_flt')-T('EOBT_1_flt')).dt.total_seconds()
d['ae']=(T('AOBT_3_flt')-T('EOBT_1_flt')).dt.total_seconds().clip(-1800,3600); d['al']=(T('AOBT_3_flt')-T('LOBT_flt')).dt.total_seconds().clip(-1800,3600)
d['res']=(T('AOBT_3_flt')-T('BLOCK_TIME_UTC_mvt')).dt.total_seconds().clip(-1200,1200)
for ap in ['LFPG','EGLL','EDDF','LEBL']:
    x=d[d.ADEP_mvt==ap]
    for nm,y in (('le<0',x[x['lme']<0]),('le>0',x[x['lme']>0]),('le==0',x[x['lme']==0])):
        X=np.c_[np.ones(len(y)),y.ae,y.al]; c=np.linalg.lstsq(X,y.res,rcond=None)[0]
        print(ap,nm,len(y),'coef ae(EOBT1)',round(c[1],3),'al(LOBT)',round(c[2],3), 'mean res',round(y.res.mean(),1))
# LFPG le<0 rows by month: res and their le magnitude
x=d[(d.ADEP_mvt=='LFPG')&(d['lme']<0)]
print(x['lme'].describe().round(0).to_dict())
