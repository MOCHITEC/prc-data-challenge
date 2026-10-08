import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt','AOBT_3_flt']
r=pd.read_parquet('data/raw/ranking.parquet',columns=cols)
tr=pd.concat([pd.read_parquet(f,columns=cols) for f in sorted(glob.glob('data/raw/training_*.parquet'))])
for d in (r,tr):
    d['t']=pd.to_datetime(d.MVT_TIME_UTC_mvt); d['m']=d.t.dt.month
    d['apt']=np.where(d.PHASE_mvt.str.upper().str.startswith('D'),d.ADEP_mvt,d.ADES_mvt)
print(r.PHASE_mvt.value_counts(), r.groupby('PHASE_mvt').TAXITIME_SEC_mvt.apply(lambda s:s.notna().mean()))
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet')
print(sub.shape, sub.columns.tolist())
dep=r[r.PHASE_mvt.str.upper().str.startswith('D')].merge(sub.rename(columns={'TAXITIME_SEC_mvt':'pred'}),on='MVT_ID_mvt')
arr=r[~r.PHASE_mvt.str.upper().str.startswith('D')]
tr=tr[tr.m.isin([1,7])]
trd=tr[tr.PHASE_mvt.str.upper().str.startswith('D')]; tra=tr[~tr.PHASE_mvt.str.upper().str.startswith('D')]
med=lambda s: s.clip(0,3600).median()
out=pd.DataFrame({
 'arr25':tra.groupby(['apt','m']).TAXITIME_SEC_mvt.apply(med),
 'arr26':arr.groupby(['apt','m']).TAXITIME_SEC_mvt.apply(med),
 'dep25':trd.groupby(['apt','m']).TAXITIME_SEC_mvt.apply(med),
 'pred26':dep.groupby(['apt','m']).pred.apply(med),
 'n26':dep.groupby(['apt','m']).size(),
 'n25':trd.groupby(['apt','m']).size(),
})
out['d_arr']=out.arr26-out.arr25; out['d_dep_pred']=out.pred26-out.dep25
print(out.round(0).to_string())
print('corr', out[['d_arr','d_dep_pred']].corr().iloc[0,1])
