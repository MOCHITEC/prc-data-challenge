import pandas as pd, numpy as np, glob
L='/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/adsb/learn/'
def load(pat): return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(L+pat))])
lab=pd.concat([pd.read_parquet(f,columns=['MVT_ID_mvt','TAXITIME_SEC_mvt']) for f in glob.glob('data/raw/training_2025-*.parquet')])
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
rows=[]
for pat,yr in (('feat_2025-01-*',2025),('feat_2025-07-*',2025),('feat_2025-10-*',2025),('feat_2026-01-*',2026),('feat_2026-07-*',2026)):
    d=load(pat+'.parquet'); d=d[d.ADEP_mvt=='LFPG']
    d=d.merge(lab if yr==2025 else sub.rename(columns={'pred':'TAXITIME_SEC_mvt'}),on='MVT_ID_mvt',how='left')
    d['proxy']=d.t_mvt-d.t_aobt3
    for k in ['t_gs3','t_gs10','t_dep1','t_move','t_glast']: d['mvt_'+k]=d.t_mvt-d[k]
    ok=d.proxy.between(0,3600)&d.mvt_t_gs3.between(60,3600)
    x=d[ok]
    rows.append(dict(p=pat,n=len(d),n_ok=ok.sum(),gfix=d.ground_fix.mean(),proxy=x.proxy.mean(),y=x.TAXITIME_SEC_mvt.mean(),mvt_gs3=x.mvt_t_gs3.mean(),mvt_gs10=x.mvt_t_gs10.mean(),
        aobt_minus_gs3=(x.t_aobt3-x.t_gs3).mean(), y_minus_gs3=(x.TAXITIME_SEC_mvt-x.mvt_t_gs3).mean(),mvt_to=(d.t_mvt-d.t_to).median()))
print(pd.DataFrame(rows).round(2).to_string())
