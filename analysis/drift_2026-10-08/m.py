import pandas as pd, numpy as np, glob
L='/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/adsb/learn/'
def load(pat):
    return pd.concat([pd.read_parquet(f,columns=['MVT_ID_mvt','ADEP_mvt','t_mvt','t_aobt3','t_push','t_gs3','t_to','ground_fix','join']) for f in sorted(glob.glob(L+pat))])
cols=['MVT_ID_mvt','BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt']
lab=pd.read_parquet('data/raw/training_2025-07-01_2025-08-01.parquet',columns=cols)
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
out=[]
for ap in ['LFPG','EDDF','EGLL','EHAM','LEMD','LSZH']:
  for yr,pat in ((2025,'feat_2025-07-*.parquet'),(2026,'feat_2026-07-*.parquet')):
    d=load(pat); d=d[(d.ADEP_mvt==ap)]
    d['mvt_push']=d.t_mvt-d.t_push; d['aobt_push']=d.t_aobt3-d.t_push; d['mvt_gs3']=d.t_mvt-d.t_gs3
    d['proxy']=d.t_mvt-d.t_aobt3
    if yr==2025: d=d.merge(lab,on='MVT_ID_mvt',how='left'); d['y']=d.TAXITIME_SEC_mvt
    else: d=d.merge(sub,on='MVT_ID_mvt',how='left'); d['y']=d.pred
    ok=d.t_push.notna()&d.proxy.between(0,3600)&d.mvt_push.between(0,3600)
    x=d[ok]; m=d.proxy.between(0,3600)
    out.append(dict(apt=ap,yr=yr,n=len(d),n_push=ok.sum(),proxy_all=d.proxy[m].mean(),y_all=d.y[m].mean(),
        mvt_push=x.mvt_push.mean(),aobt_push=x.aobt_push.mean(),proxy_push=x.proxy.mean(),y_push=x.y.mean(),
        y_minus_mvtpush=(x.y-x.mvt_push).mean()))
print(pd.DataFrame(out).round(0).to_string())
