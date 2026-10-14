import pandas as pd, numpy as np, glob
L='/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/adsb/learn/'
lab=pd.concat([pd.read_parquet(f,columns=['MVT_ID_mvt','TAXITIME_SEC_mvt']) for f in glob.glob('data/raw/training_2025-*.parquet')])
sub=pd.read_parquet('/home/user/prc-data-challenge/outputs/c15_v42_flfixm_seeds_ucmix.parquet')
both=pd.concat([lab,sub])
cs=['MVT_ID_mvt','ADEP_mvt','t_mvt','t_aobt3','t_gs3','day']
d=pd.concat([pd.read_parquet(f,columns=cs) for f in sorted(glob.glob(L+'feat_202*.parquet'))])
d=d.merge(both,on='MVT_ID_mvt',how='inner')
d['ym']=d.day.astype(str).str[:7]
d['proxy']=d.t_mvt-d.t_aobt3; d['g']=d.t_mvt-d.t_gs3
ok=d.proxy.between(0,3600)&d.g.between(60,3600)&d.TAXITIME_SEC_mvt.between(0,3600)
x=d[ok].copy(); x['block_minus_gs3']=x.g-x.TAXITIME_SEC_mvt; x['aobt_minus_gs3']=x.g-x.proxy
x['res']=x.TAXITIME_SEC_mvt-x.proxy
t=x.groupby(['ADEP_mvt','ym']).agg(n=('g','size'),block_gs3=('block_minus_gs3','mean'),aobt_gs3=('aobt_minus_gs3','mean'),res=('res','mean'))
pd.set_option('display.width',250)
for k in ['block_gs3','aobt_gs3','res']:
    print(k); print(t[k].unstack().round(0).to_string())
print(t.n.unstack().to_string())
x.to_parquet('/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/agents/drift/adsb_gs3.parquet')
