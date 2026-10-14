import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
x=pd.read_parquet('/tmp/claude-0/-home-user-prc-data-challenge/7ae154e2-3baf-5cb4-beae-354c18126ab1/scratchpad/agents/drift/adsb_gs3.parquet').drop_duplicates('MVT_ID_mvt')
cols=['MVT_ID_mvt','aobt3_minus_eobt1','aobt3_minus_sched']
f=pd.concat([pd.read_parquet('data/work_flseq/feat_train.parquet',columns=cols),pd.read_parquet('data/work_flseq/feat_rank.parquet',columns=cols)])
x=x.merge(f,on='MVT_ID_mvt')
x['de']=x.aobt3_minus_eobt1.clip(-1800,3600)
x['yr']=x.ym.str[:4]
bins=[-1e9,-300,-60,60,300,900,1e9]; x['bk']=pd.cut(x.de,bins)
for ap in ['LFPG','EHAM','LSZH','LEBL']:
    y=x[x.ADEP_mvt==ap]
    print(ap)
    print(y.groupby(['yr','bk']).agg(n=('res','size'),aobt_gs3=('aobt_minus_gs3','mean'),block_gs3=('block_minus_gs3','mean'),res=('res','mean')).round(0).unstack(0).to_string())
    for yr,z in y.groupby('yr'):
        X=np.c_[np.ones(len(z)),z.de]
        print(yr,'slope aobt_gs3',np.linalg.lstsq(X,z.aobt_minus_gs3,rcond=None)[0][1].round(3),'slope block_gs3',np.linalg.lstsq(X,z.block_minus_gs3,rcond=None)[0][1].round(3))
