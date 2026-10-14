# robustness: redo the sweep with the integrator's own LOMO q (early-stopped, judge_copyA/q_rows.parquet) instead of the production-recipe q
import polars as pl, numpy as np
np.set_printoptions(linewidth=200, suppress=True)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
Q=pl.read_parquet(f'{S}/ideas/judge_copyA/q_rows.parquet'); print(Q.schema, Q.height)
d=pl.read_parquet('lirf25.parquet').join(Q.select('MVT_ID_mvt',pl.col('q').alias('q2')),on='MVT_ID_mvt',how='inner'); print('joined',d.height, 'corr(q,q2)', np.corrcoef(d['q'],d['q2'])[0,1])
y=d['y'].to_numpy(); ms=d['ms'].to_numpy().astype(float); ml=d['mlobt'].to_numpy().astype(float); mon=d['month'].to_numpy(); imp=d['imp'].to_numpy(); p=d['final_12'].to_numpy(); copy=d['copy'].to_numpy()
lo=ml-3606; hi=ml+3606; months=np.arange(1,13)
def pred(pull):
    ov=p+pull*(ms-p); ov=np.where(ov<0,p,ov); return np.maximum(np.clip(ov,lo,hi),0)
for lab in ['q','q2']:
    q=d[lab].to_numpy(); qg=np.where(imp,0,q); S0=(pred(0.4*qg)-y)**2
    print(f'--- {lab}: calibration by bin (copy rate vs mean q):')
    edges=np.array([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]); b_=np.digitize(q,edges)-1
    print('   ', [ (round(float(q[(b_==j)&~imp].mean()),2), round(float(copy[(b_==j)&~imp].mean()),2)) for j in range(len(edges)-1)])
    for name,pull in [('w=0.65',0.65*qg),('w=0.8',0.8*qg),('q^1.5',qg**1.5),('q^2',qg**2)]:
        dd=(pred(pull)-y)**2-S0; bym=np.array([dd[mon==k].sum() for k in months])/1e6
        print(f'   {name}: dSSE {bym.sum():.1f} M, months improved {(bym<0).sum()}/12, FebJun {bym[[1,5]].sum():.1f}, JanFebJunJul {bym[[0,1,5,6]].sum():.1f}')
