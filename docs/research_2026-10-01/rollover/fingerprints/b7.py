import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
d=pl.read_parquet('dep5.parquet').with_columns(mroll=(pl.col('y')>72000)&((pl.col('y')-pl.col('ms')).abs()>60))
c=pl.col
tc=d.filter((c('src')=='train')&c('unm')&(c('ms')>=28800)&c('cross')&(c('ms')<100000))  # drop NOSOS739 (ms 131k, fill-type)
y=tc['mroll'].to_numpy(); ms=tc['ms'].to_numpy().astype(float); Y=tc['y'].to_numpy().astype(float); mon=tc['mon'].to_numpy(); g=tc['glitch'].to_numpy()
r16=tc['r16R'].to_numpy()
R=87300.; sse=lambda p: ((Y-(ms+p*(R-ms)))**2).sum()
print('n',len(y),'roll',y.sum(),'SSE pred=ms %.3e'%sse(np.zeros(len(y))))
# oracle-ish upper bound: perfect classifier
print('SSE perfect classifier %.3e'%sse(y.astype(float)))
def lomo(X,C):
    p=np.zeros(len(y))
    for m in np.unique(mon):
        tr=mon!=m; p[~tr]=LogisticRegression(C=C,max_iter=1000).fit(X[tr],y[tr]).predict_proba(X[~tr])[:,1]
    return p
# band-rate LOMO (non-parametric): bands 8-14, 14-18, 18-24h
bands=np.digitize(ms/3600,[14,18])
p=np.zeros(len(y))
for m in np.unique(mon):
    tr=mon!=m
    for b in range(3):
        sel=tr&(bands==b); p[(~tr)&(bands==b)]=(y[sel].sum()+0.5)/(sel.sum()+1)
print('LOMO band-rate SSE %.3e'%sse(p))
p2=p.copy()
for nm,X in [('ms',np.c_[ms/3600-16]),('ms+glitch',np.c_[ms/3600-16,g]),('ms+glitch+16R',np.c_[ms/3600-16,g,r16])]:
    for C in [0.3,1.0,3.0]:
        p=lomo(X,C); pr=p>=0.5
        print(f'LOMO {nm:14s} C={C}: SSE {sse(p):.3e}  logloss {-np.mean(y*np.log(p)+(1-y)*np.log(1-p)):.3f}  prec@.5 {(pr&y).sum()}/{pr.sum()}  rec {(pr&y).sum()}/{y.sum()}')
print('const-rate logloss baseline', -np.mean(y*np.log(y.mean())+(1-y)*np.log(1-y.mean())))
