import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
d=pl.read_parquet('dep5.parquet')
t=d.filter((pl.col('src')=='train')&pl.col('unm')&(pl.col('ms')>=28800)).with_columns(r=(pl.col('cls')=='roll'))
# treat NOSOS739 (y==ms and y>72000) as roll label for classification but SSE uses true y
print('n',t.height,'roll',t['r'].sum(), 'classes',t.group_by('cls').len().to_dicts())
y=t['r'].to_numpy(); Y=t['y'].to_numpy().astype(float); ms=t['ms'].to_numpy().astype(float); mon=t['mon'].to_numpy()
R=87300.
X_all={'ms':np.c_[ms/3600-16],
       'ms+glitch':np.c_[ms/3600-16,t['glitch'].to_numpy()],
       'ms+glitch+16R':np.c_[ms/3600-16,t['glitch'].to_numpy(),t['r16R'].to_numpy()],
       'ms+glitch+16R+summer':np.c_[ms/3600-16,t['glitch'].to_numpy(),t['r16R'].to_numpy(),t['summer'].to_numpy()],
       'ms+cross':np.c_[ms/3600-16,t['cross'].to_numpy()]}
def sse(p): pred=ms+p*(R-ms); return ((Y-pred)**2).sum()
print('SSE p=0 (pred ms):%.3e'%sse(np.zeros(len(y))))
# constant LOMO
p=np.zeros(len(y))
for m in np.unique(mon):
    tr=mon!=m; p[~tr]=y[tr].mean()
print('SSE const LOMO: %.3e'%sse(p))
# ms-band LOMO (12h threshold)
for C in [0.1,0.3,1.0,10.0]:
  for nm,X in X_all.items():
    p=np.zeros(len(y))
    for m in np.unique(mon):
        tr=mon!=m
        clf=LogisticRegression(C=C,max_iter=1000).fit(X[tr],y[tr]); p[~tr]=clf.predict_proba(X[~tr])[:,1]
    ll=-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))
    # precision/recall at 0.5
    pr=(p>=0.5); prec=(pr&(y==1)).sum()/max(pr.sum(),1); rec=(pr&(y==1)).sum()/y.sum()
    print(f'C={C:<4} {nm:22s} LOMO logloss {ll:.3f} SSE {sse(p):.3e} prec@0.5 {prec:.2f} rec@0.5 {rec:.2f}')
clf=LogisticRegression(C=1.0,max_iter=1000).fit(X_all['ms+glitch'],y); print('full fit ms+glitch coef',clf.coef_,clf.intercept_)
clf=LogisticRegression(C=1.0,max_iter=1000).fit(X_all['ms'],y); print('full fit ms coef',clf.coef_,clf.intercept_)
