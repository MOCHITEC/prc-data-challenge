# LOMO on the LIRF eligible set (unmatched, local date(MVT)=date(SCHED)+1): 3-class posterior-mean predictor
import polars as pl, numpy as np
j=pl.read_parquet('lirf_elig_rot.parquet').with_columns(
  inb1=((pl.col('ib_dd')==1)&(pl.col('gap_h')<12)).fill_null(False),
  msb=pl.col('ms').cut([3*3600,6*3600,12*3600,18*3600],labels=['a','b','c','d','e']).cast(pl.Utf8))
# day-level ARR anomaly at LIRF (unmatched ARR count on SCHED local day, +-1d)
dl=pl.read_parquet('daylift.parquet').filter(pl.col('APT')=='LIRF').select('day','an_unm_w','an_1d_w')
j=j.with_columns(day=pl.col('SCHED_TIME_UTC_mvt').dt.convert_time_zone('Europe/Rome').dt.date()).join(dl,on='day',how='left').with_columns(arrday=(pl.col('an_unm_w')>=3).fill_null(False))
t=j.filter(pl.col('src')=='train').to_pandas()
t['mon']=t['MVT_TIME_UTC_mvt'].dt.month
OKT=900.0; ROLLADD=1100.0
def fit_predict(tr,te,keys,alpha=1.0):
  # empirical class probs per key cell, shrunk to msb-level then global
  out=[]
  g=tr.groupby(keys)['cls'].value_counts().unstack(fill_value=0)
  base=tr['cls'].value_counts(normalize=True)
  gm=tr.groupby('msb')['cls'].value_counts().unstack(fill_value=0)
  for _,r in te.iterrows():
    prior=base.copy()
    if r['msb'] in gm.index:
      c=gm.loc[r['msb']].reindex(base.index,fill_value=0); prior=(c+2*prior)/(c.sum()+2)
    k=tuple(r[x] for x in keys) if len(keys)>1 else r[keys[0]]
    if k in g.index and keys!=['msb']:
      c=g.loc[k].reindex(base.index,fill_value=0); p=(c+alpha*2*prior)/(c.sum()+alpha*2)
    else: p=prior
    p=p.reindex(['ROLL','fill','ok'],fill_value=0)
    mu=p['ROLL']*(86400+ROLLADD)+p['fill']*r['ms']+p['ok']*OKT
    out.append((p['ROLL'],mu))
  return out
res={}
for name,keys in [('ms only',['msb']),('ms+inbound_D+1',['msb','inb1']),('ms+ARRday_unm>=3',['msb','arrday'])]:
  P=[];M=[];Y=[];C=[]
  for m in sorted(t['mon'].unique()):
    tr=t[t['mon']!=m]; te=t[t['mon']==m]
    o=fit_predict(tr,te,keys)
    P+= [a for a,b in o]; M+=[b for a,b in o]; Y+=list(te['y']); C+=list(te['cls'])
  P=np.array(P);M=np.array(M);Y=np.array(Y,float);C=np.array(C)
  yb=(C=='ROLL').astype(float)
  ll=-np.mean(yb*np.log(np.clip(P,1e-4,1))+(1-yb)*np.log(np.clip(1-P,1e-4,1)))
  sse=np.sum((Y-M)**2)
  # precision/recall at p>=0.3
  fl=P>=0.3
  print(f'{name:22s} logloss {ll:.4f} brier {np.mean((P-yb)**2):.4f} SSE {sse:.4e} RMSE_on_set {np.sqrt(sse/len(Y)):.0f}  flag(p>=.3) n={fl.sum()} TP={int((fl&(yb==1)).sum())} recall={((fl&(yb==1)).sum()/yb.sum()):.2f}')
  res[name]=sse
N=344841
for k in res: print(k,'delta SSE vs ms-only per LB row (MSE units, scaled 2025->1 period by n_rank_elig/n_train_elig=52/168):', (res[k]-res['ms only'])*52/168/N)
print('inb1 counts rank elig', j.filter(pl.col('src')=='rank').group_by('msb','inb1').len().sort('msb','inb1'))
