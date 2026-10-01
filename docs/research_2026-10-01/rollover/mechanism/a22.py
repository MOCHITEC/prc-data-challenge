from common import *
import numpy as np
D=pl.read_parquet('train_dep_cls.parquet').filter((pl.col('APT')=='LIRF')&pl.col('unm')&(pl.col('ms')>=12*3600))
D=D.with_columns(roll=(pl.col('y')-86400).is_between(-3600,7200)&(pl.col('cls')!='fill'))
ms=D['ms'].to_numpy().astype(float); y=D['y'].to_numpy().astype(float); roll=D['roll'].to_numpy(); mon=D['month'].to_numpy()
print('n',len(ms),'roll',roll.sum())
YR=87300.
def band(m, scheme):
    h=m/3600
    if scheme=='one': return np.zeros_like(h,dtype=int)
    if scheme=='3b': return np.digitize(h,[16,28])
    if scheme=='4b': return np.digitize(h,[14,16,28])
def lomo(scheme, a=0.5):
    b=band(ms,scheme); pred=np.zeros_like(ms); ps=np.zeros_like(ms)
    for m in np.unique(mon):
        tr=mon!=m; te=mon==m
        for k in np.unique(b[te]):
            sel=tr&(b==k); p=(roll[sel].sum()+a)/(sel.sum()+2*a)
            idx=te&(b==k); ps[idx]=p; pred[idx]=p*YR+(1-p)*ms[idx]
    return pred, ps
for sch in ['one','3b','4b']:
    pred,ps=lomo(sch)
    ll=-np.mean(np.where(roll,np.log(ps),np.log(1-ps)))
    print(sch,'LOMO SSE %.3e'%((pred-y)**2).sum(),'logloss %.3f'%ll)
print('baseline ms (fill) SSE %.3e'%((ms-y)**2).sum(), ' all-roll SSE %.3e'%((YR-y)**2).sum())
# full-data band estimates with Jeffreys CI
from scipy.stats import beta
b=band(ms,'4b')
for k,name in enumerate(['[12,14)','[14,16)','[16,28)','>=28']):
    s=b==k; n=s.sum(); r=roll[s].sum()
    print(name,'n',n,'roll',r,'p_mean %.3f'%((r+.5)/(n+1)),'CI90 [%.3f,%.3f]'%(beta.ppf(.05,r+.5,n-r+.5),beta.ppf(.95,r+.5,n-r+.5)))
b=band(ms,'3b')
for k,name in enumerate(['[12,16)','[16,28)','>=28']):
    s=b==k; n=s.sum(); r=roll[s].sum()
    print(name,'n',n,'roll',r,'p_mean %.3f'%((r+.5)/(n+1)),'CI90 [%.3f,%.3f]'%(beta.ppf(.05,r+.5,n-r+.5),beta.ppf(.95,r+.5,n-r+.5)))
# roll taxi
print('roll y-86400', np.sort(y[roll]-86400))
