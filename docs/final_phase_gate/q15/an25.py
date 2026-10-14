import polars as pl, numpy as np, sys
from sklearn.isotonic import IsotonicRegression
np.set_printoptions(linewidth=200, suppress=True)
d=pl.read_parquet('lirf25.parquet')
base=sys.argv[1] if len(sys.argv)>1 else 'final_12'
y=d['y'].to_numpy(); ms=d['ms'].to_numpy().astype(float); ml=d['mlobt'].to_numpy().astype(float); q=d['q'].to_numpy(); mon=d['month'].to_numpy(); imp=d['imp'].to_numpy(); p=d[base].to_numpy()
day=d['day'].cast(pl.Int32).to_numpy(); copy=d['copy'].to_numpy()
lo=ml-3606; hi=ml+3606
qg=np.where(imp,0.0,q)   # gate adopted
def pred(w, qq, revert=True, clip=True):
    ov=p+w*qq*(ms-p)
    if revert: ov=np.where(ov<0,p,ov)
    if clip: ov=np.clip(ov,lo,hi)
    return np.maximum(ov,0)
def sse(pr): return (pr-y)**2
N25=2085047
S0=sse(pred(0.4,qg)); print(f'base {base}: baseline (w=0.4,gated) SSE {S0.sum()/1e6:.1f} M; no-overlay SSE {sse(pred(0,qg)).sum()/1e6:.1f} M; overlay gain {(S0.sum()-sse(pred(0,qg)).sum())/1e6:.1f} M = {(S0.sum()-sse(pred(0,qg)).sum())/N25:.2f} s^2')
months=np.arange(1,13)
# (2) weight sweep per month
ws=np.round(np.arange(0,1.01,0.05),2)
tab=np.zeros((len(ws),12))
for i,w in enumerate(ws):
    s=sse(pred(w,qg))
    for k in months: tab[i,k-1]=(s[mon==k].sum()-S0[mon==k].sum())/1e6
print('dSSE (M s^2) vs w=0.4 by month, rows=w:'); print('w    '+' '.join(f'{k:7d}' for k in months)+'   total   FebJun  JanFebJunJul')
for i,w in enumerate(ws): print(f'{w:.2f} '+' '.join(f'{v:7.1f}' for v in tab[i])+f' {tab[i].sum():7.1f} {tab[i,[1,5]].sum():7.1f} {tab[i,[0,1,5,6]].sum():7.1f}')
# LOMO optimal w
print('LOMO: held-out month -> w chosen on other 11 months, dSSE on held-out')
tot=0; wins=0; chosen=[]
for k in months:
    tr=tab[:,months!=k].sum(axis=1); w_opt=ws[np.argmin(tr)]; dd=tab[ws==w_opt,k-1][0]; tot+=dd; wins+=dd<0; chosen.append(w_opt)
    print(f'  month {k}: w_opt {w_opt:.2f} dSSE {dd:.2f} M')
print(f'  LOMO total dSSE {tot:.1f} M = {tot*1e6/N25:.3f} s^2; months improved {wins}/12; chosen w {chosen}')
# (1) calibration: isotonic LOMO
qc=np.zeros_like(q)
for k in months:
    tr=(mon!=k)&(~imp); ir=IsotonicRegression(out_of_bounds='clip').fit(q[tr],copy[tr].astype(float)); qc[mon==k]=ir.predict(q[mon==k])
qcg=np.where(imp,0.0,qc)
print('isotonic LOMO: mean q %.4f mean qc %.4f copy %.4f; corr %.4f; rms(q-qc) %.4f' % (q[~imp].mean(), qc[~imp].mean(), copy[~imp].mean(), np.corrcoef(q,qc)[0,1], np.sqrt(((q-qc)**2)[~imp].mean())))
for w in [0.3,0.4,0.5,0.6]:
    s=sse(pred(w,qcg)); dd=s-S0; bym=np.array([dd[mon==k].sum() for k in months])/1e6
    print(f'  qcal w={w:.2f}: dSSE {bym.sum():.1f} M ({bym.sum()*1e6/N25:.3f} s^2), months improved {(bym<0).sum()}/12, FebJun {bym[[1,5]].sum():.1f}, by month {np.round(bym,1)}')
# (4) weight = q itself (mixture optimum) and c*q
print('two-point mixture: w = c*q (effective pull q*c*q):')
for c in [0.5,0.75,1.0,1.25,1.5]:
    for lab,qq in [('q',qg),('qcal',qcg)]:
        s=sse(pred(1.0,c*qq*qq)); dd=s-S0; bym=np.array([dd[mon==k].sum() for k in months])/1e6
        print(f'  c={c:.2f} {lab}: dSSE {bym.sum():.1f} M, months improved {(bym<0).sum()}/12, FebJun {bym[[1,5]].sum():.1f}')
# per-q-bin closed-form weight, LOMO
edges=np.array([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]); b_=np.digitize(q,edges)-1
x=(ms-p); r=(y-p)
print('per-q-bin closed-form weight (full-sample, info) and LOMO test:')
wfull=np.array([ (x[(b_==j)&~imp]*r[(b_==j)&~imp]).sum()/((x[(b_==j)&~imp]**2).sum()) for j in range(len(edges)-1)])
nb=np.array([((b_==j)&~imp).sum() for j in range(len(edges)-1)])
print('  bins',edges[:-1]); print('  n   ',nb); print('  w*  ',np.round(wfull,3)); print('  w*/qmid', np.round(wfull/np.array([q[(b_==j)&~imp].mean() for j in range(len(edges)-1)]),2))
wl=np.zeros_like(q)
for k in months:
    for j in range(len(edges)-1):
        tr=(b_==j)&~imp&(mon!=k); wj=(x[tr]*r[tr]).sum()/((x[tr]**2).sum()); wl[(b_==j)&(mon==k)]=wj
s=sse(pred(1.0,np.where(imp,0,wl))); dd=s-S0; bym=np.array([dd[mon==k].sum() for k in months])/1e6
print(f'  LOMO per-bin weight: dSSE {bym.sum():.1f} M ({bym.sum()*1e6/N25:.3f} s^2), months improved {(bym<0).sum()}/12, FebJun {bym[[1,5]].sum():.1f}, by month {np.round(bym,1)}')
# sign-split weights (ms>p vs ms<p), LOMO, with linear-in-q weight w = a*q
print('sign split: closed-form a in w=a*q, by sign(ms-p), full sample:')
for sgn,lab in [(1,'ms>p'),(-1,'ms<p')]:
    m_=(np.sign(x)==sgn)&~imp; a=(q[m_]*x[m_]*r[m_]).sum()/((q[m_]*x[m_])**2).sum(); print(f'  {lab}: n {m_.sum()} a {a:.3f}')
wl=np.zeros_like(q)
for k in months:
    for sgn in [1,-1]:
        tr=(np.sign(x)==sgn)&~imp&(mon!=k); a=(q[tr]*x[tr]*r[tr]).sum()/((q[tr]*x[tr])**2).sum(); te=(np.sign(x)==sgn)&(mon==k); wl[te]=a*q[te]
s=sse(pred(1.0,np.where(imp,0,wl))); dd=s-S0; bym=np.array([dd[mon==k].sum() for k in months])/1e6
print(f'  LOMO sign-split a*q: dSSE {bym.sum():.1f} M ({bym.sum()*1e6/N25:.3f} s^2), months improved {(bym<0).sum()}/12, FebJun {bym[[1,5]].sum():.1f}, by month {np.round(bym,1)}')
# (3) revert rule and clip interaction
ov=p+0.4*qg*(ms-p); neg=ov<0
print(f'revert rule: rows with ov<0: {neg.sum()}; SSE with revert {S0[neg].sum()/1e6:.3f} M vs without revert(clip0) {sse(pred(0.4,qg,revert=False))[neg].sum()/1e6:.3f} M; rows where clip binds after overlay: {((ov<lo)|(ov>hi)).sum()}, where p itself outside window: {((p<lo)|(p>hi)).sum()}')
s_noclip=sse(pred(0.4,qg,clip=False)); print(f'  clip effect on overlay rows: dSSE(clip - noclip) {(S0.sum()-s_noclip.sum())/1e6:.2f} M')
np.save(f'S0_{base}.npy',S0)
