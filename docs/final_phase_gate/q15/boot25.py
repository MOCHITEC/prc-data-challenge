import polars as pl, numpy as np, sys
np.set_printoptions(linewidth=200, suppress=True)
d=pl.read_parquet('lirf25.parquet'); base='final_12'
y=d['y'].to_numpy(); ms=d['ms'].to_numpy().astype(float); ml=d['mlobt'].to_numpy().astype(float); q=d['q'].to_numpy(); mon=d['month'].to_numpy(); imp=d['imp'].to_numpy(); p=d[base].to_numpy()
day=d['day'].cast(pl.Int32).to_numpy(); ud,dinv=np.unique(day,return_inverse=True)
lo=ml-3606; hi=ml+3606; qg=np.where(imp,0.0,q); months=np.arange(1,13); N25=2085047
def pred(pull):
    ov=p+pull*(ms-p); ov=np.where(ov<0,p,ov); return np.maximum(np.clip(ov,lo,hi),0)
S0=(pred(0.4*qg)-y)**2
def ungated_check():
    S0u=(pred(0.4*q)-y)**2
    for w in [0.5,0.55,0.65,0.8]:
        s=(pred(w*q)-y)**2; print(f'  UNGATED w={w}: dSSE vs 0.4 {(s-S0u).sum()/1e6:.1f} M  (integrator table: 0.5 -85, 0.55 -108, 0.65 -117, 0.8 -35)')
print('sanity: ungated sweep (should resemble the integrator copy_lomo table):'); ungated_check()
rng=np.random.default_rng(1)
def boot(dd,B=4000):
    perday=np.bincount(dinv,weights=dd,minlength=len(ud)); bs=np.array([perday[rng.integers(0,len(ud),len(ud))].sum() for _ in range(B)]); return np.percentile(bs,[2.5,97.5])/1e6, (bs>=0).mean()
def report(name,pull):
    s=(pred(pull)-y)**2; dd=s-S0; bym=np.array([dd[mon==k].sum() for k in months])/1e6; ci,pz=boot(dd)
    fj=bym[[1,5]].sum(); f4=bym[[0,1,5,6]].sum()
    ddfj=np.where(np.isin(mon,[2,6]),dd,0.0); cifj,_=boot(ddfj,2000)
    print(f'{name}: dSSE {bym.sum():.1f} M = {bym.sum()*1e6/N25:.3f} s^2 of 2025 MSE; CI95 [{ci[0]:.1f},{ci[1]:.1f}] M P(>=0) {pz:.4f}; months improved {(bym<0).sum()}/12; FebJun {fj:.1f} M CI [{cifj[0]:.1f},{cifj[1]:.1f}]; Jan+Feb+Jun+Jul {f4:.1f} M; by month {np.round(bym,1)}')
    srt=np.sort(dd); print(f'   top-10 rows share of total gain {srt[:10].sum()/dd.sum():.3f}; top-100 {srt[:100].sum()/dd.sum():.3f}; rows with |change|>=1: {(np.abs(pred(pull)-pred(0.4*qg))>=1).sum()}; sum sq change {((pred(pull)-pred(0.4*qg))**2).sum()/1e6:.1f} M -> ratio gain/sumsq {-dd.sum()/((pred(pull)-pred(0.4*qg))**2).sum():.2f}')
    return dd
report('A: const w=0.8 (pull=0.8q)',0.8*qg)
report('A2: const w=0.65 (pull=0.65q)',0.65*qg)
report('A3: const w=0.7',0.7*qg)
# LOMO grid over pull = c*q^a
cs=np.round(np.arange(0.6,1.61,0.1),2); as_=np.round(np.arange(1.0,2.51,0.25),2)
tab=np.zeros((len(cs),len(as_),12))
for i,c in enumerate(cs):
    for j,a in enumerate(as_):
        s=(pred(np.minimum(c*qg**a,1.0))-y)**2; dd=s-S0
        for k in months: tab[i,j,k-1]=dd[mon==k].sum()/1e6
print('full-sample dSSE (M) for pull=min(1,c*q^a), rows c, cols a=',as_)
for i,c in enumerate(cs): print(f'c={c:.1f} '+' '.join(f'{tab[i,j].sum():8.1f}' for j in range(len(as_))))
tot=0; wins=0; ch=[]
for k in months:
    tr=tab[:,:,months!=k].sum(axis=2); i,j=np.unravel_index(np.argmin(tr),tr.shape); dd=tab[i,j,k-1]; tot+=dd; wins+=dd<0; ch.append((cs[i],as_[j],round(dd,1)))
print(f'LOMO (c,a) chosen per held-out month: {ch}; LOMO total {tot:.1f} M ({tot*1e6/N25:.3f} s^2); months improved {wins}/12')
report('B: pull=q^2',qg**2)
report('B2: pull=min(1,1.2*q^2)',np.minimum(1.2*qg**2,1))
report('B3: pull=q^1.5',qg**1.5)
# per-bin LOMO weights
edges=np.array([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]); b_=np.digitize(q,edges)-1; x=(ms-p); r=(y-p)
wl=np.zeros_like(q)
for k in months:
    for j in range(len(edges)-1):
        tr=(b_==j)&~imp&(mon!=k); wl[(b_==j)&(mon==k)]=(x[tr]*r[tr]).sum()/((x[tr]**2).sum())
report('C: per-q-bin weight (LOMO)',np.where(imp,0,wl))
wfull=np.array([ (x[(b_==j)&~imp]*r[(b_==j)&~imp]).sum()/((x[(b_==j)&~imp]**2).sum()) for j in range(len(edges)-1)])
print('C full-sample bin weights (for 2026):', np.round(wfull,3))
np.save('wfull_bins.npy',wfull)
