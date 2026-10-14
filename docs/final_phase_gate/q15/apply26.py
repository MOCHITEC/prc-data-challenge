import polars as pl, numpy as np
np.set_printoptions(linewidth=200, suppress=True)
f=pl.read_parquet('lirf26.parquet'); N26=670790; N25=2085047
# validate the inverse on v2 (rec) for Feb/Jun where the chain is exact: p_pre = (rec - 0.4 q ms)/(1-0.4q) should equal padsb+incr (before clip/round)
fj=f.filter(pl.col('month').is_in([2,6])&~pl.col('imp'))
pinv=((fj['rec']-0.4*fj['qu']*fj['ms'])/(1-0.4*fj['qu'])).to_numpy(); truth=(fj['padsb']+fj['incr']).to_numpy()
e=pinv-truth; print(f'inverse check on v2 Feb/Jun non-gated rows: n {len(e)} rms {np.sqrt((e**2).mean()):.2f} s, p95|e| {np.percentile(np.abs(e),95):.2f}, max {np.abs(e).max():.1f}, rms on q>0.5 {np.sqrt((e[fj["qu"].to_numpy()>0.5]**2).mean()):.2f}')
# which q is in v4? compare inverse errors on the 1337 rows where qu!=qf, using (v4 - rec) ~ small: predicted v4 pre-overlay is unknown; use rec-consistency: rec reproduces with qu exactly (above). For v4, test which q makes (v4 - overlay_q) closest to (rec - overlay_qu) i.e. v4_pre vs v2_pre
g=f.filter(((pl.col('qu')-pl.col('qf')).abs()>1e-9)&~pl.col('imp'))
v2pre=((g['rec']-0.4*g['qu']*g['ms'])/(1-0.4*g['qu'])).to_numpy()
for qc in ['qu','qf']:
    v4pre=((g['v4']-0.4*g[qc]*g['ms'])/(1-0.4*g[qc])).to_numpy(); print(f'  rows qu!=qf: v4pre({qc}) - v2pre rms {np.sqrt(((v4pre-v2pre)**2).mean()):.2f} median|.| {np.median(np.abs(v4pre-v2pre)):.2f}')
allpre=((f['v4']-0.4*f['qu']*f['ms'])/(1-0.4*f['qu'])).to_numpy(); v2preall=((f['rec']-0.4*f['qu']*f['ms'])/(1-0.4*f['qu'])).to_numpy()
print(f'  all rows: v4pre(qu) - v2pre rms {np.sqrt(((allpre-v2preall)**2).mean()):.2f} (v4-rec rms {np.sqrt(((f["v4"]-f["rec"]).to_numpy()**2).mean()):.2f})')
# q distribution 2026 vs 2025
d25=pl.read_parquet('lirf25.parquet')
print('2025 LOMO q by month: mean, P(q>0.5), P(q>0.9):'); print(d25.filter(~pl.col('imp')).group_by('month').agg(n=pl.len(), qm=pl.col('q').mean(), q5=(pl.col('q')>0.5).mean(), q9=(pl.col('q')>0.9).mean(), copy=pl.col('copy').mean()).sort('month'))
print('2026 production q by month:'); print(f.filter(~pl.col('imp')).group_by('month').agg(n=pl.len(), qm=pl.col('qu').mean(), q5=(pl.col('qu')>0.5).mean(), q9=(pl.col('qu')>0.9).mean(), qf_m=pl.col('qf').mean()).sort('month'))
# apply candidates
q=f['qu'].to_numpy(); ms=f['ms'].to_numpy().astype(float); v4=f['v4'].to_numpy().astype(float); imp=f['imp'].to_numpy(); mon=f['month'].to_numpy(); ml=f['mlobt'].to_numpy().astype(float)
lo=np.ceil(ml-3606); hi=np.ceil(ml+3606)
ppre=(v4-0.4*q*ms)/(1-0.4*q)
print(f'p_pre stats: min {ppre.min():.0f}, rows p_pre<0: {(ppre<0).sum()}, rows p_pre outside window: {((ppre<lo)|(ppre>hi)).sum()}')
def apply(pull):
    nv=ppre+pull*(ms-ppre); nv=np.where(nv<0,ppre,nv); nv=np.clip(nv,lo,hi); nv=np.maximum(nv,0); nv=np.round(nv)
    return np.where(imp,v4,nv)
rows={}
for name,pull,ratio,g25 in [('A w=0.8',0.8*q,0.91,-335.8),('A2 w=0.65',0.65*q,2.04,-296.7),('B3 q^1.5',q**1.5,1.11,-503.6),('B q^2',q**2,1.14,-453.0),('C bins',None,0.90,-466.4)]:
    if pull is None:
        w=np.load('wfull_bins.npy'); edges=np.array([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]); pull=w[np.digitize(q,edges)-1]
    nv=apply(pull); ch=nv-v4; chg=np.abs(ch)>=1
    ss=(ch**2).sum(); bym={m:(int(chg[mon==m].sum()), round(float((ch[mon==m]**2).sum()/1e6),1)) for m in (1,2,6,7)}
    # two estimates of 4-month gain: (i) 2025 gain/sumsq ratio x sumsq26 ; (ii) per-LIRF-row 2025 gain scaled by row count
    g_i=-ratio*ss/N26/(2*234); g_ii=g25*1e6/len(d25)*len(f)/N26/(2*234)
    print(f'{name}: rows changed {chg.sum()} ({bym}); sum sq change {ss/1e6:.1f} M = {ss/N26:.1f} s^2/row26; max|ch| {np.abs(ch).max():.0f}; mean ch {ch.mean():+.2f}; expected dRMSE (i) ratio-based {g_i:+.3f} s, (ii) row-count-based {g_ii:+.3f} s')
    rows[name]=nv
    pl.DataFrame({'MVT_ID_mvt':f['MVT_ID_mvt'],'month':f['month'],'old':f['v4'],'new':nv.astype(np.int32)}).filter(pl.col('old')!=pl.col('new')).write_parquet(f'rows26_{name.split()[0]}.parquet')
