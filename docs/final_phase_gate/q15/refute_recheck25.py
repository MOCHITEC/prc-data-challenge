import polars as pl, numpy as np
np.set_printoptions(linewidth=220, suppress=True)
S='/tmp/claude-0/-home-user-prc-data-challenge/b411042e-f283-5ee9-9c9e-ffb2443b6ce1/scratchpad'
# independent rebuild from d25 + copyq_lomo_2025 (do not trust lirf25.parquet)
q=pl.read_parquet(f'{S}/ideas/ht/copyq_lomo_2025.parquet').select('MVT_ID_mvt','q')
d=(pl.scan_parquet(f'{S}/ideas/ht/d25.parquet').select('MVT_ID_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','y','matched','ms','mlobt','month','final_12','b')
   .filter(pl.col('matched')&(pl.col('ADEP_mvt')=='LIRF')).collect().join(q,on='MVT_ID_mvt',how='inner'))
print('rows',d.height,'dup ids',d['MVT_ID_mvt'].n_unique()!=d.height)
y=d['y'].to_numpy().astype(float); ms=d['ms'].to_numpy().astype(float); ml=d['mlobt'].to_numpy().astype(float); qq=d['q'].to_numpy(); mon=d['month'].to_numpy(); p=d['final_12'].to_numpy().astype(float)
imp=np.abs(ms-ml)>3606; qg=np.where(imp,0.0,qq)
day=d['MVT_TIME_UTC_mvt'].dt.date().cast(pl.Int32).to_numpy(); ud,dinv=np.unique(day,return_inverse=True)
lo=np.ceil(ml-3606); hi=np.ceil(ml+3606); N25=2085047; months=np.arange(1,13)
def pred(pull, clip=True, rnd=False):
    ov=p+pull*(ms-p); ov=np.where(ov<0,p,ov)
    if clip: ov=np.maximum(np.clip(ov,lo,hi),0)
    if rnd: ov=np.round(ov)
    return ov
base=pred(0.4*qg); S0=(base-y)**2
new=pred(qg**1.5); S1=(new-y)**2; dd=S1-S0
bym=np.array([dd[mon==k].sum() for k in months])/1e6
print('q^1.5 vs 0.4q (gated, clipped): dSSE %.1f M = %.1f s^2/N25; by month %s; improved %d/12; FebJun %.1f; JFJJ %.1f'%(dd.sum()/1e6, dd.sum()/N25, np.round(bym,1), (bym<0).sum(), bym[[1,5]].sum(), bym[[0,1,5,6]].sum()))
# with rounding (production file is integer)
ddr=(np.round(new)-y)**2-(np.round(base)-y)**2; print('  rounded version dSSE %.1f M'%(ddr.sum()/1e6))
# drop best month
print('  drop best month (Oct): %.1f M; drop best two: %.1f M'%(dd.sum()/1e6-bym.min(), dd.sum()/1e6-np.sort(bym)[:2].sum()))
srt=np.sort(dd); print('  top-10 share %.3f top-100 %.3f top-1000 %.3f; worst-10 rows (loss) %s'%(srt[:10].sum()/dd.sum(), srt[:100].sum()/dd.sum(), srt[:1000].sum()/dd.sum(), np.round(srt[-10:]/1e6,2)))
print('  rows gaining: %d, losing: %d, sum gain %.1f M, sum loss %.1f M'%((dd<0).sum(),(dd>0).sum(),dd[dd<0].sum()/1e6,dd[dd>0].sum()/1e6))
# trimmed: remove top 1% of |dd| rows to see robustness
k=int(0.01*len(dd)); o=np.argsort(np.abs(dd)); print('  dSSE excluding top-1%% |dd| rows: %.1f M; excluding top 0.1%%: %.1f M'%(dd[o[:-k]].sum()/1e6, dd[o[:-k//10]].sum()/1e6))
# bootstrap by day
rng=np.random.default_rng(7)
def boot(dd,B=3000):
    perday=np.bincount(dinv,weights=dd,minlength=len(ud)); bs=np.array([perday[rng.integers(0,len(ud),len(ud))].sum() for _ in range(B)]); return np.percentile(bs,[2.5,97.5])/1e6,(bs>=0).mean()
ci,pz=boot(dd); print('  day-boot CI95 [%.1f, %.1f] M, P(>=0)=%.4f'%(ci[0],ci[1],pz))
cifj,pfj=boot(np.where(np.isin(mon,[2,6]),dd,0.0)); print('  FebJun CI95 [%.1f, %.1f] M, P(>=0)=%.4f'%(cifj[0],cifj[1],pfj))
cij,pj=boot(np.where(mon==1,dd,0.0)); print('  Jan CI95 [%.1f, %.1f] M, P(>=0)=%.4f'%(cij[0],cij[1],pj))
ci4,p4=boot(np.where(np.isin(mon,[1,2,6,7]),dd,0.0)); print('  JFJJ CI95 [%.1f, %.1f] M, P(>=0)=%.4f'%(ci4[0],ci4[1],p4))
# month-block bootstrap (12 months, resample with replacement) -- more honest for a 4-month target
bs=np.array([bym[rng.integers(0,12,12)].sum() for _ in range(5000)]); print('  month-block boot CI95 [%.1f, %.1f] M'%tuple(np.percentile(bs,[2.5,97.5])))
# per-month per-row gain (s^2 per LIRF row) -> projected 2026
n26={1:9720,2:9204,6:13302,7:13682}; nm={k:(mon==k).sum() for k in months}
proj=sum(n26[k]*bym[k-1]*1e6/nm[k] for k in n26); print('  per-month-rate projection to 2026: dSSE %.1f M -> dMSE %.1f s^2 -> dRMSE %.3f s'%(proj/1e6, proj/670790, proj/670790/(2*234)))
print('  per-row gain by month (s^2):', {k: round(bym[k-1]*1e6/nm[k]) for k in months})
# discrepancy check with integrator table: ungated, NO clip, w=0.8 and 0.65 vs 0.4
for w in [0.5,0.65,0.8]:
    dd2=(pred(w*qq,clip=False)-y)**2-(pred(0.4*qq,clip=False)-y)**2; print('  UNGATED NO-CLIP w=%.2f: dSSE %.1f M (integrator: 0.5 -85, 0.65 -117, 0.8 -35)'%(w,dd2.sum()/1e6))
    dd3=(pred(w*qq,clip=True)-y)**2-(pred(0.4*qq,clip=True)-y)**2; print('  UNGATED CLIP    w=%.2f: dSSE %.1f M'%(w,dd3.sum()/1e6))
# per-bin optimal w on Jan vs rest
edges=np.array([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]); b_=np.digitize(qq,edges)-1; x=ms-p; r=y-p
for lab,sel in [('Jan',mon==1),('Feb',mon==2),('Jun',mon==6),('Jul',mon==7),('all',np.ones_like(imp))]:
    ws=[(x[(b_==j)&~imp&sel]*r[(b_==j)&~imp&sel]).sum()/((x[(b_==j)&~imp&sel]**2).sum()) for j in range(len(edges)-1)]
    cr=[np.mean(np.abs(y-ms)[(b_==j)&~imp&sel]<=12) for j in range(len(edges)-1)]
    print('  %s per-bin w*: %s'%(lab,np.round(ws,2)), ' copy rate:',np.round(cr,2))
# half-year split of the (c,a) grid: tune on 1-6 test 7-12 and vice versa
cs=np.round(np.arange(0.6,1.61,0.1),2); as_=np.round(np.arange(1.0,2.51,0.25),2)
tab=np.zeros((len(cs),len(as_),12))
for i,c in enumerate(cs):
    for j,a in enumerate(as_):
        s=(pred(np.minimum(c*qg**a,1.0))-y)**2-S0
        for k in months: tab[i,j,k-1]=s[mon==k].sum()/1e6
for trm,tem in [([0,1,2,3,4,5],[6,7,8,9,10,11]),([6,7,8,9,10,11],[0,1,2,3,4,5])]:
    tr=tab[:,:,trm].sum(axis=2); i,j=np.unravel_index(np.argmin(tr),tr.shape); print('  half-split: tune on months %s -> (c,a)=(%.1f,%.2f), held-out dSSE %.1f M (q^1.5 fixed: %.1f M)'%([m+1 for m in trm],cs[i],as_[j],tab[i,j,tem].sum(), tab[4,2,tem].sum()))
# base b instead of final_12
pb=d['b'].to_numpy().astype(float)
def predb(pull):
    ov=pb+pull*(ms-pb); ov=np.where(ov<0,pb,ov); return np.maximum(np.clip(ov,lo,hi),0)
ddb=(predb(qg**1.5)-y)**2-(predb(0.4*qg)-y)**2; print('base b: dSSE %.1f M, months improved %d/12'%(ddb.sum()/1e6, sum(ddb[mon==k].sum()<0 for k in months)))
np.save('dd_q15.npy',dd)
