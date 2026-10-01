import polars as pl, numpy as np, sys
d=pl.read_parquet("resid_frame2.parquet",columns=["err","genuine","is_rollover","early","ADEP_mvt","day","gap"])
err=d["err"].to_numpy(); cl=(d["ADEP_mvt"]+"_"+d["day"].cast(pl.String)).to_numpy(); u,ci=np.unique(cl,return_inverse=True)
W=np.random.default_rng(0).multinomial(len(u),np.ones(len(u))/len(u),size=300)
ref=sys.argv[1]; others=sys.argv[2].split(",")
P={k:(np.zeros(len(d)) if k=="none" else np.load(f"p_{k}.npy")) for k in [ref]+others}
ro=d["is_rollover"].to_numpy(); g=d["genuine"].to_numpy(); e=d["early"].to_numpy(); ap=d["ADEP_mvt"].to_numpy(); late=g&(d["gap"].to_numpy()>600)
masks=[("all matched non-roll",~ro),("genuine",g),("genuine early",e),("genuine late",late)]+[(f"gen {a}",g&(ap==a)) for a in ["LTFM","LIRF","LFPG","EGLL","EDDM"]]
def rm(x,m): return np.sqrt((x[m]**2).mean())
for k in others:
    print(f"== {k} vs {ref}")
    for nm,m in masks:
        a=err-P[ref]; b=err-P[k]; cnt=np.bincount(ci[m],minlength=len(u))
        sa=np.bincount(ci[m],a[m]**2,minlength=len(u)); sb=np.bincount(ci[m],b[m]**2,minlength=len(u))
        bs=[np.sqrt((w*sb).sum()/(w*cnt).sum())-np.sqrt((w*sa).sum()/(w*cnt).sum()) for w in W]
        print(f"  {nm:22s} n={m.sum():7d} {rm(a,m):7.2f} -> {rm(b,m):7.2f}  {rm(b,m)-rm(a,m):+6.2f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}]")
