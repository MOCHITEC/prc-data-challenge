import polars as pl, numpy as np, sys
d=pl.read_parquet("resid_frame.parquet",columns=["err","genuine","is_rollover","early","ADEP_mvt","day","mo","gap"])
P=np.load(sys.argv[1] if len(sys.argv)>1 else "resid_preds.npy")
names=sys.argv[2].split(",") if len(sys.argv)>2 else ["A0","A1"]
err=d["err"].to_numpy(); cl=(d["ADEP_mvt"]+"_"+d["day"].cast(pl.String)).to_numpy()
u,ci=np.unique(cl,return_inverse=True)
rng=np.random.default_rng(0)
W=rng.multinomial(len(u),np.ones(len(u))/len(u),size=300)  # cluster weights
def rep(mask,label):
    base=err[mask]; n=mask.sum(); s0=np.bincount(ci[mask],base**2,minlength=len(u)); cnt=np.bincount(ci[mask],minlength=len(u))
    line=f"{label:28s} n={n:7d} base_rmse={np.sqrt((base**2).mean()):7.2f}"
    for k,nm in enumerate(names):
        e1=err[mask]-P[k][mask]; s1=np.bincount(ci[mask],e1**2,minlength=len(u))
        diff=np.sqrt((e1**2).mean())-np.sqrt((base**2).mean())
        bs=[np.sqrt((w*s1).sum()/(w*cnt).sum())-np.sqrt((w*s0).sum()/(w*cnt).sum()) for w in W]
        line+=f" | {nm}: {diff:+6.2f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}]"
    if len(names)>=2:
        a=err[mask]-P[0][mask]; b=err[mask]-P[1][mask]
        sa=np.bincount(ci[mask],a**2,minlength=len(u)); sb=np.bincount(ci[mask],b**2,minlength=len(u))
        bs=[np.sqrt((w*sb).sum()/(w*cnt).sum())-np.sqrt((w*sa).sum()/(w*cnt).sum()) for w in W]
        line+=f" | {names[1]}-{names[0]}: {np.sqrt((b**2).mean())-np.sqrt((a**2).mean()):+6.2f} [{np.percentile(bs,2.5):+6.2f},{np.percentile(bs,97.5):+6.2f}]"
    print(line)
ro=d["is_rollover"].to_numpy(); g=d["genuine"].to_numpy(); e=d["early"].to_numpy(); ap=d["ADEP_mvt"].to_numpy()
rep(~ro,"all non-rollover matched")
rep(g,"genuine")
rep(e,"genuine early(<-10min)")
rep(g&(d["gap"].to_numpy()>600),"genuine late(>+10min)")
for a in ["LTFM","LIRF","LFPG","EGLL"]:
    rep(g&(ap==a),f"genuine {a}")
# share of early-group SSE explained
for k,nm in enumerate(names):
    print(nm,"early-group SSE reduction %.1f%%"%(100*(1-((err[e]-P[k][e])**2).sum()/(err[e]**2).sum())),
          " total matched SSE reduction %.2f%%"%(100*(1-((err[~ro]-P[k][~ro])**2).sum()/(err[~ro]**2).sum())))
