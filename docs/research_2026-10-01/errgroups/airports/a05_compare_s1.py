import sys, numpy as np, polars as pl, datetime as dt
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(S+"oof_matched_janjul.parquet")
NTOT=344841  # ranking rows; overall-RMSE translation uses total SSE of v23 ~ 252.43^2*N
def boot(d, a, b, B=1000):
    # day-block bootstrap of RMSE(b)-RMSE(a)
    e_a=(d["y"]-d[a])**2; e_b=(d["y"]-d[b])**2
    g=d.with_columns(e_a.alias("ea"),e_b.alias("eb")).group_by("day").agg(pl.col("ea").sum(),pl.col("eb").sum(),pl.len())
    ea,eb,n=g["ea"].to_numpy(),g["eb"].to_numpy(),g["len"].to_numpy()
    r=np.random.default_rng(0); D=len(n); out=[]
    for _ in range(B):
        i=r.integers(0,D,D); out.append(np.sqrt(eb[i].sum()/n[i].sum())-np.sqrt(ea[i].sum()/n[i].sum()))
    pt=np.sqrt(eb.sum()/n.sum())-np.sqrt(ea.sum()/n.sum())
    return pt, np.percentile(out,2.5), np.percentile(out,97.5), (eb.sum()-ea.sum())
for A in sys.argv[1:]:
    s=pl.read_parquet(S+f"airports/spec_{A}_s1.parquet")
    d=s.join(o.select("MVT_ID_mvt","TAXITIME_SEC_mvt","pred","genuine"),on="MVT_ID_mvt").rename({"TAXITIME_SEC_mvt":"y","pred":"G"})
    d=d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
    arms=[c for c in ["S0","S1","SN","S0p","S0w"] if c in d.columns]
    d=d.with_columns(((pl.col("G")+pl.col("S0"))/2).alias("B50"))
    if "S1" in d.columns: d=d.with_columns(((pl.col("G")+pl.col("S1"))/2).alias("B50_S1"))
    if A=="LFPG":
        ts=pl.col("MVT_TIME_UTC_mvt"); d=d.with_columns(((ts.dt.month()==1)|(ts.dt.day()<14)).alias("rep"))
    elif A=="LTFM": d=d.with_columns((pl.col("mo")==7).alias("rep"))
    else: d=d.with_columns(pl.lit(True).alias("rep"))
    print(f"===== {A} n={d.height} RMSE G={np.sqrt(((d['y']-d['G'])**2).mean()):.1f}")
    pairs=[("G","S0"),("G","B50"),("S0","S1"),("S0","SN"),("SN","S1"),("G","B50_S1"),("S0","S0p"),("S0","S0w")]
    for sub,mask in [("all",pl.lit(True)),("genuine",pl.col("genuine")),("jan",pl.col("mo")==1),("jul",pl.col("mo")==7),("rep-config",pl.col("rep"))]:
        x=d.filter(mask)
        for a,b in pairs:
            if a in x.columns and b in x.columns:
                pt,lo,hi,dsse=boot(x,a,b)
                print(f"{sub:10s} {a:>4s}->{b:<7s} n={x.height:6d} dRMSE={pt:+7.2f} [{lo:+.2f},{hi:+.2f}]  dSSE_share_of_LB={dsse/(252.43**2*NTOT)*100:+.3f}%  (~{252.43*dsse/(2*252.43**2*NTOT):+.3f}s LB if transfers)")
