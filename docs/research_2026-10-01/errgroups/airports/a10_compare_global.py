import numpy as np, polars as pl
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
H=S+"airports/"
exec(open(H+"a05_compare.py").read().split("for A in sys.argv")[0].split("o=pl.read_parquet")[0])
o=pl.read_parquet(S+"oof_matched_janjul.parquet")
NTOT=344841
src=open(H+"a05_compare.py").read(); exec(src[src.index("def boot"):src.index("for A in sys.argv")])
g=pl.read_parquet(H+"global_G2.parquet").join(pl.read_parquet(H+"global_G2R.parquet"),on="MVT_ID_mvt")
t=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select("MVT_ID_mvt","MVT_TIME_UTC_mvt").collect()
d0=o.join(g,on="MVT_ID_mvt").join(t,on="MVT_ID_mvt").rename({"TAXITIME_SEC_mvt":"y","pred":"G"}).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
print("global RMSE G",np.sqrt(((d0["y"]-d0["G"])**2).mean()),"G2",np.sqrt(((d0["y"]-d0["G2"])**2).mean()),"G2R",np.sqrt(((d0["y"]-d0["G2R"])**2).mean()))
for A in ["ALL","ALL4","LIRF","LTFM","LFPG","EGLL"]:
    if A=="ALL": d=d0
    elif A=="ALL4": d=d0.filter(pl.col("ADEP_mvt").is_in(["LIRF","LTFM","LFPG","EGLL"]))
    else:
        s=pl.read_parquet(H+f"spec_{A}.parquet").select("MVT_ID_mvt","S0","S1")
        d=d0.join(s,on="MVT_ID_mvt").with_columns(((pl.col("G2")+pl.col("S0"))/2).alias("B50"),((pl.col("G2R")+pl.col("S1"))/2).alias("B50R"))
    print("=====",A,d.height)
    pairs=[("G","G2"),("G2","G2R")]+([("G2","S0"),("G2","B50"),("G2R","B50R"),("G2","B50R")] if "S0" in d.columns else [])
    for sub,mask in [("all",pl.lit(True)),("genuine",pl.col("genuine")),("jan",pl.col("mo")==1),("jul",pl.col("mo")==7)]:
        x=d.filter(mask)
        for a,b in pairs:
            pt,lo,hi,dsse=boot(x,a,b)
            print(f"{sub:8s} {a:>4s}->{b:<5s} n={x.height:6d} dRMSE={pt:+7.2f} [{lo:+.2f},{hi:+.2f}]  ~LB {dsse/(2*252.43*NTOT):+.3f}s")
