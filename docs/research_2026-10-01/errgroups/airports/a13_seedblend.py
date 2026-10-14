# how much of the G2+S0 blend gain is plain variance reduction? compare S0 -> (S0+S0_seed1)/2 and G2->(G2+S0)/2 at LTFM
import numpy as np, polars as pl
H="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
src=open(H+"airports/a05_compare.py").read(); exec(src[src.index("def boot"):src.index("for A in sys.argv")])
o=pl.read_parquet(H+"oof_matched_janjul.parquet").select("MVT_ID_mvt","TAXITIME_SEC_mvt")
a=pl.read_parquet(H+"airports/spec_LTFM.parquet").select("MVT_ID_mvt","MVT_TIME_UTC_mvt","S0","S1")
b=pl.read_parquet(H+"airports/spec_LTFM_s1.parquet").select("MVT_ID_mvt",pl.col("S0").alias("S0s1"),pl.col("S1").alias("S1s1"))
g=pl.read_parquet(H+"airports/global_G2.parquet")
d=a.join(b,on="MVT_ID_mvt").join(g,on="MVT_ID_mvt").join(o,on="MVT_ID_mvt").rename({"TAXITIME_SEC_mvt":"y"}).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
  ((pl.col("S0")+pl.col("S0s1"))/2).alias("S0x2"),((pl.col("G2")+pl.col("S0"))/2).alias("B50"),((pl.col("S1")+pl.col("S1s1"))/2).alias("S1x2"))
d=d.with_columns(((pl.col("G2")+pl.col("S0x2"))/2).alias("B50x2"))
for p,q in [("S0","S0x2"),("G2","B50"),("G2","B50x2"),("S0x2","S1x2"),("G2","S0x2")]:
    pt,lo,hi,_=boot(d,p,q); print(f"LTFM {p}->{q}: {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
