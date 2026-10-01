import numpy as np, polars as pl
H="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
src=open(H+"airports/a05_compare.py").read(); exec(src[src.index("def boot"):src.index("for A in sys.argv")])
NTOT=344841
o=pl.read_parquet(H+"oof_matched_janjul.parquet")
s=pl.read_parquet(H+"airports/spec_LTFM_abl.parquet")
d=s.join(o.select("MVT_ID_mvt","TAXITIME_SEC_mvt"),on="MVT_ID_mvt").rename({"TAXITIME_SEC_mvt":"y"}).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
for sub,m in [("all",pl.lit(True)),("jan",pl.col("mo")==1),("jul",pl.col("mo")==7)]:
    x=d.filter(m)
    for b in ["S1day","S1win","S1arr","S1chg"]:
        pt,lo,hi,_=boot(x,"S0",b); print(f"{sub} S0->{b}: {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
