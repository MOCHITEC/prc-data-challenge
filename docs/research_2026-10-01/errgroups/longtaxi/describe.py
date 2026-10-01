import numpy as np, polars as pl
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(E+"oof_matched_janjul.parquet")
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet")
o=o.join(tx,on="MVT_ID_mvt",how="left").with_columns(pl.all().exclude(pl.String,pl.Boolean,pl.Date).fill_nan(None))
g=o.filter("genuine")
tot=(g["err"]**2).sum()
print("genuine rows",g.height,"rmse",np.sqrt((g["err"]**2).mean()))
L=g.filter(pl.col("TAXITIME_SEC_mvt")>=1800)
print("long rows share",L.height/g.height,"SSE share",(L["err"]**2).sum()/tot,"bias",L["err"].mean())
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20)
def tab(col,bins):
    x=g.with_columns(pl.col(col).cut(bins).alias("b"),(pl.col("TAXITIME_SEC_mvt")>=1800).alias("long"))
    print(col); print(x.group_by("b").agg(pl.len().alias("n"),(pl.len()/g.height*100).round(2).alias("pct"),
      ((pl.col("err")**2).sum()/tot*100).round(2).alias("sse%"),pl.col("err").mean().round(0).alias("bias"),
      (pl.col("err")**2).mean().sqrt().round(0).alias("rmse"),(pl.col("long").mean()*100).round(1).alias("long%")).sort("b"))
tab("tx_mma_c60_ratio",[0.8,1.0,1.2,1.4,1.7,2.0])
tab("tx_mma_f60_ratio",[0.8,1.0,1.2,1.4,1.7,2.0])
tab("tx_tir_c60",[-120,-30,30,120,300,600])
tab("tx_day_mma_ratio",[0.9,1.0,1.1,1.25,1.5])
tab("tx_day_tir",[-60,-20,20,60,150])
tab("tx_day_late3h",[0.01,0.02,0.05,0.1])
tab("tx_late3h_c120",[0.01,0.03,0.1,0.2])
tab("tx_tko_c60_rel",[0.5,0.8,1.0,1.2])
# daily residual: what share of long taxi SSE is day-level bias
dd=g.group_by("ADEP_mvt","lday").agg(pl.col("err").mean().alias("dm"),pl.len().alias("n"),
   (pl.col("err")**2).sum().alias("sse"),pl.col("tx_day_mma_ratio").first(),pl.col("tx_day_tir").first(),pl.col("tx_day_late3h").first(),pl.col("tx_day_unm").first(),pl.col("tx_day_n_ratio").first())
print("oracle day-debias SSE removed %", ((dd["dm"]**2*dd["n"]).sum()/tot*100))
gh=g.group_by("ADEP_mvt","lday","lhour").agg(pl.col("err").mean().alias("dm"),pl.len().alias("n"))
print("oracle day-hour debias %", ((gh["dm"]**2*gh["n"]).sum()/tot*100), "(noise floor ~ cells*var)", gh.height*float(g["err"].var())/tot*100)
print(dd.sort("sse",descending=True).head(15).with_columns((pl.col("sse")/tot*100).round(2)))
# correlation of day mean err with day observables
for c in ["tx_day_mma_ratio","tx_day_tir","tx_day_late3h","tx_day_unm","tx_day_n_ratio"]:
    print(c, round(float(np.corrcoef(dd["dm"].to_numpy(), dd[c].fill_null(0).to_numpy())[0,1]),3))
