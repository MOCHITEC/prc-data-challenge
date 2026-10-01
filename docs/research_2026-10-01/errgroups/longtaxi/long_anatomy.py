import numpy as np, polars as pl
E="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(E+"oof_matched_janjul.parquet").filter("genuine")
tx=pl.read_parquet(E+"longtaxi/tx_feats.parquet")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select("MVT_ID_mvt","mvt_minus_aobt3","aobt3_minus_eobt1","nbr_mvt_minus_aobt_mean_60m","WK_TBL_CAT_flt","RUNWAY_mvt").collect()
g=o.join(tx,on="MVT_ID_mvt").join(f,on="MVT_ID_mvt").with_columns(pl.col(pl.Float32).fill_nan(None))
L=g.filter(pl.col("TAXITIME_SEC_mvt")>=1800); tot=(L["err"]**2).sum()
print("long rows",L.height,"bias",L["err"].mean(),"median pred",L["pred"].median(),"median y",L["TAXITIME_SEC_mvt"].median())
# where does mvt-aobt3 sit for long taxis? offset = (MVT-AOBT3) - y = AOBT3 - BLOCK... y - mma = BLOCK-? 
L=L.with_columns((pl.col("TAXITIME_SEC_mvt")-pl.col("mvt_minus_aobt3")).alias("y_minus_mma"))
def t(col,bins,df=L):
    print(col); print(df.with_columns(pl.col(col).cut(bins).alias("b")).group_by("b").agg(pl.len().alias("n"),((pl.col("err")**2).sum()/tot*100).round(1).alias("sse%_of_long"),
      pl.col("err").mean().round(0).alias("bias"),pl.col("y_minus_mma").median().round(0).alias("med_y-mma")).sort("b"))
t("y_minus_mma",[-600,-60,60,300,600,1200,2400])
t("tx_mma_c60_ratio",[1.0,1.2,1.4,1.7])
t("tx_day_mma_ratio",[1.1,1.25,1.5])
t("tx_tir_c60",[30,120,300])
# what share of long-taxi SSE is in rows where mvt-aobt3 already >= 1800 (i.e., long visible) vs short visible
L2=L.with_columns((pl.col("mvt_minus_aobt3")>=1800).alias("vis_long"))
print(L2.group_by("vis_long").agg(pl.len(),((pl.col("err")**2).sum()/tot*100).round(1),pl.col("err").mean().round(0)))
# per-airport-day clustering: share of long rows' SSE in top 5% airport-hours by local congestion
