# Residual anatomy of the specialist S0 (has stand/runway cats): what is left by stand group x runway, mixed-mode, cfg change
import sys, polars as pl
S="/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/"
o=pl.read_parquet(S+"oof_matched_janjul.parquet").select("MVT_ID_mvt","TAXITIME_SEC_mvt","genuine")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet").select("MVT_ID_mvt","RUNWAY_mvt","STAND_mvt","WK_TBL_CAT_flt").collect()
rg=pl.read_parquet(S+"airports/regime_feats.parquet")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
for A in sys.argv[1:]:
    s=pl.read_parquet(S+f"airports/spec_{A}.parquet").select("MVT_ID_mvt","S0")
    d=s.join(o,on="MVT_ID_mvt").join(f,on="MVT_ID_mvt").join(rg,on="MVT_ID_mvt").filter("genuine")
    d=d.with_columns((pl.col("TAXITIME_SEC_mvt")-pl.col("S0")).alias("e"))
    tot=(d["e"]**2).sum()
    if A in ("LIRF","EGLL"): d=d.with_columns(pl.col("STAND_mvt").str.slice(0,1).alias("sg"))
    else: d=d.with_columns(pl.col("STAND_mvt").str.replace(r"\d.*$","").alias("sg"))
    print("=====",A,"genuine n",d.height,"S0 rmse",round((tot/d.height)**.5,1))
    def tab(keys):
        t=d.group_by(keys).agg(pl.len().alias("n"),pl.col("e").mean().round(0).alias("bias"),(pl.col("e")**2).mean().sqrt().round(0).alias("rmse"),
            ((pl.col("e")**2).sum()/tot*100).round(1).alias("sse%_apt"),(pl.col("e")>1800).mean().round(3).alias("p_e>30m")).filter(pl.col("n")>=150).sort("sse%_apt",descending=True)
        print(t.head(15))
    tab(["RUNWAY_mvt","sg"])
    d=d.with_columns((pl.col("rg_arr_on_own_pm30")>0).alias("mixed"),(pl.col("rg_min_since_cfgchg")<=30).alias("chg_recent"),(pl.col("rg_min_until_cfgchg")<=30).alias("chg_soon"))
    tab(["mixed"]); tab(["chg_recent"]); tab(["chg_soon"]); tab(["WK_TBL_CAT_flt"])
