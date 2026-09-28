"""P1 §5 Stage 1 [S-new]: how many non-LIRF unmatched ranking predictions the shipped
model B still puts above each cap threshold. Read-only; writes nothing."""

import polars as pl
sub = pl.read_parquet("outputs/v10_probe_reparam.parquet")
print("sub cols", sub.columns, sub.height)
rk = pl.read_parquet("data/work/feat_rank.parquet", columns=["MVT_ID_mvt","ADEP_mvt","mvt_minus_aobt3","mvt_minus_sched"])
print("rank rows", rk.height)
idc, tgt = "MVT_ID_mvt", "TAXITIME_SEC_mvt_mvt" if "TAXITIME_SEC_mvt_mvt" in sub.columns else sub.columns[1]
s = sub.rename({sub.columns[0]: idc, tgt: "pred"}) if sub.columns[0]!=idc else sub.rename({tgt:"pred"})
s = s.with_columns(pl.col(idc).cast(pl.Int64))
rk = rk.with_columns(pl.col(idc).cast(pl.Int64))
j = rk.join(s, on=idc, how="inner")
print("joined", j.height)
u = j.filter(pl.col("mvt_minus_aobt3").is_null())
print("unmatched rank rows", u.height)
nl = u.filter(pl.col("ADEP_mvt") != "LIRF")
li = u.filter(pl.col("ADEP_mvt") == "LIRF")
print("non-LIRF unmatched", nl.height, " LIRF unmatched", li.height)
for c in (1800, 2400, 3600, 7200, 14400, 43200):
    print(f"  cap {c}: non-LIRF preds > {c}: {nl.filter(pl.col('pred')>c).height}   LIRF preds > {c}: {li.filter(pl.col('pred')>c).height}")
print("non-LIRF unmatched pred quantiles:", [round(nl['pred'].quantile(q),1) for q in (0.5,0.9,0.99,0.999,1.0)])
print("non-LIRF max pred", nl["pred"].max())
print(nl.filter(pl.col("pred")>3600).select(["MVT_ID_mvt","ADEP_mvt","pred","mvt_minus_sched"]).sort("pred",descending=True).head(30))
# same for the older v7/main submission for contrast
for f in ("outputs/probe_unmatched_from_v4.parquet","outputs/main.parquet"):
    try:
        o = pl.read_parquet(f)
        o = o.rename({o.columns[0]: idc, o.columns[1]: "pred"}).with_columns(pl.col(idc).cast(pl.Int64))
        jj = rk.join(o, on=idc, how="inner").filter(pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt")!="LIRF"))
        print(f, "non-LIRF unmatched >3600:", jj.filter(pl.col("pred")>3600).height, " >7200:", jj.filter(pl.col("pred")>7200).height, "max", jj["pred"].max())
    except Exception as e:
        print(f, "ERR", e)
