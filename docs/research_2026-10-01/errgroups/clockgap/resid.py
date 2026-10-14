import polars as pl, numpy as np, lightgbm as lgb, sys
o=pl.read_parquet("../oof_matched_janjul.parquet")
f=pl.scan_parquet("/home/user/prc-data-challenge/data/work/feat_train.parquet")
sch=f.collect_schema()
num=[c for c,d in sch.items() if d in (pl.Float64,pl.Float32,pl.Int64,pl.Int32,pl.Int8,pl.Int16,pl.UInt32,pl.UInt8) and c not in ("TAXITIME_SEC_mvt","MVT_ID_mvt","FLIGHT_ID_mvt","month")]
f=f.select(["MVT_ID_mvt","MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","AOBT_3_flt"]+num).collect()
f=f.filter(pl.col("MVT_ID_mvt").is_in(o["MVT_ID_mvt"].implode()))
c=pl.read_parquet("cand_train.parquet")
c=c.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
d=o.join(f,on="MVT_ID_mvt").join(c,on="MVT_ID_mvt",how="left")
d=d.with_columns(mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
   gap=(pl.col("BLOCK_TIME_UTC_mvt")-pl.col("AOBT_3_flt")).dt.total_seconds(),ap=pl.col("ADEP_mvt").cast(pl.Categorical).to_physical())
d=d.with_columns(early=(pl.col("gap")< -600)&pl.col("genuine"))
cand=[x for x in c.columns if x!="MVT_ID_mvt"]
d.write_parquet("resid_frame.parquet")
print(d.height, len(num), cand)
