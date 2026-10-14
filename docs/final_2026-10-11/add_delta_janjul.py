"""Optional variant: re-anchor Jan/Jul matched departure rows to the final file's MVT_TIME (+1 x delta).

    python add_delta_janjul.py <in.parquet> <out.parquet>

delta = MVT_TIME_UTC_mvt(final_ranking) - MVT_TIME_UTC_mvt(old ranking), whole seconds, joined on MVT_ID_mvt; applied
once to matched (AOBT_3 non-null) DEP rows of Jan/Jul only; unmatched and Feb/Jun rows untouched; then the LOBT clip
on the final clock and the zero floor. Prints the delta distribution and the change.
"""
import sys

import polars as pl

src, dst = sys.argv[1:3]
sub = pl.read_parquet(src)
cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
new = pl.read_parquet("/home/user/prc-work/data/final/final_ranking.parquet", columns=cols).filter(pl.col("PHASE_mvt") == "DEP")
old = pl.read_parquet("/home/user/prc-work/data/raw/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
j = new.join(old.select("MVT_ID_mvt", pl.col("MVT_TIME_UTC_mvt").alias("t_old")), on="MVT_ID_mvt", how="inner")
j = j.with_columns(((pl.col("MVT_TIME_UTC_mvt") - pl.col("t_old")).dt.total_seconds()).alias("delta"),
                   pl.col("AOBT_3_flt").is_not_null().alias("mat"))
print("Jan/Jul dep rows", j.height, "matched", int(j["mat"].sum()), "delta: mean", round(float(j["delta"].mean()), 3),
      "sd", round(float(j["delta"].std()), 2), "min/max", int(j["delta"].min()), int(j["delta"].max()),
      "share nonzero", round(float((j["delta"] != 0).mean()), 3))
sh = j.filter(pl.col("mat")).select("MVT_ID_mvt", "delta")
lob = pl.read_parquet("/home/user/pb-work/data/work_final/feat_rank_unfixed.parquet", columns=["MVT_ID_mvt", "mvt_minus_lobt"])
d = sub.join(sh, on="MVT_ID_mvt", how="left", maintain_order="left").join(lob, on="MVT_ID_mvt", how="left", maintain_order="left")
p = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) + pl.col("delta").fill_null(0).cast(pl.Float64)
lo, hi = pl.col("mvt_minus_lobt") - 3606, pl.col("mvt_minus_lobt") + 3606
p2 = pl.when(pl.col("delta").is_not_null() & pl.col("mvt_minus_lobt").is_not_null()).then(pl.max_horizontal(pl.min_horizontal(p, hi.ceil()), lo.ceil())).otherwise(p)
out = d.select("MVT_ID_mvt", p2.clip(0).round(0).cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
assert out.height == sub.height and out.null_count().sum_horizontal().item() == 0
dd = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
print("rows changed", int((dd != 0).sum()), "sum d^2/N", round(float((dd ** 2).sum()) / sub.height, 2))
out.write_parquet(dst)
print("written", dst)
