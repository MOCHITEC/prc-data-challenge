"""Where does a stand-feature variant gain? Breaks the jan+jul OOF of stand_03_model.py down by
  * whether a same-stand in-block happened < 30 min before take-off (the occupancy bound is 'active')
  * clock source of the reported BLOCK (docs/06 section 2, +-60 s, simplified priority)
  * NM matched or not
Usage: python stand_10_oof_breakdown.py <variant> [tag]
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from stand_common import SCRATCH, load_all

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)


def main() -> None:
    var = sys.argv[1] if len(sys.argv) > 1 else "occ"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    a = pl.read_parquet(f"{SCRATCH}/oof_base{tag}.parquet")
    b = pl.read_parquet(f"{SCRATCH}/oof_{var}{tag}.parquet").select(["MVT_ID_mvt", pl.col("pred").alias("pb")])
    j = a.join(b, on="MVT_ID_mvt")
    raw = load_all(extra=("EOBT_1_flt", "LOBT_flt")).filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        pl.col("EOBT_1_flt").dt.epoch("s").cast(pl.Float64).alias("t_eobt1"),
        pl.col("LOBT_flt").dt.epoch("s").cast(pl.Float64).alias("t_lobt"))
    d = lambda c: (pl.col("t_blk") - pl.col(c)).abs() <= 60  # noqa: E731
    raw = raw.with_columns(
        pl.when(d("t_aobt")).then(pl.lit("aobt3")).when(d("t_sched")).then(pl.lit("sched"))
        .when(d("t_eobt1")).then(pl.lit("eobt1")).when(d("t_lobt")).then(pl.lit("lobt")).otherwise(pl.lit("genuine")).alias("src"))
    occ = pl.read_parquet(f"{SCRATCH}/stand_feats.parquet").select(["MVT_ID_mvt", "g_arr_prev1"])
    j = j.join(raw.select(["MVT_ID_mvt", "src"]), on="MVT_ID_mvt", how="left").join(occ, on="MVT_ID_mvt", how="left")
    j = j.with_columns(
        (pl.col("g_arr_prev1") < 1800).fill_null(False).alias("bound_active"),
        pl.col("mvt_minus_aobt3").is_not_null().alias("nm_matched"),
        ((pl.col("TAXITIME_SEC_mvt") - pl.col("pred")) ** 2).alias("se_a"),
        ((pl.col("TAXITIME_SEC_mvt") - pl.col("pb")) ** 2).alias("se_b"),
    )
    tot_a = j["se_a"].sum()
    for keys in (["bound_active"], ["src"], ["nm_matched"], ["nm_matched", "bound_active"], ["APT_s", "bound_active"]):
        t = j.group_by(keys).agg(
            pl.len().alias("n"), pl.col("se_a").mean().sqrt().alias("rmse_base"), pl.col("se_b").mean().sqrt().alias("rmse_new"),
            ((pl.col("se_b").sum() - pl.col("se_a").sum()) / tot_a).alias("dSSE_share_of_total"),
        ).with_columns((pl.col("rmse_new") - pl.col("rmse_base")).alias("diff")).sort(keys)
        print(f"## {var}{tag} vs base by {keys}\n", t)
    yy = j["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    print("overall", float(np.sqrt(j["se_a"].mean())), "->", float(np.sqrt(j["se_b"].mean())))


if __name__ == "__main__":
    main()
