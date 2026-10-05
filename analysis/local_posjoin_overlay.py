"""Overlay: replace the prediction of unmatched 2026 departures (no NM record, no callsign ADS-B join) with the ADS-B
taxi time MVT - t_move, where a positional join (local_adsb_posjoin.py) found the aircraft, a pushback signature was
seen (push >= PUSH_MIN) and there was no later stop >= POST_DWELL_MAX (900 s) before take-off. LIRF excluded
(schedule-fill labels). 2025 check on the same rule (225 rows): RMSE 108 s against 357 s for the model.

Rows with a later stop >= 900 s (pos/dwell_2026.parquet, local_dwell_end.py): the off-block is taken as the end of
that stop when the stop began < 100 m from the first moved fix (the aircraft never really left), else the push.
2025 check against the label (263 rows): 84 % within 5 min, RMSE 765 s. These rows move only W_LONG of the way
from the base to the ADS-B value (robust to the snow-day noise being twice the 2025 level).

    .venv/bin/python analysis/local_posjoin_overlay.py <base.parquet> <out.parquet>
"""
import glob, os, sys
import polars as pl
sys.path.insert(0, "scripts")
import adsb_stage1_feasibility as s1

S = os.environ["SCRATCH"]
base, out = sys.argv[1], sys.argv[2]
B = pl.read_parquet(base)
P = pl.concat([pl.read_parquet(f) for f in glob.glob(f"{S}/pos/v2026.*_pos.parquet")], how="diagonal_relaxed")
T = (P.filter(pl.col("t_aobt3").is_null() & pl.col("t_move").is_not_null() & (pl.col("ADEP_mvt") != "LIRF")
              & (pl.col("push").fill_null(0) >= s1.PUSH_MIN) & (pl.col("post_dwell").fill_null(0) < s1.POST_DWELL_MAX))
     .select("MVT_ID_mvt", (pl.col("t_mvt") - pl.col("t_move")).alias("a")))
W_LONG = 0.75
Dw = (pl.read_parquet(f"{S}/pos/dwell_2026.parquet").filter((pl.col("src") == "pos") & pl.col("t_aobt3").is_null())
      .select("MVT_ID_mvt", (pl.col("t_mvt") - pl.when(pl.col("dw_dist_from_push_m") < 100).then(pl.col("dw_end"))
                             .otherwise(pl.col("t_move"))).alias("a_long")))
T = T.with_columns(pl.lit(1.0).alias("w"))
L = Dw.join(T.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").rename({"a_long": "a"}).with_columns(pl.lit(W_LONG).alias("w"))
T = pl.concat([T, L])
assert T["MVT_ID_mvt"].n_unique() == T.height
O = (B.join(T, on="MVT_ID_mvt", how="left")
     .with_columns(pl.coalesce((pl.col("TAXITIME_SEC_mvt") + pl.col("w") * (pl.col("a") - pl.col("TAXITIME_SEC_mvt"))).round(0),
                              pl.col("TAXITIME_SEC_mvt")).cast(pl.Int32).alias("TAXITIME_SEC_mvt"))
     .select(B.columns))
assert O.height == B.height and O["MVT_ID_mvt"].equals(B["MVT_ID_mvt"]) and O["TAXITIME_SEC_mvt"].null_count() == 0
n = int((O["TAXITIME_SEC_mvt"] != B["TAXITIME_SEC_mvt"]).sum())
O.write_parquet(out)
print(f"{out}: {T.height} target rows ({int((T['w'] < 1).sum())} long-stop at w={W_LONG}), {n} changed; sum (new-old)^2 / N = "
      f"{float(((O['TAXITIME_SEC_mvt'].cast(pl.Float64) - B['TAXITIME_SEC_mvt'].cast(pl.Float64)) ** 2).sum() / B.height):.0f}")
