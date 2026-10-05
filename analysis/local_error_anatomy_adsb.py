"""Where the error of a learned-ADS-B submission recipe still sits (OOF on Jan+Jul 2025, 344,419 rows).

Read-only. Inputs:
  * a cv_rows parquet written by scripts/adsb_learn_cv.py (MVT_ID_mvt, day, ADEP_mvt, y, b, has, c_learned, c_rule)
  * data/work_flseq/feat_train.parquet (context columns only; no label beyond y already in cv_rows)
  * $SCRATCH/learn/feat_2025.parquet (ADS-B stage-1 status per departure)
Prediction p = b + c_learned (the recipe of v21/v22/v23 depending on the base). e = p - y.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_error_anatomy_adsb.py <cv_rows.parquet> [label]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

CV = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(CV)
SCR = os.environ["SCRATCH"]


def h(s: str) -> None:
    print(f"\n## {s}\n", flush=True)


def summ(df: pl.DataFrame, by: list[str], total_sse: float) -> pl.DataFrame:
    return (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            (pl.col("eb") ** 2).mean().sqrt().round(1).alias("rmse_base"),
            (pl.col("e") ** 2).mean().sqrt().round(1).alias("rmse_now"),
            ((pl.col("e") ** 2).sum() / total_sse * 100).round(2).alias("pct_sse"),
            pl.col("e").mean().round(1).alias("bias_now"),
            pl.col("has").mean().round(3).alias("adsb_share"),
        )
        .with_columns((pl.col("n") / df.height * 100).round(2).alias("pct_rows"))
        .sort("pct_sse", descending=True)
    )


cv = pl.read_parquet(CV)
ctx = pl.read_parquet(
    "data/work_flseq/feat_train.parquet",
    columns=["MVT_ID_mvt", "mvt_minus_aobt3", "mvt_minus_sched", "month", "hour_local", "STAND_mvt",
             "RUNWAY_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt"],
)
st = pl.read_parquet(os.path.join(SCR, "learn", "feat_2025.parquet"),
                     columns=["MVT_ID_mvt", "join", "ground_fix", "t_move", "t_push"])
D = (
    cv.join(ctx, on="MVT_ID_mvt", how="left")
    .join(st, on="MVT_ID_mvt", how="left")
    .with_columns(
        (pl.col("b") + pl.col("c_learned")).alias("p"),
        pl.when(pl.col("mvt_minus_aobt3").is_not_null()).then(pl.lit("A matched")).otherwise(pl.lit("B unmatched"))
        .alias("lane"),
        pl.when(pl.col("month") == 1).then(pl.lit("Jan")).otherwise(pl.lit("Jul")).alias("mon"),
        pl.when(~pl.col("has")).then(pl.lit("0 no ADS-B"))
        .when(~pl.col("ground_fix").fill_null(False)).then(pl.lit("1 air only"))
        .when(pl.col("t_move").is_null()).then(pl.lit("2 ground, no stand"))
        .when(pl.col("t_push").is_null()).then(pl.lit("3 stand, no push"))
        .otherwise(pl.lit("4 stand + push")).alias("adsb_status"),
    )
    .with_columns((pl.col("p") - pl.col("y")).alias("e"), (pl.col("b") - pl.col("y")).alias("eb"))
)
assert D.height == cv.height and D["p"].null_count() == 0
SSE = float((D["e"] ** 2).sum())
SSEb = float((D["eb"] ** 2).sum())
print(f"# Error anatomy: {LABEL}\n\nrows {D.height:,} | RMSE base {np.sqrt(SSEb / D.height):.3f} -> "
      f"with learned ADS-B {np.sqrt(SSE / D.height):.3f}")

h("1. Lane")
print(summ(D, ["lane"], SSE))
h("2. Lane x month")
print(summ(D, ["lane", "mon"], SSE).sort("lane", "mon"))
h("3. Airport (all lanes)")
print(summ(D, ["ADEP_mvt"], SSE))
h("4. Airport, matched lane only")
print(summ(D.filter(pl.col("lane") == "A matched"), ["ADEP_mvt"], SSE))
h("5. ADS-B status (matched lane)")
print(summ(D.filter(pl.col("lane") == "A matched"), ["adsb_status"], SSE).sort("adsb_status"))
h("6. Airport x ADS-B status (matched lane, top 25 by SSE)")
print(summ(D.filter(pl.col("lane") == "A matched"), ["ADEP_mvt", "adsb_status"], SSE).head(25))

h("7. Concentration of squared error")
e2 = np.sort((D["e"] ** 2).to_numpy())[::-1]
cum = np.cumsum(e2) / e2.sum()
for k in (10, 20, 50, 100, 200, 500, 1000, 5000, 20000):
    print(f"top {k:>6,} rows ({k / D.height * 100:6.3f} % of rows): {cum[k - 1] * 100:5.1f} % of SSE")
n = D.height
for k in (0, 20, 200, 2000):
    rm = np.sqrt(e2[k:].sum() / (n - k))
    print(f"RMSE with the {k:,} worst rows dropped: {rm:.2f}")

h("8. Label range (y) of the error")
D = D.with_columns(
    pl.when(pl.col("y") < 0).then(pl.lit("a y<0"))
    .when(pl.col("y") < 1800).then(pl.lit("b <30m"))
    .when(pl.col("y") < 3600).then(pl.lit("c 30-60m"))
    .when(pl.col("y") < 7200).then(pl.lit("d 1-2h"))
    .when(pl.col("y") < 4 * 3600).then(pl.lit("e 2-4h"))
    .when(pl.col("y") < 11 * 3600).then(pl.lit("f 4-11h"))
    .otherwise(pl.lit("g >=11h")).alias("y_band")
)
print(summ(D, ["y_band", "lane"], SSE).sort("y_band", "lane"))

h("9. Worst airport-days (share of SSE)")
ad = (D.group_by("ADEP_mvt", "day").agg(pl.len().alias("n"), ((pl.col("e") ** 2).sum() / SSE * 100).round(2)
                                         .alias("pct_sse"), (pl.col("e") ** 2).mean().sqrt().round(1).alias("rmse"))
      .sort("pct_sse", descending=True))
print(ad.head(15))
print(f"top 10 airport-days hold {ad.head(10)['pct_sse'].sum():.1f} % of SSE; "
      f"{ad.height} airport-days in all")

h("10. Hour of day (local), matched lane")
print(summ(D.filter(pl.col("lane") == "A matched"), ["hour_local"], SSE).sort("hour_local"))

h("11. The 200 worst rows")
W = D.sort(pl.col("e") ** 2, descending=True).head(200)
print(W.group_by("lane", "ADEP_mvt").agg(pl.len().alias("n"), ((pl.col("e") ** 2).sum() / SSE * 100).round(2)
                                          .alias("pct_sse")).sort("pct_sse", descending=True))
print(W.select("ADEP_mvt", "lane", "day", "y", "b", "c_learned", "p", "e", "mvt_minus_sched", "mvt_minus_aobt3",
               "adsb_status").head(25))
