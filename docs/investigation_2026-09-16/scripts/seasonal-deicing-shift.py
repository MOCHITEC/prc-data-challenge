"""Item 16 `seasonal-deicing-shift` (plan.md §1) — verification script, no training.

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/seasonal-deicing-shift.py

Sections
  A  METAR de-icing-gate hour share per airport × month, Jan 2025 – Jul 2026 (data/ref/metar_hourly.parquet);
     coverage (hours per station-month) and re-derivation of the gate from its components.
  B  OOF main-48c19bc830 joined to feat_train on MVT_ID_mvt (non-null oof only): RMSE / n / mean residual by
     deicing_gate × ADEP and by month × ADEP; Jan vs rest per airport; matched / unmatched split.
  C  Label mean / sd by month × airport for 2025 (all rows and matched-only) — the "Jan/Jul elevated" claim.
  D  Ranking rows with deicing_gate == 1 by airport × month (feat_rank), with the training Jan/Jul 2025 analogue;
     row-level gate vs METAR hour re-join check on both sets.
  E  Weather-feature null parity train vs rank.
  F  Training label by deicing_gate (does the gate move y at all?), by airport, matched/unmatched.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl  # noqa: E402

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_fmt_str_lengths(40)

ROOT = "/home/user/prc-data-challenge"
METAR = f"{ROOT}/data/ref/metar_hourly.parquet"
TRAIN = f"{ROOT}/data/work/feat_train.parquet"
RANK = f"{ROOT}/data/work/feat_rank.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"

WX_COLS = [
    "temp_c", "dewp_spread_c", "wind_ms", "gust_ms", "precip_mm", "vis_km", "frozen_wx", "deicing_gate",
    "low_vis", "temp_x_precip",
]
WX_ALL = WX_COLS + [f"{c}_prev1h" for c in WX_COLS]

ym = lambda c: pl.col(c).dt.strftime("%Y-%m").alias("ym")  # noqa: E731


def sec(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# --------------------------------------------------------------------------------------------------------------
sec("A. METAR hourly table: coverage and de-icing-gate share per station × month (Jan 2025 – Jul 2026)")
wx = pl.scan_parquet(METAR)
print("rows:", wx.select(pl.len()).collect().item())
print("time span:", wx.select(pl.col("hour_utc").min().alias("min"), pl.col("hour_utc").max().alias("max")).collect())
print("station × hour_utc unique:", wx.select(pl.struct("station", "hour_utc").n_unique()).collect().item())

# Re-derive the gate from its components exactly as hourly_weather_table does.
rederive = (
    wx.with_columns(
        ((pl.col("temp_c") <= 3.0) & ((pl.col("precip_mm") > 0) | pl.col("frozen_wx"))).cast(pl.Int8).alias("gate_re")
    )
    .select(
        (pl.col("gate_re") != pl.col("deicing_gate")).sum().alias("gate_mismatch"),
        pl.col("deicing_gate").is_null().sum().alias("gate_null"),
        pl.col("temp_c").is_null().sum().alias("temp_null"),
        pl.col("precip_mm").is_null().sum().alias("precip_null"),
        pl.col("frozen_wx").is_null().sum().alias("frozen_null"),
    )
    .collect()
)
print("gate re-derivation vs stored column:", rederive)

monthly = (
    wx.group_by("station", ym("hour_utc"))
    .agg(
        pl.len().alias("hours"),
        pl.col("deicing_gate").mean().round(4).alias("deice"),
        pl.col("frozen_wx").mean().round(4).alias("frozen"),
        (pl.col("precip_mm") > 0).mean().round(4).alias("precip_any"),
        (pl.col("temp_c") <= 3.0).mean().round(4).alias("temp_le3"),
        pl.col("low_vis").mean().round(4).alias("lowvis"),
        pl.col("temp_c").mean().round(1).alias("temp_mean"),
    )
    .sort("station", "ym")
    .collect()
)
# expected hours per month
exp_hours = (
    monthly.select("ym").unique()
    .with_columns(pl.col("ym").str.to_date("%Y-%m").alias("d"))
    .with_columns((pl.col("d").dt.offset_by("1mo") - pl.col("d")).dt.total_hours().alias("exp_hours"))
    .select("ym", "exp_hours")
)
monthly = monthly.join(exp_hours, on="ym").with_columns((pl.col("hours") / pl.col("exp_hours")).round(4).alias("coverage"))
print("\nA1. coverage: station-months with coverage < 0.99 (empty = complete)")
print(monthly.filter(pl.col("coverage") < 0.99).select("station", "ym", "hours", "exp_hours", "coverage"))
print("\nA1b. coverage summary per station")
print(monthly.group_by("station").agg(pl.col("coverage").min().alias("min_cov"), pl.col("hours").sum().alias("hours"), pl.len().alias("months")).sort("station"))

print("\nA2. de-icing-gate hour share, station × month (pivot)")
print(monthly.pivot(on="ym", index="station", values="deice").sort("station"))

print("\nA3. Jan 2025 vs Jan 2026 and Jul 2025 vs Jul 2026, all components")
cmp = monthly.filter(pl.col("ym").is_in(["2025-01", "2026-01", "2025-07", "2026-07"])).select(
    "station", "ym", "hours", "deice", "frozen", "precip_any", "temp_le3", "lowvis", "temp_mean"
)
print(cmp.sort("station", "ym"))
print("\nA3b. Jan ratio 2026/2025 of gate share, and difference in gate hours")
jan = monthly.filter(pl.col("ym").is_in(["2025-01", "2026-01"])).select("station", "ym", "deice", "hours")
jan = jan.with_columns((pl.col("deice") * pl.col("hours")).round(0).alias("gate_hours"))
janp = jan.pivot(on="ym", index="station", values=["deice", "gate_hours"]).sort("station")
print(janp.with_columns((pl.col("deice_2026-01") / pl.col("deice_2025-01")).round(2).alias("ratio_26_25")))

print("\nA4. Gate share Jan 2026 vs every 2025 month: is Jan 2026 outside the 2025 range per station?")
rng = (
    monthly.filter(pl.col("ym").str.starts_with("2025"))
    .group_by("station")
    .agg(pl.col("deice").max().alias("max_2025"), pl.col("ym").sort_by("deice").last().alias("argmax_2025"))
)
j26 = monthly.filter(pl.col("ym") == "2026-01").select("station", pl.col("deice").alias("jan_2026"))
print(rng.join(j26, on="station").with_columns((pl.col("jan_2026") > pl.col("max_2025")).alias("jan26_gt_all_2025")).sort("station"))

# --------------------------------------------------------------------------------------------------------------
sec("B. OOF main-48c19bc830 × feat_train: residuals by deicing_gate × ADEP and by month × ADEP")
tr_cols = [
    "MVT_ID_mvt", "TAXITIME_SEC_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "deicing_gate",
    "deicing_gate_prev1h", "low_vis", "frozen_wx", "temp_c", "precip_mm",
]
tr = pl.scan_parquet(TRAIN).select(tr_cols)
oof = pl.scan_parquet(OOF).filter(pl.col("oof").is_not_null())
print("oof non-null rows:", oof.select(pl.len()).collect().item(), "| MVT_ID unique:", oof.select(pl.col("MVT_ID_mvt").n_unique()).collect().item())
j = (
    oof.join(tr, on="MVT_ID_mvt", how="inner")
    .with_columns(
        (pl.col("oof") - pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)).alias("res"),
        pl.col("mvt_minus_aobt3").is_null().alias("unmatched"),
        ym("MVT_TIME_UTC_mvt"),
    )
    .with_columns((pl.col("res") ** 2).alias("se"))
    .collect()
)
print("joined rows:", j.height, "| deicing_gate null in joined rows:", j["deicing_gate"].null_count())
tot_sse = j["se"].sum()
print(f"overall RMSE {j['se'].mean() ** 0.5:.3f}  n {j.height}  mean res {j['res'].mean():.3f}  (plan: 334.58 / 672,428)")


def agg(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (
        df.group_by(keys)
        .agg(
            pl.len().alias("n"),
            pl.col("se").mean().sqrt().round(1).alias("rmse"),
            pl.col("res").mean().round(1).alias("mean_res"),
            pl.col("res").median().round(1).alias("med_res"),
            pl.col("res").abs().median().round(1).alias("med_abs_res"),
            (pl.col("se").sum() / tot_sse * 100).round(2).alias("sse_pct"),
            pl.col("TAXITIME_SEC_mvt").mean().round(0).alias("y_mean"),
            pl.col("oof").mean().round(0).alias("pred_mean"),
        )
        .sort(keys)
    )


print("\nB1. by deicing_gate (all OOF rows)")
print(agg(j, ["deicing_gate"]))
print("\nB1b. by deicing_gate × unmatched")
print(agg(j, ["unmatched", "deicing_gate"]))
print("\nB1c. by deicing_gate_prev1h")
print(agg(j, ["deicing_gate_prev1h"]))
print("\nB2. by deicing_gate × ADEP")
print(agg(j, ["ADEP_mvt", "deicing_gate"]))
print("\nB2b. by deicing_gate × ADEP, matched rows only")
print(agg(j.filter(~pl.col("unmatched")), ["ADEP_mvt", "deicing_gate"]))
print("\nB2c. by deicing_gate × ADEP, unmatched rows only")
print(agg(j.filter(pl.col("unmatched")), ["ADEP_mvt", "deicing_gate"]))
print("\nB2d. gate rows: y ≤ 7,200 only (tail removed), by ADEP × gate, matched")
print(agg(j.filter(~pl.col("unmatched") & (pl.col("TAXITIME_SEC_mvt") <= 7200)), ["ADEP_mvt", "deicing_gate"]))

print("\nB3. by month (OOF months only)")
print(agg(j, ["ym"]))
print("\nB3b. by month × unmatched")
print(agg(j, ["ym", "unmatched"]))
print("\nB4. by month × ADEP")
print(agg(j, ["ADEP_mvt", "ym"]))
print("\nB4b. by month × ADEP, matched only")
print(agg(j.filter(~pl.col("unmatched")), ["ADEP_mvt", "ym"]))
print("\nB4c. by month × ADEP, matched only, y ≤ 7,200")
print(agg(j.filter(~pl.col("unmatched") & (pl.col("TAXITIME_SEC_mvt") <= 7200)), ["ADEP_mvt", "ym"]))

print("\nB5. Jan vs rest (Jul/Nov/Dec) per ADEP, matched only")
jr = j.filter(~pl.col("unmatched")).with_columns((pl.col("ym") == "2025-01").alias("is_jan"))
print(agg(jr, ["ADEP_mvt", "is_jan"]).pivot(on="is_jan", index="ADEP_mvt", values=["n", "rmse", "mean_res", "med_abs_res"]).sort("ADEP_mvt"))

print("\nB6. Jan 2025 rows: gate share among OOF rows by ADEP, and RMSE gate vs no-gate (matched)")
print(
    jr.filter(pl.col("is_jan"))
    .group_by("ADEP_mvt")
    .agg(
        pl.len().alias("n"),
        pl.col("deicing_gate").mean().round(4).alias("gate_row_share"),
        pl.col("se").filter(pl.col("deicing_gate") == 1).mean().sqrt().round(1).alias("rmse_gate1"),
        pl.col("se").filter(pl.col("deicing_gate") == 0).mean().sqrt().round(1).alias("rmse_gate0"),
        pl.col("res").filter(pl.col("deicing_gate") == 1).mean().round(1).alias("mres_gate1"),
        pl.col("res").filter(pl.col("deicing_gate") == 0).mean().round(1).alias("mres_gate0"),
    )
    .sort("ADEP_mvt")
)

# --------------------------------------------------------------------------------------------------------------
sec("C. Label mean / sd by month × airport, 2025 training rows")
lab = pl.scan_parquet(TRAIN).select("TAXITIME_SEC_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3").with_columns(
    ym("MVT_TIME_UTC_mvt"), pl.col("mvt_minus_aobt3").is_null().alias("unmatched")
)
y = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)


def lab_agg(lf: pl.LazyFrame, keys: list[str]) -> pl.DataFrame:
    return (
        lf.group_by(keys)
        .agg(
            pl.len().alias("n"),
            y.mean().round(0).alias("mean"),
            y.std().round(0).alias("sd"),
            y.median().round(0).alias("median"),
            y.quantile(0.9).round(0).alias("p90"),
            (y >= 7200).sum().alias("n_ge_2h"),
        )
        .sort(keys)
        .collect()
    )


print("\nC1. all rows, by month (claim: Jan/Jul mean 995/1026, sd 605/745 elevated vs mid-year)")
print(lab_agg(lab, ["ym"]))
print("\nC1b. matched rows only, by month")
print(lab_agg(lab.filter(~pl.col("unmatched")), ["ym"]))
print("\nC1c. all rows with y < 7,200 only, by month")
print(lab_agg(lab.filter(y < 7200), ["ym"]))
print("\nC2. all rows, by airport (claim: LIRF sd 1,332)")
print(lab_agg(lab, ["ADEP_mvt"]))
print("\nC3. all rows, mean by airport × month (pivot)")
c3 = lab_agg(lab, ["ADEP_mvt", "ym"])
print(c3.pivot(on="ym", index="ADEP_mvt", values="mean").sort("ADEP_mvt"))
print("\nC3b. all rows, sd by airport × month (pivot)")
print(c3.pivot(on="ym", index="ADEP_mvt", values="sd").sort("ADEP_mvt"))
print("\nC3c. matched rows, mean by airport × month (pivot)")
c3m = lab_agg(lab.filter(~pl.col("unmatched")), ["ADEP_mvt", "ym"])
print(c3m.pivot(on="ym", index="ADEP_mvt", values="mean").sort("ADEP_mvt"))
print("\nC3d. matched rows, sd by airport × month (pivot)")
print(c3m.pivot(on="ym", index="ADEP_mvt", values="sd").sort("ADEP_mvt"))
print("\nC4. rank of Jan / Jul among the 12 months for mean and sd (1 = highest), all rows")
c1 = lab_agg(lab, ["ym"]).with_columns(
    pl.col("mean").rank(descending=True).alias("rank_mean"), pl.col("sd").rank(descending=True).alias("rank_sd")
)
print(c1.select("ym", "mean", "rank_mean", "sd", "rank_sd"))

# --------------------------------------------------------------------------------------------------------------
sec("D. Ranking rows with deicing_gate == 1 by airport × month; training Jan/Jul 2025 analogue; join check")
rk = pl.scan_parquet(RANK).select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "deicing_gate", "deicing_gate_prev1h", "low_vis", "temp_c", "precip_mm", "frozen_wx")


def gate_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.with_columns(ym("MVT_TIME_UTC_mvt"), pl.col("mvt_minus_aobt3").is_null().alias("unmatched"))
        .group_by("ADEP_mvt", "ym")
        .agg(
            pl.len().alias("rows"),
            (pl.col("deicing_gate") == 1).sum().alias("gate1"),
            pl.col("deicing_gate").mean().round(4).alias("gate_row_share"),
            ((pl.col("deicing_gate") == 1) & pl.col("unmatched")).sum().alias("gate1_unmatched"),
            (pl.col("low_vis") == 1).sum().alias("lowvis1"),
            pl.col("deicing_gate").is_null().sum().alias("gate_null"),
        )
        .sort("ADEP_mvt", "ym")
        .collect()
    )


rk_g = gate_counts(rk)
print("\nD1. ranking: rows with deicing_gate == 1 by ADEP × month")
print(rk_g)
print("\nD1b. ranking totals by month")
print(rk_g.group_by("ym").agg(pl.col("rows").sum(), pl.col("gate1").sum(), pl.col("gate1_unmatched").sum(), pl.col("lowvis1").sum()).sort("ym"))
tr_g = gate_counts(pl.scan_parquet(TRAIN).select(rk.collect_schema().names()).filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])))
print("\nD2. training Jan 2025 / Jul 2025: same table")
print(tr_g)
print("\nD2b. training totals by month")
print(tr_g.group_by("ym").agg(pl.col("rows").sum(), pl.col("gate1").sum(), pl.col("gate1_unmatched").sum(), pl.col("lowvis1").sum()).sort("ym"))
print("\nD3. Jan 2026 vs Jan 2025 gate row share side by side")
side = pl.concat([tr_g, rk_g]).filter(pl.col("ym").is_in(["2025-01", "2026-01"])).pivot(on="ym", index="ADEP_mvt", values=["rows", "gate1", "gate_row_share"]).sort("ADEP_mvt")
print(side.with_columns((pl.col("gate_row_share_2026-01") / pl.col("gate_row_share_2025-01")).round(2).alias("ratio_26_25")))

print("\nD4. row-level deicing_gate vs METAR re-join on truncated MVT hour (checks the cached join is the fixed one)")
wx_join = wx.select(pl.col("station").alias("ADEP_mvt"), pl.col("hour_utc").dt.replace_time_zone("UTC").alias("_h"), pl.col("deicing_gate").alias("gate_wx"), pl.col("temp_c").alias("temp_wx"))
for name, lf in (("train", pl.scan_parquet(TRAIN).select("ADEP_mvt", "MVT_TIME_UTC_mvt", "deicing_gate", "temp_c")), ("rank", rk.select("ADEP_mvt", "MVT_TIME_UTC_mvt", "deicing_gate", "temp_c"))):
    chk = (
        lf.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("_h"))
        .join(wx_join, on=["ADEP_mvt", "_h"], how="left")
        .select(
            pl.len().alias("rows"),
            pl.col("gate_wx").is_null().sum().alias("no_metar_hour"),
            (pl.col("deicing_gate") != pl.col("gate_wx")).sum().alias("gate_mismatch"),
            ((pl.col("temp_c") - pl.col("temp_wx")).abs() > 1e-6).sum().alias("temp_mismatch"),
        )
        .collect()
    )
    print(name, chk)
print("hour_utc feature dtype in feat_train:", pl.scan_parquet(TRAIN).select("hour_utc").collect_schema()["hour_utc"], "| min/max:", pl.scan_parquet(TRAIN).select(pl.col("hour_utc").min().alias("mn"), pl.col("hour_utc").max().alias("mx")).collect().row(0))

# --------------------------------------------------------------------------------------------------------------
sec("E. Weather-feature null parity train vs rank")
def null_rates(path: str) -> pl.DataFrame:
    lf = pl.scan_parquet(path).select(WX_ALL)
    n = lf.select(pl.len()).collect().item()
    return lf.select([(pl.col(c).is_null().sum() / n * 100).alias(c) for c in WX_ALL]).collect().transpose(include_header=True, header_name="col", column_names=["null_pct"]).with_columns(pl.col("null_pct").round(3))


ntr = null_rates(TRAIN).rename({"null_pct": "train_null_pct"})
nrk = null_rates(RANK).rename({"null_pct": "rank_null_pct"})
par = ntr.join(nrk, on="col").with_columns((pl.col("rank_null_pct") - pl.col("train_null_pct")).round(3).alias("diff_pp"))
print(par)
print("max |diff| pp:", par["diff_pp"].abs().max())

# --------------------------------------------------------------------------------------------------------------
sec("F. Training label by deicing_gate (all 2025 rows): does the gate move y?")
lab2 = pl.scan_parquet(TRAIN).select("TAXITIME_SEC_mvt", "ADEP_mvt", "mvt_minus_aobt3", "deicing_gate", "MVT_TIME_UTC_mvt").with_columns(pl.col("mvt_minus_aobt3").is_null().alias("unmatched"))
print("\nF1. by gate, matched / unmatched")
print(lab_agg(lab2, ["unmatched", "deicing_gate"]))
print("\nF2. by ADEP × gate, matched only")
f2 = lab_agg(lab2.filter(~pl.col("unmatched")), ["ADEP_mvt", "deicing_gate"])
print(f2.pivot(on="deicing_gate", index="ADEP_mvt", values=["n", "mean", "median", "sd"]).sort("ADEP_mvt"))
print("\nF3. by ADEP × gate, Jan 2025 only, matched")
print(lab_agg(lab2.filter(~pl.col("unmatched") & (pl.col("MVT_TIME_UTC_mvt").dt.month() == 1)), ["ADEP_mvt", "deicing_gate"]).pivot(on="deicing_gate", index="ADEP_mvt", values=["n", "mean", "median"]).sort("ADEP_mvt"))
print("\ndone")
