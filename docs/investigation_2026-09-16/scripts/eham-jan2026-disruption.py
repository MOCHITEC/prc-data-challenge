"""Item 15 `eham-jan2026-disruption` -- independent re-derivation (no training).

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/eham-jan2026-disruption.py
Inputs: data/work/feat_train.parquet, data/work/feat_rank.parquet, data/ref/metar_hourly.parquet,
        experiments/oof_main-48c19bc830.parquet, outputs/main.parquet, outputs/probe_unmatched_from_v4.parquet.
All frames are column-selected lazy scans; the 132-column frames are never collected whole.

Definitions (plan §0): unmatched = routing definition (AOBT_3_flt null -> model B); D's definition
(FLIGHT_ID_mvt null) is counted alongside. proxy = MVT_TIME - AOBT_3 (matched rows only).
day = calendar date of MVT_TIME_UTC_mvt (UTC), as in D's q4 K. Disruption day = day whose unmatched
(routing) count exceeds 3 x the median daily unmatched count of that airport in that month (median over the
calendar days of the month on which the airport has at least one departure; days with 0 unmatched count as 0).
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import math
import polars as pl

pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(60)

B = "/home/user/prc-data-challenge/"
H2, H4, H12 = 7200, 14400, 43200
MONTHS = [1, 7]


def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def hdr(t):
    print("\n" + "=" * 110 + "\n" + t + "\n" + "=" * 110)


def q(col, p, name=None):
    return pl.col(col).quantile(p, interpolation="linear").round(0).alias(name or f"{col}_p{int(p*100)}")


cols = ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched", "deicing_gate", "temp_c", "precip_mm", "vis_km"]


def load(path, with_y):
    lf = pl.scan_parquet(path).select(cols).with_columns([
        pl.col("AOBT_3_flt").is_null().alias("unm"),
        pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
        sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("proxy"),
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
        pl.col("MVT_TIME_UTC_mvt").dt.year().alias("yr"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
        (pl.col("AOBT_3_flt").dt.second() == 0).alias("aobt_round"),
    ])
    if with_y:
        lf = lf.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"))
    return lf


tr = load(B + "data/work/feat_train.parquet", True).collect()
rk = load(B + "data/work/feat_rank.parquet", False).collect()
print(f"train rows {tr.height}, unm(routing) {tr['unm'].sum()}, unm(fid) {tr['unm_fid'].sum()}")
print(f"rank  rows {rk.height}, unm(routing) {rk['unm'].sum()}, unm(fid) {rk['unm_fid'].sum()}")

# ----------------------------------------------------------------------------------------------
# S1. per-day unmatched counts and proxy sd, all 10 airports, Jan/Jul 2025 (train) and Jan/Jul 2026 (rank)
# ----------------------------------------------------------------------------------------------
def per_day(df):
    return (df.filter(pl.col("mo").is_in(MONTHS))
            .group_by(["ADEP_mvt", "yr", "mo", "day"]).agg([
                pl.len().alias("n_dep"),
                pl.col("unm").sum().alias("n_unm"),
                pl.col("unm_fid").sum().alias("n_unm_fid"),
                pl.col("proxy").std().round(0).alias("proxy_sd"),
                pl.col("proxy").median().alias("proxy_med"),
                pl.col("proxy").quantile(0.99).round(0).alias("proxy_p99"),
                pl.col("aobt_round").mean().round(3).alias("aobt_round"),
            ]).sort(["ADEP_mvt", "yr", "mo", "day"]))


pd_tr = per_day(tr)
pd_rk = per_day(rk)
pd_all = pl.concat([pd_tr, pd_rk])

# month-level medians of the daily unmatched count and the 3x rule
med = (pd_all.group_by(["ADEP_mvt", "yr", "mo"]).agg([
    pl.len().alias("n_days"),
    pl.col("n_unm").median().alias("med_unm"),
    pl.col("n_unm").max().alias("max_unm"),
    pl.col("n_unm").sum().alias("sum_unm"),
    pl.col("n_dep").sum().alias("sum_dep"),
    pl.col("proxy_sd").median().alias("med_proxy_sd"),
    pl.col("proxy_sd").max().alias("max_proxy_sd"),
]).sort(["ADEP_mvt", "yr", "mo"]))
pd_all = pd_all.join(med.select(["ADEP_mvt", "yr", "mo", "med_unm"]), on=["ADEP_mvt", "yr", "mo"]).with_columns(
    (pl.col("n_unm") > 3 * pl.col("med_unm")).alias("disrupt"),
    (pl.col("n_unm") / pl.col("med_unm")).round(2).alias("ratio"),
)

hdr("S1a. Airport x month summary (Jan/Jul 2025 train, Jan/Jul 2026 rank): daily unmatched count median/max, proxy sd")
print(med.join(pd_all.group_by(["ADEP_mvt", "yr", "mo"]).agg(pl.col("disrupt").sum().alias("n_disrupt_days")),
               on=["ADEP_mvt", "yr", "mo"]).sort(["ADEP_mvt", "yr", "mo"]))

hdr("S1b. EHAM January per-day table: 2025 (train) and 2026 (rank)")
e = pd_all.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).with_columns(pl.col("day").dt.day().alias("d"))
print(e.pivot(on="yr", index="d", values=["n_unm", "n_unm_fid", "proxy_sd", "proxy_med", "aobt_round", "disrupt"]).sort("d"))
print("EHAM Jan totals:", e.group_by("yr").agg(pl.col("n_unm").sum(), pl.col("n_unm_fid").sum(), pl.col("n_dep").sum(),
                                               pl.col("aobt_round").mean().round(3).alias("aobt_round_daymean")).sort("yr"))
print("EHAM AOBT_3 second==0 share by month (row-weighted):")
print(pl.concat([tr.filter(pl.col("ADEP_mvt") == "EHAM").group_by(["yr", "mo"]).agg(pl.col("aobt_round").mean().round(3), pl.len()),
                 rk.filter(pl.col("ADEP_mvt") == "EHAM").group_by(["yr", "mo"]).agg(pl.col("aobt_round").mean().round(3), pl.len())])
      .filter(pl.col("mo").is_in(MONTHS)).sort(["yr", "mo"]))
print("All airports AOBT_3 second==0 share, Jan 2025 vs Jan 2026 (row-weighted):")
print(pl.concat([tr.filter(pl.col("mo") == 1).group_by("ADEP_mvt").agg(pl.col("aobt_round").mean().round(3).alias("r2025")),
                 ], how="vertical").join(
    rk.filter(pl.col("mo") == 1).group_by("ADEP_mvt").agg(pl.col("aobt_round").mean().round(3).alias("r2026")), on="ADEP_mvt").sort("ADEP_mvt"))

hdr("S1c. EHAM July per-day table: 2025 (train) and 2026 (rank)")
e7 = pd_all.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 7)).with_columns(pl.col("day").dt.day().alias("d"))
print(e7.pivot(on="yr", index="d", values=["n_unm", "proxy_sd", "disrupt"]).sort("d"))

# ----------------------------------------------------------------------------------------------
# S2. disruption days (n_unm > 3 x monthly median), all airports x 4 months
# ----------------------------------------------------------------------------------------------
hdr("S2. Disruption days (n_unm > 3 x median daily n_unm of the airport-month), all airports, 4 months")
dis = pd_all.filter(pl.col("disrupt")).sort(["yr", "mo", "ADEP_mvt", "day"])
print(dis.select(["ADEP_mvt", "yr", "mo", "day", "n_dep", "n_unm", "med_unm", "ratio", "proxy_sd", "proxy_med", "aobt_round"]))
print("disruption-day counts by set/airport:")
print(dis.group_by(["yr", "ADEP_mvt"]).agg(pl.len().alias("n_days"), pl.col("n_unm").sum().alias("unm_rows"), pl.col("n_dep").sum().alias("dep_rows")).sort(["yr", "ADEP_mvt"]))
print("airport-months with med_unm <= 2 (3x rule triggers on small counts):")
print(med.filter(pl.col("med_unm") <= 2).select(["ADEP_mvt", "yr", "mo", "med_unm", "max_unm"]))

# keys for joins
dis_keys = dis.select(["ADEP_mvt", "day"]).with_columns(pl.lit(True).alias("dis_day"))

# Extension: disruption days over ALL 12 training months (same rule per airport-month) for a larger training analogue set
pd_tr_all = (tr.group_by(["ADEP_mvt", "yr", "mo", "day"]).agg([pl.len().alias("n_dep"), pl.col("unm").sum().alias("n_unm"),
                                                               pl.col("proxy").std().round(0).alias("proxy_sd")])
             .with_columns(pl.col("n_unm").median().over(["ADEP_mvt", "mo"]).alias("med_unm"))
             .with_columns((pl.col("n_unm") > 3 * pl.col("med_unm")).alias("disrupt"), (pl.col("n_unm") / pl.col("med_unm")).round(2).alias("ratio")))
hdr("S2b. Extension: training disruption days over all 12 months of 2025 (same rule per airport-month)")
dis_tr_all = pd_tr_all.filter(pl.col("disrupt")).sort(["ADEP_mvt", "day"])
print(dis_tr_all.select(["ADEP_mvt", "day", "n_dep", "n_unm", "med_unm", "ratio", "proxy_sd"]))
print(dis_tr_all.group_by("ADEP_mvt").agg(pl.len().alias("n_days"), pl.col("n_unm").sum().alias("unm_rows"), pl.col("n_unm").max().alias("max_n_unm")).sort("ADEP_mvt"))
print("training days with n_unm >= 68 (the lower end of EHAM Jan 2026 disruption-day counts), any airport:")
print(pd_tr_all.filter(pl.col("n_unm") >= 68).sort("n_unm", descending=True).select(["ADEP_mvt", "day", "n_dep", "n_unm", "med_unm", "proxy_sd"]))
print("EHAM training: top-15 days by n_unm:")
print(pd_tr_all.filter(pl.col("ADEP_mvt") == "EHAM").sort("n_unm", descending=True).head(15).select(["day", "n_dep", "n_unm", "med_unm", "ratio", "proxy_sd"]))
dis_tr_all_keys = dis_tr_all.select(["ADEP_mvt", "day"]).with_columns(pl.lit(True).alias("dis_day_all"))

# ----------------------------------------------------------------------------------------------
# S3. training disruption days vs normal days: OOF residual RMSE (matched / unmatched) and label quantiles
# ----------------------------------------------------------------------------------------------
oof = pl.scan_parquet(B + "experiments/oof_main-48c19bc830.parquet").filter(pl.col("oof").is_not_null()).collect()
t = (tr.join(oof, on="MVT_ID_mvt", how="left")
     .join(dis_keys, on=["ADEP_mvt", "day"], how="left").with_columns(pl.col("dis_day").fill_null(False))
     .join(dis_tr_all_keys, on=["ADEP_mvt", "day"], how="left").with_columns(pl.col("dis_day_all").fill_null(False))
     .with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se")))
print(f"\nOOF non-null rows {oof.height}; months present in OOF: "
      f"{sorted(t.filter(pl.col('oof').is_not_null())['mo'].unique().to_list())}; "
      f"OOF RMSE all rows {math.sqrt(t['se'].sum() / t['oof'].is_not_null().sum()):.2f}")


def lab_stats(df, keys):
    return (df.group_by(keys).agg([
        pl.len().alias("n"),
        pl.col("oof").is_not_null().sum().alias("n_oof"),
        (pl.col("se").sum() / pl.col("oof").is_not_null().sum()).sqrt().round(1).alias("oof_rmse"),
        (pl.col("oof") - pl.col("y")).filter(pl.col("oof").is_not_null()).median().round(0).alias("oof_bias_med"),
        q("y", 0.10), q("y", 0.50), q("y", 0.90), q("y", 0.99), pl.col("y").max().alias("y_max"),
        (pl.col("y") >= H2).sum().alias("n_y>=2h"),
        ((pl.col("y") - pl.col("ms")).abs() <= 60).mean().round(3).alias("y_eq_ms60"),
        pl.col("ms").median().alias("ms_med"),
    ]).sort(keys))


hdr("S3a. Training Jan+Jul 2025, disruption days (4-month rule) vs normal days: OOF RMSE and label quantiles, matched/unmatched")
tj = t.filter(pl.col("mo").is_in(MONTHS))
print(lab_stats(tj, ["dis_day", "unm"]))
hdr("S3b. Same, by airport (only airports having >= 1 disruption day in Jan/Jul 2025)")
apts = dis.filter(pl.col("yr") == 2025)["ADEP_mvt"].unique().to_list()
print(lab_stats(tj.filter(pl.col("ADEP_mvt").is_in(apts)), ["ADEP_mvt", "dis_day", "unm"]))
hdr("S3c. EHAM training, Jan 2025: disruption days vs normal days (both unmatched definitions); D quoted Jan 5 & 18 (n=72) median 1,006 / p90 3,580")
ej = t.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1))
print(lab_stats(ej, ["dis_day", "unm"]))
print("D's day set {5, 18}:")
print(lab_stats(ej.with_columns(pl.col("day").dt.day().is_in([5, 18]).alias("d5_18")), ["d5_18", "unm_fid"]))
hdr("S3d. Extension: all 12 training months, disruption days (per airport-month rule) vs normal, matched/unmatched")
print(lab_stats(t, ["dis_day_all", "unm"]))
print("per airport, unmatched rows only:")
print(lab_stats(t.filter(pl.col("unm")), ["ADEP_mvt", "dis_day_all"]))
hdr("S3e. Extension: EHAM unmatched training rows by daily unmatched count band (all 2025)")
t2 = t.join(pd_tr_all.select(["ADEP_mvt", "day", "n_unm"]), on=["ADEP_mvt", "day"], how="left")
eb = t2.filter((pl.col("ADEP_mvt") == "EHAM") & pl.col("unm")).with_columns(
    pl.when(pl.col("n_unm") < 10).then(pl.lit("a <10")).when(pl.col("n_unm") < 20).then(pl.lit("b 10-19"))
    .when(pl.col("n_unm") < 40).then(pl.lit("c 20-39")).when(pl.col("n_unm") < 68).then(pl.lit("d 40-67")).otherwise(pl.lit("e >=68")).alias("band"))
print(lab_stats(eb, ["band"]))
print("all airports, unmatched rows, by daily unmatched count band:")
ab = t2.filter(pl.col("unm")).with_columns(
    pl.when(pl.col("n_unm") < 10).then(pl.lit("a <10")).when(pl.col("n_unm") < 20).then(pl.lit("b 10-19"))
    .when(pl.col("n_unm") < 40).then(pl.lit("c 20-39")).when(pl.col("n_unm") < 68).then(pl.lit("d 40-67")).otherwise(pl.lit("e >=68")).alias("band"))
print(lab_stats(ab, ["band"]))
print("matched rows, EHAM, by same band (label = y, proxy residual sd):")
mb = t2.filter((pl.col("ADEP_mvt") == "EHAM") & ~pl.col("unm")).with_columns(
    pl.when(pl.col("n_unm") < 10).then(pl.lit("a <10")).when(pl.col("n_unm") < 20).then(pl.lit("b 10-19"))
    .when(pl.col("n_unm") < 40).then(pl.lit("c 20-39")).when(pl.col("n_unm") < 68).then(pl.lit("d 40-67")).otherwise(pl.lit("e >=68")).alias("band"))
print(mb.group_by("band").agg([pl.len().alias("n"), (pl.col("se").sum() / pl.col("oof").is_not_null().sum()).sqrt().round(1).alias("oof_rmse"),
                               q("y", 0.5), q("y", 0.9), (pl.col("y") - pl.col("proxy")).std().round(0).alias("y_minus_proxy_sd"),
                               pl.col("proxy").std().round(0).alias("proxy_sd")]).sort("band"))

# ----------------------------------------------------------------------------------------------
# S4. ranking rows on ranking disruption days, by airport x matched/unmatched
# ----------------------------------------------------------------------------------------------
hdr("S4. Ranking rows on ranking (2026) disruption days by airport x matched/unmatched")
r = rk.join(dis_keys, on=["ADEP_mvt", "day"], how="left").with_columns(pl.col("dis_day").fill_null(False))
print(r.filter(pl.col("dis_day")).group_by(["ADEP_mvt", "mo", "unm"]).agg(pl.len().alias("n"), pl.col("day").n_unique().alias("n_days"),
                                                                              pl.col("ms").median().alias("ms_med"), (pl.col("ms") >= H2).sum().alias("n_ms>=2h")).sort(["mo", "ADEP_mvt", "unm"]))
print("totals:", r.group_by(["dis_day", "unm"]).agg(pl.len()).sort(["dis_day", "unm"]))
print("share of all ranking unmatched rows that fall on disruption days:",
      round(r.filter(pl.col("unm") & pl.col("dis_day")).height / r.filter(pl.col("unm")).height, 4))
print("EHAM Jan 2026 unmatched: on disruption days", r.filter(pl.col("unm") & pl.col("dis_day") & (pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).height,
      "of", r.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).height)

# ----------------------------------------------------------------------------------------------
# S5. METAR conditions on disruption days vs normal days (data/ref/metar_hourly.parquet)
# ----------------------------------------------------------------------------------------------
wx = (pl.scan_parquet(B + "data/ref/metar_hourly.parquet")
      .select(["station", "hour_utc", "temp_c", "precip_mm", "vis_km", "frozen_wx", "deicing_gate", "low_vis", "wind_ms"])
      .with_columns([pl.col("hour_utc").dt.date().alias("day"), pl.col("hour_utc").dt.year().alias("yr"), pl.col("hour_utc").dt.month().alias("mo")])
      .filter(pl.col("mo").is_in(MONTHS))
      .group_by(["station", "yr", "mo", "day"]).agg([
          pl.len().alias("hrs"),
          pl.col("temp_c").mean().round(1).alias("t_mean"), pl.col("temp_c").min().round(1).alias("t_min"),
          pl.col("precip_mm").sum().round(1).alias("precip_sum"), (pl.col("precip_mm") > 0).sum().alias("precip_hrs"),
          pl.col("vis_km").min().round(2).alias("vis_min"), pl.col("low_vis").sum().alias("lowvis_hrs"),
          pl.col("frozen_wx").sum().alias("frozen_hrs"), pl.col("deicing_gate").sum().alias("deice_hrs"),
          pl.col("wind_ms").max().round(1).alias("wind_max"),
      ]).rename({"station": "ADEP_mvt"}).collect())
wxd = wx.join(pd_all.select(["ADEP_mvt", "day", "n_unm", "disrupt", "proxy_sd"]), on=["ADEP_mvt", "day"], how="left")
hdr("S5a. EHAM January per-day METAR, 2025 and 2026, with unmatched count and disruption flag")
print(wxd.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).sort("day")
      .select(["day", "n_unm", "disrupt", "proxy_sd", "t_mean", "t_min", "precip_sum", "precip_hrs", "vis_min", "lowvis_hrs", "frozen_hrs", "deice_hrs", "wind_max"]))
hdr("S5b. METAR on disruption days vs normal days, per airport-month (mean of daily values), all airports with a disruption day")
agg = (wxd.filter(pl.col("disrupt").is_not_null())
       .group_by(["ADEP_mvt", "yr", "mo", "disrupt"]).agg([
           pl.len().alias("days"), pl.col("t_mean").mean().round(1).alias("t_mean"), pl.col("t_min").mean().round(1).alias("t_min"),
           pl.col("precip_sum").mean().round(1).alias("precip_mm/day"), pl.col("vis_min").mean().round(2).alias("vis_min"),
           pl.col("lowvis_hrs").mean().round(1).alias("lowvis_hrs"), pl.col("frozen_hrs").mean().round(1).alias("frozen_hrs"),
           pl.col("deice_hrs").mean().round(1).alias("deice_hrs"), pl.col("wind_max").mean().round(1).alias("wind_max")])
       .sort(["yr", "mo", "ADEP_mvt", "disrupt"]))
keep = agg.filter(pl.col("disrupt")).select(["ADEP_mvt", "yr", "mo"]).unique()
print(agg.join(keep, on=["ADEP_mvt", "yr", "mo"]).sort(["yr", "mo", "ADEP_mvt", "disrupt"]))
hdr("S5c. Monthly METAR shares (hour-weighted) EHAM: deicing_gate, frozen_wx, low_vis, mean temp (D: 12.8 % vs 5.1 %)")
print(pl.scan_parquet(B + "data/ref/metar_hourly.parquet").filter(pl.col("station") == "EHAM")
      .group_by(pl.col("hour_utc").dt.strftime("%Y-%m").alias("ym")).agg([pl.len().alias("hrs"), pl.col("deicing_gate").mean().round(3), pl.col("frozen_wx").mean().round(3),
                                                                             pl.col("low_vis").mean().round(3), pl.col("temp_c").mean().round(1), pl.col("precip_mm").sum().round(1)])
      .filter(pl.col("ym").is_in(["2025-01", "2025-07", "2026-01", "2026-07"])).sort("ym").collect())
hdr("S5d. Deicing-gate flag on the departure rows themselves (feature deicing_gate in feat_*), EHAM Jan, disruption vs normal, unmatched/matched")
print(pl.concat([
    t.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).group_by(["yr", "dis_day", "unm"]).agg(pl.len().alias("n"), pl.col("deicing_gate").mean().round(3).alias("deice_share"), pl.col("temp_c").mean().round(1).alias("temp_mean"), pl.col("precip_mm").mean().round(2).alias("precip_mean"), pl.col("vis_km").mean().round(1).alias("vis_mean")),
    r.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1)).group_by(["yr", "dis_day", "unm"]).agg(pl.len().alias("n"), pl.col("deicing_gate").mean().round(3).alias("deice_share"), pl.col("temp_c").mean().round(1).alias("temp_mean"), pl.col("precip_mm").mean().round(2).alias("precip_mean"), pl.col("vis_km").mean().round(1).alias("vis_mean")),
]).sort(["yr", "dis_day", "unm"]))
print("training EHAM unmatched rows by deicing_gate flag (all 2025): label and OOF")
print(lab_stats(t.filter((pl.col("ADEP_mvt") == "EHAM") & pl.col("unm")), ["deicing_gate"]))

# ----------------------------------------------------------------------------------------------
# S6. what outputs/main.parquet (v7) and outputs/probe_unmatched_from_v4.parquet (v9) predict for EHAM Jan 2026
# ----------------------------------------------------------------------------------------------
main = pl.scan_parquet(B + "outputs/main.parquet").rename({"TAXITIME_SEC_mvt": "p_main"}).collect()
probe = pl.scan_parquet(B + "outputs/probe_unmatched_from_v4.parquet").rename({"TAXITIME_SEC_mvt": "p_probe"}).collect()
rp = r.join(main, on="MVT_ID_mvt", how="left").join(probe, on="MVT_ID_mvt", how="left")
print("joined:", rp.height, "null p_main", rp["p_main"].null_count(), "null p_probe", rp["p_probe"].null_count())


def pred_stats(df, keys):
    return (df.group_by(keys).agg([
        pl.len().alias("n"),
        q("p_main", 0.1), q("p_main", 0.5), q("p_main", 0.9), pl.col("p_main").max().alias("p_main_max"), (pl.col("p_main") >= H2).sum().alias("main>=2h"),
        q("p_probe", 0.1), q("p_probe", 0.5), q("p_probe", 0.9), pl.col("p_probe").max().alias("p_probe_max"), (pl.col("p_probe") >= H2).sum().alias("probe>=2h"),
        ((pl.col("p_main") - pl.col("p_probe")) ** 2).mean().sqrt().round(0).alias("rms(main-probe)"),
        (pl.col("p_main") - pl.col("p_probe")).median().round(0).alias("med(main-probe)"),
        pl.col("ms").median().alias("ms_med"),
    ]).sort(keys))


hdr("S6a. EHAM Jan 2026 predictions: disruption vs normal days x unmatched/matched (main = v7, probe = v9: v4 on unmatched rows)")
ej26 = rp.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1))
print(pred_stats(ej26, ["dis_day", "unm"]))
hdr("S6b. EHAM Jan 2026 unmatched rows: per disruption day")
print(pred_stats(ej26.filter(pl.col("unm") & pl.col("dis_day")), ["day"]))
hdr("S6c. Comparison: training EHAM Jan 2025 unmatched labels and OOF, disruption vs normal (same stats as S6a for scale)")
print(lab_stats(t.filter((pl.col("ADEP_mvt") == "EHAM") & (pl.col("mo") == 1) & pl.col("unm")), ["dis_day"]))
hdr("S6d. All ranking unmatched rows on 2026 disruption days vs normal days: predictions by airport")
print(pred_stats(rp.filter(pl.col("unm")), ["dis_day"]))
print(pred_stats(rp.filter(pl.col("unm") & pl.col("dis_day")), ["ADEP_mvt", "mo"]))
hdr("S6e. SSE-share bound: EHAM Jan 2026 disruption-day unmatched rows as share of all ranking rows / all unmatched ranking rows")
n_all, n_unm = rp.height, rp.filter(pl.col("unm")).height
n_ed = ej26.filter(pl.col("unm") & pl.col("dis_day")).height
print(f"rows {n_ed} / {n_all} all = {n_ed / n_all:.4%}; / {n_unm} unmatched = {n_ed / n_unm:.2%}")
for rmse_hyp in (1000, 2000, 3000, 5000):
    print(f"  if RMSE on these {n_ed} rows were {rmse_hyp}: contribution to overall MSE = {n_ed * rmse_hyp**2 / n_all:.0f} s^2 "
          f"(= {math.sqrt(n_ed * rmse_hyp**2 / n_all):.1f} s RMSE-equivalent if the rest were perfect; on top of 305^2 -> {math.sqrt(305**2 + n_ed * rmse_hyp**2 / n_all):.1f})")
print("done")
