"""Item 2 `block-day-rollover` (docs/investigation_2026-09-16/plan.md §1).

Independent re-derivation, no training. POLARS_MAX_THREADS=1, lazy scans, only needed columns.

Sections
  A  training: label tail counts under several thresholds; full list of rows y > 20 h
  B  training: per-row rollover diagnostics (y-86400, BLOCK-SCHED, date gaps, alt. interpretations)
  C  training: same-callsign row on the BLOCK date (alt. mechanism: BLOCK taken from previous day's flight)
  D  training: ms bands 20-28 h / > 28 h, matched vs unmatched (both definitions)
  E  OOF main-48c19bc830: SSE shares under the three definitions used by D / M / H
  F  ranking: ms bands 20-28 h / > 28 h; LIRF unmatched mms >= 12 h; BLOCK availability
  G  training LIRF unmatched mms >= 12 h: sched-filled vs rollover vs other; visible-feature split
"""
import os

os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(320)
pl.Config.set_fmt_str_lengths(40)

B = "/home/user/prc-data-challenge/"
TR = B + "data/work/feat_train.parquet"
RK = B + "data/work/feat_rank.parquet"
OOF = B + "experiments/oof_main-48c19bc830.parquet"
MAIN = B + "outputs/main.parquet"

DAY = 86_400
H2, H4, H12, H20, H28 = 7_200, 14_400, 43_200, 72_000, 100_800

sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

base_cols = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
    "AOBT_3_flt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "mvt_minus_sched", "month", "hour_utc", "dow",
]


def enrich(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
        pl.col("mvt_minus_aobt3").is_null().alias("unm_route"),  # routing definition (H, P, M)
        pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),  # D's definition
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
        sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("blk_minus_sched"),
        sec("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt").alias("mvt_minus_blk"),
        (pl.col("MVT_TIME_UTC_mvt").dt.date() - pl.col("BLOCK_TIME_UTC_mvt").dt.date()).dt.total_days().alias("d_mvt_blk_days"),
        (pl.col("MVT_TIME_UTC_mvt").dt.date() - pl.col("SCHED_TIME_UTC_mvt").dt.date()).dt.total_days().alias("d_mvt_sch_days"),
        (pl.col("BLOCK_TIME_UTC_mvt").dt.date() - pl.col("SCHED_TIME_UTC_mvt").dt.date()).dt.total_days().alias("d_blk_sch_days"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.second().alias("blk_sec"),
    ).with_columns(
        (pl.col("y") - DAY).alias("y_minus_day"),
        (pl.col("y") % DAY).alias("y_mod_day"),
        (pl.col("blk_minus_sched").abs() <= 10).alias("fill10"),  # D's fill
        (pl.col("blk_minus_sched").abs() <= 6).alias("fill6"),  # M's fill
        ((pl.col("y") - pl.col("ms")).abs() <= 60).alias("y_eq_ms60"),
        # alternative interpretations of a ~1-day label
        (pl.col("blk_minus_sched") + DAY).alias("blkplus1d_minus_sched"),  # if BLOCK date is 1 day early
        (pl.col("ms") - DAY).alias("mvtminus1d_minus_sched"),  # if MVT date is 1 day late
    )


tr = enrich(pl.scan_parquet(TR).select(base_cols))

print("=" * 100)
print("A. TRAINING: phase check, label-tail counts under each threshold")
print("=" * 100)
print(tr.group_by("PHASE_mvt").len().collect())
print(
    tr.select(
        pl.len().alias("rows"),
        pl.col("unm_route").sum().alias("unm_route"),
        pl.col("unm_fid").sum().alias("unm_fid"),
        (pl.col("y") >= H2).sum().alias("y_ge_2h"),
        (pl.col("y") >= H4).sum().alias("y_ge_4h"),
        (pl.col("y") >= H12).sum().alias("y_ge_12h"),
        (pl.col("y") > H20).sum().alias("y_gt_20h"),
        (pl.col("y") >= 80_000).sum().alias("y_ge_80000"),
        (pl.col("y") >= DAY).sum().alias("y_ge_86400"),
        (pl.col("y") >= H28).sum().alias("y_ge_28h"),
        (pl.col("y") <= -H20).sum().alias("y_le_minus20h"),
        (pl.col("y") < 0).sum().alias("y_neg"),
        ((pl.col("y") - pl.col("mvt_minus_blk")).abs() > 1).sum().alias("identity_viol"),
    ).collect()
)
print("--- y >= 12 h by unmatched definition ---")
print(
    tr.filter(pl.col("y") >= H12)
    .group_by(["unm_route", "unm_fid"])
    .agg(pl.len(), (pl.col("y") > H20).sum().alias("gt20h"), (pl.col("y") >= H4).sum().alias("ge4h"))
    .collect()
)

roll = tr.filter(pl.col("y") > H20).sort("y", descending=True).collect()
print(f"--- FULL LIST of training rows y > 20 h (n={roll.height}) ---")
print(
    roll.select(
        "ADEP_mvt", "FLIGHT_mvt", "unm_route", "unm_fid", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "MVT_TIME_UTC_mvt", "y", "ms", "y_minus_day", "blk_minus_sched", "d_mvt_blk_days", "d_mvt_sch_days",
        "d_blk_sch_days", "blkplus1d_minus_sched", "mvtminus1d_minus_sched", "fill10", "AIRCRAFT_TYPE_mvt",
        "RUNWAY_mvt", "AOBT_3_flt", "mvt_minus_aobt3", "blk_sec", "MVT_ID_mvt",
    )
)

print("=" * 100)
print("B. TRAINING: rollover diagnostics on the y > 20 h rows")
print("=" * 100)
print("y - 86400 sorted:", sorted(roll["y_minus_day"].to_list()))
print("n with 602 <= y-86400 <= 1992:", ((roll["y_minus_day"] >= 602) & (roll["y_minus_day"] <= 1992)).sum())
print("n with 0 < y-86400 < 7200 (plausible taxi after removing a day):", ((roll["y_minus_day"] > 0) & (roll["y_minus_day"] < H2)).sum())
print("n with d_mvt_blk_days == 1:", (roll["d_mvt_blk_days"] == 1).sum(), " ==2:", (roll["d_mvt_blk_days"] == 2).sum())
print("n with BLOCK date == SCHED date:", (roll["d_blk_sch_days"] == 0).sum())
print("n with MVT date == SCHED date:", (roll["d_mvt_sch_days"] == 0).sum(), " MVT date == SCHED date+1:", (roll["d_mvt_sch_days"] == 1).sum())
print("n with |ms| < 2 h (MVT consistent with SCHED):", (roll["ms"].abs() < H2).sum())
print("n with BLOCK before SCHED by > 1 h:", (roll["blk_minus_sched"] < -3600).sum())
print("n fill10:", roll["fill10"].sum(), " fill6:", roll["fill6"].sum())
print("n with |ms - y| <= 60 (y == ms):", roll["y_eq_ms60"].sum())
print("BLOCK seconds-of-minute values:", sorted(roll["blk_sec"].to_list()))
print("per airport:", roll.group_by("ADEP_mvt").len().sort("len", descending=True).to_dicts())
print("per month:", roll.group_by("month").len().sort("month").to_dicts())
print("per unm_route:", roll.group_by("unm_route").len().to_dicts(), " per unm_fid:", roll.group_by("unm_fid").len().to_dicts())
print("--- interpretation test: under 'BLOCK date -1' hypothesis, implied true BLOCK-SCHED (s); under 'MVT date +1', implied MVT-SCHED (s) ---")
print(roll.select("ADEP_mvt", "y", "ms", "blk_minus_sched", "blkplus1d_minus_sched", "mvtminus1d_minus_sched", "y_minus_day"))
print("--- 12 h <= y <= 20 h rows (are there rollovers hiding below 20 h?) ---")
mid = tr.filter((pl.col("y") >= H12) & (pl.col("y") <= H20)).sort("y", descending=True).collect()
print(mid.select("ADEP_mvt", "unm_route", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y", "ms", "blk_minus_sched", "d_mvt_blk_days", "fill10", "y_eq_ms60"))

print("=" * 100)
print("C. TRAINING: alt. mechanism - is there a DEP row of the same (ADEP, FLIGHT) whose MVT date == this row's BLOCK date?")
print("=" * 100)
keys = roll.select(
    "MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", pl.col("BLOCK_TIME_UTC_mvt").dt.date().alias("blk_date"), pl.col("BLOCK_TIME_UTC_mvt").alias("roll_block")
)
cand = (
    tr.select("MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "y")
    .filter(pl.col("FLIGHT_mvt").is_in(keys["FLIGHT_mvt"].implode()) & pl.col("ADEP_mvt").is_in(keys["ADEP_mvt"].implode()))
    .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("mvt_date"))
    .collect()
)
m = keys.join(cand.rename({"MVT_ID_mvt": "other_id", "y": "other_y"}), left_on=["ADEP_mvt", "FLIGHT_mvt", "blk_date"], right_on=["ADEP_mvt", "FLIGHT_mvt", "mvt_date"], how="left")
m = m.filter(pl.col("other_id") != pl.col("MVT_ID_mvt"))
m = m.with_columns((pl.col("roll_block") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("rollblk_minus_otherblk"))
print(f"rollover rows with a same-(ADEP,FLIGHT) departure on the BLOCK date: {m['MVT_ID_mvt'].n_unique()} of {roll.height}")
print(m.select("ADEP_mvt", "FLIGHT_mvt", "blk_date", "roll_block", "other_id", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "other_y", "rollblk_minus_otherblk"))

print("=" * 100)
print("D. TRAINING: ms bands 20-28 h / > 28 h by matched/unmatched (both definitions)")
print("=" * 100)
band = pl.when(pl.col("ms") > H28).then(pl.lit(">28h")).when(pl.col("ms") > H20).then(pl.lit("20-28h")).when(pl.col("ms") >= H12).then(pl.lit("12-20h")).otherwise(pl.lit("<12h"))
for d in ("unm_route", "unm_fid"):
    print(f"--- by {d} ---")
    print(
        tr.with_columns(band.alias("ms_band"))
        .filter(pl.col("ms_band") != "<12h")
        .group_by(["ms_band", d])
        .agg(
            pl.len(),
            (pl.col("y") >= H2).mean().round(3).alias("y_ge2h_share"),
            (pl.col("y") > H20).sum().alias("n_y_gt20h"),
            pl.col("y_eq_ms60").mean().round(3).alias("y_eq_ms_share"),
            pl.col("fill10").mean().round(3).alias("fill10_share"),
            pl.col("y").median().alias("y_med"),
            (pl.col("ADEP_mvt") == "LIRF").sum().alias("n_lirf"),
        )
        .sort(["ms_band", d])
        .collect()
    )

print("=" * 100)
print("E. OOF main-48c19bc830: SSE shares of rollover rows under each definition")
print("=" * 100)
oof = pl.scan_parquet(OOF).drop_nulls("oof")
j = tr.join(oof, on="MVT_ID_mvt", how="inner").with_columns(((pl.col("y") - pl.col("oof")) ** 2).alias("se")).collect()
tot = j["se"].sum()
n = j.height
print(f"OOF rows {n}, RMSE {(tot / n) ** 0.5:.2f}")
for d in ("unm_route", "unm_fid"):
    u = j.filter(pl.col(d))
    print(f"  unmatched({d}): n={u.height}, RMSE={(u['se'].mean()) ** 0.5:.1f}, SSE share of total={u['se'].sum() / tot:.4f}")
    mt = j.filter(~pl.col(d))
    print(f"  matched({d}):   n={mt.height}, RMSE={(mt['se'].mean()) ** 0.5:.1f}")
sse_unm_route = j.filter(pl.col("unm_route"))["se"].sum()
sse_unm_fid = j.filter(pl.col("unm_fid"))["se"].sum()


def rep(mask, name):
    s = j.filter(mask)
    if s.height == 0:
        print(f"{name:75s} n=0")
        return
    su = s.filter(pl.col("unm_route"))
    print(
        f"{name:75s} n={s.height:3d} (unm_route {su.height:3d}) | SSE/total={s['se'].sum() / tot:.4f} | "
        f"SSE(unm rows only)/SSE_unm_route={su['se'].sum() / sse_unm_route:.4f} | SSE(all these rows)/SSE_unm_route={s['se'].sum() / sse_unm_route:.4f} | "
        f"RMSE={s['se'].mean() ** 0.5:.0f} | pred_med={s['oof'].median():.0f} | y_med={s['y'].median():.0f}"
    )


rep(pl.col("y") > H20, "D: y > 20 h (matched+unmatched)")
rep((pl.col("y") > H20) & pl.col("unm_route"), "D-subset: y > 20 h, unmatched (routing)")
rep((pl.col("y_minus_day") > 0) & (pl.col("y_minus_day") < H2), "M: y = 86400 + taxi, 0 < y-86400 < 7200 (all rows)")
rep((pl.col("y_minus_day") > 0) & (pl.col("y_minus_day") < H2) & pl.col("unm_route"), "M: y = 86400 + taxi, unmatched (routing)")
rep((pl.col("y_minus_day") >= 602) & (pl.col("y_minus_day") <= 1992), "D's 602..1992 band")
rep((pl.col("y") > H20) & pl.col("unm_route") & ((pl.col("y") - pl.col("ms")).abs() >= 1800), "M-reconstruction: y>20h, unmatched, NOT sched-filled by |y-mms|<1800")
rep((pl.col("y") > H20) & ((pl.col("y") - pl.col("ms")).abs() < 1800), "y>20h rows that M's |y-mms|<1800 rule would call sched-filled")
rep(pl.col("y") >= 80_000, "H: y >= ~1 day (>= 80,000)")
rep(pl.col("y") >= DAY, "y >= 86,400")
rep(pl.col("y") >= H12, "y >= 12 h")
rep(pl.col("y") >= H4, "y >= 4 h")
rep(pl.col("y") >= H2, "y >= 2 h")
rep((pl.col("y") > H20) & (pl.col("ADEP_mvt") == "LIRF"), "y > 20 h at LIRF")
rep((pl.col("y") > H20) & (pl.col("ADEP_mvt") != "LIRF"), "y > 20 h not LIRF")
print("--- FULL LIST of OOF rows y > 20 h with prediction ---")
print(
    j.filter(pl.col("y") > H20)
    .sort("se", descending=True)
    .select("ADEP_mvt", "FLIGHT_mvt", "unm_route", "unm_fid", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y", "ms", "y_minus_day", "blk_minus_sched", "oof", "se", (pl.col("se") / tot).round(4).alias("share_tot"), (pl.col("se") / sse_unm_route).round(4).alias("share_unm"))
)
print("--- top-25 SSE rows among unmatched(routing): share of unmatched SSE, and how many are rollovers ---")
top = j.filter(pl.col("unm_route")).sort("se", descending=True).head(25)
print(f"top-25 unmatched SSE share = {top['se'].sum() / sse_unm_route:.4f}; rollovers (y>20h) among them = {(top['y'] > H20).sum()}; y>=4h among them = {(top['y'] >= H4).sum()}")
print(top.select("ADEP_mvt", "unm_fid", "y", "ms", "oof", "se", "fill10", "y_eq_ms60"))
print("--- SSE share of OOF rows y > 20 h if removed: RMSE without them ---")
rest = j.filter(pl.col("y") <= H20)
print(f"RMSE all={((tot / n) ** 0.5):.2f}; RMSE excluding y>20h rows={((rest['se'].sum() / rest.height) ** 0.5):.2f}")
ru = j.filter(pl.col("unm_route") & (pl.col("y") <= H20))
print(f"unmatched RMSE all={(j.filter(pl.col('unm_route'))['se'].mean() ** 0.5):.1f}; excluding y>20h rows={(ru['se'].mean() ** 0.5):.1f}")
print("--- which training months are in the OOF (fold coverage) ---")
print(j.group_by("month").len().sort("month").to_dicts())
print("training rollover rows per month vs OOF rollover rows per month:")
print(roll.group_by("month").len().sort("month").rename({"len": "train_n"}).join(j.filter(pl.col("y") > H20).group_by("month").len().rename({"len": "oof_n"}), on="month", how="left"))

print("=" * 100)
print("F. RANKING: ms bands, LIRF unmatched mms >= 12 h, BLOCK availability")
print("=" * 100)
rk = enrich(pl.scan_parquet(RK).select(base_cols))
print(rk.select(pl.len(), pl.col("unm_route").sum().alias("unm_route"), pl.col("unm_fid").sum().alias("unm_fid"), pl.col("BLOCK_TIME_UTC_mvt").is_null().sum().alias("block_null"), pl.col("TAXITIME_SEC_mvt").is_null().sum().alias("y_null"), pl.col("ms").is_null().sum().alias("ms_null")).collect())
for d in ("unm_route", "unm_fid"):
    print(f"--- ranking ms bands by {d} ---")
    print(rk.with_columns(band.alias("ms_band")).filter(pl.col("ms_band") != "<12h").group_by(["ms_band", d]).agg(pl.len(), (pl.col("ADEP_mvt") == "LIRF").sum().alias("n_lirf"), pl.col("ms").median().alias("ms_med")).sort(["ms_band", d]).collect())
lirf12 = rk.filter(pl.col("unm_route") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= H12)).sort("ms", descending=True)
try:
    main = pl.scan_parquet(MAIN).rename({"TAXITIME_SEC_mvt": "pred_main"})
    lirf12 = lirf12.join(main, on="MVT_ID_mvt", how="left")
except Exception as e:  # noqa: BLE001
    print("main.parquet not joined:", e)
lirf12 = lirf12.collect()
print(f"--- ranking LIRF unmatched(routing) rows with ms >= 12 h: n={lirf12.height} ---")
print(lirf12.select([c for c in ["ADEP_mvt", "FLIGHT_mvt", "month", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "ms", "hour_utc", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "pred_main", "MVT_ID_mvt"] if c in lirf12.columns]))
print("ranking LIRF unmatched ms >= 2h / 4h / 12h:", rk.filter(pl.col("unm_route") & (pl.col("ADEP_mvt") == "LIRF")).select((pl.col("ms") >= H2).sum().alias("ge2h"), (pl.col("ms") >= H4).sum().alias("ge4h"), (pl.col("ms") >= H12).sum().alias("ge12h")).collect().row(0))
print("ranking non-LIRF unmatched ms >= 12h:", rk.filter(pl.col("unm_route") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("ms") >= H12)).select(pl.len()).collect().item())

print("=" * 100)
print("G. TRAINING LIRF unmatched(routing) mms >= 12 h: sched-filled / rollover / other, and visible-feature split")
print("=" * 100)
cls = (
    pl.when(pl.col("y") > H20).then(pl.lit("rollover(y>20h)"))
    .when(pl.col("y_eq_ms60")).then(pl.lit("sched-filled(|y-ms|<=60)"))
    .otherwise(pl.lit("other"))
)
g = tr.filter(pl.col("unm_route") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= H12)).with_columns(cls.alias("cls")).collect()
print(f"n={g.height}")
print(g.group_by("cls").agg(pl.len(), pl.col("y").median().alias("y_med"), pl.col("ms").median().alias("ms_med")).sort("cls"))
print(g.sort("ms").select("cls", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "y", "ms", "blk_minus_sched", "hour_utc", "dow", "month", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt"))
print("--- by ms hour bucket ---")
print(g.with_columns((pl.col("ms") // 3600).alias("ms_h")).group_by("ms_h").agg(pl.len(), (pl.col("cls") == "rollover(y>20h)").sum().alias("n_roll"), (pl.col("cls").str.starts_with("sched")).sum().alias("n_sf")).sort("ms_h"))
print("--- by take-off hour UTC ---")
print(g.group_by("hour_utc").agg(pl.len(), (pl.col("cls") == "rollover(y>20h)").sum().alias("n_roll")).sort("hour_utc"))
print("--- same, all airports, unmatched(routing), ms >= 12 h ---")
g2 = tr.filter(pl.col("unm_route") & (pl.col("ms") >= H12)).with_columns(cls.alias("cls")).collect()
print(g2.group_by(["ADEP_mvt", "cls"]).len().sort(["ADEP_mvt", "cls"]))
print("--- rollover rows: where do they sit in ms? (all 16, matched+unmatched) ---")
print(roll.select("ADEP_mvt", "unm_route", "ms", (pl.col("ms") // 3600).alias("ms_h"), "y"))

print("=" * 100)
print("H. OTHER OOF FILES in experiments/: SSE share of y > 20 h rows (to locate H's 24.1 % figure)")
print("=" * 100)
import glob
lab = tr.select("MVT_ID_mvt", "y", "unm_route", "ADEP_mvt").collect()
for f in sorted(glob.glob(B + "experiments/oof_*.parquet")):
    try:
        o = pl.read_parquet(f).drop_nulls("oof")
    except Exception as e:  # noqa: BLE001
        print(os.path.basename(f), "skip:", e)
        continue
    jj = lab.join(o, on="MVT_ID_mvt", how="inner").with_columns(((pl.col("y") - pl.col("oof")) ** 2).alias("se"))
    t = jj["se"].sum()
    r = jj.filter(pl.col("y") > H20)
    u = jj.filter(pl.col("unm_route"))
    ru = r.filter(pl.col("unm_route"))
    print(
        f"{os.path.basename(f):45s} n={jj.height:7d} RMSE={(t / jj.height) ** 0.5:7.2f} unmRMSE={(u['se'].mean()) ** 0.5:7.1f} | y>20h: n={r.height:2d} share_tot={r['se'].sum() / t:.3f} "
        f"share_unm(unm rows)={ru['se'].sum() / u['se'].sum():.3f} | y>=80000 n={(jj['y'] >= 80000).sum()} share={jj.filter(pl.col('y') >= 80000)['se'].sum() / t:.3f} | y>=4h n={(jj['y'] >= H4).sum()} share={jj.filter(pl.col('y') >= H4)['se'].sum() / t:.3f} | y>=2h n={(jj['y'] >= H2).sum()} share={jj.filter(pl.col('y') >= H2)['se'].sum() / t:.3f}"
    )
