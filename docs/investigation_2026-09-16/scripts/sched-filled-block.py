"""Verification of investigation item 1 `sched-filled-block` (plan.md §1).

Independent re-derivation of every number in the item's "Verified" block, under each of the
definitions listed in plan.md §0 / §4 where they differ:

  unmatched:   routing  = mvt_minus_aobt3 (== AOBT_3_flt) null   -> model B
               fid      = FLIGHT_ID_mvt null                       -> D's definition
  fill:        f10   = |BLOCK - SCHED| <= 10 s      (D)
               f6    = |BLOCK - SCHED| <= 6 s       (M, "BLOCK == SCHED (±6 s)")
               f120  = |BLOCK - SCHED| <= 120 s     (M)
               r1800 = |y - ms| < 1800 s            (M)
               r60   = |y - ms| <= 60 s             (D)
  ms = mvt_minus_sched = MVT - SCHED.  Note y - ms = SCHED - BLOCK identically, so f_k == r_k for equal k.

No training. POLARS_MAX_THREADS=1. Only the needed columns are scanned.
"""
import os

os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(60)

ROOT = "/home/user/prc-data-challenge"
FT = f"{ROOT}/data/work/feat_train.parquet"
FR = f"{ROOT}/data/work/feat_rank.parquet"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"
MAIN = f"{ROOT}/outputs/main.parquet"

sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "mvt_minus_aobt3", "mvt_minus_sched",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
]

tr = (
    pl.scan_parquet(FT)
    .select(COLS)
    .with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
        sec("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt").alias("mvt_minus_block"),
        sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs"),
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms_raw"),
        pl.col("mvt_minus_sched").alias("ms"),
        pl.col("mvt_minus_aobt3").is_null().alias("unm_routing"),
        pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.second().alias("blk_sec"),
        (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
    )
    .with_columns(
        (pl.col("bs").abs() <= 10).alias("f10"),
        (pl.col("bs").abs() <= 6).alias("f6"),
        (pl.col("bs").abs() <= 120).alias("f120"),
        ((pl.col("y") - pl.col("ms")).abs() < 1800).alias("r1800"),
        ((pl.col("y") - pl.col("ms")).abs() <= 60).alias("r60"),
        (pl.col("y") - pl.col("ms")).alias("resid"),
        pl.min_horizontal(pl.col("blk_sec"), 60 - pl.col("blk_sec")).alias("blk_sec_dist"),
    )
    .collect()
)
N = tr.height
print(f"### training departures: {N}")

# ------------------------------------------------------------------ A. identity
print("\n=== A. label identity y == MVT - BLOCK ===")
print(
    tr.select(
        pl.len().alias("n"),
        (pl.col("y") == pl.col("mvt_minus_block")).sum().alias("exact_equal"),
        ((pl.col("y") - pl.col("mvt_minus_block")).abs() <= 1).sum().alias("within_1s"),
        pl.col("y").is_null().sum().alias("y_null"),
        pl.col("BLOCK_TIME_UTC_mvt").is_null().sum().alias("block_null"),
        pl.col("SCHED_TIME_UTC_mvt").is_null().sum().alias("sched_null"),
        (pl.col("ms") == pl.col("ms_raw")).sum().alias("feat_ms_equals_raw"),
        # algebraic check: y - ms == SCHED - BLOCK == -bs
        ((pl.col("resid") + pl.col("bs")).abs() == 0).sum().alias("resid_eq_minus_bs"),
    )
)

# ------------------------------------------------------------------ B. unmatched definitions
print("\n=== B. unmatched definitions (training) ===")
print(
    tr.select(
        pl.col("unm_routing").sum().alias("routing_null_aobt3"),
        pl.col("unm_fid").sum().alias("fid_null"),
        (pl.col("unm_routing") & ~pl.col("unm_fid")).sum().alias("routing_only"),
        (~pl.col("unm_routing") & pl.col("unm_fid")).sum().alias("fid_only"),
    )
)

# ------------------------------------------------------------------ C. label buckets
print("\n=== C. label buckets: n, unmatched share (both defs), fill share (each def), BLOCK>1h before SCHED ===")
bucket = (
    pl.when(pl.col("y") < 7200).then(pl.lit("0:<2h"))
    .when(pl.col("y") < 14400).then(pl.lit("1:2-4h"))
    .when(pl.col("y") < 72000).then(pl.lit("2:4-20h"))
    .otherwise(pl.lit("3:>20h"))
    .alias("bucket")
)
trb = tr.with_columns(bucket)
print(
    trb.group_by("bucket")
    .agg(
        pl.len().alias("n"),
        pl.col("unm_routing").mean().round(3).alias("unm_routing"),
        pl.col("unm_fid").mean().round(3).alias("unm_fid"),
        pl.col("f10").mean().round(3).alias("f10"),
        pl.col("f6").mean().round(3).alias("f6"),
        pl.col("f120").mean().round(3).alias("f120"),
        pl.col("r1800").mean().round(3).alias("r1800"),
        (pl.col("bs") < -3600).mean().round(3).alias("blk_gt1h_before_sched"),
        pl.col("lirf").mean().round(3).alias("lirf_share"),
    )
    .sort("bucket")
)

# ------------------------------------------------------------------ D. y >= 2h: |y - ms| <= 60 by matched status
print("\n=== D. y >= 2h rows: share with |y - ms| <= 60 s / < 1800 s, by unmatched def ===")
for defn in ("unm_routing", "unm_fid"):
    print(f"-- def {defn}")
    print(
        tr.filter(pl.col("y") >= 7200)
        .group_by(defn)
        .agg(
            pl.len().alias("n"),
            pl.col("r60").mean().round(3).alias("r60"),
            pl.col("r1800").mean().round(3).alias("r1800"),
            pl.col("f10").mean().round(3).alias("f10"),
            pl.col("lirf").mean().round(3).alias("lirf_share"),
        )
        .sort(defn)
    )

# ------------------------------------------------------------------ E. y >= 4h unmatched: residual quantiles
print("\n=== E. unmatched rows y >= 4h: residual y - ms quantiles, within 1800, BLOCK==SCHED, LIRF count ===")
for defn in ("unm_routing", "unm_fid"):
    g = tr.filter((pl.col("y") >= 14400) & pl.col(defn))
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(f"-- def {defn}: n={g.height}")
    print("   resid quantiles (0,.1,.25,.5,.75,.9,1):", [int(g["resid"].quantile(p, "lower")) for p in q])
    print(
        g.select(
            pl.col("r1800").sum().alias("within_1800"),
            pl.col("resid").median().alias("resid_median"),
            pl.col("f6").sum().alias("f6"),
            pl.col("f10").sum().alias("f10"),
            pl.col("f120").sum().alias("f120"),
            pl.col("lirf").sum().alias("n_lirf"),
            (pl.col("lirf") & pl.col("r1800")).sum().alias("lirf_and_r1800"),
            (~pl.col("lirf") & pl.col("r1800")).sum().alias("nonlirf_and_r1800"),
            (pl.col("r1800") & ~pl.col("f6")).sum().alias("r1800_but_not_f6"),
            pl.col("AIRCRAFT_TYPE_mvt").is_null().sum().alias("actype_null"),
        )
    )
    print("   by ADEP:")
    print(
        g.group_by("ADEP_mvt")
        .agg(pl.len().alias("n"), pl.col("r1800").sum().alias("r1800"), pl.col("f10").sum().alias("f10"),
             pl.col("resid").median().alias("resid_med"), pl.col("y").median().alias("y_med"))
        .sort("n", descending=True)
    )
    print("   rows y >= 4h NOT within 1800 (the non-sched-filled part of the tail):")
    print(
        g.filter(~pl.col("r1800"))
        .select("ADEP_mvt", "y", "ms", "resid", "bs", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
        .sort("y", descending=True)
    )
print("-- 2h <= y < 4h unmatched (routing): share within 1800")
print(
    tr.filter((pl.col("y") >= 7200) & (pl.col("y") < 14400) & pl.col("unm_routing"))
    .select(pl.len().alias("n"), pl.col("r1800").mean().round(4).alias("r1800"), pl.col("f10").mean().round(4).alias("f10"),
            pl.col("lirf").sum().alias("n_lirf"), (pl.col("lirf") & pl.col("r1800")).sum().alias("lirf_r1800"),
            (~pl.col("lirf") & pl.col("r1800")).sum().alias("nonlirf_r1800"))
)
print("   non-LIRF unmatched 2h<=y<4h rows within 1800 by ADEP:")
print(
    tr.filter((pl.col("y") >= 7200) & (pl.col("y") < 14400) & pl.col("unm_routing") & ~pl.col("lirf") & pl.col("r1800"))
    .group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("f10").sum().alias("f10")).sort("n", descending=True)
)

# ------------------------------------------------------------------ F. fill rate by airport x matched
print("\n=== F. fill rate by ADEP x unmatched (routing def): f10 / f6 / f120; filled rows y median, share >= 2h ===")
print(
    tr.group_by("ADEP_mvt", "unm_routing")
    .agg(
        pl.len().alias("n"),
        pl.col("f10").mean().round(4).alias("f10"),
        pl.col("f6").mean().round(4).alias("f6"),
        pl.col("f120").mean().round(4).alias("f120"),
        pl.col("y").filter(pl.col("f10")).median().alias("y_med_f10"),
        (pl.col("y").filter(pl.col("f10")) >= 7200).mean().round(3).alias("ge2h_f10"),
        pl.col("y").filter(~pl.col("f10")).median().alias("y_med_nonfill"),
        (pl.col("y").filter(~pl.col("f10")) >= 7200).mean().round(4).alias("ge2h_nonfill"),
    )
    .sort("ADEP_mvt", "unm_routing")
)
print("-- same, FLIGHT_ID def, LIRF only")
print(
    tr.filter(pl.col("lirf"))
    .group_by("unm_fid")
    .agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"), pl.col("f120").mean().round(4).alias("f120"),
         pl.col("y").filter(pl.col("f10")).median().alias("y_med_f10"),
         (pl.col("y").filter(pl.col("f10")) >= 7200).mean().round(3).alias("ge2h_f10"))
    .sort("unm_fid")
)

# ------------------------------------------------------------------ G. ms bands x ADEP x unmatched
print("\n=== G. ms band 2-20h [7200, 72000): per ADEP x unmatched(routing): n, f10, r60, y median ===")
print(
    tr.filter((pl.col("ms") >= 7200) & (pl.col("ms") < 72000))
    .group_by("ADEP_mvt", "unm_routing")
    .agg(pl.len().alias("n"), pl.col("f10").mean().round(3).alias("f10"), pl.col("r60").mean().round(3).alias("r60"),
         pl.col("f120").mean().round(3).alias("f120"), pl.col("y").median().alias("y_med"),
         (pl.col("y") >= 7200).mean().round(3).alias("y_ge2h"))
    .sort("ADEP_mvt", "unm_routing")
)
print("-- ms >= 7200 (no upper bound), LIRF vs others, unmatched only, both defs")
for defn in ("unm_routing", "unm_fid"):
    print(f"   def {defn}")
    print(
        tr.filter((pl.col("ms") >= 7200) & pl.col(defn))
        .group_by("lirf")
        .agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"), pl.col("f6").mean().round(4).alias("f6"),
             pl.col("f120").mean().round(4).alias("f120"), pl.col("r1800").mean().round(4).alias("r1800"),
             pl.col("y").median().alias("y_med"))
        .sort("lirf")
    )
print("-- non-LIRF unmatched (routing), ms >= 7200: max fill rate over airports")
print(
    tr.filter((pl.col("ms") >= 7200) & pl.col("unm_routing") & ~pl.col("lirf"))
    .group_by("ADEP_mvt")
    .agg(pl.len().alias("n"), pl.col("f10").mean().round(4).alias("f10"), pl.col("f120").mean().round(4).alias("f120"),
         pl.col("r1800").mean().round(4).alias("r1800"))
    .sort("ADEP_mvt")
)
print("-- LIRF unmatched (routing): fill rate by ms band (finer)")
msband = (
    pl.when(pl.col("ms") < 0).then(pl.lit("a:<0"))
    .when(pl.col("ms") < 3600).then(pl.lit("b:0-1h"))
    .when(pl.col("ms") < 7200).then(pl.lit("c:1-2h"))
    .when(pl.col("ms") < 14400).then(pl.lit("d:2-4h"))
    .when(pl.col("ms") < 43200).then(pl.lit("e:4-12h"))
    .when(pl.col("ms") < 72000).then(pl.lit("f:12-20h"))
    .otherwise(pl.lit("g:>=20h"))
    .alias("msband")
)
print(
    tr.filter(pl.col("lirf") & pl.col("unm_routing"))
    .with_columns(msband)
    .group_by("msband")
    .agg(pl.len().alias("n"), pl.col("f10").mean().round(3).alias("f10"), pl.col("r1800").mean().round(3).alias("r1800"),
         pl.col("y").median().alias("y_med"), (pl.col("y") >= 7200).mean().round(3).alias("y_ge2h"),
         (pl.col("y") >= 72000).sum().alias("n_y_ge20h"))
    .sort("msband")
)

# ------------------------------------------------------------------ H. BLOCK seconds-of-minute
print("\n=== H. BLOCK seconds-of-minute within 6 s of a minute boundary (dist = min(sec, 60-sec) <= 6) ===")
print(
    tr.group_by("lirf", "unm_routing")
    .agg(pl.len().alias("n"), (pl.col("blk_sec_dist") <= 6).mean().round(4).alias("within6"),
         (pl.col("blk_sec") == 0).mean().round(4).alias("sec_eq_0"),
         (pl.col("SCHED_TIME_UTC_mvt").dt.second() == 0).mean().round(4).alias("sched_sec_eq_0"),
         (pl.col("MVT_TIME_UTC_mvt").dt.second() == 0).mean().round(4).alias("mvt_sec_eq_0"))
    .sort("lirf", "unm_routing")
)
print("-- non-LIRF unmatched (routing), by ADEP")
print(
    tr.filter(~pl.col("lirf") & pl.col("unm_routing"))
    .group_by("ADEP_mvt")
    .agg(pl.len().alias("n"), (pl.col("blk_sec_dist") <= 6).mean().round(3).alias("within6"))
    .sort("ADEP_mvt")
)

# ------------------------------------------------------------------ I. LIRF unmatched profile
print("\n=== I. LIRF unmatched (routing) profile ===")
lu = tr.filter(pl.col("lirf") & pl.col("unm_routing"))
print(
    lu.select(
        pl.len().alias("n"), pl.col("AIRCRAFT_TYPE_mvt").is_null().sum().alias("actype_null"),
        pl.col("y").median().alias("y_med"), (pl.col("y") >= 7200).mean().round(4).alias("y_ge2h"),
        (pl.col("y") >= 14400).sum().alias("n_y_ge4h"), pl.col("f10").mean().round(4).alias("f10"),
        pl.col("f120").mean().round(4).alias("f120"),
    )
)
print("-- LIRF matched (routing) actype null:", tr.filter(pl.col("lirf") & ~pl.col("unm_routing"))["AIRCRAFT_TYPE_mvt"].is_null().sum())
print("-- LIRF unmatched: does fill co-occur with y >= 2h?  2x2 (f10 x y>=2h)")
print(lu.group_by("f10", (pl.col("y") >= 7200).alias("y_ge2h")).agg(pl.len().alias("n"), pl.col("ms").median().alias("ms_med"), pl.col("y").median().alias("y_med")).sort("f10", "y_ge2h"))
print("-- LIRF unmatched: fill by month")
print(lu.group_by(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).agg(pl.len().alias("n"), pl.col("f10").mean().round(3).alias("f10"), (pl.col("y") >= 7200).mean().round(3).alias("y_ge2h"), (pl.col("ms") >= 7200).mean().round(3).alias("ms_ge2h")).sort("mo"))

# ------------------------------------------------------------------ J. OOF SSE share of sched-filled rows
print("\n=== J. OOF main-48c19bc830: SSE share of sched-filled rows among unmatched rows ===")
oof_all = pl.read_parquet(OOF)
oof = oof_all.filter(pl.col("oof").is_not_null() & pl.col("oof").is_not_nan())
print(f"oof file rows: {oof_all.height}; rows with a non-null OOF prediction (scored): {oof.height}")
j = oof.join(tr.select("MVT_ID_mvt", "y", "unm_routing", "unm_fid", "lirf", "f10", "f6", "f120", "r1800", "ms", "resid", "ADEP_mvt"), on="MVT_ID_mvt", how="left")
print("unjoined:", j["y"].is_null().sum())
j = j.with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
tot = j["se"].sum()
print(f"total SSE {tot:.4e}  RMSE all {(tot / j.height) ** 0.5:.2f}")
for defn in ("unm_routing", "unm_fid"):
    u = j.filter(pl.col(defn))
    usse = u["se"].sum()
    print(f"-- def {defn}: unmatched rows {u.height}, RMSE {(usse / u.height) ** 0.5:.1f}, SSE share of total {usse / tot:.3f}")
    for f in ("f6", "f10", "f120", "r1800"):
        s = u.filter(pl.col(f))
        print(f"   {f}: n={s.height}, SSE share of unmatched {s['se'].sum() / usse:.4f}, of total {s['se'].sum() / tot:.4f}, RMSE {(s['se'].sum() / max(s.height, 1)) ** 0.5:.1f}")
    s = u.filter(pl.col("f10") & (pl.col("ms") >= 7200))
    print(f"   f10 & ms>=2h: n={s.height}, share of unmatched {s['se'].sum() / usse:.4f}, RMSE {(s['se'].sum() / max(s.height, 1)) ** 0.5:.1f}, pred/ms median {(s['oof'] / s['ms']).median():.3f}")
    s = u.filter(~pl.col("f10") & (pl.col("ms") >= 7200))
    print(f"   not f10 & ms>=2h: n={s.height}, share of unmatched {s['se'].sum() / usse:.4f}, RMSE {(s['se'].sum() / max(s.height, 1)) ** 0.5:.1f}")
    s = u.filter(pl.col("lirf") & (pl.col("ms") >= 7200))
    print(f"   LIRF & ms>=2h: n={s.height}, share of unmatched {s['se'].sum() / usse:.4f}, of total {s['se'].sum() / tot:.4f}; f10 among them {s['f10'].sum()}")
# oracle: if sched-filled rows were predicted exactly y (= ms), what RMSE would remain on unmatched rows?
u = j.filter(pl.col("unm_routing"))
usse = u["se"].sum()
for f in ("f10", "r1800"):
    rem = u.filter(~pl.col(f))["se"].sum()
    print(f"-- oracle '{f} rows predicted exactly': unmatched RMSE {(usse / u.height) ** 0.5:.1f} -> {(rem / u.height) ** 0.5:.1f}")
# replacing pred by ms on f10 rows (what "deterministic given mvt_minus_sched" would give if the flag were known)
rep = u.with_columns(pl.when(pl.col("f10")).then(pl.col("ms")).otherwise(pl.col("oof")).alias("p2")).with_columns(((pl.col("p2") - pl.col("y")) ** 2).alias("se2"))
print(f"-- replace pred by ms on f10 rows (oracle flag): unmatched RMSE {(rep['se2'].sum() / rep.height) ** 0.5:.1f}")
# what happens if pred := ms on ALL LIRF unmatched rows with ms >= 2h (no oracle flag)
rep2 = u.with_columns(pl.when(pl.col("lirf") & (pl.col("ms") >= 7200)).then(pl.col("ms")).otherwise(pl.col("oof")).alias("p2")).with_columns(((pl.col("p2") - pl.col("y")) ** 2).alias("se2"))
print(f"-- replace pred by ms on all LIRF unmatched ms>=2h rows (no flag): unmatched RMSE {(rep2['se2'].sum() / rep2.height) ** 0.5:.1f}")

# ------------------------------------------------------------------ K. ranking
print("\n=== K. ranking: LIRF unmatched ms >= 2h, main.parquet predictions; BLOCK availability ===")
rk = (
    pl.scan_parquet(FR)
    .select("MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "mvt_minus_aobt3", "mvt_minus_sched", "BLOCK_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt")
    .with_columns(pl.col("mvt_minus_aobt3").is_null().alias("unm_routing"), pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
                  (pl.col("ADEP_mvt") == "LIRF").alias("lirf"), pl.col("mvt_minus_sched").alias("ms"))
    .collect()
)
print(rk.select(pl.len().alias("n"), pl.col("BLOCK_TIME_UTC_mvt").is_null().sum().alias("block_null"), pl.col("TAXITIME_SEC_mvt").is_null().sum().alias("label_null"),
                pl.col("ms").is_null().sum().alias("ms_null"), pl.col("unm_routing").sum().alias("unm_routing"), pl.col("unm_fid").sum().alias("unm_fid")))
main = pl.read_parquet(MAIN).rename({"TAXITIME_SEC_mvt": "pred"})
rk = rk.join(main, on="MVT_ID_mvt", how="left")
print("main.parquet unjoined:", rk["pred"].is_null().sum())
for defn in ("unm_routing", "unm_fid"):
    s = rk.filter(pl.col("lirf") & pl.col(defn) & (pl.col("ms") >= 7200))
    print(f"-- def {defn}: LIRF unmatched ms>=2h rows {s.height}; pred median {s['pred'].median()}, n pred>=2h {(s['pred'] >= 7200).sum()}, "
          f"pred/ms median {(s['pred'] / s['ms']).median():.3f}; actype null {s['AIRCRAFT_TYPE_mvt'].is_null().sum()}")
    print("   by month:", s.group_by(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).agg(pl.len().alias("n"), (pl.col("ms") >= 14400).sum().alias("ms_ge4h"), (pl.col("ms") >= 43200).sum().alias("ms_ge12h")).sort("mo").to_dicts())
print("-- LIRF unmatched (routing) ranking total:", rk.filter(pl.col("lirf") & pl.col("unm_routing")).height,
      " training LIRF unmatched Jan+Jul 2025:", lu.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])).height)
print("-- training-implied fill count for the 181 ranking rows: rate f10 among LIRF unmatched ms>=2h training =",
      round(tr.filter(pl.col("lirf") & pl.col("unm_routing") & (pl.col("ms") >= 7200))["f10"].mean(), 4))

# ------------------------------------------------------------------ L. LIRF unmatched ms >= 2h, non-filled rows: what are their labels?
print("\n=== L. LIRF unmatched (routing) ms >= 2h: filled vs non-filled label distribution ===")
s = tr.filter(pl.col("lirf") & pl.col("unm_routing") & (pl.col("ms") >= 7200))
print(
    s.group_by("f10")
    .agg(pl.len().alias("n"), pl.col("y").quantile(0.1).alias("y_p10"), pl.col("y").median().alias("y_med"),
         pl.col("y").quantile(0.9).alias("y_p90"), pl.col("y").max().alias("y_max"),
         (pl.col("y") < 3600).mean().round(3).alias("y_lt1h"), (pl.col("y") >= 7200).mean().round(3).alias("y_ge2h"),
         (pl.col("y") >= 72000).sum().alias("n_ge20h"), pl.col("ms").median().alias("ms_med"))
    .sort("f10")
)
