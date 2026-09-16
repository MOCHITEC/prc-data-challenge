"""Pre-check for investigation item 31 `shared-b-no-transfer` (plan.md §1; the training run is item 30).

No training. POLARS_MAX_THREADS=1. Lazy scans, only the needed columns.

Questions answered on data/work/feat_train.parquet (and, for prediction-time availability only, feat_rank):
  B  per airport, label distribution (median, p90, p99, share y >= 2 h, share y >= 4 h) matched vs unmatched
  C  fill rate |BLOCK - SCHED| <= 10 s (and <= 120 s) matched vs unmatched by airport
  D  AIRCRAFT_TYPE_mvt null rate by airport x matched
  E  matched rows carrying the visible signature of the LIRF unmatched extreme rows (LIRF, actype null,
     ms >= 2 h) and their labels; relaxations of the signature (LIRF & ms >= 2 h with actype present)
  F  P(y >= 2 h | ADEP, ms band) matched vs unmatched -- what the pooled rows say about the ms -> y mapping
  G  cols_b (the exact prefix rule of src/prc/train.py) and categoricals: non-null rate on matched / unmatched
     training rows and on unmatched ranking rows
  H  extra "extreme" examples (y >= 1 h / 2 h / 4 h / 12 h) that pooling adds per airport, and the profile of
     the matched y >= 2 h rows

Definitions (plan §0): unmatched = routing definition (mvt_minus_aobt3 null); D's FLIGHT_ID definition is
reported alongside where the two differ. y = TAXITIME_SEC_mvt; ms = mvt_minus_sched = MVT - SCHED.
"""
import os

os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

pl.Config.set_tbl_rows(300)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_float_precision(4)

ROOT = "/home/user/prc-data-challenge"
FT = f"{ROOT}/data/work/feat_train.parquet"
FR = f"{ROOT}/data/work/feat_rank.parquet"

H1, H2, H4, H12 = 3600, 7200, 14400, 43200
sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

# ---------------------------------------------------------------------------------------------- main frame
COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt", "AOBT_3_flt", "mvt_minus_aobt3", "mvt_minus_sched", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt",
    "STAND_mvt", "month", "nm_matched",
]
tr = (
    pl.scan_parquet(FT)
    .select(COLS)
    .with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
        sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("bs"),
        sec("BLOCK_TIME_UTC_mvt", "AOBT_3_flt").alias("block_minus_aobt3"),
        pl.col("mvt_minus_sched").alias("ms"),
        pl.col("mvt_minus_aobt3").is_null().alias("unm"),
        pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().alias("act_null"),
        (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
    )
    .with_columns(
        (pl.col("bs").abs() <= 10).alias("f10"),
        (pl.col("bs").abs() <= 120).alias("f120"),
        (pl.col("y") >= H1).alias("y1h"),
        (pl.col("y") >= H2).alias("y2h"),
        (pl.col("y") >= H4).alias("y4h"),
        (pl.col("y") >= H12).alias("y12h"),
        pl.when(pl.col("unm")).then(pl.lit("unmatched")).otherwise(pl.lit("matched")).alias("grp"),
    )
    .collect()
)
N = tr.height
print(f"### training departures: {N}")

# ---------------------------------------------------------------------------------------------- A. sanity
print("\n=== A. sanity ===")
print(
    tr.select(
        pl.len().alias("n"),
        pl.col("unm").sum().alias("unmatched_routing"),
        pl.col("unm_fid").sum().alias("unmatched_fid"),
        (pl.col("unm") & ~pl.col("unm_fid")).sum().alias("routing_only"),
        (~pl.col("unm") & pl.col("unm_fid")).sum().alias("fid_only"),
        (pl.col("nm_matched") == (~pl.col("unm")).cast(pl.Int8)).sum().alias("nm_matched_flag_equals_not_unm"),
        pl.col("y").is_null().sum().alias("y_null"),
        pl.col("ms").is_null().sum().alias("ms_null"),
        (pl.col("y") - pl.col("ms") + pl.col("bs")).abs().max().alias("max_abs(y-ms+bs)"),
    )
)

# ---------------------------------------------------------------------------------------------- B. labels
print("\n=== B. label distribution per ADEP x group (routing definition) ===")
lab = (
    tr.group_by("ADEP_mvt", "grp")
    .agg(
        pl.len().alias("n"),
        pl.col("y").median().alias("y_med"),
        pl.col("y").mean().alias("y_mean"),
        pl.col("y").quantile(0.9).alias("y_p90"),
        pl.col("y").quantile(0.99).alias("y_p99"),
        pl.col("y").max().alias("y_max"),
        pl.col("y2h").mean().alias("share_2h"),
        pl.col("y4h").mean().alias("share_4h"),
        pl.col("y1h").sum().alias("n_1h"),
        pl.col("y2h").sum().alias("n_2h"),
        pl.col("y4h").sum().alias("n_4h"),
        pl.col("y12h").sum().alias("n_12h"),
    )
    .sort("ADEP_mvt", "grp")
)
print(lab)
print("\n--- pooled over airports ---")
print(
    tr.group_by("grp")
    .agg(
        pl.len().alias("n"),
        pl.col("y").median().alias("y_med"),
        pl.col("y").mean().alias("y_mean"),
        pl.col("y").quantile(0.9).alias("y_p90"),
        pl.col("y").quantile(0.99).alias("y_p99"),
        pl.col("y2h").mean().alias("share_2h"),
        pl.col("y4h").mean().alias("share_4h"),
        pl.col("y1h").sum().alias("n_1h"),
        pl.col("y2h").sum().alias("n_2h"),
        pl.col("y4h").sum().alias("n_4h"),
        pl.col("y12h").sum().alias("n_12h"),
    )
    .sort("grp")
)
print("\n--- same, FLIGHT_ID definition (only cells that differ from routing are of interest) ---")
print(
    tr.with_columns(pl.when(pl.col("unm_fid")).then(pl.lit("unmatched")).otherwise(pl.lit("matched")).alias("g2"))
    .group_by("g2")
    .agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y2h").sum().alias("n_2h"), pl.col("y4h").sum().alias("n_4h"))
    .sort("g2")
)

# ---------------------------------------------------------------------------------------------- C. fill rate
print("\n=== C. fill rate |BLOCK - SCHED| <= 10 s / <= 120 s per ADEP x group ===")
fill = (
    tr.group_by("ADEP_mvt", "grp")
    .agg(pl.len().alias("n"), pl.col("f10").mean().alias("f10"), pl.col("f120").mean().alias("f120"),
         pl.col("f10").sum().alias("n_f10"),
         pl.col("y").filter(pl.col("f10")).median().alias("y_med_f10"),
         pl.col("y2h").filter(pl.col("f10")).mean().alias("share_2h_f10"),
         (pl.col("f10") & pl.col("y2h")).sum().alias("n_f10_and_2h"))
    .sort("ADEP_mvt", "grp")
)
print(fill)
print("\n--- wide: matched vs unmatched f10 side by side ---")
print(
    fill.pivot(on="grp", index="ADEP_mvt", values=["f10", "n_f10_and_2h"])
    .sort("ADEP_mvt")
)

# ---------------------------------------------------------------------------------------------- D. actype null
print("\n=== D. AIRCRAFT_TYPE_mvt null rate per ADEP x group ===")
print(
    tr.group_by("ADEP_mvt", "grp")
    .agg(pl.len().alias("n"), pl.col("act_null").mean().alias("actype_null_rate"), pl.col("act_null").sum().alias("n_actype_null"))
    .sort("ADEP_mvt", "grp")
    .pivot(on="grp", index="ADEP_mvt", values=["actype_null_rate", "n_actype_null"])
    .sort("ADEP_mvt")
)
print("matched rows with actype null, all airports:", tr.filter(~pl.col("unm") & pl.col("act_null")).height)
print("unmatched rows with actype null, all airports:", tr.filter(pl.col("unm") & pl.col("act_null")).height,
      "of", tr.filter(pl.col("unm")).height)
print("\n--- matched actype-null rows: labels ---")
print(
    tr.filter(~pl.col("unm") & pl.col("act_null"))
    .group_by("ADEP_mvt")
    .agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").max().alias("y_max"),
         pl.col("y2h").sum().alias("n_2h"), (pl.col("ms") >= H2).sum().alias("n_ms2h"), pl.col("f10").mean().alias("f10"))
    .sort("n", descending=True)
)

# ---------------------------------------------------------------------------------------------- E. signature
print("\n=== E. the LIRF unmatched extreme signature (LIRF & actype null & ms >= 2 h) ===")
sig = pl.col("lirf") & pl.col("act_null") & (pl.col("ms") >= H2)


def dist(df: pl.DataFrame, name: str) -> pl.DataFrame:
    return df.select(
        pl.lit(name).alias("subset"),
        pl.len().alias("n"),
        pl.col("act_null").mean().alias("actype_null"),
        pl.col("f10").mean().alias("f10"),
        pl.col("y").quantile(0.1).alias("y_p10"),
        pl.col("y").median().alias("y_med"),
        pl.col("y").quantile(0.9).alias("y_p90"),
        pl.col("y").max().alias("y_max"),
        pl.col("ms").median().alias("ms_med"),
        pl.col("y2h").mean().alias("share_2h"),
        pl.col("y4h").mean().alias("share_4h"),
        pl.col("y2h").sum().alias("n_2h"),
        pl.col("y4h").sum().alias("n_4h"),
        pl.col("y12h").sum().alias("n_12h"),
        ((pl.col("y") - pl.col("ms")).abs() <= 60).mean().alias("share_y_eq_ms_60s"),
    )


print(
    pl.concat(
        [
            dist(tr.filter(pl.col("unm") & sig), "unmatched: LIRF & actype null & ms>=2h  (the regime)"),
            dist(tr.filter(~pl.col("unm") & sig), "matched:   LIRF & actype null & ms>=2h  (same signature)"),
            dist(tr.filter(~pl.col("unm") & pl.col("lirf") & (pl.col("ms") >= H2)), "matched:   LIRF & ms>=2h (actype present)"),
            dist(tr.filter(pl.col("unm") & pl.col("lirf") & (pl.col("ms") >= H2)), "unmatched: LIRF & ms>=2h (any actype)"),
            dist(tr.filter(~pl.col("unm") & pl.col("lirf") & (pl.col("ms") >= H2) & pl.col("f10")), "matched:   LIRF & ms>=2h & f10"),
            dist(tr.filter(~pl.col("unm") & pl.col("lirf") & (pl.col("ms") >= H2) & ~pl.col("f10")), "matched:   LIRF & ms>=2h & not f10"),
            dist(tr.filter(~pl.col("unm") & ~pl.col("lirf") & (pl.col("ms") >= H2)), "matched:   non-LIRF & ms>=2h"),
            dist(tr.filter(pl.col("unm") & ~pl.col("lirf") & (pl.col("ms") >= H2)), "unmatched: non-LIRF & ms>=2h"),
            dist(tr.filter(~pl.col("unm") & pl.col("lirf")), "matched:   LIRF all"),
            dist(tr.filter(pl.col("unm") & pl.col("lirf")), "unmatched: LIRF all"),
        ]
    )
)
print("\n--- matched LIRF & ms >= 2 h & f10 rows: BLOCK - AOBT_3 (s) quantiles and labels ---")
sub = tr.filter(~pl.col("unm") & pl.col("lirf") & (pl.col("ms") >= H2) & pl.col("f10"))
print(sub.select(
    pl.len().alias("n"),
    pl.col("block_minus_aobt3").quantile(0.1).alias("bma_p10"),
    pl.col("block_minus_aobt3").median().alias("bma_med"),
    pl.col("block_minus_aobt3").quantile(0.9).alias("bma_p90"),
    (pl.col("block_minus_aobt3").abs() > 3600).sum().alias("n_|BLK-AOBT|>1h"),
    pl.col("y").median().alias("y_med"), pl.col("y2h").sum().alias("n_2h"), pl.col("y4h").sum().alias("n_4h"),
))

# ---------------------------------------------------------------------------------------------- F. ms -> y
print("\n=== F. P(y >= 2 h | ms band) and y median, LIRF vs non-LIRF, matched vs unmatched ===")
band = (
    pl.when(pl.col("ms") < 0).then(pl.lit("a <0"))
    .when(pl.col("ms") < H1).then(pl.lit("b 0-1h"))
    .when(pl.col("ms") < H2).then(pl.lit("c 1-2h"))
    .when(pl.col("ms") < H4).then(pl.lit("d 2-4h"))
    .when(pl.col("ms") < H12).then(pl.lit("e 4-12h"))
    .when(pl.col("ms") < 72000).then(pl.lit("f 12-20h"))
    .otherwise(pl.lit("g >=20h"))
    .alias("ms_band")
)
print(
    tr.with_columns(band, pl.when(pl.col("lirf")).then(pl.lit("LIRF")).otherwise(pl.lit("non-LIRF")).alias("apt"))
    .group_by("apt", "grp", "ms_band")
    .agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").quantile(0.9).alias("y_p90"),
         pl.col("y2h").mean().alias("P(y>=2h)"), pl.col("y2h").sum().alias("n_2h"), pl.col("f10").mean().alias("f10"),
         (pl.col("y") / pl.col("ms")).median().alias("med(y/ms)"))
    .sort("apt", "ms_band", "grp")
)

# ---------------------------------------------------------------------------------------------- G. cols_b
print("\n=== G. cols_b (train.py prefix rule) and categoricals: non-null rate by group; ranking unmatched ===")
NUM_TYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32, pl.UInt8, pl.Int16)
BLANKED = {"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}
LABEL = "TAXITIME_SEC_mvt"
CANDIDATE_COLS = {"queue_ahead", "bank_position", "stand_rwy_share", "noise_control", "active_runways_60m",
                  "dep_arr_ratio_30m", "congestion_vs_usual_30m", "dest_distance_km", "planned_flight_s"}
B_PREFIXES = ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")
CATEGORICALS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
                "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_mvt", "ADES_REGION"]
schema = pl.scan_parquet(FT).collect_schema()
num_cols = [c for c, dt in schema.items()
            if c not in BLANKED | {"MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "_t"} and dt in NUM_TYPES]
visible = [c for c in num_cols if c not in (LABEL, "month") and c not in CANDIDATE_COLS]  # main.yaml: no excludes, no candidates
cols_b = [c for c in visible if not c.startswith(B_PREFIXES)]
cols_a_only = [c for c in visible if c.startswith(B_PREFIXES)]
print(f"numeric feature columns in frame: {len(num_cols)}; visible under configs/main.yaml: {len(visible)}; "
      f"cols_b: {len(cols_b)}; hidden from B by prefix: {len(cols_a_only)}")
print("hidden from model B by prefix:", cols_a_only)
print("Boolean columns in frame (not numeric -> never features):", [c for c, dt in schema.items() if dt == pl.Boolean])
print("fold-derived encodings (attached at fit time, not in the parquet): ref_p10, ref_p20, ref_med, ref_n, arh_med, arh_n, "
      "apt_med, ref_p10_fb, ref_missing -- keyed on (ADEP, STAND, RUNWAY), (ADEP, RUNWAY, hour_local), ADEP")

check_cols = cols_b + CATEGORICALS
nn_tr = (
    pl.scan_parquet(FT)
    .select(["mvt_minus_aobt3", *check_cols])
    .with_columns(pl.when(pl.col("mvt_minus_aobt3").is_null()).then(pl.lit("unmatched")).otherwise(pl.lit("matched")).alias("grp"))
    .group_by("grp")
    .agg([pl.col(c).is_not_null().mean().alias(c) for c in check_cols])
    .collect()
)
nn_rk = (
    pl.scan_parquet(FR)
    .select(["mvt_minus_aobt3", *check_cols])
    .filter(pl.col("mvt_minus_aobt3").is_null())
    .select([pl.col(c).is_not_null().mean().alias(c) for c in check_cols])
    .collect()
)
# constant-ness on unmatched training rows (a column that is non-null but single-valued carries nothing)
nuniq_unm = (
    pl.scan_parquet(FT)
    .select(["mvt_minus_aobt3", *check_cols])
    .filter(pl.col("mvt_minus_aobt3").is_null())
    .select([pl.col(c).n_unique().alias(c) for c in check_cols])
    .collect()
)
nuniq_mat = (
    pl.scan_parquet(FT)
    .select(["mvt_minus_aobt3", *check_cols])
    .filter(pl.col("mvt_minus_aobt3").is_not_null())
    .select([pl.col(c).n_unique().alias(c) for c in check_cols])
    .collect()
)
m = nn_tr.filter(pl.col("grp") == "matched").drop("grp").transpose(include_header=True, header_name="column", column_names=["nonnull_matched_tr"])
u = nn_tr.filter(pl.col("grp") == "unmatched").drop("grp").transpose(include_header=True, header_name="column", column_names=["nonnull_unmatched_tr"])
r = nn_rk.transpose(include_header=True, header_name="column", column_names=["nonnull_unmatched_rank"])
qu = nuniq_unm.transpose(include_header=True, header_name="column", column_names=["nuniq_unmatched_tr"])
qm = nuniq_mat.transpose(include_header=True, header_name="column", column_names=["nuniq_matched_tr"])
avail = (
    m.join(u, on="column").join(r, on="column").join(qu, on="column").join(qm, on="column")
    .with_columns(
        pl.when(pl.col("column").is_in(CATEGORICALS)).then(pl.lit("categorical")).otherwise(pl.lit("cols_b numeric")).alias("kind"),
        pl.when((pl.col("nonnull_matched_tr") >= 0.99) & (pl.col("nonnull_unmatched_tr") >= 0.99) & (pl.col("nuniq_unmatched_tr") > 1))
        .then(pl.lit("both >=99% & varies on unmatched"))
        .when((pl.col("nonnull_matched_tr") >= 0.99) & (pl.col("nonnull_unmatched_tr") >= 0.99))
        .then(pl.lit("both >=99% but CONSTANT on unmatched"))
        .when(pl.col("nonnull_unmatched_tr") < 0.01)
        .then(pl.lit("null on unmatched (<1%)"))
        .when(pl.col("nonnull_unmatched_tr") < pl.col("nonnull_matched_tr") - 0.05)
        .then(pl.lit("partly null on unmatched"))
        .otherwise(pl.lit("other"))
        .alias("class"),
    )
    .sort("kind", "class", "column")
)
print(avail)
print("\n--- class counts ---")
print(avail.group_by("kind", "class").agg(pl.len().alias("n_cols"), pl.col("column").sort().str.join(", ").alias("columns")).sort("kind", "class"))

# regime indicators inside cols_b: flag columns whose value differs between the two groups
print("\n--- G2. flag columns in cols_b: min/max per group (a column constant within each group but different "
      "across groups separates the regimes in one split) ---")
FLAGS = ["nm_matched", "eobt1_null", "lobt_null", "anchor_ok", "runway_null", "stand_null", "actype_mismatch", "diverted"]
fl = (
    pl.scan_parquet(FT)
    .select(["mvt_minus_aobt3", "FLIGHT_ID_mvt", *FLAGS])
    .with_columns(pl.col("mvt_minus_aobt3").is_null().alias("unm"))
)
print(
    fl.group_by("unm")
    .agg([pl.len().alias("n")] + [pl.col(c).min().alias(f"{c}_min") for c in FLAGS] + [pl.col(c).max().alias(f"{c}_max") for c in FLAGS])
    .sort("unm")
    .collect()
)
print(fl.filter(pl.col("unm")).group_by("nm_matched").agg(pl.len().alias("n"), pl.col("FLIGHT_ID_mvt").is_null().sum().alias("fid_null")).sort("nm_matched").collect())

# ---------------------------------------------------------------------------------------------- H. extra extremes
print("\n=== H. extra 'extreme' training examples pooling would add to model B, per airport ===")
ext = (
    tr.group_by("ADEP_mvt", "grp")
    .agg(pl.col("y1h").sum().alias("n_1h"), pl.col("y2h").sum().alias("n_2h"), pl.col("y4h").sum().alias("n_4h"), pl.col("y12h").sum().alias("n_12h"), pl.len().alias("n"))
    .pivot(on="grp", index="ADEP_mvt", values=["n", "n_1h", "n_2h", "n_4h", "n_12h"])
    .fill_null(0)
    .with_columns(
        (pl.col("n_2h_matched") / (pl.col("n_2h_unmatched") + pl.col("n_2h_matched"))).alias("matched_share_of_pooled_2h"),
        (pl.col("n_4h_matched") / (pl.col("n_4h_unmatched") + pl.col("n_4h_matched"))).alias("matched_share_of_pooled_4h"),
    )
    .sort("ADEP_mvt")
)
print(ext.select("ADEP_mvt", "n_unmatched", "n_matched", "n_1h_unmatched", "n_1h_matched", "n_2h_unmatched", "n_2h_matched",
                 "matched_share_of_pooled_2h", "n_4h_unmatched", "n_4h_matched", "matched_share_of_pooled_4h", "n_12h_unmatched", "n_12h_matched"))
print("\n--- totals ---")
print(tr.group_by("grp").agg(pl.len().alias("n"), pl.col("y1h").sum().alias("n_1h"), pl.col("y2h").sum().alias("n_2h"),
                              pl.col("y4h").sum().alias("n_4h"), pl.col("y12h").sum().alias("n_12h")).sort("grp"))
print("\n--- thresholds at which the pooled set holds ~2,000 rows (docs/README '33 -> ~2,000' claim) ---")
for t in (1800, 2700, 3600, 5400, 7200, 14400, 43200):
    a = tr.filter(pl.col("y") >= t)
    print(f"  y >= {t:>6d} s: all rows {a.height:>7d}  unmatched {a.filter(pl.col('unm')).height:>5d}  matched {a.filter(~pl.col('unm')).height:>7d}")

print("\n--- profile of the matched y >= 2 h rows (what pooling adds at the 2 h tail) ---")
m2 = tr.filter(~pl.col("unm") & pl.col("y2h"))
print(
    m2.group_by("ADEP_mvt")
    .agg(pl.len().alias("n"), pl.col("act_null").sum().alias("n_actype_null"), pl.col("f10").mean().alias("f10"),
         (pl.col("ms") >= H2).mean().alias("share_ms>=2h"), ((pl.col("y") - pl.col("ms")).abs() <= 60).mean().alias("y_eq_ms_60s"),
         (pl.col("block_minus_aobt3").abs() > 3600).mean().alias("share_|BLK-AOBT|>1h"),
         pl.col("y").median().alias("y_med"), pl.col("ms").median().alias("ms_med"), pl.col("y4h").sum().alias("n_4h"))
    .sort("n", descending=True)
)
print("\n--- profile of the unmatched y >= 2 h rows (the target regime), for comparison ---")
u2 = tr.filter(pl.col("unm") & pl.col("y2h"))
print(
    u2.group_by("ADEP_mvt")
    .agg(pl.len().alias("n"), pl.col("act_null").sum().alias("n_actype_null"), pl.col("f10").mean().alias("f10"),
         (pl.col("ms") >= H2).mean().alias("share_ms>=2h"), ((pl.col("y") - pl.col("ms")).abs() <= 60).mean().alias("y_eq_ms_60s"),
         pl.col("y").median().alias("y_med"), pl.col("ms").median().alias("ms_med"), pl.col("y4h").sum().alias("n_4h"))
    .sort("n", descending=True)
)

print("\n--- matched y >= 4 h rows, listed (all of them) ---")
print(
    tr.filter(~pl.col("unm") & pl.col("y4h"))
    .select("MVT_ID_mvt", "ADEP_mvt", "month", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "y", "ms", "bs", "block_minus_aobt3", "f10")
    .sort("y", descending=True)
)

# pooled-set composition seen by a shared model B under main.yaml (no weights in _fit): unmatched share
print("\n--- pooled composition: unmatched share of rows / of y>=2h rows / of y>=4h rows ---")
print(tr.select(
    pl.col("unm").mean().alias("unm_share_rows"),
    ((pl.col("unm") & pl.col("y2h")).sum() / pl.col("y2h").sum()).alias("unm_share_of_2h_rows"),
    ((pl.col("unm") & pl.col("y4h")).sum() / pl.col("y4h").sum()).alias("unm_share_of_4h_rows"),
    ((pl.col("unm") & pl.col("y12h")).sum() / pl.col("y12h").sum()).alias("unm_share_of_12h_rows"),
))
print("\ndone")
