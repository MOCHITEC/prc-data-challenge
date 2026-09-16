"""Item 8 `cond-mean-vs-median` -- independent re-derivation (no training).

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/cond-mean-vs-median.py
Inputs: data/work/feat_train.parquet (9 columns), data/work/feat_rank.parquet (7 columns),
        experiments/oof_main-48c19bc830.parquet, outputs/main.parquet (v7), outputs/det_a.parquet (v8),
        outputs/ab_split.parquet, scratchpad uploaded v4.parquet (optional).
Everything is column-selected through scan_parquet; no model is trained.

Question: on LIRF unmatched rows binned by ms = MVT - SCHED (2-4 h, 4-12 h, >= 12 h), is the label
two-valued, what RMSE do constant rules give (predict ms / bin mean / bin median / current OOF), does the
in-sample ranking of rules survive leave-one-month-out, and what does each rule imply on the 11 ranking
LIRF rows with ms >= 12 h under the training mixture proportions?
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
import math
import polars as pl

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
pl.Config.set_fmt_str_lengths(40); pl.Config.set_fmt_float("full")

B = "/home/user/prc-data-challenge/"
UP = "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/uploaded/"
H2, H4, H12, H20, DAY = 7200, 14400, 43200, 72000, 86400
N_RANK = 344841
BINS = [("2-4h", H2, H4), ("4-12h", H4, H12), (">=12h", H12, 10**12)]


def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


def hdr(t):
    print("\n" + "=" * 110 + "\n" + t + "\n" + "=" * 110)


def rmse(err):
    err = [float(e) for e in err]
    return math.sqrt(sum(e * e for e in err) / len(err)) if err else float("nan")


def binname(ms):
    for name, lo, hi in BINS:
        if lo <= ms < hi:
            return name
    return None


# ----------------------------------------------------------------------------------------------
# S0. LIRF unmatched training rows + OOF join
# ----------------------------------------------------------------------------------------------
tr = (pl.scan_parquet(B + "data/work/feat_train.parquet")
      .select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
               "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched"])
      .filter(pl.col("ADEP_mvt") == "LIRF")
      .with_columns([
          pl.col("AOBT_3_flt").is_null().alias("unm"),
          pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
          sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
          sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("blk_minus_sched"),
          pl.col("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
          pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
      ])
      .with_columns([(pl.col("y") - pl.col("ms")).alias("res"),
                     pl.col("mvt_minus_sched").cast(pl.Int64).alias("ms_feat")])
      .collect())

oof = pl.scan_parquet(B + "experiments/oof_main-48c19bc830.parquet").filter(pl.col("oof").is_not_null()).collect()
lu = (tr.filter(pl.col("unm")).join(oof, on="MVT_ID_mvt", how="left")
      .with_columns([
          pl.when(pl.col("ms") >= H12).then(pl.lit(">=12h"))
            .when(pl.col("ms") >= H4).then(pl.lit("4-12h"))
            .when(pl.col("ms") >= H2).then(pl.lit("2-4h")).otherwise(pl.lit("<2h")).alias("bin"),
          (pl.col("res").abs() <= 60).alias("y_eq_ms"),          # sched-filled (label == ms within 60 s)
          (pl.col("y") >= H20).alias("rollover"),                 # y > 20 h (day rollover)
      ])
      .with_columns(pl.when(pl.col("y_eq_ms")).then(pl.lit("filled(y=ms)"))
                      .when(pl.col("rollover")).then(pl.lit("rollover(y>20h)"))
                      .otherwise(pl.lit("other")).alias("cls")))

hdr("S0. LIRF unmatched training rows: identity checks")
print(f"LIRF departures {tr.height}; LIRF unmatched (AOBT_3 null) {tr['unm'].sum()}; (FLIGHT_ID null) {tr['unm_fid'].sum()}; "
      f"AOBT-null & FID-present {(tr['unm'] & ~tr['unm_fid']).sum()}")
print(f"ms (MVT-SCHED) == feature mvt_minus_sched on all LIRF unmatched rows: {(lu['ms'] == lu['ms_feat']).all()}; "
      f"y == MVT-BLOCK identity implies res = y-ms = -(BLOCK-SCHED): {(lu['res'] == -lu['blk_minus_sched']).all()}")
print(f"OOF file rows {oof.height}; LIRF unmatched rows with OOF {lu['oof'].is_not_null().sum()} "
      f"(months {sorted(lu.filter(pl.col('oof').is_not_null())['mo'].unique().to_list())})")
print("filled(y=ms) is |y-ms| <= 60 s; rollover is y >= 72,000 s; 'other' is the remainder (mostly ordinary taxi labels).")

# ----------------------------------------------------------------------------------------------
# S1. bin composition (routing definition; also M's strict '> 43,200' variant)
# ----------------------------------------------------------------------------------------------
hdr("S1. LIRF unmatched rows per ms bin: composition")
comp = (lu.group_by("bin").agg([
    pl.len().alias("n"),
    pl.col("y_eq_ms").sum().alias("n_filled"), (pl.col("cls") == "rollover(y>20h)").sum().alias("n_rollover"),
    (pl.col("cls") == "other").sum().alias("n_other"),
    pl.col("y_eq_ms").mean().round(4).alias("p_filled"), (pl.col("cls") == "rollover(y>20h)").mean().round(4).alias("p_rollover"),
    (pl.col("rollover") & pl.col("y_eq_ms")).sum().alias("n_filled_and_y>=20h"),
    pl.col("ms").min().alias("ms_min"), pl.col("ms").median().alias("ms_med"), pl.col("ms").max().alias("ms_max"),
    pl.col("y").mean().round(0).alias("y_mean"), pl.col("y").median().alias("y_med"),
    pl.col("res").mean().round(0).alias("res_mean"), pl.col("res").median().alias("res_med"),
    pl.col("oof").is_not_null().sum().alias("n_oof"),
]).sort("ms_min"))
print(comp)
strict = lu.filter(pl.col("ms") > H12)
print(f"M's bin (43,200, inf) strict: n {strict.height}, filled {strict['y_eq_ms'].sum()} ({strict['y_eq_ms'].mean():.4f}), "
      f"rollover {(strict['cls']=='rollover(y>20h)').sum()}, other {(strict['cls']=='other').sum()}")
print("per-month composition, bin >=12h:")
print(lu.filter(pl.col("bin") == ">=12h").group_by("mo").agg([pl.len().alias("n"), pl.col("y_eq_ms").sum().alias("filled"),
                                                             (pl.col("cls") == "rollover(y>20h)").sum().alias("rollover")]).sort("mo"))
print("per-month composition, bin 4-12h:")
print(lu.filter(pl.col("bin") == "4-12h").group_by("mo").agg([pl.len().alias("n"), pl.col("y_eq_ms").sum().alias("filled"),
                                                             (pl.col("cls") == "rollover(y>20h)").sum().alias("rollover"),
                                                             (pl.col("cls") == "other").sum().alias("other")]).sort("mo"))

# ----------------------------------------------------------------------------------------------
# S2. full histograms of y and y - ms per bin (bimodality check)
# ----------------------------------------------------------------------------------------------
hdr("S2. histograms per bin (counts). y in 1-hour cells; y-ms in fixed cells")
Y_EDGES = [-10**9, 0, 1800, 3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600, 43200,
           50400, 57600, 64800, 72000, 79200, 86400, 88200, 90000, 10**9]
Y_LAB = ["<0", "[0,0.5h)", "[0.5,1h)", "[1,2h)", "[2,3h)", "[3,4h)", "[4,5h)", "[5,6h)", "[6,7h)", "[7,8h)", "[8,9h)",
         "[9,10h)", "[10,11h)", "[11,12h)", "[12,14h)", "[14,16h)", "[16,18h)", "[18,20h)", "[20,22h)", "[22,24h)",
         "[24h,24.5h)", "[24.5h,25h)", ">=25h"]
R_EDGES = [-10**9, -86400, -72000, -43200, -14400, -7200, -1800, -60, 61, 1801, 7201, 14401, 43201, 72001, 86401, 10**9]
R_LAB = ["<-24h", "[-24h,-20h)", "[-20h,-12h)", "[-12h,-4h)", "[-4h,-2h)", "[-2h,-30m)", "[-30m,-60s)", "[-60s,+60s]",
         "(60s,30m]", "(30m,2h]", "(2h,4h]", "(4h,12h]", "(12h,20h]", "(20h,24h]", ">24h"]


def hist(vals, edges, labels):
    out = []
    for i in range(len(edges) - 1):
        c = sum(1 for v in vals if edges[i] <= v < edges[i + 1])
        if c:
            out.append((labels[i], c))
    return out


for name, lo, hi in BINS:
    d = lu.filter(pl.col("bin") == name)
    print(f"\n--- bin {name}: n={d.height}, ms range [{d['ms'].min()}, {d['ms'].max()}] ---")
    print("  y      :", hist(d["y"].to_list(), Y_EDGES, Y_LAB))
    print("  y - ms :", hist(d["res"].to_list(), R_EDGES, R_LAB))
    # finer view of the two candidate modes
    r = d["res"].to_list(); y = d["y"].to_list()
    print(f"  |y-ms|<=10 s: {sum(1 for v in r if abs(v) <= 10)}; <=60 s: {sum(1 for v in r if abs(v) <= 60)}; "
          f"<=1800 s: {sum(1 for v in r if abs(v) <= 1800)}; y in [86400+300, 86400+2400]: "
          f"{sum(1 for v in y if 86700 <= v <= 88800)}; y < 3600 (ordinary taxi): {sum(1 for v in y if v < 3600)}; "
          f"y in [3600, ms-1800) i.e. neither mode: {sum(1 for v, m in zip(y, d['ms'].to_list()) if 3600 <= v < m - 1800)}")

hdr("S2b. bin >=12h: every row (sorted by ms)")
print(lu.filter(pl.col("bin") == ">=12h").sort("ms").select(
    ["MVT_ID_mvt", "mo", "ms", "y", "res", "cls", pl.col("oof").round(0)]))
hdr("S2c. bin 4-12h: the non-filled rows")
print(lu.filter((pl.col("bin") == "4-12h") & ~pl.col("y_eq_ms")).sort("ms").select(
    ["MVT_ID_mvt", "mo", "ms", "y", "res", "cls", pl.col("oof").round(0)]))

# ----------------------------------------------------------------------------------------------
# S3. RMSE of constant rules per bin, in-sample (all rows) and on the OOF subset
# ----------------------------------------------------------------------------------------------
hdr("S3. RMSE per bin of constant rules (in-sample statistics). 'const' = one number per bin; 'ms+' = ms plus a bin offset")


def rule_table(d, stats_src, label):
    """d: rows to score; stats_src: rows the bin statistics are computed from."""
    y = d["y"].to_list(); ms = d["ms"].to_list()
    mean_y = stats_src["y"].mean(); med_y = stats_src["y"].median()
    mean_r = stats_src["res"].mean(); med_r = stats_src["res"].median()
    rows = {
        "pred = ms": rmse([yy - mm for yy, mm in zip(y, ms)]),
        "pred = const bin mean(y)": rmse([yy - mean_y for yy in y]),
        "pred = const bin median(y)": rmse([yy - med_y for yy in y]),
        "pred = ms + mean(y-ms)": rmse([yy - (mm + mean_r) for yy, mm in zip(y, ms)]),
        "pred = ms + median(y-ms)": rmse([yy - (mm + med_r) for yy, mm in zip(y, ms)]),
    }
    if d["oof"].is_not_null().all():
        rows["pred = OOF main-48c19bc830"] = rmse([yy - o for yy, o in zip(y, d["oof"].to_list())])
    print(f"  [{label}] n={d.height}; stats: mean(y)={mean_y:.0f}, median(y)={med_y:.0f}, mean(y-ms)={mean_r:.0f}, median(y-ms)={med_r:.0f}")
    for k, v in rows.items():
        print(f"      {k:32s} RMSE {v:10.1f}")
    return rows


insample = {}
for name, lo, hi in BINS:
    d = lu.filter(pl.col("bin") == name)
    print(f"\n--- bin {name} ---")
    insample[name] = rule_table(d, d, "all rows, stats from all rows (in-sample)")
    do = d.filter(pl.col("oof").is_not_null())
    rule_table(do, d, "OOF rows only (months 1,7,11,12), stats from all rows (in-sample)")

# ----------------------------------------------------------------------------------------------
# S4. leave-one-month-out: statistics from the other 11 months, scored on the held-out month
# ----------------------------------------------------------------------------------------------
hdr("S4. leave-one-month-out (LOMO) RMSE per bin; OOF and 'pred = ms' need no statistics and are unchanged")


def lomo(d, groups):
    """groups: list of lists of months to hold out together. Returns dict rule -> RMSE over all held-out rows."""
    err = {k: [] for k in ["pred = ms", "pred = const bin mean(y)", "pred = const bin median(y)",
                           "pred = ms + mean(y-ms)", "pred = ms + median(y-ms)", "pred = OOF main-48c19bc830"]}
    per_fold = []
    for g in groups:
        test = d.filter(pl.col("mo").is_in(g)); train = d.filter(~pl.col("mo").is_in(g))
        if test.height == 0:
            continue
        if train.height == 0:
            mean_y = med_y = mean_r = med_r = float("nan")
        else:
            mean_y, med_y, mean_r, med_r = train["y"].mean(), train["y"].median(), train["res"].mean(), train["res"].median()
        for yy, mm, o in zip(test["y"].to_list(), test["ms"].to_list(), test["oof"].to_list()):
            err["pred = ms"].append(yy - mm)
            err["pred = const bin mean(y)"].append(yy - mean_y)
            err["pred = const bin median(y)"].append(yy - med_y)
            err["pred = ms + mean(y-ms)"].append(yy - (mm + mean_r))
            err["pred = ms + median(y-ms)"].append(yy - (mm + med_r))
            if o is not None:
                err["pred = OOF main-48c19bc830"].append(yy - o)
        per_fold.append((g, test.height, train.height, mean_y, med_y, mean_r, med_r))
    return {k: rmse(v) for k, v in err.items() if v}, per_fold


for name, lo, hi in BINS:
    d = lu.filter(pl.col("bin") == name)
    print(f"\n--- bin {name}, n={d.height} (OOF rows {d['oof'].is_not_null().sum()}) ---")
    res_all, folds = lomo(d, [[m] for m in range(1, 13)])
    print("  LOMO (12 folds), all rows:")
    for k, v in res_all.items():
        print(f"      {k:32s} RMSE {v:10.1f}" + ("   [OOF rows only]" if "OOF" in k else ""))
    print("  per held-out month: (months, n_test, n_stats, mean_y, median_y, mean_res, median_res)")
    for f in folds:
        print("      ", f[0], f[1], f[2], *(round(x) if x == x else x for x in f[3:]))
    # same rules scored only on the OOF rows (so that all six rules share one denominator)
    do = d.filter(pl.col("oof").is_not_null())
    # LOMO statistics come from all other months of d (not only the OOF months); scored on OOF rows only:
    err = {k: [] for k in ["pred = ms", "pred = const bin mean(y)", "pred = const bin median(y)",
                           "pred = ms + mean(y-ms)", "pred = ms + median(y-ms)", "pred = OOF main-48c19bc830"]}
    for m in sorted(do["mo"].unique().to_list()):
        test = do.filter(pl.col("mo") == m); train = d.filter(pl.col("mo") != m)
        mean_y, med_y, mean_r, med_r = train["y"].mean(), train["y"].median(), train["res"].mean(), train["res"].median()
        for yy, mm, o in zip(test["y"].to_list(), test["ms"].to_list(), test["oof"].to_list()):
            err["pred = ms"].append(yy - mm); err["pred = const bin mean(y)"].append(yy - mean_y)
            err["pred = const bin median(y)"].append(yy - med_y); err["pred = ms + mean(y-ms)"].append(yy - (mm + mean_r))
            err["pred = ms + median(y-ms)"].append(yy - (mm + med_r)); err["pred = OOF main-48c19bc830"].append(yy - o)
    print(f"  LOMO scored on the {do.height} OOF rows only (common denominator with the model):")
    for k, v in err.items():
        print(f"      {k:32s} RMSE {rmse(v):10.1f}")
    # CV-fold-shaped LOMO (jan / jul / forward = Nov+Dec) for comparability with M's forward-fold numbers
    res_cv, folds_cv = lomo(d, [[1], [7], [11, 12]])
    print("  fold-shaped hold-outs (jan / jul / forward=Nov+Dec), statistics from the other months, all rows in those months:")
    for k, v in res_cv.items():
        print(f"      {k:32s} RMSE {v:10.1f}")
    for f in folds_cv:
        print("      ", f[0], "n_test", f[1], "n_stats", f[2], "mean_y", round(f[3]), "median_y", round(f[4]), "mean_res", round(f[5]), "median_res", round(f[6]))

hdr("S4b. M's forward-fold statement: LIRF unmatched, ms >= 4 h, months 11+12 (n, mean y, mean model, bin-mean from other months, mean ms)")
fw = lu.filter((pl.col("ms") >= H4) & pl.col("mo").is_in([11, 12]))
oth = lu.filter((pl.col("ms") >= H4) & ~pl.col("mo").is_in([11, 12]))
print(f"n={fw.height}, all have OOF: {fw['oof'].is_not_null().all()}; mean y {fw['y'].mean():.0f}; mean OOF {fw['oof'].mean():.0f}; "
      f"bin-mean(y) from other months (single bin ms>=4h) {oth['y'].mean():.0f}; mean ms {fw['ms'].mean():.0f}")
for name in ["4-12h", ">=12h"]:
    a = fw.filter(pl.col("bin") == name); b = oth.filter(pl.col("bin") == name)
    print(f"   sub-bin {name}: n={a.height}, mean y {a['y'].mean():.0f}, mean OOF {a['oof'].mean():.0f}, other-month bin mean {b['y'].mean():.0f}, mean ms {a['ms'].mean():.0f}")
print(f"RMSE on these {fw.height} rows: OOF {rmse((fw['y']-fw['oof']).to_list()):.0f}; ms {rmse((fw['y']-fw['ms']).to_list()):.0f}; "
      f"const other-month mean {rmse((fw['y']-oth['y'].mean()).to_list()):.0f}; ms+other-month mean res {rmse((fw['y']-fw['ms']-oth['res'].mean()).to_list()):.0f}")

# ----------------------------------------------------------------------------------------------
# S5. analytic two-point mixture (M's [supposed] numbers) and the empirical version
# ----------------------------------------------------------------------------------------------
hdr("S5. two-point mixture arithmetic: y = m w.p. pi, y = R (rollover) w.p. 1-pi")
b12 = lu.filter(pl.col("bin") == ">=12h")
pi = b12["y_eq_ms"].mean()
roll = b12.filter(pl.col("cls") == "rollover(y>20h)")["y"].to_list()
other12 = b12.filter(pl.col("cls") == "other")["y"].to_list()
R_mean = sum(roll) / len(roll); R_sd = math.sqrt(sum((r - R_mean) ** 2 for r in roll) / len(roll))
print(f"bin >=12h: pi(filled) = {pi:.4f} ({b12['y_eq_ms'].sum()}/{b12.height}); rollover rows {len(roll)}: y mean {R_mean:.0f}, sd {R_sd:.0f}, "
      f"min {min(roll)}, max {max(roll)}; 'other' rows {len(other12)}: {other12}")
print("Analytic (R treated as the constant R_mean), for a row with ms = m:")
print("   RMSE(pred=m)            = sqrt(1-pi) * (R-m)")
print("   RMSE(pred=mixture mean) = sqrt(pi(1-pi)) * (R-m)   [mixture mean = pi*m + (1-pi)*R = m + (1-pi)(R-m)]")
print("   RMSE(pred=c)            = sqrt(pi (m-c)^2 + (1-pi)(R-c)^2)")
for m in [47337, 50000, 60000, 80000, 100000, 111654]:
    mix = pi * m + (1 - pi) * R_mean
    r_ms = math.sqrt(1 - pi) * abs(R_mean - m); r_mix = math.sqrt(pi * (1 - pi)) * abs(R_mean - m)
    r35 = math.sqrt(pi * (m - 35000) ** 2 + (1 - pi) * (R_mean - 35000) ** 2)
    r_med = r_ms  # median of the two-point law with pi>0.5 is m
    print(f"   m={m:7d}: mixture mean {mix:8.0f} (= m + {mix-m:6.0f}); RMSE pred=m {r_ms:8.0f}; pred=mean {r_mix:8.0f}; pred=35,000 plateau {r35:8.0f}; "
          f"median-optimal pred = m (pi>0.5): RMSE {r_med:8.0f}")
print(f"M's [supposed] line (m = 60k, pi = 0.62, R ~ 87k): mixture mean ~70k, RMSE ~13k (mean) vs ~16.6k (ms) -- recomputed above with pi={pi:.3f}, R={R_mean:.0f}.")
print("Analytic lines above treat R as a constant; the empirical rollover spread (sd %.0f) adds in quadrature: sqrt(RMSE^2 + (1-pi) sd^2)." % R_sd)

# ----------------------------------------------------------------------------------------------
# S6. the 11 ranking LIRF unmatched rows with ms >= 12 h: expected SSE per rule under training proportions
# ----------------------------------------------------------------------------------------------
hdr("S6. ranking LIRF unmatched rows with ms >= 12 h: predictions on file and expected SSE per rule")
rk = (pl.scan_parquet(B + "data/work/feat_rank.parquet")
      .select(["MVT_ID_mvt", "ADEP_mvt", "AOBT_3_flt", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
      .filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
      .with_columns([sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")])
      .collect())
subs = {"v7(outputs/main)": B + "outputs/main.parquet", "v8(det_a)": B + "outputs/det_a.parquet",
        "ab_split": B + "outputs/ab_split.parquet", "v2": B + "outputs/v2_two_clock.parquet"}
if os.path.exists(UP + "v4.parquet"):
    subs["v4(uploaded)"] = UP + "v4.parquet"
if os.path.exists(UP + "v3.parquet"):
    subs["v3(uploaded)"] = UP + "v3.parquet"
for k, p in subs.items():
    rk = rk.join(pl.scan_parquet(p).select(["MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").alias(k)]).collect(), on="MVT_ID_mvt", how="left")
r12 = rk.filter(pl.col("ms") >= H12).sort("ms")
print(f"ranking LIRF unmatched: {rk.height} rows; ms >= 12 h: {r12.height}; ms >= 4 h: {(rk['ms'] >= H4).sum()}; ms >= 2 h: {(rk['ms'] >= H2).sum()}")
print(r12.select(["MVT_ID_mvt", "mo", "RUNWAY_mvt", "ms"] + list(subs.keys())))

# rules evaluated on ranking rows; bin statistics from ALL training rows of the bin (that is what a deployed rule would use)
stats = {name: (lu.filter(pl.col("bin") == name)["y"].mean(), lu.filter(pl.col("bin") == name)["y"].median(),
                lu.filter(pl.col("bin") == name)["res"].mean(), lu.filter(pl.col("bin") == name)["res"].median())
         for name, _, _ in BINS}


def expected_sse(ms_vals, preds, pi_, comp_vals):
    """Mixture: y = ms w.p. pi_, else y drawn uniformly from the empirical list comp_vals (training non-filled labels)."""
    tot = 0.0
    for m, p in zip(ms_vals, preds):
        e_other = sum((c - p) ** 2 for c in comp_vals) / len(comp_vals)
        tot += pi_ * (m - p) ** 2 + (1 - pi_) * e_other
    return tot


def mixture_table(rows, name, pi_, comp_vals, label):
    ms_vals = rows["ms"].to_list(); n = len(ms_vals)
    mean_y, med_y, mean_r, med_r = stats[name]
    comp_mean = sum(comp_vals) / len(comp_vals)
    rules = {
        "pred = ms": ms_vals,
        "pred = const bin mean(y)": [mean_y] * n,
        "pred = const bin median(y)": [med_y] * n,
        "pred = ms + mean(y-ms)": [m + mean_r for m in ms_vals],
        "pred = ms + median(y-ms)": [m + med_r for m in ms_vals],
        "pred = per-row mixture mean pi*ms+(1-pi)*E[R] (oracle-pi)": [pi_ * m + (1 - pi_) * comp_mean for m in ms_vals],
    }
    for k in subs:
        rules[f"pred = {k}"] = [float(v) for v in rows[k].to_list()]
    print(f"\n  [{label}] n={n}, pi={pi_:.4f}, non-filled component: {len(comp_vals)} training values, mean {comp_mean:.0f}")
    print(f"  bin statistics used: mean(y) {mean_y:.0f}, median(y) {med_y:.0f}, mean(y-ms) {mean_r:.0f}, median(y-ms) {med_r:.0f}")
    print(f"  {'rule':62s} {'E[SSE]':>14s} {'E[RMSE] on rows':>16s} {'E[SSE]/N_rank':>14s} {'dLB@305.03':>11s} {'dLB@343.80':>11s}")
    out = {}
    base = None
    for k, p in rules.items():
        s = expected_sse(ms_vals, p, pi_, comp_vals)
        out[k] = s
    ref = out["pred = ms"]
    for k, s in out.items():
        d = (s - ref) / N_RANK  # change in LB MSE relative to the 'pred = ms' rule
        print(f"  {k:62s} {s:14.4e} {math.sqrt(s/n):16.0f} {s/N_RANK:14.1f} {math.sqrt(305.03**2 + d) - 305.03:+11.2f} {math.sqrt(343.80**2 + d) - 343.80:+11.2f}")
    print("  (dLB columns: change in LB RMSE relative to the 'pred = ms' rule when the LB RMSE without this change is 305.03 / 343.80; "
          "positive = worse; all other rows assumed unchanged)")
    return out


comp12 = b12.filter(~pl.col("y_eq_ms"))["y"].to_list()           # every non-filled training label in the bin (rollover + other)
res_main = mixture_table(r12, ">=12h", pi, comp12, "11 ranking rows, training proportions of bin >=12h")
print("\n  sensitivity to pi (same rules, same rollover component):")
for pi_s in [0.0, 0.25, 0.5, 0.621, 0.75, 0.9, 1.0]:
    ms_vals = r12["ms"].to_list(); n = len(ms_vals); mean_y, med_y, mean_r, med_r = stats[">=12h"]
    cm = sum(comp12) / len(comp12)
    line = f"   pi={pi_s:5.3f}: "
    for k, p in [("ms", ms_vals), ("const mean", [mean_y] * n), ("const median", [med_y] * n), ("ms+mean res", [m + mean_r for m in ms_vals]),
                 ("mix-mean(pi)", [pi_s * m + (1 - pi_s) * cm for m in ms_vals])] + [(k, [float(v) for v in r12[k].to_list()]) for k in subs]:
        line += f"{k} {math.sqrt(expected_sse(ms_vals, p, pi_s, comp12)/n):7.0f} | "
    print(line)

hdr("S6b. v8 - v4 on the 11 rows under the two-valued label assumption, against the LB unmatched-side gap (v9 identity)")
LB_V4, LB_V8, LB_V9 = 305.027, 343.7964, 306.4555   # experiments/submissions.jsonl
gap_total = (LB_V8 ** 2 - LB_V4 ** 2) * N_RANK
gap_unm = (LB_V8 ** 2 - LB_V9 ** 2) * N_RANK        # v9 = v8 on matched rows, v4 on unmatched rows (sibling item 3 verified)
print(f"LB SSE(v8)-SSE(v4) = {gap_total:.4e}; unmatched-side part SSE(v8)-SSE(v9) = {gap_unm:.4e}; matched-side part {gap_total-gap_unm:.4e}")
if "v4(uploaded)" in subs:
    v4p = [float(v) for v in r12["v4(uploaded)"].to_list()]; v8p = [float(v) for v in r12["v8(det_a)"].to_list()]; msv = r12["ms"].to_list()
    R_lo, R_hi = min(roll), max(roll)
    tot_min = tot_max = tot_fill = tot_roll = 0.0
    print(f"  per row: gap if y = ms; gap if y = R (R in [{R_lo}, {R_hi}], shown at R_mean {R_mean:.0f}); gap = (v8-y)^2 - (v4-y)^2")
    for m, a, b in zip(msv, v4p, v8p):
        g_fill = (b - m) ** 2 - (a - m) ** 2
        g_roll = [(b - R) ** 2 - (a - R) ** 2 for R in (R_lo, R_mean, R_hi)]
        tot_fill += g_fill; tot_roll += g_roll[1]
        tot_min += min(g_fill, min(g_roll)); tot_max += max(g_fill, max(g_roll))
        print(f"    ms {m:7d}: v4 {a:7.0f} v8 {b:7.0f}  gap|y=ms {g_fill:+.3e}  gap|y=R {g_roll[1]:+.3e} (range {min(g_roll):+.3e} .. {max(g_roll):+.3e})")
    print(f"  sum if all 11 filled: {tot_fill:.4e} ({tot_fill/gap_unm*100:.0f} % of the LB unmatched gap); all 11 rollover: {tot_roll:.4e} ({tot_roll/gap_unm*100:.0f} %)")
    print(f"  expectation at training pi={pi:.4f}: {pi*tot_fill+(1-pi)*tot_roll:.4e} ({(pi*tot_fill+(1-pi)*tot_roll)/gap_unm*100:.0f} %)")
    print(f"  minimum over all 2^11 assignments (each row at its most v8-favourable outcome and R): {tot_min:.4e} ({tot_min/gap_unm*100:.0f} %); maximum {tot_max:.4e} ({tot_max/gap_unm*100:.0f} %)")

# extension: the same arithmetic for the ranking rows in the 4-12 h bin (training proportions of that bin)
b412 = lu.filter(pl.col("bin") == "4-12h")
r412 = rk.filter((pl.col("ms") >= H4) & (pl.col("ms") < H12)).sort("ms")
mixture_table(r412, "4-12h", b412["y_eq_ms"].mean(), b412.filter(~pl.col("y_eq_ms"))["y"].to_list(),
              f"{r412.height} ranking rows in bin 4-12h, training proportions of that bin (extension, same assumptions)")

hdr("S7. OOF rows in bin >=12h: current prediction vs each rule (the rows the CV actually scores)")
o12 = b12.filter(pl.col("oof").is_not_null()).sort("ms")
mean_y, med_y, mean_r, med_r = stats[">=12h"]
print(o12.select(["mo", "ms", "y", "cls", pl.col("oof").round(0),
                  (pl.col("ms") + mean_r).round(0).alias("ms+mean_res"), pl.lit(round(mean_y)).alias("const_mean"),
                  (pl.col("y") - pl.col("oof")).round(0).alias("err_oof"), (pl.col("y") - pl.col("ms")).alias("err_ms"),
                  (pl.col("y") - pl.col("ms") - mean_r).round(0).alias("err_ms+mean")]))
print("done")
