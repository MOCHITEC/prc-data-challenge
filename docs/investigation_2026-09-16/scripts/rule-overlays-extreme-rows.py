"""Verification of investigation item 12 `rule-overlays-extreme-rows` (plan.md §1, source E H5).

Rules under test (as quoted from external.md A3), with the local reading used here:

  Z  zestful-fountain / Phoenix-Ops v11 "clock overlay":
        trigger  AOBT_3 - LOBT > 7200  and  both proxies in [-7200, 172800]
        predict  max(0, MVT - LOBT)
     "both proxies" is read as  MVT - AOBT_3  and  MVT - LOBT  (the two NM clock proxies).
     Needs AOBT_3 and LOBT -> can only fire on rows with a non-null AOBT_3, i.e. on model-A (matched) rows.

  O  OpenAir Stage5:  "OPDI ground time >= 7200 s -> shrink 50 % toward median"
     OPDI data is not in this repository (grep: only the hypothesis docs mention it), so the trigger is
     tested through local analogues of "ground time":
        O-a  MVT - AOBT_3 >= 7200   (matched rows only)
        O-l  MVT - LOBT   >= 7200   (rows with LOBT)
        O-s  MVT - SCHED  >= 7200   (all rows; the only one available on every ranking row)
     median = per-airport median of y computed on training months NOT in the OOF file (2,3,4,5,6,8,9,10),
     and, as a second variant, the global median from the same months.
     shrink:  pred' = 0.5 * pred + 0.5 * median.

  S  sergiuv11 "LIRF overlay at 14,400 s" -- the source gives no formula. Readings tested:
        S-cap   LIRF rows: pred' = min(pred, 14,400)
        S-ms    LIRF rows with MVT - SCHED >= 14,400: pred' = MVT - SCHED
        S-msU   as S-ms but unmatched LIRF rows only
        S-floor LIRF rows with MVT - SCHED >= 14,400: pred' = max(pred, 14,400)

For every trigger: rows selected in training (matched / unmatched, by airport); on the OOF-scored rows the
RMSE of (a) current OOF prediction, (b) rule prediction, (c) MVT - AOBT_3, (d) MVT - SCHED; the
counterfactual total OOF RMSE with the rule applied. Ranking: rows selected in feat_rank and what
outputs/main.parquet (v7) and outputs/probe_unmatched_from_v4.parquet (v9) predict there.

No training. POLARS_MAX_THREADS=1. Lazy scans, only the needed columns.
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
PROBE = f"{ROOT}/outputs/probe_unmatched_from_v4.parquet"

OOF_MONTHS = [1, 7, 11, 12]
LO, HI = -7200, 172800

sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)

COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "month", "hour_utc",
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
    "AOBT_3_flt", "LOBT_flt", "mvt_minus_aobt3", "mvt_minus_lobt", "mvt_minus_sched",
]


def base(path, with_label):
    cols = COLS if with_label else [c for c in COLS if c not in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")]
    lf = pl.scan_parquet(path).select(cols).with_columns(
        sec("AOBT_3_flt", "LOBT_flt").alias("aobt3_minus_lobt"),
        sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("pa_raw"),
        sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("pl_raw"),
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms_raw"),
        pl.col("mvt_minus_aobt3").alias("pa"),
        pl.col("mvt_minus_lobt").alias("plob"),
        pl.col("mvt_minus_sched").alias("ms"),
        pl.col("mvt_minus_aobt3").is_null().alias("unm"),
        pl.col("LOBT_flt").is_null().alias("lobt_null"),
        (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
    )
    if with_label:
        lf = lf.with_columns(
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            sec("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt").alias("y_raw"),
        )
    # triggers
    lf = lf.with_columns(
        (pl.col("aobt3_minus_lobt") > 7200).fill_null(False).alias("z_gap"),
        pl.col("pa").is_between(LO, HI).fill_null(False).alias("z_pa_in"),
        pl.col("plob").is_between(LO, HI).fill_null(False).alias("z_pl_in"),
        (pl.col("pa") >= 7200).fill_null(False).alias("oa"),
        (pl.col("plob") >= 7200).fill_null(False).alias("ol"),
        (pl.col("ms") >= 7200).fill_null(False).alias("os"),
        (pl.col("lirf") & (pl.col("ms") >= 14400)).alias("s_ms"),
    ).with_columns(
        (pl.col("z_gap") & pl.col("z_pa_in") & pl.col("z_pl_in")).alias("Z"),
        pl.max_horizontal(pl.lit(0.0), pl.col("plob")).alias("z_pred"),
        (pl.col("s_ms") & pl.col("unm")).alias("s_msU"),
    )
    return lf


tr = base(FT, True).collect()
N = tr.height
print(f"### training departures: {N}")

# ------------------------------------------------------------------ A. identities and availability
print("\n=== A. identities / column availability (training) ===")
print(tr.select(
    pl.len().alias("n"),
    (pl.col("y") == pl.col("y_raw")).sum().alias("y_eq_mvt_minus_block"),
    ((pl.col("pa") - pl.col("pa_raw")).abs() < 1e-6).sum().alias("feat_pa_eq_raw"),
    pl.col("pa").is_null().sum().alias("pa_null"),
    ((pl.col("plob") - pl.col("pl_raw")).abs() < 1e-6).sum().alias("feat_pl_eq_raw"),
    pl.col("plob").is_null().sum().alias("pl_null"),
    ((pl.col("ms") - pl.col("ms_raw")).abs() < 1e-6).sum().alias("feat_ms_eq_raw"),
    pl.col("ms").is_null().sum().alias("ms_null"),
    pl.col("unm").sum().alias("unmatched_routing"),
    pl.col("FLIGHT_ID_mvt").is_null().sum().alias("fid_null"),
    pl.col("lobt_null").sum().alias("lobt_null"),
    (pl.col("lobt_null") & pl.col("unm")).sum().alias("lobt_null_and_unm"),
    (pl.col("lobt_null") & ~pl.col("unm")).sum().alias("lobt_null_and_matched"),
    (~pl.col("lobt_null") & pl.col("unm")).sum().alias("lobt_present_and_unm"),
))
print("aobt3_minus_lobt on matched rows, quantiles:")
print(tr.filter(~pl.col("unm")).select(
    pl.col("aobt3_minus_lobt").quantile(q).alias(f"q{q}") for q in [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
))

# ------------------------------------------------------------------ B. trigger counts (training)
print("\n=== B. trigger counts on training rows ===")
TRIG = ["Z", "z_gap", "oa", "ol", "os", "s_ms", "s_msU", "lirf"]
print(tr.select([pl.col(t).sum().alias(t) for t in TRIG]))
print("Z sensitivity: gap>7200 alone; gap & pa in range; gap & pl in range; gap & both (=Z)")
print(tr.select(
    pl.col("z_gap").sum().alias("gap"),
    (pl.col("z_gap") & pl.col("z_pa_in")).sum().alias("gap_pa_in"),
    (pl.col("z_gap") & pl.col("z_pl_in")).sum().alias("gap_pl_in"),
    pl.col("Z").sum().alias("Z"),
    (pl.col("z_gap") & ~pl.col("z_pa_in")).sum().alias("gap_pa_out"),
    (pl.col("z_gap") & ~pl.col("z_pl_in")).sum().alias("gap_pl_out"),
))


def by_airport(df, flag, label):
    print(f"\n-- {label}: rows by ADEP x unmatched (training) --")
    g = (
        df.filter(pl.col(flag))
        .group_by("ADEP_mvt", "unm")
        .agg(
            pl.len().alias("n"),
            pl.col("y").median().alias("y_med"),
            (pl.col("y") >= 7200).mean().round(3).alias("y_ge2h"),
            pl.col("pa").median().alias("pa_med"),
            pl.col("plob").median().alias("pl_med"),
            pl.col("ms").median().alias("ms_med"),
            pl.col("aobt3_minus_lobt").median().alias("a3_minus_lobt_med"),
        )
        .sort("n", descending=True)
    )
    print(g)
    tot = df.filter(pl.col(flag)).select(
        pl.len().alias("n"), pl.col("unm").sum().alias("n_unm"), (~pl.col("unm")).sum().alias("n_matched"),
        pl.col("y").median().alias("y_med"), (pl.col("y") >= 7200).sum().alias("n_y_ge2h"),
        (pl.col("y") >= 14400).sum().alias("n_y_ge4h"),
    )
    print(tot)


for f, lab in [("Z", "Rule Z (AOBT_3-LOBT>7200 & both proxies in range)"), ("oa", "O-a (MVT-AOBT_3>=7200)"),
               ("ol", "O-l (MVT-LOBT>=7200)"), ("os", "O-s (MVT-SCHED>=7200)"), ("s_ms", "S-ms (LIRF & ms>=14400)")]:
    by_airport(tr, f, lab)

# Rule Z: how does rule prediction relate to y on all training trigger rows?
print("\n-- Rule Z on all training trigger rows: y vs rule prediction vs proxies --")
zt = tr.filter(pl.col("Z"))
print(zt.select(
    pl.len().alias("n"),
    ((pl.col("z_pred") - pl.col("y")) ** 2).mean().sqrt().alias("rmse_rule"),
    ((pl.col("pa") - pl.col("y")) ** 2).mean().sqrt().alias("rmse_pa"),
    ((pl.col("ms") - pl.col("y")) ** 2).mean().sqrt().alias("rmse_ms"),
    ((pl.col("z_pred") - pl.col("y")).abs() <= 600).mean().round(3).alias("rule_within_10min"),
    ((pl.col("pa") - pl.col("y")).abs() <= 600).mean().round(3).alias("pa_within_10min"),
    ((pl.col("ms") - pl.col("y")).abs() <= 600).mean().round(3).alias("ms_within_10min"),
    (pl.col("y") - pl.col("z_pred")).median().alias("y_minus_rule_med"),
))
print(zt.select("MVT_ID_mvt", "ADEP_mvt", "month", "y", "pa", "plob", "ms", "aobt3_minus_lobt", "z_pred").sort("y", descending=True).head(60))

# ------------------------------------------------------------------ C. medians for the shrink rule
print("\n=== C. medians for shrink (training months not in OOF) ===")
nonoof = tr.filter(~pl.col("month").is_in(OOF_MONTHS))
med_ap = nonoof.group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_ap"))
med_ap_unm = nonoof.filter(pl.col("unm")).group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_ap_unm"))
med_glob = float(nonoof["y"].median())
print(med_ap.join(med_ap_unm, on="ADEP_mvt").sort("ADEP_mvt"))
print(f"global median (non-OOF months): {med_glob}")

# ------------------------------------------------------------------ D. OOF
print("\n=== D. OOF main-48c19bc830 ===")
oof_all = pl.read_parquet(OOF)
oof = oof_all.filter(pl.col("oof").is_not_null() & pl.col("oof").is_not_nan())
print(f"oof file rows: {oof_all.height}; scored rows: {oof.height}")
j = oof.join(
    tr.select("MVT_ID_mvt", "ADEP_mvt", "month", "y", "pa", "plob", "ms", "unm", "lirf", "aobt3_minus_lobt",
              "Z", "z_gap", "z_pred", "oa", "ol", "os", "s_ms", "s_msU"),
    on="MVT_ID_mvt", how="left",
).join(med_ap, on="ADEP_mvt", how="left")
assert j["y"].null_count() == 0
j = j.with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"), pl.lit(med_glob).alias("med_glob"))
NO = j.height
SSE0 = float(j["se"].sum())
RMSE0 = (SSE0 / NO) ** 0.5
print(f"scored rows {NO}; total RMSE {RMSE0:.4f}; months {sorted(j['month'].unique().to_list())}")
u = j.filter(pl.col("unm"))
m = j.filter(~pl.col("unm"))
print(f"unmatched {u.height} RMSE {(u['se'].mean()) ** 0.5:.2f}; matched {m.height} RMSE {(m['se'].mean()) ** 0.5:.3f}")


def rmse(expr_pred, df):
    e = df.select(((expr_pred - pl.col("y")) ** 2).mean().sqrt()).item()
    return e


def evaluate(name, trig, pred_expr, df=None):
    """trig: boolean column name; pred_expr: expression giving the rule prediction on trigger rows."""
    if df is None:
        df = j
    sub = df.filter(pl.col(trig))
    n = sub.height
    if n == 0:
        print(f"\n[{name}] trigger selects 0 OOF rows")
        return
    n_unm = int(sub["unm"].sum())
    print(f"\n[{name}] OOF trigger rows n={n} (unmatched {n_unm}, matched {n - n_unm}); y median {sub['y'].median()}, "
          f"y>=2h {int((sub['y'] >= 7200).sum())}, y>=4h {int((sub['y'] >= 14400).sum())}")
    print(sub.group_by("ADEP_mvt", "unm").agg(pl.len().alias("n")).sort("n", descending=True).head(12))
    r_a = rmse(pl.col("oof"), sub)
    r_b = rmse(pred_expr, sub)
    sub_pa = sub.filter(pl.col("pa").is_not_null())
    r_c = rmse(pl.col("pa"), sub_pa) if sub_pa.height else float("nan")
    r_d = rmse(pl.col("ms"), sub)
    print(f"   RMSE on trigger rows: (a) current OOF {r_a:.1f}  (b) rule {r_b:.1f}  "
          f"(c) MVT-AOBT_3 {r_c:.1f} [n={sub_pa.height}]  (d) MVT-SCHED {r_d:.1f}")
    print(f"   medians on trigger rows: oof {sub['oof'].median():.0f}, rule {sub.select(pred_expr.median()).item():.0f}, "
          f"pa {sub['pa'].median()}, ms {sub['ms'].median():.0f}, y {sub['y'].median():.0f}")
    # counterfactual total
    cf = df.with_columns(pl.when(pl.col(trig)).then(pred_expr).otherwise(pl.col("oof")).alias("p2"))
    cf = cf.with_columns(((pl.col("p2") - pl.col("y")) ** 2).alias("se2"))
    sse2 = float(cf["se2"].sum())
    r_tot = (sse2 / NO) ** 0.5
    cu = cf.filter(pl.col("unm"))
    cm = cf.filter(~pl.col("unm"))
    print(f"   counterfactual total OOF RMSE {r_tot:.4f} (baseline {RMSE0:.4f}, delta {r_tot - RMSE0:+.4f}); "
          f"unmatched {(cu['se2'].mean()) ** 0.5:.2f} (baseline {(cu['se'].mean()) ** 0.5:.2f}); "
          f"matched {(cm['se2'].mean()) ** 0.5:.3f} (baseline {(cm['se'].mean()) ** 0.5:.3f})")
    print(f"   SSE on trigger rows: before {float(sub['se'].sum()):.4e} ({float(sub['se'].sum()) / SSE0:.4%} of total), "
          f"after {float(cf.filter(pl.col(trig))['se2'].sum()):.4e}")
    # rows where the rule helps / hurts
    d = cf.filter(pl.col(trig)).with_columns((pl.col("se2") - pl.col("se")).alias("dse"))
    print(f"   rows improved {int((d['dse'] < 0).sum())}, worsened {int((d['dse'] > 0).sum())}, unchanged {int((d['dse'] == 0).sum())}")
    return cf


# Rule Z
cfZ = evaluate("Z: predict max(0, MVT-LOBT)", "Z", pl.col("z_pred"))
if cfZ is not None:
    print(j.filter(pl.col("Z")).select("MVT_ID_mvt", "ADEP_mvt", "month", "y", "oof", "z_pred", "pa", "ms", "aobt3_minus_lobt")
          .with_columns((pl.col("oof") - pl.col("y")).alias("err_oof"), (pl.col("z_pred") - pl.col("y")).alias("err_rule"))
          .sort("y", descending=True))
# Z variant: gap only, no range condition
evaluate("Z-gap-only: AOBT_3-LOBT>7200, predict max(0, MVT-LOBT)", "z_gap", pl.col("z_pred"))

# Rule O analogues, airport median and global median
shr_ap = 0.5 * pl.col("oof") + 0.5 * pl.col("med_ap")
shr_gl = 0.5 * pl.col("oof") + 0.5 * pl.col("med_glob")
for t, lab in [("oa", "MVT-AOBT_3>=7200"), ("ol", "MVT-LOBT>=7200"), ("os", "MVT-SCHED>=7200")]:
    evaluate(f"O-{t}: {lab}, 0.5*pred + 0.5*airport median", t, shr_ap)
    evaluate(f"O-{t}: {lab}, 0.5*pred + 0.5*global median", t, shr_gl)
# O-s restricted to unmatched rows (the OpenAir stage acts after their LIRF-unmatched stage)
j = j.with_columns((pl.col("os") & pl.col("unm")).alias("osU"), (pl.col("os") & ~pl.col("unm")).alias("osM"))
evaluate("O-osU: MVT-SCHED>=7200 & unmatched, airport median", "osU", shr_ap)
evaluate("O-osM: MVT-SCHED>=7200 & matched, airport median", "osM", shr_ap)

# Rule S readings
j = j.with_columns(pl.col("lirf").alias("S_lirf_all"))
evaluate("S-cap: LIRF, min(pred, 14400)", "S_lirf_all", pl.min_horizontal(pl.col("oof"), pl.lit(14400.0)))
evaluate("S-ms: LIRF & ms>=14400, pred := ms", "s_ms", pl.col("ms"))
evaluate("S-msU: LIRF unmatched & ms>=14400, pred := ms", "s_msU", pl.col("ms"))
evaluate("S-floor: LIRF & ms>=14400, max(pred, 14400)", "s_ms", pl.max_horizontal(pl.col("oof"), pl.lit(14400.0)))
print("\nS-ms rows detail (OOF):")
print(j.filter(pl.col("s_ms")).select("MVT_ID_mvt", "ADEP_mvt", "month", "unm", "y", "oof", "ms", "pa")
      .with_columns((pl.col("y") - pl.col("ms")).alias("y_minus_ms")).sort("ms", descending=True))

# ------------------------------------------------------------------ E. Ranking
print("\n=== E. ranking set ===")
rk = base(FR, False).collect()
NR = rk.height
main = pl.read_parquet(MAIN).rename({"TAXITIME_SEC_mvt": "v7"})
probe = pl.read_parquet(PROBE).rename({"TAXITIME_SEC_mvt": "v9"})
rk = rk.join(main, on="MVT_ID_mvt", how="left").join(probe, on="MVT_ID_mvt", how="left").join(med_ap, on="ADEP_mvt", how="left")
print(f"ranking rows {NR}; v7 null {rk['v7'].null_count()}; v9 null {rk['v9'].null_count()}")
print(rk.select(
    pl.col("unm").sum().alias("unmatched_routing"),
    pl.col("lobt_null").sum().alias("lobt_null"),
    (pl.col("lobt_null") & pl.col("unm")).sum().alias("lobt_null_and_unm"),
    (pl.col("lobt_null") & ~pl.col("unm")).sum().alias("lobt_null_and_matched"),
    pl.col("pa").is_null().sum().alias("pa_null"),
    pl.col("ms").is_null().sum().alias("ms_null"),
    ((pl.col("pa") - pl.col("pa_raw")).abs() < 1e-6).sum().alias("feat_pa_eq_raw"),
    ((pl.col("plob") - pl.col("pl_raw")).abs() < 1e-6).sum().alias("feat_pl_eq_raw"),
    ((pl.col("ms") - pl.col("ms_raw")).abs() < 1e-6).sum().alias("feat_ms_eq_raw"),
))
print("trigger counts (ranking):")
print(rk.select([pl.col(t).sum().alias(t) for t in ["Z", "z_gap", "oa", "ol", "os", "s_ms", "s_msU", "lirf"]]))
print(rk.select(
    pl.col("z_gap").sum().alias("gap"),
    (pl.col("z_gap") & pl.col("z_pa_in")).sum().alias("gap_pa_in"),
    (pl.col("z_gap") & pl.col("z_pl_in")).sum().alias("gap_pl_in"),
    pl.col("Z").sum().alias("Z"),
))


def rank_eval(name, trig, pred_expr):
    sub = rk.filter(pl.col(trig))
    n = sub.height
    print(f"\n[{name}] ranking trigger rows n={n} (unmatched {int(sub['unm'].sum())}, matched {n - int(sub['unm'].sum())})")
    if n == 0:
        return
    print(sub.group_by("ADEP_mvt", "unm", "month").agg(pl.len().alias("n")).sort("n", descending=True).head(14))
    s = sub.with_columns(pred_expr.alias("rule"))
    print(s.select(
        pl.col("v7").median().alias("v7_med"), pl.col("v7").min().alias("v7_min"), pl.col("v7").max().alias("v7_max"),
        pl.col("v9").median().alias("v9_med"), pl.col("v9").min().alias("v9_min"), pl.col("v9").max().alias("v9_max"),
        pl.col("rule").median().alias("rule_med"), pl.col("rule").min().alias("rule_min"), pl.col("rule").max().alias("rule_max"),
        pl.col("pa").median().alias("pa_med"), pl.col("ms").median().alias("ms_med"),
        ((pl.col("rule") - pl.col("v7")) ** 2).mean().sqrt().alias("rms_rule_minus_v7"),
        ((pl.col("rule") - pl.col("v9")) ** 2).mean().sqrt().alias("rms_rule_minus_v9"),
        ((pl.col("rule") - pl.col("v7")) ** 2).sum().alias("sum_sq_change_v7"),
        (pl.col("v7") >= 7200).sum().alias("v7_ge2h"), (pl.col("v9") >= 7200).sum().alias("v9_ge2h"),
        (pl.col("rule") >= 7200).sum().alias("rule_ge2h"),
    ))
    return s


sZ = rank_eval("Z", "Z", pl.col("z_pred"))
if sZ is not None:
    print(sZ.select("MVT_ID_mvt", "ADEP_mvt", "month", "unm", "v7", "v9", "z_pred", "pa", "plob", "ms", "aobt3_minus_lobt").sort("z_pred", descending=True))
rank_eval("Z-gap-only", "z_gap", pl.col("z_pred"))
for t, lab in [("oa", "MVT-AOBT_3>=7200"), ("ol", "MVT-LOBT>=7200"), ("os", "MVT-SCHED>=7200")]:
    rank_eval(f"O-{t} ({lab}) shrink toward airport median (applied to v7)", t, 0.5 * pl.col("v7") + 0.5 * pl.col("med_ap"))
rk = rk.with_columns((pl.col("os") & pl.col("unm")).alias("osU"), pl.col("lirf").alias("S_lirf_all"))
rank_eval("O-osU (MVT-SCHED>=7200 & unmatched) shrink toward airport median (applied to v7)", "osU", 0.5 * pl.col("v7") + 0.5 * pl.col("med_ap"))
rank_eval("S-cap LIRF min(v7, 14400)", "S_lirf_all", pl.min_horizontal(pl.col("v7"), pl.lit(14400)))
rank_eval("S-ms LIRF & ms>=14400, rule := ms", "s_ms", pl.col("ms"))
rank_eval("S-msU LIRF unmatched & ms>=14400, rule := ms", "s_msU", pl.col("ms"))
print("\nS-ms ranking rows detail:")
print(rk.filter(pl.col("s_ms")).select("MVT_ID_mvt", "ADEP_mvt", "month", "unm", "v7", "v9", "ms", "pa").sort("ms", descending=True))
print("\nLIRF ranking rows with v7 >= 14400 or v9 >= 14400:")
print(rk.filter(pl.col("lirf") & ((pl.col("v7") >= 14400) | (pl.col("v9") >= 14400)))
      .select("MVT_ID_mvt", "month", "unm", "v7", "v9", "ms", "pa").sort("ms", descending=True))
print("\nTraining LIRF rows: how many have y >= 14400 and how many pred>=14400 in OOF:")
print(j.filter(pl.col("lirf")).select(pl.len().alias("n"), (pl.col("y") >= 14400).sum().alias("y_ge4h"), (pl.col("oof") >= 14400).sum().alias("oof_ge4h"),
                                        ((pl.col("y") >= 14400) & (pl.col("oof") >= 14400)).sum().alias("both")))
# ------------------------------------------------------------------ F. S-msU decomposition and threshold sweep
print("\n=== F. LIRF unmatched OOF rows, pred := ms when ms >= T: decomposition by row category / month ===")
lu = j.filter(pl.col("lirf") & pl.col("unm")).with_columns(
    pl.when((pl.col("y") - pl.col("ms")).abs() <= 1800).then(pl.lit("y~ms (filled, +-1800)"))
    .when(pl.col("y") > 72000).then(pl.lit("rollover y>20h"))
    .otherwise(pl.lit("other")).alias("cat")
)
print(f"LIRF unmatched OOF rows: {lu.height}; SSE {float(lu['se'].sum()):.4e} ({float(lu['se'].sum()) / SSE0:.2%} of total)")
for T in [7200, 14400, 43200]:
    cf = lu.with_columns(pl.when(pl.col("ms") >= T).then(pl.col("ms")).otherwise(pl.col("oof")).alias("p2"))
    cf = cf.with_columns(((pl.col("p2") - pl.col("y")) ** 2).alias("se2"))
    trig = cf.filter(pl.col("ms") >= T)
    sb, sa = float(trig["se"].sum()), float(trig["se2"].sum())
    print(f"\nT={T}: trigger n={trig.height}; SSE before {sb:.4e} after {sa:.4e}; total OOF RMSE {RMSE0:.4f} -> {((SSE0 - sb + sa) / NO) ** 0.5:.4f}; "
          f"unmatched RMSE -> {((float(u['se'].sum()) - sb + sa) / u.height) ** 0.5:.1f}")
    print(trig.group_by("cat").agg(pl.len().alias("n"), pl.col("se").sum().alias("sse_before"), pl.col("se2").sum().alias("sse_after"),
                                   pl.col("y").median().alias("y_med"), pl.col("ms").median().alias("ms_med"), pl.col("oof").median().alias("oof_med")).sort("n", descending=True))
    print(trig.group_by("month").agg(pl.len().alias("n"), pl.col("se").sum().alias("sse_before"), pl.col("se2").sum().alias("sse_after")).sort("month"))
    # per-month total RMSE before/after
    jm = j.join(cf.select("MVT_ID_mvt", "se2"), on="MVT_ID_mvt", how="left").with_columns(pl.coalesce(pl.col("se2"), pl.col("se")).alias("se2"))
    print(jm.group_by("month").agg(pl.len().alias("n"), pl.col("se").mean().sqrt().alias("rmse_before"), pl.col("se2").mean().sqrt().alias("rmse_after")).sort("month"))

# Rule Z: bound on the ranking-set RMSE change implied by the prediction change (labels unknown)
zr = rk.filter(pl.col("Z"))
ssq = float(((zr["z_pred"] - zr["v7"]) ** 2).sum())
for lb in [305.03, 342.59]:
    print(f"Rule Z ranking: sum of squared prediction change vs v7 = {ssq:.4e}; if LB RMSE were {lb}, the change is bounded by "
          f"[{(lb ** 2 - ssq / NR) ** 0.5 - lb:+.3f}, {(lb ** 2 + ssq / NR) ** 0.5 - lb:+.3f}] s (extremes: rule right/wrong on every row)")
print("done")
