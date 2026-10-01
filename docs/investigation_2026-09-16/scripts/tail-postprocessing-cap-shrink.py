"""Item 13 `tail-postprocessing-cap-shrink` -- independent re-derivation (no training).

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/tail-postprocessing-cap-shrink.py
Inputs (all column-selected lazy scans):
  data/work/feat_train.parquet          labels, ADEP, routing flags, hour
  data/work/feat_rank.parquet           ADEP, routing flags for the ranking set
  experiments/oof_main-48c19bc830.parquet  (MVT_ID_mvt, oof); 672,428 non-null of 2,085,047
  outputs/main.parquet, outputs/probe_unmatched_from_v4.parquet   submissions (v7, v9)

Counterfactuals evaluated on the OOF file (labels known):
  (a) cap unmatched predictions at 3,600 / 7,200 / 14,400 / 43,200 / 86,400  (+ cap-quantile sweep)
  (b) shrink unmatched predictions: lam*pred + (1-lam)*airport-median, lam in {0.25, 0.5, 0.75} (+ lam=0)
  (c) same toward the (airport, hour_utc) median  (+ hour_local as sensitivity)
  (d) caps on non-LIRF unmatched rows only
  (e) caps on matched rows only
Ranking set (labels unknown): number of predictions > 4 h / > 12 h per file vs training label frequency.
"Unmatched" is computed under both definitions of plan.md section 0 (routing: mvt_minus_aobt3 null; D: FLIGHT_ID_mvt null).
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import math
import polars as pl

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)

B = "/home/user/prc-data-challenge/"
H1, H2, H4, H12, DAY = 3600, 7200, 14400, 43200, 86400
CAPS = [H1, H2, H4, H12, DAY]
LAMS = [0.25, 0.5, 0.75]


def hdr(t):
    print("\n" + "=" * 110 + "\n" + t + "\n" + "=" * 110)


def rmse(se_sum, n):
    return math.sqrt(se_sum / n) if n else float("nan")


# --------------------------------------------------------------------------------------------------
# 0. training frame (needed columns only) and OOF join
# --------------------------------------------------------------------------------------------------
tr = (pl.scan_parquet(B + "data/work/feat_train.parquet")
      .select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "mvt_minus_aobt3", "TAXITIME_SEC_mvt",
               "hour_utc", "hour_local", "MVT_TIME_UTC_mvt"])
      .with_columns([
          pl.col("mvt_minus_aobt3").is_null().alias("unm"),        # routing definition (model B)
          pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),      # D's definition
          pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
          pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
          (pl.col("ADEP_mvt") == "LIRF").alias("lirf"),
      ])
      .drop(["FLIGHT_ID_mvt", "mvt_minus_aobt3", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"])
      .collect())

hdr("0. training frame")
print(tr.select([pl.len().alias("n_train"), pl.col("unm").sum().alias("unm_routing"), pl.col("unm_fid").sum().alias("unm_fid"),
                 pl.col("MVT_ID_mvt").n_unique().alias("mvt_id_nunique"),
                 pl.col("hour_utc").is_null().sum().alias("hour_utc_null"), pl.col("hour_local").is_null().sum().alias("hour_local_null"),
                 pl.col("hour_utc").min().alias("hour_utc_min"), pl.col("hour_utc").max().alias("hour_utc_max")]))
print("hour_utc dtype:", tr.schema["hour_utc"], " hour_local dtype:", tr.schema["hour_local"])

oof = (pl.scan_parquet(B + "experiments/oof_main-48c19bc830.parquet")
       .filter(pl.col("oof").is_not_null()).collect())
print("OOF rows non-null:", oof.height, " ids unique:", oof["MVT_ID_mvt"].n_unique())
df = oof.join(tr, on="MVT_ID_mvt", how="inner")
print("joined rows:", df.height, " months:", sorted(df["mo"].unique().to_list()))
df = df.with_columns(((pl.col("oof") - pl.col("y")) ** 2).alias("se"))
N = df.height
SSE0 = df["se"].sum()

# --------------------------------------------------------------------------------------------------
# training label medians (for shrink targets). Two variants: all training rows, and training rows in
# months NOT covered by this OOF (2-6, 8-10) so that the target is not computed on the scored rows.
# Medians are computed on unmatched (routing) training rows because they are applied to unmatched rows.
# --------------------------------------------------------------------------------------------------
oof_months = sorted(df["mo"].unique().to_list())
tr_unm = tr.filter(pl.col("unm"))
tr_unm_out = tr_unm.filter(~pl.col("mo").is_in(oof_months))
med_ap = tr_unm.group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_ap"), pl.len().alias("n_ap"))
med_ap_out = tr_unm_out.group_by("ADEP_mvt").agg(pl.col("y").median().alias("med_ap_out"), pl.len().alias("n_ap_out"))
med_aph = tr_unm.group_by(["ADEP_mvt", "hour_utc"]).agg(pl.col("y").median().alias("med_aph"), pl.len().alias("n_aph"))
med_aph_out = tr_unm_out.group_by(["ADEP_mvt", "hour_utc"]).agg(pl.col("y").median().alias("med_aph_out"), pl.len().alias("n_aph_out"))
med_aphl = tr_unm.group_by(["ADEP_mvt", "hour_local"]).agg(pl.col("y").median().alias("med_aphl"), pl.len().alias("n_aphl"))
# mean (oracle-ish, M's "oracle airport-mean") on OOF unmatched rows themselves
mean_ap_oof = df.filter(pl.col("unm")).group_by("ADEP_mvt").agg(pl.col("y").mean().alias("mean_ap_oof"))

hdr("0b. shrink targets: per-airport medians of unmatched training labels (all months / non-OOF months)")
print(med_ap.join(med_ap_out, on="ADEP_mvt", how="left").join(mean_ap_oof, on="ADEP_mvt", how="left").sort("ADEP_mvt"))
print("(airport, hour_utc) cells:", med_aph.height, " min n:", med_aph["n_aph"].min(), " cells with n<10:", (med_aph["n_aph"] < 10).sum(),
      " | non-OOF-month cells:", med_aph_out.height, " min n:", med_aph_out["n_aph_out"].min())

df = (df.join(med_ap, on="ADEP_mvt", how="left").join(med_ap_out, on="ADEP_mvt", how="left")
        .join(med_aph, on=["ADEP_mvt", "hour_utc"], how="left").join(med_aph_out, on=["ADEP_mvt", "hour_utc"], how="left")
        .join(med_aphl, on=["ADEP_mvt", "hour_local"], how="left").join(mean_ap_oof, on="ADEP_mvt", how="left"))
print("null shrink targets on OOF unmatched rows: med_aph", df.filter(pl.col("unm"))["med_aph"].null_count(),
      " med_aph_out", df.filter(pl.col("unm"))["med_aph_out"].null_count(),
      " med_aphl", df.filter(pl.col("unm"))["med_aphl"].null_count())

# --------------------------------------------------------------------------------------------------
# 1. baseline
# --------------------------------------------------------------------------------------------------
hdr("1. baseline OOF main-48c19bc830")


def summary(frame, pred_col, label):
    """total / unmatched(routing) / unmatched(fid) / matched(routing) RMSE for a prediction column."""
    f = frame.with_columns(((pl.col(pred_col) - pl.col("y")) ** 2).alias("_se"))
    tot = f["_se"].sum()
    u = f.filter(pl.col("unm")); uf = f.filter(pl.col("unm_fid")); m = f.filter(~pl.col("unm"))
    nl = f.filter(pl.col("unm") & ~pl.col("lirf")); l = f.filter(pl.col("unm") & pl.col("lirf"))
    return dict(variant=label, total=rmse(tot, f.height), unm_routing=rmse(u["_se"].sum(), u.height),
                unm_fid=rmse(uf["_se"].sum(), uf.height), matched=rmse(m["_se"].sum(), m.height),
                unm_lirf=rmse(l["_se"].sum(), l.height), unm_nonlirf=rmse(nl["_se"].sum(), nl.height),
                n_changed=int((f[pred_col] != f["oof"]).sum()))


rows = [summary(df, "oof", "baseline (model)")]
print(f"N={N}; unmatched routing {df['unm'].sum()}, fid {df['unm_fid'].sum()}, matched {(~df['unm']).sum()}; "
      f"LIRF unmatched {(df['unm'] & df['lirf']).sum()}")
print(pl.DataFrame(rows))
print("unmatched-side SSE share:", round(df.filter(pl.col("unm"))["se"].sum() / SSE0, 4))
print("unmatched OOF prediction quantiles:", df.filter(pl.col("unm"))["oof"].quantile(0.5), df.filter(pl.col("unm"))["oof"].quantile(0.9),
      df.filter(pl.col("unm"))["oof"].quantile(0.99), df.filter(pl.col("unm"))["oof"].max())
print("matched OOF prediction max:", df.filter(~pl.col("unm"))["oof"].max(), " n matched pred > 3600:", (df.filter(~pl.col("unm"))["oof"] > H1).sum(),
      " > 7200:", (df.filter(~pl.col("unm"))["oof"] > H2).sum())

# --------------------------------------------------------------------------------------------------
# (a) caps on unmatched (routing) predictions; also under fid definition and on all rows
# --------------------------------------------------------------------------------------------------
hdr("(a) cap unmatched predictions at c: pred := min(pred, c) on unmatched rows")
tab_a = []
for c in CAPS:
    f = df.with_columns(pl.when(pl.col("unm")).then(pl.min_horizontal(pl.col("oof"), pl.lit(float(c)))).otherwise(pl.col("oof")).alias("p"))
    s = summary(f, "p", f"cap unm @ {c}")
    aff = f.filter(pl.col("unm") & (pl.col("oof") > c))
    helped = aff.filter((pl.col("p") - pl.col("y")).abs() < (pl.col("oof") - pl.col("y")).abs())
    s.update(n_affected=aff.height, n_helped=helped.height, y_mean_affected=round(aff["y"].mean(), 0) if aff.height else None,
             y_median_affected=aff["y"].median() if aff.height else None,
             sse_delta=(f.with_columns(((pl.col("p") - pl.col("y")) ** 2).alias("s2"))["s2"].sum() - SSE0))
    tab_a.append(s); rows.append(s)
    # fid definition
    ff = df.with_columns(pl.when(pl.col("unm_fid")).then(pl.min_horizontal(pl.col("oof"), pl.lit(float(c)))).otherwise(pl.col("oof")).alias("p"))
    rows.append(summary(ff, "p", f"cap unm(fid) @ {c}"))
    # all rows
    fa = df.with_columns(pl.min_horizontal(pl.col("oof"), pl.lit(float(c))).alias("p"))
    rows.append(summary(fa, "p", f"cap ALL rows @ {c}"))
print(pl.DataFrame(tab_a))

hdr("(a2) cap-quantile sweep: cap = quantile q of unmatched OOF predictions (routing)")
u_pred = df.filter(pl.col("unm"))["oof"]
tab_q = []
for q in [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0]:
    c = float(u_pred.quantile(q))
    f = df.with_columns(pl.when(pl.col("unm")).then(pl.min_horizontal(pl.col("oof"), pl.lit(c))).otherwise(pl.col("oof")).alias("p"))
    s = summary(f, "p", f"cap unm @ q{q}")
    s.update(cap_value=round(c, 0), n_affected=int(((df["unm"]) & (df["oof"] > c)).sum()))
    tab_q.append(s)
print(pl.DataFrame(tab_q))

# --------------------------------------------------------------------------------------------------
# (b) shrink toward airport median
# --------------------------------------------------------------------------------------------------
hdr("(b) shrink unmatched: lam*pred + (1-lam)*airport-median (median of unmatched training labels per ADEP)")
tab_b = []
for tgt, name in [("med_ap", "airport-median (all train months)"), ("med_ap_out", "airport-median (non-OOF months)"),
                  ("mean_ap_oof", "airport-mean of OOF labels (oracle, M)")]:
    for lam in [0.0] + LAMS + [1.0]:
        f = df.with_columns(pl.when(pl.col("unm")).then(lam * pl.col("oof") + (1 - lam) * pl.col(tgt)).otherwise(pl.col("oof")).alias("p"))
        s = summary(f, "p", f"lam={lam} -> {name}")
        tab_b.append(s); rows.append(s)
print(pl.DataFrame(tab_b))

# reconciliation with M's figures (constant airport-median -> 3,724; 0.5 shrink -> 2,825): alternative
# definitions of "airport-median" that M may have used.
print("-- (b2) alternative airport-median definitions (lam=0 constant / lam=0.5 shrink), unmatched (routing) RMSE:")
alts = {
    "median of ALL training rows per ADEP (matched+unmatched)": tr.group_by("ADEP_mvt").agg(pl.col("y").median().alias("t")),
    "median of OOF unmatched labels per ADEP": df.filter(pl.col("unm")).group_by("ADEP_mvt").agg(pl.col("y").median().alias("t")),
    "median of OOF all-row labels per ADEP": df.group_by("ADEP_mvt").agg(pl.col("y").median().alias("t")),
    "median of OOF unmatched PREDICTIONS per ADEP": df.filter(pl.col("unm")).group_by("ADEP_mvt").agg(pl.col("oof").median().alias("t")),
    "mean of unmatched training labels per ADEP": tr_unm.group_by("ADEP_mvt").agg(pl.col("y").mean().alias("t")),
    "global median of unmatched training labels": tr.select(pl.lit("x").alias("k")).head(0),
}
for name, t in alts.items():
    if name.startswith("global"):
        g = float(tr_unm["y"].median())
        f = df.with_columns(pl.lit(g).alias("t"))
    else:
        f = df.join(t, on="ADEP_mvt", how="left")
    r0 = summary(f.with_columns(pl.when(pl.col("unm")).then(pl.col("t")).otherwise(pl.col("oof")).alias("p")), "p", "")["unm_routing"]
    r5 = summary(f.with_columns(pl.when(pl.col("unm")).then(0.5 * pl.col("oof") + 0.5 * pl.col("t")).otherwise(pl.col("oof")).alias("p")), "p", "")["unm_routing"]
    lirf_t = f.filter(pl.col("lirf"))["t"][0]
    print(f"   {name:60s} LIRF target {lirf_t:8.1f} | lam=0: {r0:7.1f} | lam=0.5: {r5:7.1f}")

# --------------------------------------------------------------------------------------------------
# (c) shrink toward (airport, hour) median
# --------------------------------------------------------------------------------------------------
hdr("(c) shrink unmatched: lam*pred + (1-lam)*(airport,hour)-median")
tab_c = []
for tgt, name in [("med_aph", "(ADEP,hour_utc)-median (all train months)"), ("med_aph_out", "(ADEP,hour_utc)-median (non-OOF months)"),
                  ("med_aphl", "(ADEP,hour_local)-median (all train months)")]:
    for lam in [0.0] + LAMS:
        # fall back to airport median where the cell is missing
        f = df.with_columns(pl.when(pl.col("unm")).then(lam * pl.col("oof") + (1 - lam) * pl.coalesce(pl.col(tgt), pl.col("med_ap")))
                            .otherwise(pl.col("oof")).alias("p"))
        s = summary(f, "p", f"lam={lam} -> {name}")
        tab_c.append(s); rows.append(s)
print(pl.DataFrame(tab_c))

# --------------------------------------------------------------------------------------------------
# (d) caps on non-LIRF unmatched rows only
# --------------------------------------------------------------------------------------------------
hdr("(d) cap non-LIRF unmatched predictions only")
tab_d = []
for c in CAPS:
    f = df.with_columns(pl.when(pl.col("unm") & ~pl.col("lirf")).then(pl.min_horizontal(pl.col("oof"), pl.lit(float(c)))).otherwise(pl.col("oof")).alias("p"))
    s = summary(f, "p", f"cap non-LIRF unm @ {c}")
    aff = f.filter(pl.col("unm") & ~pl.col("lirf") & (pl.col("oof") > c))
    helped = aff.filter((pl.col("p") - pl.col("y")).abs() < (pl.col("oof") - pl.col("y")).abs())
    s.update(n_affected=aff.height, n_helped=helped.height, y_mean_affected=round(aff["y"].mean(), 0) if aff.height else None,
             y_max_affected=aff["y"].max() if aff.height else None)
    tab_d.append(s); rows.append(s)
print(pl.DataFrame(tab_d))
print("-- non-LIRF unmatched OOF rows with pred > 7,200: label distribution")
nl = df.filter(pl.col("unm") & ~pl.col("lirf") & (pl.col("oof") > H2))
print(nl.select([pl.len(), pl.col("y").median().alias("y_med"), pl.col("y").mean().alias("y_mean"), (pl.col("y") >= H2).sum().alias("n_y>=2h"), pl.col("y").max().alias("y_max"), pl.col("oof").median().alias("pred_med")]))
print(nl.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("y").max().alias("y_max"), pl.col("oof").median().alias("pred_med")).sort("n", descending=True))
print("-- (d2) finer cap sweep on non-LIRF unmatched rows, with per-month SSE delta (robustness across the 4 OOF months)")
tab_d3 = []
for c in [1800, 2400, 3000, 3600, 4200, 4800, 5400, 7200]:
    f = df.with_columns(pl.when(pl.col("unm") & ~pl.col("lirf")).then(pl.min_horizontal(pl.col("oof"), pl.lit(float(c)))).otherwise(pl.col("oof")).alias("p"))
    f = f.with_columns(((pl.col("p") - pl.col("y")) ** 2 - pl.col("se")).alias("dse"))
    s = summary(f, "p", f"cap non-LIRF unm @ {c}")
    aff = f.filter(pl.col("unm") & ~pl.col("lirf") & (pl.col("oof") > c))
    per_mo = f.group_by("mo").agg(pl.col("dse").sum()).sort("mo")
    s.update(n_affected=aff.height, n_helped=int(((aff["p"] - aff["y"]).abs() < (aff["oof"] - aff["y"]).abs()).sum()),
             n_hurt=int(((aff["p"] - aff["y"]).abs() > (aff["oof"] - aff["y"]).abs()).sum()),
             sse_delta_total=f["dse"].sum(), **{f"dSSE_mo{int(m)}": round(v, 0) for m, v in zip(per_mo["mo"], per_mo["dse"])})
    tab_d3.append(s)
print(pl.DataFrame(tab_d3).drop(["unm_fid", "matched", "unm_lirf"]))
print("-- cap non-LIRF unm @ 3600: affected rows by airport (n, helped, y median, pred median, SSE delta)")
f = df.with_columns(pl.when(pl.col("unm") & ~pl.col("lirf")).then(pl.min_horizontal(pl.col("oof"), pl.lit(3600.0))).otherwise(pl.col("oof")).alias("p"))
aff = f.filter(pl.col("unm") & ~pl.col("lirf") & (pl.col("oof") > 3600)).with_columns(((pl.col("p") - pl.col("y")) ** 2 - pl.col("se")).alias("dse"))
print(aff.group_by("ADEP_mvt").agg(pl.len().alias("n"), ((pl.col("p") - pl.col("y")).abs() < (pl.col("oof") - pl.col("y")).abs()).sum().alias("helped"),
                                   pl.col("y").median().alias("y_med"), pl.col("y").max().alias("y_max"), pl.col("oof").median().alias("pred_med"), pl.col("dse").sum().round(0).alias("dSSE")).sort("dSSE"))
print("   the 4 hurt rows:"); print(aff.filter(pl.col("dse") > 0).select(["MVT_ID_mvt", "ADEP_mvt", "mo", "y", "oof", "dse"]))
print("-- LIRF unmatched only, caps (complement of (d)):")
tab_d2 = []
for c in CAPS:
    f = df.with_columns(pl.when(pl.col("unm") & pl.col("lirf")).then(pl.min_horizontal(pl.col("oof"), pl.lit(float(c)))).otherwise(pl.col("oof")).alias("p"))
    s = summary(f, "p", f"cap LIRF unm @ {c}")
    aff = f.filter(pl.col("unm") & pl.col("lirf") & (pl.col("oof") > c))
    s.update(n_affected=aff.height, y_mean_affected=round(aff["y"].mean(), 0) if aff.height else None)
    tab_d2.append(s)
print(pl.DataFrame(tab_d2))

# --------------------------------------------------------------------------------------------------
# (e) caps on matched rows only
# --------------------------------------------------------------------------------------------------
hdr("(e) cap matched (routing) predictions only")
tab_e = []
for c in CAPS:
    f = df.with_columns(pl.when(~pl.col("unm")).then(pl.min_horizontal(pl.col("oof"), pl.lit(float(c)))).otherwise(pl.col("oof")).alias("p"))
    s = summary(f, "p", f"cap matched @ {c}")
    aff = f.filter(~pl.col("unm") & (pl.col("oof") > c))
    helped = aff.filter((pl.col("p") - pl.col("y")).abs() < (pl.col("oof") - pl.col("y")).abs())
    s.update(n_affected=aff.height, n_helped=helped.height, y_mean_affected=round(aff["y"].mean(), 0) if aff.height else None,
             y_max_affected=aff["y"].max() if aff.height else None)
    tab_e.append(s); rows.append(s)
print(pl.DataFrame(tab_e))
print("-- matched rows with pred > 3,600 (largest 15 of them):")
print(df.filter(~pl.col("unm") & (pl.col("oof") > H1)).select(["MVT_ID_mvt", "ADEP_mvt", "mo", "hour_utc", "y", "oof"]).sort("oof", descending=True).head(15))
print("   ... (534 rows; only the 15 largest predictions shown)")
print("-- matched OOF rows with y >= 4h (labels), and their predictions:")
print(df.filter(~pl.col("unm") & (pl.col("y") >= H4)).select(["MVT_ID_mvt", "ADEP_mvt", "mo", "y", "oof"]))

# --------------------------------------------------------------------------------------------------
# (f) tail-frequency calibration on OOF (labels known): predictions vs labels above thresholds
# --------------------------------------------------------------------------------------------------
hdr("(f) OOF: number of predictions vs labels above thresholds (routing definition)")
for grp, mask in [("all", pl.lit(True)), ("unmatched", pl.col("unm")), ("matched", ~pl.col("unm")), ("LIRF unm", pl.col("unm") & pl.col("lirf")),
                  ("non-LIRF unm", pl.col("unm") & ~pl.col("lirf"))]:
    g = df.filter(mask)
    print(f"{grp:14s} n={g.height:8d} | pred>2h {(g['oof'] > H2).sum():4d} y>=2h {(g['y'] >= H2).sum():4d} | pred>4h {(g['oof'] > H4).sum():4d} y>=4h {(g['y'] >= H4).sum():4d} | "
          f"pred>12h {(g['oof'] > H12).sum():3d} y>=12h {(g['y'] >= H12).sum():3d} | pred>=4h {(g['oof'] >= H4).sum():4d} pred>=12h {(g['oof'] >= H12).sum():3d}")
# joint: among rows predicted > 4h, how many labels >= 4h; and vice versa
for thr, nm in [(H4, "4h"), (H12, "12h")]:
    u = df.filter(pl.col("unm"))
    pp = u.filter(pl.col("oof") > thr); yy = u.filter(pl.col("y") >= thr)
    print(f"unmatched, {nm}: pred>{nm} n={pp.height}, of which y>={nm}: {(pp['y'] >= thr).sum()}, y median {pp['y'].median()}; "
          f"y>={nm} n={yy.height}, of which pred>{nm}: {(yy['oof'] > thr).sum()}, pred median {yy['oof'].median()}, y mean {yy['y'].mean():.0f}, pred mean {yy['oof'].mean():.0f}")

# --------------------------------------------------------------------------------------------------
# 2. ranking set: predictions above thresholds vs training label frequency
# --------------------------------------------------------------------------------------------------
hdr("2. ranking set: predictions > 4h / > 12h in outputs/main.parquet and outputs/probe_unmatched_from_v4.parquet")
rk = (pl.scan_parquet(B + "data/work/feat_rank.parquet")
      .select(["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_ID_mvt", "mvt_minus_aobt3", "MVT_TIME_UTC_mvt"])
      .with_columns([pl.col("mvt_minus_aobt3").is_null().alias("unm"), pl.col("FLIGHT_ID_mvt").is_null().alias("unm_fid"),
                     (pl.col("ADEP_mvt") == "LIRF").alias("lirf"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")])
      .drop(["FLIGHT_ID_mvt", "mvt_minus_aobt3", "MVT_TIME_UTC_mvt"]).collect())
print("ranking rows:", rk.height, " unm routing:", rk["unm"].sum(), " unm fid:", rk["unm_fid"].sum(), " matched routing:", (~rk["unm"]).sum(),
      " matched fid:", (~rk["unm_fid"]).sum(), " months:", sorted(rk["mo"].unique().to_list()))

subs = {"main.parquet (v7)": B + "outputs/main.parquet", "probe_unmatched_from_v4.parquet (v9)": B + "outputs/probe_unmatched_from_v4.parquet"}
sub_tab = []
for name, path in subs.items():
    s = pl.scan_parquet(path).select(["MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred")]).collect()
    j = rk.join(s, on="MVT_ID_mvt", how="inner")
    assert j.height == rk.height, (name, j.height)
    for grp, mask in [("all rows", pl.lit(True)), ("unmatched (routing)", pl.col("unm")), ("unmatched (fid)", pl.col("unm_fid")),
                      ("matched (routing)", ~pl.col("unm")), ("matched (fid)", ~pl.col("unm_fid")), ("LIRF unmatched (routing)", pl.col("unm") & pl.col("lirf"))]:
        g = j.filter(mask)
        sub_tab.append(dict(file=name, group=grp, n=g.height,
                            **{"pred>2h": int((g["pred"] > H2).sum()), "pred>=2h": int((g["pred"] >= H2).sum()),
                               "pred>4h": int((g["pred"] > H4).sum()), "pred>=4h": int((g["pred"] >= H4).sum()),
                               "pred>12h": int((g["pred"] > H12).sum()), "pred>=12h": int((g["pred"] >= H12).sum()),
                               "pred_max": g["pred"].max()}))
    # how many ranking predictions each cap would change (fact, no labels)
    print(f"-- {name}: rows changed by cap on unmatched (routing) rows:",
          {c: int(((j["unm"]) & (j["pred"] > c)).sum()) for c in CAPS}, " | on matched rows:", {c: int(((~j["unm"]) & (j["pred"] > c)).sum()) for c in CAPS})
print(pl.DataFrame(sub_tab))

hdr("2b. training label frequency above thresholds (both unmatched definitions) and implied ranking counts")
freq_tab = []
for grp, mask, n_rank in [("all rows", pl.lit(True), rk.height),
                          ("unmatched (routing)", pl.col("unm"), int(rk["unm"].sum())),
                          ("unmatched (fid)", pl.col("unm_fid"), int(rk["unm_fid"].sum())),
                          ("matched (routing)", ~pl.col("unm"), int((~rk["unm"]).sum())),
                          ("matched (fid)", ~pl.col("unm_fid"), int((~rk["unm_fid"]).sum())),
                          ("LIRF unmatched (routing)", pl.col("unm") & pl.col("lirf"), int((rk["unm"] & rk["lirf"]).sum()))]:
    g = tr.filter(mask)
    gj = g.filter(pl.col("mo").is_in([1, 7]))
    n_rank_jj = n_rank
    for thr, nm in [(H2, "2h"), (H4, "4h"), (H12, "12h")]:
        k = int((g["y"] >= thr).sum()); kj = int((gj["y"] >= thr).sum())
        freq_tab.append(dict(group=grp, thr=nm, n_train=g.height, n_y_ge=k, rate=k / g.height, n_rank=n_rank, implied=round(k / g.height * n_rank, 1),
                             n_train_JanJul=gj.height, n_y_ge_JanJul=kj, implied_JanJul_rate=round(kj / gj.height * n_rank, 1) if gj.height else None,
                             n_y_gt=int((g["y"] > thr).sum())))
print(pl.DataFrame(freq_tab))

# --------------------------------------------------------------------------------------------------
# 3. consolidated table
# --------------------------------------------------------------------------------------------------
hdr("3. consolidated OOF counterfactual table (RMSE; unmatched = routing unless stated)")
out = pl.DataFrame(rows).select(["variant", "total", "unm_routing", "unm_fid", "matched", "unm_lirf", "unm_nonlirf", "n_changed"])
print(out.with_columns(pl.col(["total", "unm_routing", "unm_fid", "matched", "unm_lirf", "unm_nonlirf"]).round(1)))
