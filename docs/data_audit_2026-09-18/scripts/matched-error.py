"""Data audit 2026-09-18 — lens: structure of model A's remaining (matched) error.

No model training. Reads only:
  experiments/oof_main-48c19bc830.parquet  (MVT_ID_mvt, oof; non-null on 672,428 rows)
  data/work/feat_train.parquet             (2,085,047 DEP rows)

Definitions (plan.md 2026-09-16 §0):
  y      = TAXITIME_SEC_mvt          = MVT - BLOCK          (exact, verified there)
  proxy  = mvt_minus_aobt3           = MVT - AOBT_3
  d      = y - proxy                 = AOBT_3 - BLOCK       (the inter-system clock disagreement)
  ms     = mvt_minus_sched           = MVT - SCHED
  as_    = aobt3_minus_sched
  err    = y - oof                   (model A residual; positive = under-prediction)
  matched= mvt_minus_aobt3 not null  (routing definition)
  fill   = |BLOCK - SCHED| <= 10 s   = |ms - y| <= 10  (label-derived; diagnosis only)

Run: POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/matched-error.py
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "1")
import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)

ROOT = "/home/user/prc-data-challenge"
OOF = f"{ROOT}/experiments/oof_main-48c19bc830.parquet"
FEAT = f"{ROOT}/data/work/feat_train.parquet"

COLS = [
    "MVT_ID_mvt", "TAXITIME_SEC_mvt",
    "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt", "ADES_REGION", "ADES_mvt",
    "month", "dow", "hour_utc",
    "mvt_minus_aobt3", "mvt_minus_sched", "aobt3_minus_sched", "aobt3_minus_eobt1",
    "mvt_minus_eobt1", "mvt_minus_lobt",
    "queue_ahead", "dep_tko_prev_30m", "rwy_headway_prev_s", "arr_taxi_in_mean_30m",
    "nbr_mvt_minus_aobt_mean_30m", "nbr_aobt_delay_mean_30m",
    "stand_rwy_share", "planned_flight_s", "dest_distance_km",
    "deicing_gate", "low_vis", "active_runways_60m", "bank_position",
]

def hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)

oof = pl.scan_parquet(OOF).filter(pl.col("oof").is_not_null())
df = (
    pl.scan_parquet(FEAT).select(COLS)
    .join(oof, on="MVT_ID_mvt", how="inner")
    .with_columns(
        y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64),
        proxy=pl.col("mvt_minus_aobt3"),
        ms=pl.col("mvt_minus_sched"),
        as_=pl.col("aobt3_minus_sched"),
    )
    .with_columns(
        err=pl.col("y") - pl.col("oof"),
        d=pl.col("y") - pl.col("mvt_minus_aobt3"),
    )
    .collect()
)

hdr("0. Frame")
print("oof non-null rows joined:", df.height)
matched = df.filter(pl.col("proxy").is_not_null())
unm = df.filter(pl.col("proxy").is_null())
print("matched:", matched.height, " unmatched:", unm.height)
for nm, f in (("all", df), ("matched", matched), ("unmatched", unm)):
    sse = float((f["err"] ** 2).sum())
    print(f"  {nm:9s} n={f.height:7d} RMSE={(sse/f.height)**0.5:10.2f} SSE={sse:.6e} mean_err={float(f['err'].mean()):8.2f}")
TOT_SSE = float((df["err"] ** 2).sum())
M = matched
N = M.height
M_SSE = float((M["err"] ** 2).sum())
print(f"matched SSE share of total: {M_SSE/TOT_SSE:.4f}")
print("months present:", sorted(M["month"].unique().to_list()))
print("rows per month:", M.group_by("month").len().sort("month").to_dicts())

# ---------------------------------------------------------------- derived keys
M = M.with_columns(
    stand=pl.col("STAND_mvt").fill_null("<null>"),
    rwy=pl.col("RUNWAY_mvt").fill_null("<null>"),
    actype=pl.col("AIRCRAFT_TYPE_mvt").fill_null("<null>"),
    wake=pl.col("WK_TBL_CAT_flt").fill_null("<null>"),
    op=pl.col("AIRCRAFT_OPERATOR_flt").fill_null("<null>"),
    seg=pl.col("MARKET_SEGMENT_flt").fill_null("<null>"),
    ftype=pl.col("FLIGHT_TYPE_flt").fill_null("<null>"),
    region=pl.col("ADES_REGION").fill_null("<null>"),
    fill=(pl.col("ms") - pl.col("y")).abs() <= 10,
)
def bin_expr(c, cuts, name):
    e = pl.when(pl.col(c).is_null()).then(pl.lit("null"))
    for i, cu in enumerate(cuts):
        e = e.when(pl.col(c) < cu).then(pl.lit(f"{i}:<{cu}"))
    return e.otherwise(pl.lit(f"{len(cuts)}:>={cuts[-1]}")).alias(name)

M = M.with_columns(
    bin_expr("ms", [0, 900, 1800, 3600, 7200], "ms_bin"),
    bin_expr("as_", [-900, 0, 900, 1800, 3600, 7200], "as_bin"),
    bin_expr("proxy", [400, 600, 800, 1000, 1200, 1500, 2000, 3000], "proxy_bin"),
    bin_expr("proxy", [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1350, 1500, 1750, 2000, 2500, 3500], "proxy_bin16"),
    bin_expr("queue_ahead", [5, 10, 15, 20, 30], "queue_bin"),
)

# ============================================================ 1. residual decomposition
def decomp(frame, cols, label, col="err"):
    """In-sample bias/noise split of sum of squares of `col` around 0."""
    g = frame.group_by(cols).agg(n=pl.len(), m=pl.col(col).mean())
    G = g.height
    between = float((g["n"] * g["m"] ** 2).sum())
    tot = float((frame[col] ** 2).sum())
    n = frame.height
    within_rmse = ((tot - between) / n) ** 0.5
    sigma2 = (tot - between) / n
    chance = G * sigma2                       # E[between] if every true group mean were 0
    return dict(grouping=label, groups=G, n=n,
                rmse=(tot / n) ** 0.5,
                between_share=between / tot,
                chance_share=chance / tot,
                excess_share=(between - chance) / tot,
                within_rmse=within_rmse)

def lomo(frame, cols, label, col="err", shrink=0.0, target_mode="offset"):
    """Leave-one-month-out honest gain from subtracting group means of `col`.
    target_mode 'offset': predict col by group mean (fallback global train mean)."""
    months = sorted(frame["month"].unique().to_list())
    sse_before = 0.0
    sse_after = 0.0
    n_tot = 0
    cov = 0
    for mth in months:
        tr = frame.filter(pl.col("month") != mth)
        te = frame.filter(pl.col("month") == mth)
        gm = float(tr[col].mean())
        g = tr.group_by(cols).agg(n=pl.len(), m=pl.col(col).mean())
        if shrink > 0:
            g = g.with_columns(m=(pl.col("m") * pl.col("n") + gm * shrink) / (pl.col("n") + shrink))
        te2 = te.join(g.select(cols + ["m"]), on=cols, how="left")
        cov += int(te2["m"].is_not_null().sum())
        pred = te2["m"].fill_null(gm)
        sse_before += float((te2[col] ** 2).sum())
        sse_after += float(((te2[col] - pred) ** 2).sum())
        n_tot += te.height
    return dict(grouping=label, n=n_tot,
                rmse_before=(sse_before / n_tot) ** 0.5,
                rmse_after=(sse_after / n_tot) ** 0.5,
                gain=(sse_before / n_tot) ** 0.5 - (sse_after / n_tot) ** 0.5,
                sse_reduction=1 - sse_after / sse_before,
                cover=cov / n_tot)

GROUPINGS = [
    (["ADEP_mvt"], "airport"),
    (["ADEP_mvt", "stand"], "airport x stand"),
    (["ADEP_mvt", "rwy"], "airport x runway"),
    (["ADEP_mvt", "stand", "rwy"], "airport x stand x runway"),
    (["op"], "operator"),
    (["seg"], "market segment"),
    (["actype"], "aircraft type"),
    (["wake"], "wake category"),
    (["ftype"], "flight type"),
    (["region"], "ADES region"),
    (["ADES_mvt"], "destination"),
    (["hour_utc"], "hour"),
    (["dow"], "weekday"),
    (["month"], "month"),
    (["ADEP_mvt", "hour_utc"], "airport x hour"),
    (["ADEP_mvt", "op"], "airport x operator"),
    (["ADEP_mvt", "actype"], "airport x actype"),
    (["ADEP_mvt", "wake"], "airport x wake"),
    (["ADEP_mvt", "seg"], "airport x segment"),
    (["ADEP_mvt", "stand", "rwy", "hour_utc"], "airport x stand x runway x hour"),
    (["ADEP_mvt", "stand", "rwy", "hour_utc", "actype"], "airport x stand x runway x hour x actype"),
    (["ADEP_mvt", "op", "hour_utc"], "airport x operator x hour"),
    (["ADEP_mvt", "stand", "op"], "airport x stand x operator"),
    (["ms_bin"], "ms bin"),
    (["as_bin"], "aobt3-sched bin"),
    (["proxy_bin"], "proxy bin"),
    (["ADEP_mvt", "proxy_bin"], "airport x proxy bin"),
    (["ADEP_mvt", "ms_bin"], "airport x ms bin"),
    (["ADEP_mvt", "op", "as_bin"], "airport x operator x aobt3-sched bin"),
    (["queue_bin"], "queue bin"),
    (["ADEP_mvt", "hour_utc", "dow"], "airport x hour x weekday"),
    (["ADEP_mvt", "month"], "airport x month"),
]

hdr("1. In-sample bias/noise split of matched squared error (err = y - oof, around 0)")
print(f"matched n={N}  RMSE={(M_SSE/N)**0.5:.2f}  mean err={float(M['err'].mean()):.2f}")
rows = [decomp(M, c, l) for c, l in GROUPINGS]
print(pl.DataFrame(rows).sort("excess_share", descending=True))

hdr("2. Leave-one-month-out honest recoverable bias (subtract group mean of err)")
rows = []
for c, l in GROUPINGS:
    rows.append(lomo(M, c, l))
print(pl.DataFrame(rows).sort("sse_reduction", descending=True))

hdr("2b. Same, with shrinkage k=50 toward the global mean")
rows = [lomo(M, c, l, shrink=50.0) for c, l in GROUPINGS]
print(pl.DataFrame(rows).sort("sse_reduction", descending=True))

# ============================================================ 3. share of SSE by group
hdr("3. Share of matched squared error, within-group RMSE, bias, by airport")
g = (M.group_by("ADEP_mvt").agg(
        n=pl.len(), sse=(pl.col("err") ** 2).sum(),
        mean=pl.col("err").mean(), med=pl.col("err").median(),
        sd=pl.col("err").std(), sd_d=pl.col("d").std(), mean_d=pl.col("d").mean())
     .with_columns(rmse=(pl.col("sse") / pl.col("n")) ** 0.5,
                   sse_share=pl.col("sse") / M_SSE,
                   n_share=pl.col("n") / N,
                   bias_share_of_own_mse=(pl.col("mean") ** 2) / (pl.col("sse") / pl.col("n")))
     .sort("sse_share", descending=True))
print(g)

for key, lab, topn, minn in [("seg", "market segment", 20, 1),
                             ("wake", "wake category", 20, 1),
                             ("ftype", "flight type", 20, 1),
                             ("hour_utc", "hour", 24, 1),
                             ("dow", "weekday", 7, 1),
                             ("month", "month", 12, 1),
                             ("region", "ADES region", 20, 1)]:
    hdr(f"3. by {lab}")
    print(M.group_by(key).agg(n=pl.len(), sse=(pl.col("err") ** 2).sum(),
                              mean=pl.col("err").mean(), sd=pl.col("err").std(),
                              mean_d=pl.col("d").mean(), sd_d=pl.col("d").std())
           .with_columns(rmse=(pl.col("sse") / pl.col("n")) ** 0.5,
                         sse_share=pl.col("sse") / M_SSE)
           .sort(key).head(topn))

hdr("3. top-20 operators by SSE share (n >= 500)")
print(M.group_by("op").agg(n=pl.len(), sse=(pl.col("err") ** 2).sum(),
                           mean=pl.col("err").mean(), sd=pl.col("err").std(),
                           mean_d=pl.col("d").mean(), sd_d=pl.col("d").std())
       .filter(pl.col("n") >= 500)
       .with_columns(rmse=(pl.col("sse") / pl.col("n")) ** 0.5, sse_share=pl.col("sse") / M_SSE)
       .sort("sse_share", descending=True).head(20))

hdr("3. top-20 aircraft types by SSE share (n >= 500)")
print(M.group_by("actype").agg(n=pl.len(), sse=(pl.col("err") ** 2).sum(),
                               mean=pl.col("err").mean(), sd=pl.col("err").std())
       .filter(pl.col("n") >= 500)
       .with_columns(rmse=(pl.col("sse") / pl.col("n")) ** 0.5, sse_share=pl.col("sse") / M_SSE)
       .sort("sse_share", descending=True).head(20))

# ============================================================ 4. stable bias subgroups
hdr("4. Subgroups with a large AND month-stable mean residual")
def stable(frame, cols, label, min_n=400, min_month_n=40):
    piv = (frame.group_by(cols + ["month"]).agg(n=pl.len(), m=pl.col("err").mean()))
    ok = (piv.group_by(cols).agg(
            months=pl.len(), min_mn=pl.col("n").min(),
            mmin=pl.col("m").min(), mmax=pl.col("m").max())
          .filter((pl.col("months") == 4) & (pl.col("min_mn") >= min_month_n)))
    tot = (frame.group_by(cols).agg(n=pl.len(), mean=pl.col("err").mean(),
                                    sd=pl.col("err").std(), sse=(pl.col("err") ** 2).sum())
           .filter(pl.col("n") >= min_n))
    j = (ok.join(tot, on=cols, how="inner")
           .with_columns(se=pl.col("sd") / pl.col("n").sqrt(),
                         same_sign=(pl.col("mmin") > 0) | (pl.col("mmax") < 0),
                         sse_share=pl.col("sse") / M_SSE,
                         bias_sse=pl.col("n") * pl.col("mean") ** 2)
           .with_columns(t=pl.col("mean") / pl.col("se"),
                         bias_share_of_matched_sse=pl.col("bias_sse") / M_SSE))
    j = j.filter(pl.col("same_sign")).sort("bias_sse", descending=True)
    print(f"--- {label}: {j.height} groups with n>={min_n}, all 4 months present (n>={min_month_n}) and same-sign monthly means")
    print(j.select(cols + ["n", "mean", "mmin", "mmax", "sd", "t", "sse_share", "bias_share_of_matched_sse"]).head(25))
    print(f"    total bias SSE of these groups: {float(j['bias_sse'].sum())/M_SSE:.4%} of matched SSE")
    return j

stable(M, ["ADEP_mvt"], "airport", min_n=400)
stable(M, ["ADEP_mvt", "rwy"], "airport x runway")
stable(M, ["ADEP_mvt", "stand"], "airport x stand", min_n=400, min_month_n=40)
stable(M, ["ADEP_mvt", "op"], "airport x operator")
stable(M, ["ADEP_mvt", "hour_utc"], "airport x hour")
stable(M, ["ADEP_mvt", "actype"], "airport x aircraft type")
stable(M, ["ADEP_mvt", "stand", "rwy"], "airport x stand x runway", min_n=300, min_month_n=30)
stable(M, ["op"], "operator")
stable(M, ["ADEP_mvt", "seg"], "airport x segment")

# ============================================================ 5. is d = AOBT_3 - BLOCK predictable?
hdr("5. The clock disagreement d = AOBT_3 - BLOCK = y - proxy")
dd = M["d"]
print(f"n={M.height} mean={float(dd.mean()):.2f} median={float(dd.median()):.1f} sd={float(dd.std()):.2f}")
print("BLOCK - AOBT_3 (= -d): mean %.2f median %.1f" % (-float(dd.mean()), -float(dd.median())))
q = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
print("quantiles of d:", {str(p): round(float(dd.quantile(p)), 1) for p in q})
print("share |d| <= 60 s:", float((dd.abs() <= 60).mean()))
print("share |d| <= 30 s:", float((dd.abs() <= 30).mean()))
print("share |d| > 600 s:", float((dd.abs() > 600).mean()))
print("sd of d excluding |d|>3600:", float(M.filter(pl.col("d").abs() <= 3600)["d"].std()),
      " n excl:", int((dd.abs() > 3600).sum()))
print("sd of y:", float(M["y"].std()), " sd of proxy:", float(M["proxy"].std()))
print("corr(y, proxy):", float(pl.DataFrame({"a": M["y"], "b": M["proxy"]}).select(pl.corr("a", "b")).item()))

hdr("5a. corr(d, visible numeric columns)")
num = ["proxy", "ms", "as_", "aobt3_minus_eobt1", "mvt_minus_eobt1", "mvt_minus_lobt",
       "queue_ahead", "dep_tko_prev_30m", "rwy_headway_prev_s", "arr_taxi_in_mean_30m",
       "nbr_mvt_minus_aobt_mean_30m", "nbr_aobt_delay_mean_30m", "stand_rwy_share",
       "planned_flight_s", "dest_distance_km", "active_runways_60m", "bank_position",
       "hour_utc", "dow", "month", "deicing_gate", "low_vis"]
out = []
for c in num:
    sub = M.select([pl.col("d"), pl.col("err"), pl.col(c).cast(pl.Float64).alias("x")]).drop_nulls()
    sub = sub.filter(pl.col("x").is_finite())
    r = sub.select(pl.corr("d", "x")).item()
    r2 = sub.select(pl.corr("err", "x")).item()
    out.append(dict(col=c, n=sub.height, corr_d=r, corr_err=r2))
print(pl.DataFrame(out).sort("corr_d"))

hdr("5b. Group-mean R2 of d (in-sample) and honest LOMO R2")
DG = [(["ADEP_mvt"], "airport"), (["op"], "operator"), (["hour_utc"], "hour"),
      (["actype"], "aircraft type"), (["wake"], "wake category"), (["seg"], "market segment"),
      (["ADEP_mvt", "stand"], "airport x stand"), (["ADEP_mvt", "rwy"], "airport x runway"),
      (["ADEP_mvt", "op"], "airport x operator"), (["ADEP_mvt", "hour_utc"], "airport x hour"),
      (["ADEP_mvt", "actype"], "airport x actype"),
      (["ADEP_mvt", "op", "hour_utc"], "airport x operator x hour"),
      (["ADEP_mvt", "stand", "rwy"], "airport x stand x runway"),
      (["ADEP_mvt", "op", "as_bin"], "airport x operator x aobt3-sched bin"),
      (["as_bin"], "aobt3-sched bin"), (["ms_bin"], "ms bin"),
      (["ADEP_mvt", "as_bin"], "airport x aobt3-sched bin"),
      (["ADEP_mvt", "op", "hour_utc", "as_bin"], "airport x op x hour x as bin"),
      (["ADEP_mvt", "stand", "rwy", "hour_utc", "actype"], "airport x stand x runway x hour x actype")]
d_var = float((M["d"] ** 2).sum())
rows = []
for c, l in DG:
    ins = decomp(M, c, l, col="d")
    lo = lomo(M, c, l, col="d")
    los = lomo(M, c, l, col="d", shrink=50.0)
    rows.append(dict(grouping=l, groups=ins["groups"],
                     insample_R2=ins["between_share"],
                     chance_R2=ins["chance_share"],
                     lomo_R2=lo["sse_reduction"], lomo_R2_shrunk=los["sse_reduction"],
                     rms_d_before=lo["rmse_before"], rms_d_after=min(lo["rmse_after"], los["rmse_after"]),
                     cover=lo["cover"]))
print(pl.DataFrame(rows).sort("lomo_R2_shrunk", descending=True))

hdr("5d. How much of d the model already recovers (implicit prediction = oof - proxy)")
dhat = M["oof"] - M["proxy"]
ref = float((((M["d"] - M["d"].mean()) ** 2).mean())) ** 0.5
print(f"RMS of d about its own mean (= RMSE of 'proxy + constant') : {ref:.2f}")
print(f"RMSE of the model on matched rows                          : {(M_SSE/N)**0.5:.2f}")
print(f"=> share of the clock-disagreement variance the model explains: {1 - (M_SSE/N)/ref**2:.4f}")
print("corr(oof - proxy, d) =", float(pl.DataFrame({"a": dhat, "b": M["d"]}).select(pl.corr("a", "b")).item()))
print("corr(err, d)         =", float(M.select(pl.corr("err", "d")).item()))
print("mean(oof - proxy) =", float(dhat.mean()), " mean(d) =", float(M["d"].mean()))
g = (M.with_columns(dhat=pl.col("oof") - pl.col("proxy"))
      .group_by("ADEP_mvt")
      .agg(n=pl.len(), sd_d=pl.col("d").std(), mean_d=pl.col("d").mean(),
           corr=pl.corr("dhat", "d"), mse=(pl.col("err") ** 2).mean())
      .with_columns(rmse=pl.col("mse") ** 0.5,
                    R2_vs_proxy_plus_const=1 - pl.col("mse") / (pl.col("sd_d") ** 2))
      .sort("R2_vs_proxy_plus_const", descending=True))
print(g.drop("mse"))

hdr("5c. d by airport and by ms/as bin")
print(M.group_by("ADEP_mvt").agg(n=pl.len(), mean_d=pl.col("d").mean(), med_d=pl.col("d").median(),
                                 sd_d=pl.col("d").std(),
                                 p01=pl.col("d").quantile(0.01), p99=pl.col("d").quantile(0.99),
                                 within60=(pl.col("d").abs() <= 60).mean())
       .sort("sd_d", descending=True))
print(M.group_by("as_bin").agg(n=pl.len(), mean_d=pl.col("d").mean(), med_d=pl.col("d").median(),
                               sd_d=pl.col("d").std(), rmse=((pl.col("err") ** 2).mean()) ** 0.5).sort("as_bin"))

# ============================================================ 6. ceiling
hdr("6. Ceiling: oracle group-mean predictors (in-sample and LOMO)")
def oracle(frame, cols, label):
    """RMSE of (a) group mean of y, (b) proxy + group mean of d; in-sample and LOMO."""
    n = frame.height
    ga = frame.group_by(cols).agg(my=pl.col("y").mean(), md=pl.col("d").mean(), n=pl.len())
    j = frame.join(ga, on=cols, how="left")
    ins_y = float(((j["y"] - j["my"]) ** 2).mean()) ** 0.5
    ins_d = float(((j["d"] - j["md"]) ** 2).mean()) ** 0.5
    months = sorted(frame["month"].unique().to_list())
    sy = sd_ = syk = sdk = 0.0
    for mth in months:
        tr = frame.filter(pl.col("month") != mth)
        te = frame.filter(pl.col("month") == mth)
        gy, gd = float(tr["y"].mean()), float(tr["d"].mean())
        g = tr.group_by(cols).agg(my=pl.col("y").mean(), md=pl.col("d").mean(), n=pl.len())
        gk = g.with_columns(myk=(pl.col("my") * pl.col("n") + gy * 50) / (pl.col("n") + 50),
                            mdk=(pl.col("md") * pl.col("n") + gd * 50) / (pl.col("n") + 50))
        t = te.join(gk.select(cols + ["my", "md", "myk", "mdk"]), on=cols, how="left")
        sy += float(((t["y"] - t["my"].fill_null(gy)) ** 2).sum())
        sd_ += float(((t["d"] - t["md"].fill_null(gd)) ** 2).sum())
        syk += float(((t["y"] - t["myk"].fill_null(gy)) ** 2).sum())
        sdk += float(((t["d"] - t["mdk"].fill_null(gd)) ** 2).sum())
    return dict(grouping=label, groups=ga.height,
                insample_meanY=ins_y, lomo_meanY=(sy / n) ** 0.5, lomo_meanY_shrunk=(syk / n) ** 0.5,
                insample_proxy_plus_meanD=ins_d, lomo_proxy_plus_meanD=(sd_ / n) ** 0.5,
                lomo_proxy_plus_meanD_shrunk=(sdk / n) ** 0.5)

OG = [(["ADEP_mvt"], "airport"),
      (["ADEP_mvt", "stand"], "airport x stand"),
      (["ADEP_mvt", "stand", "rwy"], "airport x stand x runway"),
      (["ADEP_mvt", "stand", "rwy", "hour_utc"], "airport x stand x runway x hour"),
      (["ADEP_mvt", "stand", "rwy", "hour_utc", "actype"], "airport x stand x runway x hour x actype"),
      (["ADEP_mvt", "op"], "airport x operator"),
      (["ADEP_mvt", "op", "hour_utc"], "airport x operator x hour"),
      (["proxy_bin16"], "proxy bin (16)"),
      (["ADEP_mvt", "proxy_bin16"], "airport x proxy bin"),
      (["ADEP_mvt", "stand", "rwy", "proxy_bin16"], "airport x stand x runway x proxy bin"),
      (["ADEP_mvt", "stand", "rwy", "hour_utc", "actype", "proxy_bin16"], "airport x stand x rwy x hour x actype x proxy bin"),
      (["ADEP_mvt", "op", "as_bin"], "airport x operator x aobt3-sched bin"),
      (["ADEP_mvt", "op", "hour_utc", "as_bin"], "airport x op x hour x aobt3-sched bin"),
      (["ADEP_mvt", "ms_bin", "as_bin"], "airport x ms bin x aobt3-sched bin"),
      (["ADEP_mvt", "stand", "rwy", "as_bin"], "airport x stand x runway x aobt3-sched bin")]
print(pl.DataFrame([oracle(M, c, l) for c, l in OG]))
print("\nreference: matched model RMSE = %.2f ; RMSE of the raw proxy as a predictor of y = %.2f"
      % ((M_SSE / N) ** 0.5, float(((M["y"] - M["proxy"]) ** 2).mean()) ** 0.5))
print("RMSE of proxy + global mean d = %.2f" % float((((M["d"] - M["d"].mean()) ** 2).mean()) ** 0.5))

# ============================================================ 7. tail concentration
hdr("7. Concentration of matched squared error in the tail")
Ms = M.with_columns(ae=pl.col("err").abs()).sort("ae", descending=True)
for frac, lab in [(0.0001, "top 0.01%"), (0.001, "top 0.1%"), (0.01, "top 1%"), (0.05, "top 5%"), (0.10, "top 10%")]:
    k = max(1, int(round(N * frac)))
    top = Ms.head(k)
    sse = float((top["err"] ** 2).sum())
    print(f"{lab:11s} k={k:6d}  SSE share of matched = {sse/M_SSE:7.4f}  cutoff |err| = {float(top['ae'].min()):8.1f}"
          f"  RMSE within = {(sse/k)**0.5:9.1f}  mean err = {float(top['err'].mean()):9.1f}")
k1 = max(1, int(round(N * 0.01)))
print("\nComposition of the top 1 pct (" + str(k1) + " rows):")
t1 = Ms.head(k1)
for key in ["ADEP_mvt", "seg", "ms_bin", "fill"]:
    print(f"-- {key}")
    print(t1.group_by(key).agg(n=pl.len(), sse=(pl.col("err") ** 2).sum(), mean=pl.col("err").mean())
            .with_columns(sse_share_of_matched=pl.col("sse") / M_SSE,
                          n_share_of_this_tail=pl.col("n") / k1).sort("sse", descending=True).head(12))
print("baseline airport mix of all matched rows:")
print(M.group_by("ADEP_mvt").agg(n=pl.len()).with_columns(share=pl.col("n") / N).sort("share", descending=True))

print("\nFlags in the tails (share of rows):")
flagdefs = [("fill (|BLOCK-SCHED|<=10s)", pl.col("fill")),
            ("|d| > 3600 s", pl.col("d").abs() > 3600),
            ("|d| > 1800 s", pl.col("d").abs() > 1800),
            ("|d| > 600 s", pl.col("d").abs() > 600),
            ("y >= 7200 s", pl.col("y") >= 7200),
            ("y >= 3600 s", pl.col("y") >= 3600),
            ("ms >= 7200 s", pl.col("ms") >= 7200),
            ("deicing_gate=1", pl.col("deicing_gate") == 1),
            ("LIRF", pl.col("ADEP_mvt") == "LIRF")]
rows = []
for lab, e in flagdefs:
    r = dict(flag=lab, all=float(M.select(e.mean()).item()))
    for frac, nm in [(0.01, "top1%"), (0.001, "top0.1%"), (0.0001, "top0.01%")]:
        k = max(1, int(round(N * frac)))
        r[nm] = float(Ms.head(k).select(e.mean()).item())
    rows.append(r)
print(pl.DataFrame(rows))

print("\nThe worst matched rows (top 0.01 pct):")
k = max(1, int(round(N * 0.0001)))
print(Ms.head(k).select(["ADEP_mvt", "month", "hour_utc", "op", "seg", "y", "oof", "err", "proxy", "d", "ms", "as_", "fill"]).head(40))

hdr("8. Error with the tail removed")
for cut in [600, 900, 1200, 1800, 3600]:
    sub = M.filter(pl.col("err").abs() <= cut)
    print(f"|err| <= {cut:5d}: n={sub.height:7d} ({sub.height/N:.4%}) RMSE={float(((sub['err']**2).mean()))**0.5:7.2f} "
          f"SSE share kept={float((sub['err']**2).sum())/M_SSE:.4f}")
for cut in [3600, 7200]:
    sub = M.filter(pl.col("d").abs() <= cut)
    print(f"|d|   <= {cut:5d}: n={sub.height:7d} ({sub.height/N:.4%}) RMSE={float(((sub['err']**2).mean()))**0.5:7.2f} "
          f"SSE share kept={float((sub['err']**2).sum())/M_SSE:.4f}")
sub = M.filter(~pl.col("fill"))
print(f"not filled     : n={sub.height:7d} RMSE={float(((sub['err']**2).mean()))**0.5:7.2f} SSE share kept={float((sub['err']**2).sum())/M_SSE:.4f}")

hdr("9. How much of the matched RMSE would month-stable group bias correction buy (combined)")
# sequential: subtract LOMO-shrunk group means for a few groupings, in order, measuring cumulative gain
def seq_lomo(frame, spec_list, k=50.0):
    """Cumulative honest gain from subtracting shrunk group means, one grouping at a time.
    Group means are estimated on the other three OOF months only."""
    months = sorted(frame["month"].unique().to_list())
    sse0 = 0.0
    n_tot = 0
    cum = {lab: 0.0 for _, lab in spec_list}
    for mth in months:
        tr = frame.filter(pl.col("month") != mth).with_columns(r=pl.col("err"))
        te = frame.filter(pl.col("month") == mth).with_columns(r=pl.col("err"))
        sse0 += float((te["err"] ** 2).sum())
        n_tot += te.height
        for cols, lab in spec_list:
            gm = float(tr["r"].mean())
            g = (tr.group_by(cols).agg(m=pl.col("r").mean(), gn=pl.len())
                   .with_columns(m=(pl.col("m") * pl.col("gn") + gm * k) / (pl.col("gn") + k))
                   .select(cols + ["m"]))
            tr = (tr.join(g, on=cols, how="left")
                    .with_columns(r=pl.col("r") - pl.col("m").fill_null(gm)).drop("m"))
            te = (te.join(g, on=cols, how="left")
                    .with_columns(r=pl.col("r") - pl.col("m").fill_null(gm)).drop("m"))
            cum[lab] += float((te["r"] ** 2).sum())
    print(f"baseline matched RMSE {(sse0/n_tot)**0.5:.2f}")
    for _, lab in spec_list:
        print(f"  after {lab:50s} -> {(cum[lab]/n_tot)**0.5:.2f}")


seq_lomo(M, [(["ADEP_mvt", "stand", "rwy"], "airport x stand x runway"),
             (["ADEP_mvt", "op"], "airport x operator"),
             (["ADEP_mvt", "hour_utc"], "airport x hour"),
             (["ADEP_mvt", "actype"], "airport x actype"),
             (["as_bin"], "aobt3-sched bin"),
             (["ADEP_mvt", "proxy_bin16"], "airport x proxy bin")])

print("\nDONE")
