"""Item 5 `matched-proxy-outliers` — independent re-derivation (no training).

Inputs : data/work/feat_train.parquet, data/work/feat_rank.parquet,
         experiments/oof_main-48c19bc830.parquet, outputs/main.parquet
Rules  : POLARS_MAX_THREADS=1, scan_parquet with explicit column selection only.

Definitions (plan.md §0):
  y      = TAXITIME_SEC_mvt (= MVT_TIME - BLOCK_TIME)
  proxy  = MVT_TIME - AOBT_3
  ms     = MVT_TIME - SCHED_TIME
  gap    = BLOCK_TIME - AOBT_3   (D's blk_minus_aobt; >0 means BLOCK later than AOBT_3)
  fill   = |BLOCK - SCHED| <= 10 s
  matched (routing) = AOBT_3 not null ; matched (D) = FLIGHT_ID_mvt not null
"""
import os
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
import polars as pl

pl.Config.set_tbl_rows(120); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260)
B = "/home/user/prc-data-challenge/"
H1, H2 = 3600, 7200

def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)

COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "TAXITIME_SEC_mvt", "AIRCRAFT_OPERATOR_flt",
        "FLIGHT_mvt", "month", "mvt_minus_aobt3", "mvt_minus_sched"]

def derive(lf):
    return lf.select(COLS).with_columns([
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        sec("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt").alias("y_chk"),
        sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("proxy"),
        sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("ms"),
        sec("BLOCK_TIME_UTC_mvt", "AOBT_3_flt").alias("gap"),
        sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt_sched"),
        sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt_eobt"),
        (sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").abs() <= 10).alias("fill"),
        pl.col("AOBT_3_flt").is_not_null().alias("m_route"),
        pl.col("FLIGHT_ID_mvt").is_not_null().alias("m_fid"),
    ])

# ----------------------------------------------------------------------------- A. training
print("################ A. TRAINING (feat_train.parquet) ################")
tr = derive(pl.scan_parquet(B + "data/work/feat_train.parquet")).collect()
print("rows", tr.height, "| PHASE values", tr["PHASE_mvt"].unique().to_list())
print("identity y == MVT-BLOCK violations (|diff|>1s):", (tr["y"] - tr["y_chk"]).abs().gt(1).sum(),
      "| y null:", tr["y"].is_null().sum())
print("matched routing (AOBT_3 not null):", tr["m_route"].sum(), "| matched D (FLIGHT_ID not null):", tr["m_fid"].sum(),
      "| FLIGHT_ID not null & AOBT_3 null:", (tr["m_fid"] & ~tr["m_route"]).sum())
# proxy feature consistency
print("proxy vs mvt_minus_aobt3 feature: max |diff| =",
      (tr["proxy"] - tr["mvt_minus_aobt3"]).abs().max(), "| ms vs mvt_minus_sched max |diff| =",
      (tr["ms"] - tr["mvt_minus_sched"]).abs().max())

mm = tr.filter(pl.col("m_route"))  # rows model A actually sees
print("\n--- A1. matched(routing) rows: |gap| thresholds ---")
print(mm.select([pl.len().alias("n_matched"),
                 (pl.col("gap").abs() > H1).sum().alias("n_gap_gt_1h"),
                 (pl.col("gap").abs() > 1800).sum().alias("n_gap_gt_30m"),
                 (pl.col("gap").abs() > 600).sum().alias("n_gap_gt_10m"),
                 (pl.col("gap").abs() > 600).mean().round(4).alias("rate_gap_gt_10m"),
                 (pl.col("gap").abs() > H1).mean().round(5).alias("rate_gap_gt_1h")]))
mmf = tr.filter(pl.col("m_fid") & pl.col("m_route"))
print("same with D's definition (FLIGHT_ID not null, AOBT_3 not null):",
      mmf.select([pl.len(), (pl.col("gap").abs() > H1).sum().alias("gt1h"), (pl.col("gap").abs() > 1800).sum().alias("gt30m")]).row(0))

print("\n--- A2. matched |gap| > 1h, by airport: what is wrong, BLOCK or AOBT_3? ---")
g1 = mm.filter(pl.col("gap").abs() > H1)
print(g1.group_by("ADEP_mvt").agg([
    pl.len().alias("n"),
    pl.col("y").median().alias("y_med"),
    pl.col("proxy").median().alias("proxy_med"),
    pl.col("gap").median().alias("gap_med"),
    (pl.col("gap") > 0).mean().round(2).alias("share_gap_pos(BLOCK>AOBT)"),
    pl.col("fill").mean().round(2).alias("fill_rate"),
    (pl.col("y") >= H2).mean().round(2).alias("share_y_ge_2h"),
    (pl.col("proxy") >= H2).mean().round(2).alias("share_proxy_ge_2h"),
    pl.col("aobt_sched").median().alias("aobt-sched_med"),
    pl.col("aobt_eobt").median().alias("aobt-eobt_med"),
    pl.col("ms").median().alias("ms_med"),
]).sort("n", descending=True))
print("total rows |gap|>1h:", g1.height, "| y>=2h among them:", (g1["y"] >= H2).sum(),
      "| proxy>=2h among them:", (g1["proxy"] >= H2).sum())

# airport baseline medians for context
print("\n--- A2b. matched baseline per airport (all rows): y_med, proxy_med, gap sd ---")
print(mm.group_by("ADEP_mvt").agg([pl.len().alias("n"), pl.col("y").median().alias("y_med"),
                                    pl.col("proxy").median().alias("proxy_med"),
                                    pl.col("gap").median().alias("gap_med"), pl.col("gap").std().round(0).alias("gap_sd"),
                                    (pl.col("gap").abs() > H1).sum().alias("n_gap_gt1h"),
                                    (pl.col("proxy") >= H2).sum().alias("n_proxy_ge2h"),
                                    (pl.col("y") >= H2).sum().alias("n_y_ge2h")]).sort("n", descending=True))

print("\n--- A3. LTFM |gap|>1h: sign split and label ---")
lt = g1.filter(pl.col("ADEP_mvt") == "LTFM")
print(lt.group_by(pl.col("gap") > 0).agg([pl.len(), pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"),
                                            pl.col("aobt_sched").median().alias("aobt-sched_med"), pl.col("aobt_eobt").median().alias("aobt-eobt_med"),
                                            pl.col("fill").mean().round(2).alias("fill")]))
print("LTFM |gap|>1h by month:", lt.group_by("month").len().sort("month").to_dict(as_series=False))
print("LTFM |gap|>1h top operators:", lt.group_by("AIRCRAFT_OPERATOR_flt").len().sort("len", descending=True).head(6).to_dict(as_series=False))

print("\n--- A4. EGLL/LFPG/EDDM |gap|>1h: sign split, fill, label ---")
for a in ["EGLL", "LFPG", "EDDM", "LIRF"]:
    s = g1.filter(pl.col("ADEP_mvt") == a)
    print(a, s.group_by(pl.col("gap") > 0).agg([pl.len(), pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"),
                                               pl.col("fill").mean().round(2).alias("fill"), pl.col("ms").median().alias("ms_med")]).sort("gap"))

print("\n--- A5. matched rows with y >= 2h (M: 103) and with proxy >= 2h ---")
ym = mm.filter(pl.col("y") >= H2)
print("matched y>=2h: n", ym.height, "| (y-proxy) median", (ym["y"] - ym["proxy"]).median(),
      "| fill rate", round(ym["fill"].mean(), 3), "| |gap|>1h share", round((ym["gap"].abs() > H1).mean(), 3))
print(ym.group_by("ADEP_mvt").len().sort("len", descending=True).head(8).to_dict(as_series=False))
pm = mm.filter(pl.col("proxy") >= H2)
print("matched proxy>=2h: n", pm.height, "| y median", pm["y"].median(), "| y>=2h share", round((pm["y"] >= H2).mean(), 3),
      "| |gap|>1h share", round((pm["gap"].abs() > H1).mean(), 3))
print(pm.group_by("ADEP_mvt").agg([pl.len(), pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"),
                                    (pl.col("y") >= H2).mean().round(2).alias("y_ge2h_share")]).sort("len", descending=True))

# ----------------------------------------------------------------------------- B. OOF
print("\n################ B. OOF (experiments/oof_main-48c19bc830.parquet) ################")
oof = pl.scan_parquet(B + "experiments/oof_main-48c19bc830.parquet").drop_nulls("oof").collect()
print("oof rows (non-null)", oof.height)
j = tr.join(oof, on="MVT_ID_mvt", how="inner").with_columns(((pl.col("y") - pl.col("oof")) ** 2).alias("se"))
tot = j["se"].sum()
print("joined rows", j.height, "| total RMSE", round((tot / j.height) ** 0.5, 2))
jm = j.filter(pl.col("m_route"))
totm = jm["se"].sum()
print("matched(routing) OOF rows", jm.height, "| matched RMSE", round((totm / jm.height) ** 0.5, 2),
      "| matched SSE share of total", round(totm / tot, 4))

def share(df, name):
    n = df.height
    print(f"{name:55s} n={n:6d} sse/total={df['se'].sum()/tot:.4f} sse/matched={df['se'].sum()/totm:.4f} "
          f"rmse_in={(df['se'].mean()**0.5 if n else float('nan')):8.0f} pred_med={df['oof'].median()} y_med={df['y'].median()} proxy_med={df['proxy'].median()}")

print("\n--- B1. SSE shares ---")
share(jm.filter((pl.col("proxy") - pl.col("y")).abs() > H1), "matched |proxy-y|>1h")
share(jm.filter(pl.col("gap").abs() > H1), "matched |BLOCK-AOBT_3|>1h")
share(jm.filter(pl.col("gap").abs() > 1800), "matched |BLOCK-AOBT_3|>30m")
share(jm.filter(pl.col("y") >= H2), "matched y>=2h")
share(jm.filter(pl.col("proxy") >= H2), "matched proxy>=2h  (EXTENSION)")
share(jm.filter((pl.col("proxy") >= H2) & (pl.col("y") < H2)), "  proxy>=2h & y<2h")
share(jm.filter((pl.col("proxy") >= H2) & (pl.col("y") >= H2)), "  proxy>=2h & y>=2h")
share(jm.filter((pl.col("proxy") < H2) & (pl.col("y") >= H2)), "  proxy<2h & y>=2h")
share(jm.filter(pl.col("oof") >= H2), "matched pred>=2h")
share(jm.filter((pl.col("oof") >= H2) & (pl.col("y") < H1)), "matched pred>=2h & y<1h (false alarm)")

print("\n--- B2. matched |gap|>1h in OOF, by airport ---")
print(jm.filter(pl.col("gap").abs() > H1).group_by("ADEP_mvt").agg([
    pl.len().alias("n"), (pl.col("se").sum() / tot).round(4).alias("sse_share_total"),
    (pl.col("se").mean() ** 0.5).round(0).alias("rmse_in"),
    pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"), pl.col("oof").median().alias("pred_med"),
    (pl.col("oof") - pl.col("proxy")).abs().median().alias("|pred-proxy|_med"),
    (pl.col("oof") - pl.col("y")).abs().median().alias("|pred-y|_med"),
]).sort("n", descending=True))

print("\n--- B3. EXTENSION: matched rows with proxy > 7,200 in OOF — RMSE vs y, by airport ---")
px = jm.filter(pl.col("proxy") > H2)
print("n", px.height, "| RMSE(pred, y)", round(px["se"].mean() ** 0.5, 1),
      "| RMSE(proxy, y)", round(((px["proxy"] - px["y"]) ** 2).mean() ** 0.5, 1),
      "| RMSE(pred, proxy)", round(((px["oof"] - px["proxy"]) ** 2).mean() ** 0.5, 1),
      "| sse share of total", round(px["se"].sum() / tot, 4))
print(px.group_by("ADEP_mvt").agg([pl.len().alias("n"), (pl.col("se").mean() ** 0.5).round(0).alias("rmse_pred_y"),
                                    pl.col("y").median().alias("y_med"), pl.col("proxy").median().alias("proxy_med"),
                                    pl.col("oof").median().alias("pred_med"), (pl.col("y") >= H2).sum().alias("n_y_ge2h"),
                                    (pl.col("oof") >= H2).sum().alias("n_pred_ge2h")]).sort("n", descending=True))
print("proxy>7200 rows, sorted by se, top 15:")
print(px.sort("se", descending=True).select(["MVT_ID_mvt", "ADEP_mvt", "month", "y", "proxy", "ms", "gap", "oof", "fill"]).head(15))

# ----------------------------------------------------------------------------- C. ranking
print("\n################ C. RANKING (feat_rank.parquet + outputs/main.parquet) ################")
rk = derive(pl.scan_parquet(B + "data/work/feat_rank.parquet")).collect()
main = pl.scan_parquet(B + "outputs/main.parquet").rename({"TAXITIME_SEC_mvt": "pred"}).collect()
print("feat_rank rows", rk.height, "| PHASE", rk["PHASE_mvt"].unique().to_list(), "| main.parquet rows", main.height)
print("ranking DEP BLOCK_TIME null rate", round(rk["BLOCK_TIME_UTC_mvt"].is_null().mean(), 4),
      "| TAXITIME null rate", round(rk["TAXITIME_SEC_mvt"].is_null().mean(), 4))
r = rk.join(main, on="MVT_ID_mvt", how="inner").with_columns(pl.col("pred").cast(pl.Float64))
print("joined", r.height, "| matched routing", r["m_route"].sum(), "| matched D", r["m_fid"].sum(),
      "| matched(D) with AOBT_3 null:", (r["m_fid"] & ~r["m_route"]).sum(), "| unmatched routing", (~r["m_route"]).sum())

print("\n--- C1. ranking matched rows predicted >= 2h: counts under each definition ---")
for nm, mask in [("routing (AOBT_3 not null)", pl.col("m_route")), ("D (FLIGHT_ID not null)", pl.col("m_fid"))]:
    s = r.filter(mask & (pl.col("pred") >= H2))
    s2 = r.filter(mask & (pl.col("pred") >= H2) & (pl.col("ms") >= H2))
    print(f"{nm}: pred>=2h n={s.height}  | with ms>=2h filter (as D q5 Q): n={s2.height}")
    print("   by airport:", s.group_by("ADEP_mvt").len().sort("len", descending=True).to_dict(as_series=False))
print("unmatched(routing) pred>=2h n =", r.filter(~pl.col("m_route") & (pl.col("pred") >= H2)).height,
      "| all rows pred>=2h n =", (r["pred"] >= H2).sum())

print("\n--- C2. EXTENSION: list of ranking matched (routing) rows predicted >= 2h ---")
lst = r.filter(pl.col("m_route") & (pl.col("pred") >= H2)).select(
    ["MVT_ID_mvt", "ADEP_mvt", "month", "FLIGHT_mvt", "AIRCRAFT_OPERATOR_flt", "pred", "proxy", "ms", "aobt_sched", "aobt_eobt"]
).with_columns((pl.col("pred") - pl.col("proxy")).alias("pred-proxy")).sort(["ADEP_mvt", "pred"], descending=[False, True])
print(lst)
print("summary: pred-proxy median", lst["pred-proxy"].median(), "| |pred-proxy| max", lst["pred-proxy"].abs().max(),
      "| proxy>=2h among them", (lst["proxy"] >= H2).sum(), "| ms>=2h among them", (lst["ms"] >= H2).sum())
print("matched(D) rows with AOBT_3 null (routed to model B):")
print(r.filter(pl.col("m_fid") & ~pl.col("m_route")).select(["MVT_ID_mvt", "ADEP_mvt", "month", "ms", "pred"]))

print("\n--- C3. ranking matched proxy >= 2h: count by airport, and how many predicted >= 2h ---")
rp = r.filter(pl.col("m_route") & (pl.col("proxy") >= H2))
print("ranking matched proxy>=2h n =", rp.height, "| rate", round(rp.height / r["m_route"].sum(), 6),
      "| training matched proxy>=2h rate", round(pm.height / mm.height, 6))
print(rp.group_by("ADEP_mvt").agg([pl.len().alias("n"), pl.col("proxy").median().alias("proxy_med"), pl.col("pred").median().alias("pred_med"),
                                    (pl.col("pred") >= H2).sum().alias("n_pred_ge2h"), pl.col("ms").median().alias("ms_med")]).sort("n", descending=True))

print("\n--- C4. 'same kind?' — training matched rows in the same (airport, proxy>=2h) cell: y outcome & OOF pred ---")
cell = jm.filter(pl.col("proxy") >= H2).group_by("ADEP_mvt").agg([
    pl.len().alias("n_oof"), pl.col("y").median().alias("y_med"), (pl.col("y") >= H2).mean().round(2).alias("y_ge2h_share"),
    pl.col("oof").median().alias("pred_med"), (pl.col("oof") >= H2).mean().round(2).alias("pred_ge2h_share"),
    (pl.col("se").mean() ** 0.5).round(0).alias("rmse")])
print(cell.sort("n_oof", descending=True))
print("training (all folds) matched proxy>=2h by airport, y>=2h share:")
print(pm.group_by("ADEP_mvt").agg([pl.len().alias("n_train"), (pl.col("y") >= H2).mean().round(2).alias("y_ge2h_share"),
                                    pl.col("y").median().alias("y_med")]).sort("n_train", descending=True))

print("\n--- C5. ranking matched |proxy - ms| — do the 22 have AOBT_3 consistent with SCHED? (training reference: proxy>=2h rows) ---")
print("ranking 22: aobt-sched quantiles", lst["aobt_sched"].quantile(0.1), lst["aobt_sched"].median(), lst["aobt_sched"].quantile(0.9))
print("training matched proxy>=2h: aobt-sched quantiles", pm["aobt_sched"].quantile(0.1), pm["aobt_sched"].median(), pm["aobt_sched"].quantile(0.9),
      "| split y>=2h vs y<2h aobt-sched median:",
      pm.filter(pl.col("y") >= H2)["aobt_sched"].median(), pm.filter(pl.col("y") < H2)["aobt_sched"].median())
print("DONE")

# ----------------------------------------------------------------------------- D. signature check
print("\n################ D. SIGNATURE of the 22 ranking rows, looked up in training ################")
print("Signature S1 (EGLL/LIRF-type): matched, proxy < 2,000 s, AOBT_3 - SCHED >= 5,400 s (1.5 h)")
sig1 = (pl.col("m_route")) & (pl.col("proxy") < 2000) & (pl.col("aobt_sched") >= 5400)
def sig_table(df, name, has_pred):
    aggs = [pl.len().alias("n"), pl.col("y").median().alias("y_med"), (pl.col("y") >= H2).mean().round(3).alias("y_ge2h_share"),
            ((pl.col("y") - pl.col("ms")).abs() <= 1800).mean().round(3).alias("y_within30m_of_ms"),
            (pl.col("gap").abs() > H1).mean().round(3).alias("gap_gt1h_share"), pl.col("fill").mean().round(3).alias("fill"),
            pl.col("ms").median().alias("ms_med"), pl.col("proxy").median().alias("proxy_med")]
    if has_pred:
        aggs += [pl.col("oof").median().alias("pred_med"), (pl.col("oof") >= H2).mean().round(3).alias("pred_ge2h_share"),
                 (pl.col("se").mean() ** 0.5).round(0).alias("rmse")]
    print(name); print(df.group_by("ADEP_mvt").agg(aggs).sort("n", descending=True))
sig_table(tr.filter(sig1), "training, S1:", False)
sig_table(j.filter(sig1), "OOF, S1:", True)
print("ranking S1 count by airport:", r.filter(sig1).group_by("ADEP_mvt").agg([pl.len(), (pl.col("pred") >= H2).sum().alias("n_pred_ge2h"), pl.col("pred").median().alias("pred_med")]).sort("len", descending=True).to_dict(as_series=False))
print("training S1 total", tr.filter(sig1).height, "of", tr["m_route"].sum(), "| ranking S1 total", r.filter(sig1).height, "of", r["m_route"].sum())
print("\nSignature S2 (EHAM-Jan-type): matched, 4,000 <= proxy < 7,200")
sig2 = (pl.col("m_route")) & (pl.col("proxy") >= 4000) & (pl.col("proxy") < H2)
sig_table(tr.filter(sig2), "training, S2:", False)
sig_table(j.filter(sig2), "OOF, S2:", True)
print("ranking S2 count by airport:", r.filter(sig2).group_by("ADEP_mvt").agg([pl.len(), (pl.col("pred") >= H2).sum().alias("n_pred_ge2h"), pl.col("pred").median().alias("pred_med")]).sort("len", descending=True).to_dict(as_series=False))
print("EHAM ranking S2 rows, month split:", r.filter(sig2 & (pl.col("ADEP_mvt") == "EHAM")).group_by("month").len().to_dict(as_series=False))
print("\nEGLL training S1 rows by y-bucket (is y ~ ms or ~ proxy?):")
e = tr.filter(sig1 & (pl.col("ADEP_mvt") == "EGLL")).with_columns(
    pl.when((pl.col("y") - pl.col("proxy")).abs() <= 1800).then(pl.lit("y~proxy"))
      .when((pl.col("y") - pl.col("ms")).abs() <= 1800).then(pl.lit("y~ms")).otherwise(pl.lit("neither")).alias("kind"))
print(e.group_by("kind").agg([pl.len(), pl.col("y").median().alias("y_med"), pl.col("gap").median().alias("gap_med")]).sort("len", descending=True))
for a in ["LIRF", "LFPG", "LTFM"]:
    e = tr.filter(sig1 & (pl.col("ADEP_mvt") == a)).with_columns(
        pl.when((pl.col("y") - pl.col("proxy")).abs() <= 1800).then(pl.lit("y~proxy"))
          .when((pl.col("y") - pl.col("ms")).abs() <= 1800).then(pl.lit("y~ms")).otherwise(pl.lit("neither")).alias("kind"))
    print(a, e.group_by("kind").agg([pl.len(), pl.col("y").median().alias("y_med")]).sort("len", descending=True).to_dict(as_series=False))
print("DONE-D")
