"""What the large-`ms` rows are (2026-09-24). Read-only, no training.

`ms = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` (= `mvt_minus_sched`) is visible on
100 % of scored rows. `y = TAXITIME_SEC_mvt = MVT - BLOCK` exactly, so
`bs := BLOCK - SCHED = ms - y` -- a label-side quantity (BLOCK is null on all
344,841 scored rows). This script maps the ms >= 1800 s population, classifies it
by what the label says happened, and then asks whether anything visible at
prediction time separates the classes.

Inputs (never written):
  data/work/feat_train.parquet          2,085,047 rows, all PHASE_mvt == "DEP"
  data/work/feat_rank.parquet             344,841 rows
  experiments/oof_main_reparam-bc60017878.parquet  (+ _folds sidecar)

Run: POLARS_MAX_THREADS=4 .venv/bin/python \
     docs/competitor_research_2026-09-21/scripts/large_ms_population.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

BASE = "/home/user/prc-data-challenge/"
TRAIN = BASE + "data/work/feat_train.parquet"
RANK = BASE + "data/work/feat_rank.parquet"
OOF = BASE + "experiments/oof_main_reparam-bc60017878.parquet"
FOLDS = BASE + "experiments/oof_main_reparam-bc60017878_folds.parquet"

# ms bands, seconds. Edges chosen finer where the mass is (30 min - 2 h).
EDGES = [1800, 2700, 3600, 4500, 5400, 7200, 10800, 14400, 21600, 28800, 43200, 10**9]
LABELS = ["30-45m", "45-60m", "60-75m", "75-90m", "90-120m", "2-3h", "3-4h",
          "4-6h", "6-8h", "8-12h", ">=12h"]


def band_expr(col: str = "mvt_minus_sched") -> pl.Expr:
    e = pl.when(pl.col(col) < EDGES[0]).then(pl.lit("<30m"))
    for i, lab in enumerate(LABELS):
        e = e.when((pl.col(col) >= EDGES[i]) & (pl.col(col) < EDGES[i + 1])).then(pl.lit(lab))
    return e.otherwise(pl.lit("?")).alias("band")


BAND_ORDER = ["<30m"] + LABELS


def band_sort(df: pl.DataFrame, extra: list[str] | None = None) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("band").map_elements(lambda s: BAND_ORDER.index(s), return_dtype=pl.Int32).alias("_o")
    )
    return df.sort(["_o"] + (extra or [])).drop("_o")


# ---------------------------------------------------------------- kinds
# Ordered, mutually exclusive, defined ON THE LABEL (training only).
#   rollover  y > 72,000 (20 h)                     -- corrupted BLOCK date
#   schedfill |bs| <= 10 s (BLOCK == SCHED)         -- y == ms by construction
#   longtaxi  y >= 3,600                            -- genuine >1 h taxi
#   modtaxi   1,800 <= y < 3,600                    -- 30-60 min taxi
#   latepush  y < 1,800                             -- ms is pushback delay
def kind_expr() -> pl.Expr:
    return (
        pl.when(pl.col("y") > 72000).then(pl.lit("rollover"))
        .when(pl.col("bs").abs() <= 10).then(pl.lit("schedfill"))
        .when(pl.col("y") >= 3600).then(pl.lit("longtaxi"))
        .when(pl.col("y") >= 1800).then(pl.lit("modtaxi"))
        .otherwise(pl.lit("latepush")).alias("kind")
    )


KIND_ORDER = ["latepush", "modtaxi", "longtaxi", "schedfill", "rollover"]


def kind_sort(df: pl.DataFrame, extra: list[str] | None = None) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("kind").map_elements(lambda s: KIND_ORDER.index(s), return_dtype=pl.Int32).alias("_k")
    )
    return df.sort(["_k"] + (extra or [])).drop("_k")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC of `score` for separating label==1 from label==0."""
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n1, n0 = label.sum(), (1 - label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


TRAIN_COLS = [
    "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
    "FLIGHT_mvt", "FLIGHT_RULE_mvt", "TAXITIME_SEC_mvt", "mvt_minus_sched", "mvt_minus_aobt3",
    "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt", "nm_matched", "stand_null",
    "runway_null", "month", "dow", "hour_utc", "dep_tko_prev_30m", "arr_land_prev_30m",
    "congestion_vs_usual_30m", "queue_ahead", "rwy_headway_prev_s", "dest_distance_km",
    "planned_flight_s", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
]
RANK_COLS = [c for c in TRAIN_COLS]


def prep(path: str, train: bool) -> pl.DataFrame:
    lf = pl.scan_parquet(path).select(TRAIN_COLS if train else RANK_COLS)
    lf = lf.with_columns(
        pl.col("mvt_minus_aobt3").is_null().alias("unm"),
        pl.col("mvt_minus_sched").alias("ms"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("actype_null"),
        pl.col("FLIGHT_mvt").str.slice(0, 3).alias("prefix"),
    )
    if train:
        lf = lf.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
        lf = lf.with_columns((pl.col("ms") - pl.col("y")).alias("bs"))
        lf = lf.with_columns(kind_expr())
    lf = lf.with_columns(band_expr("ms"))
    return lf.collect()


def hr(t: str) -> None:
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def q(df: pl.DataFrame, c: str, p: float) -> float:
    v = df[c].quantile(p)
    return float("nan") if v is None else float(v)


def main() -> None:
    tr = prep(TRAIN, True)
    rk = prep(RANK, False)

    hr("S0  sanity + lane sizes")
    viol = (pl.scan_parquet(TRAIN)
            .select(((pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds()
                     - pl.col("TAXITIME_SEC_mvt")).abs().gt(1).sum()).collect().item())
    print(f"rows train={tr.height} rank={rk.height}; |y-(MVT-BLOCK)|>1s violations = {viol}")
    print(f"train unmatched={int(tr['unm'].sum())}  rank unmatched={int(rk['unm'].sum())}")
    print("NM clocks present on unmatched training rows:",
          {c: int(tr.filter(pl.col("unm"))[c].is_not_null().sum())
           for c in ["mvt_minus_aobt3", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt"]})
    print("nm_matched x unm:", tr.group_by("nm_matched", "unm").len().sort("nm_matched", "unm").to_dicts())
    print("_flt categoricals non-null on unmatched rows:",
          {c: int(tr.filter(pl.col("unm"))[c].is_not_null().sum())
           for c in ["AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt"]})

    hr("S1  the mass: ms band x lane -- n, y distribution, bs distribution")
    rows = []
    for band in BAND_ORDER:
        for unm in [False, True]:
            s = tr.filter((pl.col("band") == band) & (pl.col("unm") == unm))
            if s.height == 0:
                continue
            rows.append(dict(
                band=band, lane="B unm" if unm else "A mat", n=s.height,
                pct_of_lane=100 * s.height / max(1, tr.filter(pl.col("unm") == unm).height),
                y_p50=q(s, "y", .5), y_p90=q(s, "y", .9), y_p99=q(s, "y", .99),
                y_max=float(s["y"].max()),
                n_y_ge1800=int((s["y"] >= 1800).sum()), n_y_ge3600=int((s["y"] >= 3600).sum()),
                bs_p10=q(s, "bs", .1), bs_p50=q(s, "bs", .5), bs_p90=q(s, "bs", .9),
                frac_bs_neg=float((s["bs"] < 0).mean()),
            ))
    print(band_sort(pl.DataFrame(rows), ["lane"]))

    hr("S1b  same, per airport, ms >= 1800 only (n and share of the airport's ms>=1800 mass)")
    big = tr.filter(pl.col("ms") >= 1800)
    t = (big.group_by("ADEP_mvt", "band").agg(
            pl.len().alias("n"), pl.col("unm").sum().alias("n_unm"),
            pl.col("y").median().alias("y_p50"), (pl.col("y") >= 1800).sum().alias("n_y_ge1800"))
         .sort("ADEP_mvt"))
    print(band_sort(t, ["ADEP_mvt"]).pivot(on="band", index="ADEP_mvt", values="n", aggregate_function="sum"))
    print("\ntotals per airport, ms>=1800:")
    print(big.group_by("ADEP_mvt").agg(
        pl.len().alias("n"), pl.col("unm").sum().alias("n_unm"),
        (pl.col("y") >= 1800).sum().alias("n_y_ge1800"),
        (pl.col("bs").abs() <= 10).sum().alias("n_schedfill"),
        pl.col("y").median().alias("y_p50"), pl.col("bs").median().alias("bs_p50"),
    ).sort("n", descending=True))

    hr("S2  kinds: counts, share of band, per lane")
    t = (big.group_by("band", "kind").agg(pl.len().alias("n"), pl.col("unm").sum().alias("n_unm"))
         .with_columns((pl.col("n") / pl.col("n").sum().over("band")).alias("share_of_band")))
    print(band_sort(kind_sort(t), ["band"]).sort("band"))
    print("\noverall ms>=1800:")
    tot = big.height
    t2 = (big.group_by("kind").agg(pl.len().alias("n"), pl.col("unm").sum().alias("n_unm"),
                                   pl.col("y").median().alias("y_p50"), pl.col("bs").median().alias("bs_p50"))
          .with_columns((pl.col("n") / tot).alias("share")))
    print(kind_sort(t2))

    hr("S2b  kinds per airport (ms >= 1800), counts and per-airport rate")
    piv = big.group_by("ADEP_mvt", "kind").len().pivot(
        on="kind", index="ADEP_mvt", values="len", aggregate_function="sum").fill_null(0)
    print(piv.sort("ADEP_mvt"))
    print("\nrates (kind n / airport ms>=1800 n):")
    tot_ap = big.group_by("ADEP_mvt").len().rename({"len": "tot"})
    pivr = piv.join(tot_ap, on="ADEP_mvt")
    for k in KIND_ORDER:
        if k in pivr.columns:
            pivr = pivr.with_columns((pl.col(k) / pl.col("tot")).round(5).alias(k + "_r"))
    print(pivr.select(["ADEP_mvt", "tot"] + [k + "_r" for k in KIND_ORDER if k + "_r" in pivr.columns]).sort("ADEP_mvt"))

    hr("S2c  kinds x lane, and the unmatched lane broken out per airport")
    print(big.group_by("unm", "kind").agg(pl.len().alias("n")).sort("unm", "kind"))
    print("\nunmatched lane only, ms>=1800, per airport x kind:")
    print(big.filter(pl.col("unm")).group_by("ADEP_mvt", "kind").len().pivot(
        on="kind", index="ADEP_mvt", values="len", aggregate_function="sum").fill_null(0).sort("ADEP_mvt"))

    # ------------------------------------------------------------ S3 identifiability
    hr("S3  MATCHED lane: does AOBT_3 resolve the split? (ms>=1800, matched)")
    m = big.filter(~pl.col("unm"))
    d = (m["mvt_minus_aobt3"] - m["y"]).to_numpy()
    print(f"n={m.height}  mvt_minus_aobt3 - y : p1={np.percentile(d,1):.0f} p50={np.median(d):.0f} "
          f"p99={np.percentile(d,99):.0f}  |d|<=120s frac={np.mean(np.abs(d)<=120):.4f}  "
          f"|d|<=300s frac={np.mean(np.abs(d)<=300):.4f}  corr={np.corrcoef(m['mvt_minus_aobt3'].to_numpy(), m['y'].to_numpy())[0,1]:.4f}")
    for thr, name in [(1800, "y>=1800"), (3600, "y>=3600")]:
        lab = (m["y"].to_numpy() >= thr).astype(int)
        print(f"  AUC for {name}:  mvt_minus_aobt3 {auc(m['mvt_minus_aobt3'].to_numpy(), lab):.4f}"
              f" | ms {auc(m['ms'].to_numpy(), lab):.4f}"
              f" | mvt_minus_lobt {auc(m['mvt_minus_lobt'].to_numpy().astype(float), lab):.4f}"
              f" | mvt_minus_eobt1 {auc(m['mvt_minus_eobt1'].to_numpy().astype(float), lab):.4f}"
              f"   (base rate {lab.mean():.5f})")
    # simple rule: classify latepush vs taxi by AOBT3
    for thr in [1800, 3600]:
        pred = m["mvt_minus_aobt3"].to_numpy() >= thr
        act = m["y"].to_numpy() >= thr
        print(f"  rule mvt_minus_aobt3>={thr}: TP={int((pred&act).sum())} FP={int((pred&~act).sum())} "
              f"FN={int((~pred&act).sum())} TN={int((~pred&~act).sum())}")

    hr("S4  UNMATCHED lane: is anything visible? (ms>=1800, unmatched)")
    u = big.filter(pl.col("unm"))
    print(f"n={u.height}. Visible numeric candidates tested by AUC against each kind-vs-rest.")
    num_feats = ["ms", "hour_utc", "dow", "month", "dep_tko_prev_30m", "arr_land_prev_30m",
                 "congestion_vs_usual_30m", "queue_ahead", "rwy_headway_prev_s",
                 "dest_distance_km", "planned_flight_s", "actype_null", "stand_null", "runway_null"]
    for target in ["schedfill", "rollover", "longtaxi"]:
        lab = (u["kind"] == target).to_numpy().astype(int)
        if lab.sum() == 0:
            continue
        aucs = {f: auc(u[f].to_numpy().astype(float), lab) for f in num_feats}
        top = sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))
        print(f"  target={target} n1={int(lab.sum())} base={lab.mean():.4f} :",
              ", ".join(f"{k} {v:.3f}" for k, v in top[:8]))
    print("\n  ALSO: y>=1800 (any long taxi) in the unmatched lane:")
    lab = (u["y"] >= 1800).to_numpy().astype(int)
    aucs = {f: auc(u[f].to_numpy().astype(float), lab) for f in num_feats}
    print("   ", ", ".join(f"{k} {v:.3f}" for k, v in sorted(aucs.items(), key=lambda kv: -abs(kv[1] - 0.5))[:8]),
          f" base={lab.mean():.4f}")

    hr("S4b  the schedfill kind: where it lives and what marks it")
    sf = big.filter(pl.col("kind") == "schedfill")
    print("per airport x lane:")
    print(sf.group_by("ADEP_mvt", "unm").len().sort("len", descending=True))
    print("\nLIRF unmatched ms>=1800: schedfill rate vs visible splits")
    lirf = u.filter(pl.col("ADEP_mvt") == "LIRF")
    print(f"  n={lirf.height}, schedfill={int((lirf['kind']=='schedfill').sum())} "
          f"({(lirf['kind']=='schedfill').mean():.4f})")
    for col in ["RUNWAY_mvt", "prefix"]:
        g = (lirf.group_by(col).agg(pl.len().alias("n"), (pl.col("kind") == "schedfill").mean().alias("sf_rate"),
                                    pl.col("y").median().alias("y_p50"))
             .filter(pl.col("n") >= 5).sort("n", descending=True))
        print(f"\n  by {col}:"); print(g.head(20))
    lab = (lirf["kind"] == "schedfill").to_numpy().astype(int)
    print("\n  LIRF-only AUCs:", ", ".join(
        f"{f} {auc(lirf[f].to_numpy().astype(float), lab):.3f}" for f in
        ["ms", "hour_utc", "dow", "month", "dep_tko_prev_30m", "congestion_vs_usual_30m", "dest_distance_km"]))
    print("  ms-band contingency (LIRF unmatched):")
    print(band_sort(lirf.group_by("band").agg(
        pl.len().alias("n"), (pl.col("kind") == "schedfill").sum().alias("schedfill"),
        (pl.col("kind") == "rollover").sum().alias("rollover"),
        (pl.col("kind") == "latepush").sum().alias("latepush"),
        (pl.col("kind") == "longtaxi").sum().alias("longtaxi"),
        (pl.col("kind") == "modtaxi").sum().alias("modtaxi"))))

    hr("S4c  the 'long taxi at large ms' question: P(y>=1800 | band, lane)")
    t = (big.group_by("band", "unm").agg(
        pl.len().alias("n"), (pl.col("y") >= 1800).mean().alias("p_y1800"),
        (pl.col("y") >= 3600).mean().alias("p_y3600"),
        (pl.col("y") >= 7200).mean().alias("p_y7200"), pl.col("y").mean().alias("y_mean")))
    print(band_sort(t, ["unm"]))
    print("\nbaseline over ALL training rows: P(y>=1800)=%.5f P(y>=3600)=%.5f mean y=%.1f" %
          (float((tr["y"] >= 1800).mean()), float((tr["y"] >= 3600).mean()), float(tr["y"].mean())))

    # ------------------------------------------------------------ S5 OOF
    hr("S5  OOF: per band and per kind -- n, RMSE, bias, SSE share")
    oof = pl.read_parquet(OOF).filter(pl.col("oof").is_not_null())
    folds = pl.read_parquet(FOLDS)
    j = (tr.join(oof, on="MVT_ID_mvt", how="inner").join(folds, on="MVT_ID_mvt", how="left")
         .with_columns((pl.col("oof") - pl.col("y")).alias("resid"))
         .with_columns((pl.col("resid") ** 2).alias("sse")))
    TOT = float(j["sse"].sum())
    N = j.height
    RMSE = float(np.sqrt(TOT / N))
    print(f"OOF rows={N} RMSE={RMSE:.4f} total SSE={TOT:.4e}; 1 s of OOF RMSE = {N*(2*RMSE+1):.4e} SSE")

    def agg(df, keys):
        return (df.group_by(keys).agg(
            pl.len().alias("n"), pl.col("sse").sum().alias("SSE"),
            pl.col("resid").mean().alias("bias"), pl.col("y").mean().alias("y_mean"),
            pl.col("oof").mean().alias("pred_mean"))
            .with_columns((pl.col("SSE") / pl.col("n")).sqrt().alias("RMSE"),
                          (pl.col("SSE") / TOT).alias("sse_share"),
                          (pl.col("n") * pl.col("bias") ** 2).alias("bias_SSE")))

    print("\nby band x lane:")
    print(band_sort(agg(j, ["band", "unm"]), ["unm"]))
    jb = j.filter(pl.col("ms") >= 1800)
    print("\nby kind (ms>=1800) x lane:")
    print(kind_sort(agg(jb, ["kind", "unm"]), ["unm"]))
    print("\nby kind x band (ms>=1800), kinds with n>=20:")
    print(band_sort(kind_sort(agg(jb, ["kind", "band"]).filter(pl.col("n") >= 20)), ["band"]))
    print("\nlatepush only (the bulk), by band x lane -- is the model biased there?")
    print(band_sort(agg(jb.filter(pl.col("kind") == "latepush"), ["band", "unm"]), ["unm"]))
    print("\nlatepush, matched lane, by band, split on whether AOBT_3 agrees:")
    lp = jb.filter((pl.col("kind") == "latepush") & (~pl.col("unm")))
    print(band_sort(agg(lp, ["band"])))
    print("\nschedfill rows in OOF, per airport:")
    print(agg(jb.filter(pl.col("kind") == "schedfill"), ["ADEP_mvt"]).sort("SSE", descending=True))

    hr("S5b  the biggest single lever check: bias by band, matched lane, latepush")
    print("sum over ms>=1800 matched latepush: n=%d bias=%.2f biasSSE=%.3e (%.4f of total, %.4f s of OOF RMSE)"
          % (lp.height, float(lp["resid"].mean()), lp.height * float(lp["resid"].mean()) ** 2,
             lp.height * float(lp["resid"].mean()) ** 2 / TOT,
             lp.height * float(lp["resid"].mean()) ** 2 / (N * (2 * RMSE + 1))))

    # ------------------------------------------------------------ S6 rank
    hr("S6  the 2026 scored frame: same bands, same lanes, same airports")
    t = (rk.group_by("band", "unm").agg(pl.len().alias("n"), pl.col("ms").median().alias("ms_p50")))
    print(band_sort(t, ["unm"]))
    print(f"\nrank ms>=1800: {rk.filter(pl.col('ms')>=1800).height} of {rk.height} "
          f"({rk.filter(pl.col('ms')>=1800).height/rk.height:.4f}); "
          f"ms>=3600: {rk.filter(pl.col('ms')>=3600).height}")
    print("\nper airport, rank, ms>=1800:")
    print(rk.filter(pl.col("ms") >= 1800).group_by("ADEP_mvt").agg(
        pl.len().alias("n"), pl.col("unm").sum().alias("n_unm"),
        pl.col("actype_null").sum().alias("n_actype_null")).sort("n", descending=True))
    print("\nrank LIRF unmatched by band (the schedfill-candidate population):")
    print(band_sort(rk.filter((pl.col("ADEP_mvt") == "LIRF") & pl.col("unm") & (pl.col("ms") >= 1800))
                    .group_by("band").agg(pl.len().alias("n"))))
    print("\nrank matched rows ms>=1800 with mvt_minus_aobt3 >= 1800 (AOBT_3 says long taxi):")
    r2 = rk.filter((pl.col("ms") >= 1800) & (~pl.col("unm")))
    print(f"  n(ms>=1800, matched)={r2.height}; aobt3>=1800 -> {int((r2['mvt_minus_aobt3']>=1800).sum())}; "
          f"aobt3>=3600 -> {int((r2['mvt_minus_aobt3']>=3600).sum())}")

    # ------------------------------------------------------------ S7 pricing
    hr("S7  pricing")
    NB, RB = 344841, 282.7371
    per_s = NB * (2 * RB + 1)
    print(f"board: N={NB} RMSE={RB} -> 1 s of RMSE = {per_s:.4e} SSE")
    # per-kind per-row SSE from OOF, scaled by rank-frame population where identifiable
    oof_rate = agg(jb, ["kind"]).with_columns((pl.col("SSE") / pl.col("n")).alias("sse_per_row"))
    print(kind_sort(oof_rate.select("kind", "n", "RMSE", "bias", "sse_per_row", "bias_SSE")))
    rk_big = rk.filter(pl.col("ms") >= 1800).height
    print(f"\nrank ms>=1800 rows = {rk_big}")
    for row in oof_rate.iter_rows(named=True):
        share = row["n"] / jb.height
        n_rank = share * rk_big
        print(f"  kind={row['kind']:<10} oof share {share:.4f} -> ~{n_rank:8.0f} rank rows; "
              f"perfect = {n_rank*row['sse_per_row']/per_s:7.3f} s; "
              f"bias-only = {n_rank*row['bias']**2/per_s:7.4f} s; "
              f"10 % of SSE = {0.10*n_rank*row['sse_per_row']/per_s:7.3f} s")
    print("\nWhole ms>=1800 population:")
    sse_big = float(jb["sse"].sum())
    print(f"  OOF ms>=1800: n={jb.height} SSE={sse_big:.4e} = {sse_big/TOT:.4f} of total")
    print(f"  scaled to {rk_big} rank rows: perfect = {sse_big/jb.height*rk_big/per_s:.3f} s")


if __name__ == "__main__":
    main()
