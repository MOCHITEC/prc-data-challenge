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

    hr("S3b  is |bs|<=10 a real spike, or just the local density of bs?")
    # A 21-s window. Compare its count with the count in equal-width windows
    # displaced by +-600 s and +-1200 s at the same airport (ms >= 1800).
    rows = []
    for ap in sorted(big["ADEP_mvt"].unique().to_list()):
        s_ap = big.filter(pl.col("ADEP_mvt") == ap)
        at0 = int((s_ap["bs"].abs() <= 10).sum())
        nb = [int(((s_ap["bs"] - d).abs() <= 10).sum()) for d in (-1200, -600, 600, 1200)]
        base = float(np.mean(nb))
        rows.append(dict(ADEP=ap, n=s_ap.height, at0=at0, nbr_mean=round(base, 1),
                         excess=at0 - base, ratio=at0 / max(base, 0.5),
                         at0_unm=int((s_ap.filter(pl.col("unm"))["bs"].abs() <= 10).sum())))
    print(pl.DataFrame(rows).sort("excess", descending=True))
    print("exact bs == 0 count per airport (ms>=1800):")
    print(big.filter(pl.col("bs") == 0).group_by("ADEP_mvt").len().sort("len", descending=True))

    hr("S4c  the 'long taxi at large ms' question: P(y>=1800 | band, lane)")
    t = (big.group_by("band", "unm").agg(
        pl.len().alias("n"), (pl.col("y") >= 1800).mean().alias("p_y1800"),
        (pl.col("y") >= 3600).mean().alias("p_y3600"),
        (pl.col("y") >= 7200).mean().alias("p_y7200"), pl.col("y").mean().alias("y_mean")))
    print(band_sort(t, ["unm"]))
    print("\nbaseline over ALL training rows: P(y>=1800)=%.5f P(y>=3600)=%.5f mean y=%.1f" %
          (float((tr["y"] >= 1800).mean()), float((tr["y"] >= 3600).mean()), float(tr["y"].mean())))

    hr("S4d  per-band identifiability: AUC for y>=1800, each lane, each candidate")
    cand_mat = ["ms", "mvt_minus_aobt3", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt",
                "hour_utc", "dep_tko_prev_30m", "congestion_vs_usual_30m", "queue_ahead",
                "dest_distance_km", "planned_flight_s", "stand_null", "runway_null"]
    cand_unm = ["ms", "hour_utc", "dow", "dep_tko_prev_30m", "arr_land_prev_30m",
                "congestion_vs_usual_30m", "queue_ahead", "dest_distance_km",
                "planned_flight_s", "actype_null", "stand_null", "runway_null"]
    for unm, cands in [(False, cand_mat), (True, cand_unm)]:
        print(f"\n  lane = {'B unmatched' if unm else 'A matched'}")
        out = []
        for band in LABELS:
            s_b = big.filter((pl.col("band") == band) & (pl.col("unm") == unm))
            if s_b.height < 50:
                continue
            lab = (s_b["y"] >= 1800).to_numpy().astype(int)
            if lab.sum() < 5 or lab.sum() == len(lab):
                continue
            d = dict(band=band, n=s_b.height, base=round(float(lab.mean()), 4))
            for c in cands:
                d[c] = round(auc(s_b[c].to_numpy().astype(float), lab), 3)
            out.append(d)
        print(band_sort(pl.DataFrame(out)))
    print("\n  matched lane, RMSE of using mvt_minus_aobt3 directly as yhat (ms>=1800):")
    mm = big.filter(~pl.col("unm"))
    e = (mm["mvt_minus_aobt3"] - mm["y"]).to_numpy()
    print(f"    n={len(e)} RMSE={np.sqrt((e**2).mean()):.1f} bias={e.mean():.1f} "
          f"p25={np.percentile(e,25):.0f} p75={np.percentile(e,75):.0f}")

    hr("S4e  unmatched lane: does airport x band already say it? (lift table, no label)")
    u2 = big.filter(pl.col("unm")).with_columns(
        (pl.col("ADEP_mvt") == "LIRF").alias("is_lirf"))
    t = (u2.group_by("is_lirf", "band").agg(
            pl.len().alias("n"), (pl.col("y") >= 1800).mean().alias("p_y1800"),
            pl.col("y").mean().alias("y_mean"), pl.col("y").median().alias("y_p50"))
         .filter(pl.col("n") >= 20))
    print(band_sort(t, ["is_lirf"]))
    print("\n  and with actype_null (LIRF's unmatched rows are 99.9 % actype-null):")
    t = (u2.group_by("actype_null", "band").agg(
            pl.len().alias("n"), (pl.col("y") >= 1800).mean().alias("p_y1800"),
            pl.col("y").mean().alias("y_mean")).filter(pl.col("n") >= 20))
    print(band_sort(t, ["actype_null"]))

    # ------------------------------------------------------------ S5 OOF
    hr("S5  OOF: per band and per kind -- n, RMSE, bias, SSE share")
    oof = pl.read_parquet(OOF).filter(pl.col("oof").is_not_null())
    # NB: the folds sidecar has 863,141 rows for 672,428 OOF rows -- `jul` and
    # `jul_fwd` list the same 190,713 MVT_IDs, so joining it naively duplicates
    # every July row. Deduplicate to one fold label per MVT_ID.
    folds = pl.read_parquet(FOLDS).unique(subset=["MVT_ID_mvt"], keep="first")
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
    print("\nUNMATCHED lane by band x (LIRF / not) -- the cross-airport contamination check:")
    ju = jb.filter(pl.col("unm")).with_columns((pl.col("ADEP_mvt") == "LIRF").alias("is_lirf"))
    print(band_sort(agg(ju, ["is_lirf", "band"]).filter(pl.col("n") >= 10), ["is_lirf"]))
    print("\nMATCHED lane by band -- bias only:")
    print(band_sort(agg(jb.filter(~pl.col("unm")), ["band"])))
    print("\ntop-20 single rows by SSE within ms>=1800:")
    print(jb.sort("sse", descending=True).head(20).select(
        "ADEP_mvt", "FLIGHT_mvt", "band", "kind", "unm", "ms", "bs", "y", "oof", "sse"))
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

    hr("S8  the LIRF fill: is it stable, and is the CALLSIGN PREFIX a visible handle?")
    # FLIGHT_mvt is present on scored rows; AIRCRAFT_OPERATOR_flt (a model categorical)
    # is 100 % null in the unmatched lane, so the operator is invisible to model B today.
    print("FLIGHT_mvt null count: train %d / %d, rank %d / %d"
          % (int(tr["FLIGHT_mvt"].is_null().sum()), tr.height,
             int(rk["FLIGHT_mvt"].is_null().sum()), rk.height))
    lirf = big.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
    print("\nLIRF unmatched ms>=1800 fill rate by month:")
    print(lirf.group_by("month").agg(pl.len().alias("n"),
        (pl.col("kind") == "schedfill").mean().alias("fill_rate"),
        pl.col("y").mean().alias("y_mean")).sort("month"))
    print("\nrank frame months:", rk.group_by("month").len().sort("month").to_dicts())
    print("LIRF unmatched ALL ms (incl <30m) fill rate by month:")
    la = tr.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
    print(la.group_by("month").agg(pl.len().alias("n"),
        (pl.col("bs").abs() <= 10).mean().alias("fill_rate")).sort("month"))

    # out-of-time test of the prefix handle: fit the prefix fill-rate on months 1-6,
    # score months 7-12 (and the reverse).
    for tr_months, te_months, name in [([1,2,3,4,5,6], [7,8,9,10,11,12], "H1->H2"),
                                       ([7,8,9,10,11,12], [1,2,3,4,5,6], "H2->H1")]:
        A = lirf.filter(pl.col("month").is_in(tr_months))
        Bt = lirf.filter(pl.col("month").is_in(te_months))
        tab = (A.group_by("prefix").agg(pl.len().alias("n"),
               (pl.col("kind") == "schedfill").mean().alias("r")).filter(pl.col("n") >= 5))
        prior = float((A["kind"] == "schedfill").mean())
        sc = (Bt.join(tab, on="prefix", how="left")
                .with_columns(pl.col("r").fill_null(prior).alias("score")))
        lab = (sc["kind"] == "schedfill").to_numpy().astype(int)
        cov = float(sc["r"].is_not_null().mean())
        print(f"\n  {name}: fit n={A.height} ({len(tab)} prefixes n>=5), test n={Bt.height}, "
              f"prefix known on {cov:.3f}; AUC(prefix rate)={auc(sc['score'].to_numpy(), lab):.4f} "
              f"base={lab.mean():.4f}")
        # compare with ADES (a categorical the model already has) and with ms
        tab2 = (A.group_by("ADES_mvt").agg(pl.len().alias("n"),
                (pl.col("kind") == "schedfill").mean().alias("r2")).filter(pl.col("n") >= 5))
        sc2 = (Bt.join(tab2, on="ADES_mvt", how="left")
                 .with_columns(pl.col("r2").fill_null(prior).alias("s2")))
        print(f"       AUC(ADES rate)={auc(sc2['s2'].to_numpy(), lab):.4f}  "
              f"AUC(ms)={auc(Bt['ms'].to_numpy().astype(float), lab):.4f}  "
              f"AUC(hour)={auc(Bt['hour_utc'].to_numpy().astype(float), lab):.4f}")
        # what a prefix-conditional mixture prediction would give vs the model's own
        # (label-free at prediction time apart from the fitted table)
        q3 = sc.with_columns(pl.when(pl.col("score") >= 0.5).then(pl.col("ms"))
                             .otherwise(pl.lit(1000.0)).alias("mix"))
        e_mix = (q3["mix"] - q3["y"]).to_numpy()
        print(f"       hard-assign RMSE (ms if rate>=.5 else 1000s) = {np.sqrt((e_mix**2).mean()):.0f} "
              f"on {len(e_mix)} rows;  sd(y) = {float(q3['y'].std()):.0f}")

    print("\n  prefix table fitted on the WHOLE year, LIRF unmatched ms>=1800, n>=8:")
    print(lirf.group_by("prefix").agg(pl.len().alias("n"),
          (pl.col("kind") == "schedfill").mean().round(3).alias("fill_rate"),
          pl.col("y").median().alias("y_p50"), pl.col("ms").median().alias("ms_p50"))
          .filter(pl.col("n") >= 8).sort("n", descending=True))

    hr("S8b  is the model already at the mixture optimum on the LIRF unmatched cells?")
    ju2 = jb.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
    out = []
    for band in LABELS:
        c = ju2.filter(pl.col("band") == band)
        if c.height < 10:
            continue
        ybar = float(c["y"].mean())
        out.append(dict(band=band, n=c.height,
                        model_RMSE=float(np.sqrt((c["resid"] ** 2).mean())),
                        cellmean_RMSE=float(np.sqrt(((c["y"] - ybar) ** 2).mean())),
                        bias=float(c["resid"].mean()),
                        model_SSE=float(c["sse"].sum()),
                        cellmean_SSE=float(((c["y"] - ybar) ** 2).sum())))
    print(band_sort(pl.DataFrame(out)))
    print("NB cellmean uses the cell's own realised mean -- an in-sample oracle, a LOWER bound on SSE.")

    hr("S8c  rank-frame population of the identifiable cells")
    rl = rk.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= 1800))
    print(f"rank LIRF unmatched ms>=1800: n={rl.height}")
    tab = (lirf.group_by("prefix").agg(pl.len().alias("n_tr"),
           (pl.col("kind") == "schedfill").mean().alias("fill_rate")))
    jr = rl.join(tab, on="prefix", how="left")
    print(f"  prefix seen in training LIRF-unmatched table: {int(jr['fill_rate'].is_not_null().sum())} of {rl.height}")
    print(jr.group_by(pl.when(pl.col("fill_rate").is_null()).then(pl.lit("unseen"))
                      .when(pl.col("fill_rate") >= 0.8).then(pl.lit("fill>=0.8"))
                      .when(pl.col("fill_rate") >= 0.5).then(pl.lit("fill 0.5-0.8"))
                      .when(pl.col("fill_rate") >= 0.2).then(pl.lit("fill 0.2-0.5"))
                      .otherwise(pl.lit("fill<0.2")).alias("cls"))
          .agg(pl.len().alias("n"), pl.col("ms").median().alias("ms_p50")).sort("n", descending=True))
    print("\nrank unmatched non-LIRF ms>=1800 by band:")
    print(band_sort(rk.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF") & (pl.col("ms") >= 1800))
                    .group_by("band").agg(pl.len().alias("n"))))

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
    hr("S7b  PRICING BY VISIBLE CELL (lane x LIRF x band) -- the only honest unit")
    cells = jb.with_columns(pl.when(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).then(pl.lit("B-LIRF"))
                            .when(pl.col("unm")).then(pl.lit("B-other"))
                            .otherwise(pl.lit("A-matched")).alias("cell"))
    rcells = (rk.filter(pl.col("ms") >= 1800)
              .with_columns(pl.when(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF")).then(pl.lit("B-LIRF"))
                            .when(pl.col("unm")).then(pl.lit("B-other"))
                            .otherwise(pl.lit("A-matched")).alias("cell")))
    a = agg(cells, ["cell", "band"]).join(
        rcells.group_by("cell", "band").len().rename({"len": "n_rank"}), on=["cell", "band"], how="left")
    a = a.with_columns(
        pl.col("n_rank").fill_null(0),
        (pl.col("SSE") / pl.col("n")).alias("sse_per_row"))
    a = a.with_columns(
        (pl.col("n_rank") * pl.col("sse_per_row") / per_s).alias("s_if_perfect"),
        (pl.col("n_rank") * pl.col("bias") ** 2 / per_s).alias("s_if_bias_closed"))
    print(band_sort(a.select("cell", "band", "n", "RMSE", "bias", "sse_per_row", "n_rank",
                             "s_if_perfect", "s_if_bias_closed"), ["cell"]).sort("cell"))
    print("\ncell totals:")
    b = agg(cells, ["cell"]).join(rcells.group_by("cell").len().rename({"len": "n_rank"}), on="cell", how="left")
    b = b.with_columns((pl.col("SSE") / pl.col("n")).alias("sse_per_row"))
    print(b.with_columns((pl.col("n_rank") * pl.col("sse_per_row") / per_s).alias("s_if_perfect"),
                         (pl.col("n_rank") * pl.col("bias") ** 2 / per_s).alias("s_if_bias_closed"))
          .select("cell", "n", "RMSE", "bias", "n_rank", "s_if_perfect", "s_if_bias_closed"))
    print("\nsum over cells of s_if_bias_closed (per-band cells): %.4f s" % float(a["s_if_bias_closed"].sum()))

    hr("S7c  is the LIRF-unmatched over-prediction real? SE-adjusted, and per fold/month")
    # E[bhat^2] = b^2 + SE^2, so the unbiased estimate of the recoverable squared bias
    # is max(0, bhat^2 - SE^2). Anything at or below zero is noise, not bias.
    aa = a.filter(pl.col("cell") == "B-LIRF").with_columns(
        (pl.col("RMSE") / pl.col("n").sqrt()).alias("se"))
    aa = aa.with_columns((pl.col("bias").abs() / pl.col("se")).alias("t"),
        ((pl.col("bias") ** 2 - pl.col("se") ** 2).clip(0) * pl.col("n_rank") / per_s).alias("s_shrunk"))
    print(band_sort(aa.select("band", "n", "bias", "se", "t", "n_rank", "s_if_bias_closed", "s_shrunk")))
    print("  sum s_shrunk over B-LIRF bands 75-90m..4-6h = %.3f s; incl >=12h = %.3f s"
          % (float(aa.filter(pl.col("band").is_in(["75-90m", "90-120m", "2-3h", "3-4h", "4-6h"]))["s_shrunk"].sum()),
             float(aa["s_shrunk"].sum())))
    print("\n  LIRF unmatched ms>=1800, OOF bias per fold and per month:")
    ju3 = jb.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
    print(agg(ju3, ["fold"]).select("fold", "n", "RMSE", "bias", "y_mean", "pred_mean").sort("fold"))
    print(agg(ju3, ["month"]).select("month", "n", "RMSE", "bias", "y_mean", "pred_mean").sort("month"))
    print("\n  same for B-other (non-LIRF unmatched):")
    print(agg(jb.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF")), ["month"])
          .select("month", "n", "RMSE", "bias", "y_mean", "pred_mean").sort("month"))
    print("\n  realised schedfill rate among LIRF unmatched ms>=1800 by month, train vs OOF months:")
    print(big.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
          .group_by("month").agg(pl.len().alias("n"), (pl.col("kind") == "schedfill").mean().alias("fill"))
          .sort("month"))

    hr("S7d  months 1+7 (= the scored months) and a leave-one-month-out shift test")
    ju4 = jb.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))
    j17 = ju4.filter(pl.col("month").is_in([1, 7]))
    b17 = float(j17["resid"].mean()); r17 = float(np.sqrt((j17["resid"] ** 2).mean()))
    se17 = r17 / np.sqrt(j17.height)
    print(f"OOF months 1+7, LIRF unmatched ms>=1800: n={j17.height} RMSE={r17:.0f} "
          f"bias={b17:+.0f} SE={se17:.0f} t={abs(b17)/se17:.2f}")
    nr = rk.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= 1800)).height
    print(f"  value of removing that mean shift on the {nr} scored rows: "
          f"naive {nr*b17**2/per_s:.3f} s, SE-adjusted {nr*max(0.0, b17**2-se17**2)/per_s:.3f} s")
    print("\n  leave-one-month-out (estimate the shift on one month, apply to the other):")
    for est, app in [(1, 7), (7, 1)]:
        e = ju4.filter(pl.col("month") == est); ap = ju4.filter(pl.col("month") == app)
        sh = float(e["resid"].mean())
        before = float((ap["resid"] ** 2).sum()); after = float(((ap["resid"] - sh) ** 2).sum())
        print(f"    shift from month {est} ({sh:+.0f} s) applied to month {app} (n={ap.height}): "
              f"SSE {before:.3e} -> {after:.3e}  ({(before-after)/before:+.1%}); "
              f"scaled to {nr} scored rows = {(before-after)/ap.height*nr/per_s:+.3f} s")
    print("\n  and the same shift test on ALL LIRF unmatched rows (any ms) and on B-other:")
    for name, fr in [("LIRF unm all ms", j.filter(pl.col("unm") & (pl.col("ADEP_mvt") == "LIRF"))),
                     ("non-LIRF unm ms>=1800", jb.filter(pl.col("unm") & (pl.col("ADEP_mvt") != "LIRF")))]:
        for est, app in [(1, 7), (7, 1)]:
            e = fr.filter(pl.col("month") == est); ap = fr.filter(pl.col("month") == app)
            if e.height < 5 or ap.height < 5:
                continue
            sh = float(e["resid"].mean())
            before = float((ap["resid"] ** 2).sum()); after = float(((ap["resid"] - sh) ** 2).sum())
            print(f"    {name}: shift {sh:+.0f} from m{est} -> m{app} (n={ap.height}): "
                  f"{(before-after)/before:+.2%} of that cell's SSE")

    print("\nWhole ms>=1800 population:")
    sse_big = float(jb["sse"].sum())
    print(f"  OOF ms>=1800: n={jb.height} SSE={sse_big:.4e} = {sse_big/TOT:.4f} of total")
    print(f"  scaled to {rk_big} rank rows: perfect = {sse_big/jb.height*rk_big/per_s:.3f} s")


if __name__ == "__main__":
    main()
