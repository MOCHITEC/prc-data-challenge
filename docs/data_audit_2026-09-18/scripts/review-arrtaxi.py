"""Adversarial review of the claim "arr_taxi_in_mean_* is contaminated by the LIRF BLOCK:=SCHED fill".

No training. POLARS_MAX_THREADS=1. Lazy scans, minimal columns.

Sections
  1  is the feature shipped? exact column names, membership in num_cols / cols_a / cols_b
  2  is the LIRF arrival BLOCK==SCHED bucket a FILL or a coincidence of PUNCTUALITY?
     - per-airport census, taxi-in of the bucket vs the neighbouring-minute bands
     - taxi-in-stratified spike test (the decisive one)
     - where the extreme taxi-in values actually live (inside the bucket or outside?)
  3  how much does dropping the bucket move arr_taxi_in_mean_{30,60}m? (train and rank)
     - baseline recomputation is validated against the shipped feature cache first
  4  do the moved rows have unusual OOF residuals?
  5  would the fix starve any (airport, window) cell of rows?

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/data_audit_2026-09-18/scripts/review-arrtaxi.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
OOF = ROOT / "experiments" / "oof_main-48c19bc830.parquet"
TOL = 10  # seconds; the audit's f10.  f6 reported alongside.
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def hr(s: str) -> None:
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}")


def q(x: np.ndarray, ps=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> str:
    x = x[~np.isnan(x)]
    if not len(x):
        return "(empty)"
    return " ".join(f"{np.quantile(x, p):>9.1f}" for p in ps)


# --------------------------------------------------------------------------------------------------
# 1. Is the feature real and shipped?
# --------------------------------------------------------------------------------------------------
def section1() -> list[str]:
    hr("1. IS THE FEATURE SHIPPED?")
    from prc.features import CANDIDATE_COLS, numeric_feature_columns
    from prc.train import LABEL, PROXY, TrainConfig, visible_numeric_columns

    spec = yaml.safe_load((ROOT / "configs" / "main.yaml").read_text())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates", "unmatched_sched_reparam_airports"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    cfg = TrainConfig(**fields)
    print(f"configs/main.yaml -> exclude_features={cfg.exclude_features!r} include_candidates={cfg.include_candidates!r}")
    print(f"use_encodings={cfg.use_encodings} use_offset_encoding={getattr(cfg, 'use_offset_encoding', None)}")

    head = pl.scan_parquet(WORK / "feat_train.parquet").head(0).collect()
    num_cols = numeric_feature_columns(head)
    vis = visible_numeric_columns(head, cfg)
    cols_a = [c for c in vis if c != PROXY] + [PROXY]
    cols_b = [
        c
        for c in vis
        if not c.startswith(("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt"))
    ]
    fam = [c for c in head.columns if c.startswith("arr_taxi_in")]
    print(f"\ncolumns in the cached feature frame matching ^arr_taxi_in: {fam}")
    print(f"len(numeric_feature_columns)={len(num_cols)}  len(visible_numeric_columns)={len(vis)}  "
          f"len(cols_a)={len(cols_a)}  len(cols_b)={len(cols_b)}")
    print(f"LABEL={LABEL}  PROXY={PROXY}  (CANDIDATE_COLS has {len(CANDIDATE_COLS)} gated cols; "
          f"arr_taxi_in in CANDIDATE_COLS: {any(c.startswith('arr_taxi_in') for c in CANDIDATE_COLS)})")
    print("\n| column | in numeric_feature_columns | in visible (model A/B pool) | in cols_a (model A) | in cols_b (model B) |")
    print("|---|---|---|---|---|")
    for c in fam:
        print(f"| `{c}` | {c in num_cols} | {c in vis} | {c in cols_a} | {c in cols_b} |")
    rk = pl.scan_parquet(WORK / "feat_rank.parquet").head(0).collect()
    print(f"\nsame columns present in feat_rank.parquet: {[c for c in rk.columns if c.startswith('arr_taxi_in')]}")
    return fam


# --------------------------------------------------------------------------------------------------
# raw arrival / departure loaders
# --------------------------------------------------------------------------------------------------
def arrivals(which: str) -> pl.DataFrame:
    src = (
        sorted(RAW.glob("training_*.parquet")) if which == "train" else [RAW / "ranking.parquet"]
    )
    d = (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()
    return (
        pl.scan_parquet(src)
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            pl.col("ADES_mvt"),
            pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t_inblock"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("taxi_in"),
            d.cast(pl.Float64).alias("d"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon"),
        )
        .collect()
    )


def departures(which: str) -> pl.DataFrame:
    src = (
        sorted(RAW.glob("training_*.parquet")) if which == "train" else [RAW / "ranking.parquet"]
    )
    return (
        pl.scan_parquet(src)
        .filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            pl.col("MVT_ID_mvt"),
            pl.col("ADEP_mvt"),
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"),
        )
        .collect()
    )


# --------------------------------------------------------------------------------------------------
# 2. fill or punctuality?
# --------------------------------------------------------------------------------------------------
def section2(arr_tr: pl.DataFrame, arr_rk: pl.DataFrame) -> None:
    hr("2. FILL OR PUNCTUALITY?  (the audit's central inference)")

    for name, a in (("TRAIN", arr_tr), ("RANK", arr_rk)):
        print(f"\n--- {name}: per-airport census of the BLOCK==SCHED bucket ---")
        print("| apt | n arr | n f6 | n f10 | rate f10 | local dens (+-1,2 min, f6) | spike f6 "
              "| taxi-in p50 f10 | p50 non-f10 | p95 f10 | p95 non-f10 | f10 >30m | f10 >1h | f10 >2h |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for apt in APTS:
            s = a.filter(pl.col("ADES_mvt") == apt)
            d = s["d"].to_numpy()
            ti = s["taxi_in"].to_numpy()
            f6, f10 = np.abs(d) <= 6, np.abs(d) <= TOL
            nb = np.mean([np.sum(np.abs(d - k) <= 6) for k in (-120, -60, 60, 120)])
            spike = np.sum(f6) / nb if nb else np.nan
            tif, tin = ti[f10], ti[~f10]
            print(f"| {apt} | {len(s):,} | {f6.sum():,} | {f10.sum():,} | {f10.mean():.4f} | {nb:,.1f} | {spike:.2f} "
                  f"| {np.nanmedian(tif):.0f} | {np.nanmedian(tin):.0f} | {np.nanquantile(tif, 0.95):.0f} "
                  f"| {np.nanquantile(tin, 0.95):.0f} | {np.nansum(tif > 1800):,} | {np.nansum(tif > 3600):,} "
                  f"| {np.nansum(tif > 7200):,} |")

    # ---- the decisive test: is the excess mass at d==0 concentrated in implausible taxi-in values?
    print("\n--- 2b. TAXI-IN-STRATIFIED SPIKE TEST ---")
    print("Within a taxi-in stratum, count rows with |d|<=6 against the mean count in the four")
    print("neighbouring-minute buckets (|d-/+60|<=6, |d-/+120|<=6).  PUNCTUALITY predicts a")
    print("stratum-independent ratio; a FILL predicts the ratio to climb with taxi-in, because a filled")
    print("row's 'taxi-in' is SCHED-landing (schedule buffer), not a taxi time.")
    bands = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1800), (1800, 3600), (3600, 1e9)]
    for name, a in (("TRAIN", arr_tr), ("RANK", arr_rk)):
        for apt in ("LIRF", "EGLL", "LTFM"):
            s = a.filter(pl.col("ADES_mvt") == apt)
            d, ti = s["d"].to_numpy(), s["taxi_in"].to_numpy()
            print(f"\n{name} {apt}:")
            print("| taxi-in band (s) | n in band | n |d|<=6 | local dens | spike | share of band that is f6 |")
            print("|---|---|---|---|---|---|")
            for lo, hi in bands:
                m = (ti >= lo) & (ti < hi)
                dm = d[m]
                n0 = np.sum(np.abs(dm) <= 6)
                nb = np.mean([np.sum(np.abs(dm - k) <= 6) for k in (-120, -60, 60, 120)])
                r = n0 / nb if nb else np.nan
                print(f"| {lo:,}-{hi:,.0f} | {m.sum():,} | {n0:,} | {nb:,.1f} | {r:>6.2f} | "
                      f"{(n0 / m.sum() if m.sum() else np.nan):.4f} |")

    # ---- what the excess rows carry
    print("\n--- 2c. IMPLIED PROFILE OF THE EXCESS ROWS AT LIRF ---")
    print("Assume the genuine (punctual) d~0 rows look like the neighbouring-minute rows.  Then")
    print("n_excess = n0 - n_local and mean_excess = (n0*mean0 - n_local*mean_local)/n_excess.")
    for name, a in (("TRAIN", arr_tr), ("RANK", arr_rk)):
        s = a.filter(pl.col("ADES_mvt") == "LIRF")
        d, ti = s["d"].to_numpy(), s["taxi_in"].to_numpy()
        m0 = np.abs(d) <= 6
        mnb = np.any([np.abs(d - k) <= 6 for k in (-120, -60, 60, 120)], axis=0)
        n0, nnb = m0.sum(), mnb.sum()
        n_local = nnb / 4.0
        n_ex = n0 - n_local
        mean0, mean_nb = np.nanmean(ti[m0]), np.nanmean(ti[mnb])
        mean_ex = (n0 * mean0 - n_local * mean_nb) / n_ex
        print(f"\n{name} LIRF: n(|d|<=6)={n0:,}  neighbour-minute rows={nnb:,} -> local density={n_local:,.1f}")
        print(f"  excess rows ~= {n_ex:,.0f}  ({n_ex / len(s):.4f} of all LIRF arrivals)")
        print(f"  mean taxi-in: bucket {mean0:.1f} s, neighbour minutes {mean_nb:.1f} s -> implied excess {mean_ex:.1f} s")
        print(f"  taxi-in quantiles (p01 p05 p25 p50 p75 p95 p99):")
        print(f"    bucket        {q(ti[m0])}")
        print(f"    neighbour min {q(ti[mnb])}")
        print(f"    all non-f10   {q(ti[np.abs(d) > TOL])}")

    # ---- where do the extreme values live?
    print("\n--- 2d. WHERE THE EXTREME TAXI-IN VALUES ACTUALLY LIVE ---")
    print("The claim is that values 'up to 87,244 s flow into arr_taxi_in_mean_*' BECAUSE of the fill.")
    print("If the extremes sit OUTSIDE the BLOCK==SCHED bucket, excluding the bucket does not remove them.")
    for name, a in (("TRAIN", arr_tr), ("RANK", arr_rk)):
        print(f"\n{name}: arrivals by taxi-in threshold, split by f10, all airports and LIRF only")
        print("| threshold | n all | of which f10 | share f10 | n LIRF | LIRF f10 | max value | max is f10? | apt of max |")
        print("|---|---|---|---|---|---|---|---|---|")
        d, ti, ap = a["d"].to_numpy(), a["taxi_in"].to_numpy(), a["ADES_mvt"].to_numpy()
        f10 = np.abs(d) <= TOL
        for thr in (1800, 3600, 7200, 43200, 86400):
            m = ti > thr
            ml = m & (ap == "LIRF")
            if m.sum() == 0:
                print(f"| >{thr:,} | 0 | - | - | 0 | - | - | - | - |")
                continue
            i = np.nanargmax(np.where(m, ti, np.nan))
            print(f"| >{thr:,} | {m.sum():,} | {(m & f10).sum():,} | {(m & f10).sum() / m.sum():.3f} | "
                  f"{ml.sum():,} | {(ml & f10).sum():,} | {ti[i]:,.0f} | {bool(f10[i])} | {ap[i]} |")


    # ---- the named extreme value, row by row
    print("\n--- 2e. THE INDIVIDUAL EXTREME ARRIVALS (taxi-in > 2 h), train, with their d = BLOCK - SCHED ---")
    s = arr_tr.filter(pl.col("taxi_in") > 7200).sort("taxi_in", descending=True)
    print(f"n = {s.height}")
    print("| apt | taxi_in | d | in filled bucket? |")
    print("|---|---|---|---|")
    for r in s.head(12).iter_rows(named=True):
        print(f"| {r['ADES_mvt']} | {r['taxi_in']:,.0f} | {r['d']:,.0f} | {abs(r['d']) <= TOL} |")
    print(s.group_by("ADES_mvt").agg(pl.len(), (pl.col("d").abs() <= TOL).sum().alias("f10")).sort("len", descending=True))


# --------------------------------------------------------------------------------------------------
# 3. how much does it move the feature?
# --------------------------------------------------------------------------------------------------
def recompute(dep: pl.DataFrame, arr: pl.DataFrame, drop_filled: bool) -> pl.DataFrame:
    """arr_taxi_in_mean_{30,60}m / _n_ by the shipped code path, optionally dropping the filled arrivals."""
    from prc.features import _window_mean

    out = []
    for apt in dep["ADEP_mvt"].unique().to_list():
        dpred = pl.col("ADEP_mvt").is_null() if apt is None else (pl.col("ADEP_mvt") == apt)
        apred = pl.col("ADES_mvt").is_null() if apt is None else (pl.col("ADES_mvt") == apt)
        tgt = dep.filter(dpred)
        a = arr.filter(apred)
        if drop_filled:
            a = a.filter(pl.col("d").abs() > TOL)
        t = tgt["t"].to_numpy()
        times, vals = a["t_inblock"].to_numpy(), a["taxi_in"].to_numpy()
        cols = {"MVT_ID_mvt": tgt["MVT_ID_mvt"].to_numpy()}
        for w in (30, 60):
            mean, n = _window_mean(t, times, vals, w * 60, lag_sec=0)
            cols[f"arr_taxi_in_mean_{w}m"] = mean
            cols[f"arr_taxi_in_n_{w}m"] = n
        out.append(pl.DataFrame(cols))
    return pl.concat(out, how="diagonal")


def section3(dep_tr, arr_tr, dep_rk, arr_rk) -> tuple[pl.DataFrame, pl.DataFrame]:
    hr("3. HOW MUCH DOES DROPPING THE FILLED ARRIVALS MOVE THE FEATURE?")

    res = {}
    for name, dep, arr, cache in (
        ("TRAIN", dep_tr, arr_tr, WORK / "feat_train.parquet"),
        ("RANK", dep_rk, arr_rk, WORK / "feat_rank.parquet"),
    ):
        base = recompute(dep, arr, drop_filled=False)
        fixed = recompute(dep, arr, drop_filled=True)
        shipped = (
            pl.scan_parquet(cache)
            .select("MVT_ID_mvt", "ADEP_mvt", "arr_taxi_in_mean_30m", "arr_taxi_in_mean_60m",
                    "arr_taxi_in_n_30m", "arr_taxi_in_n_60m")
            .collect()
        )
        # validation: does my recomputation reproduce the shipped cache?
        chk = shipped.join(base, on="MVT_ID_mvt", how="inner", suffix="_re")
        print(f"\n--- {name}: validation of the recomputation against the shipped cache "
              f"({chk.height:,} of {shipped.height:,} rows joined) ---")
        for c in ("arr_taxi_in_mean_30m", "arr_taxi_in_mean_60m", "arr_taxi_in_n_30m", "arr_taxi_in_n_60m"):
            a_, b_ = chk[c].cast(pl.Float64).to_numpy(), chk[f"{c}_re"].cast(pl.Float64).to_numpy()
            both_nan = np.isnan(a_) & np.isnan(b_)
            dd = np.abs(a_ - b_)
            worst = np.nanmax(np.where(both_nan, 0.0, dd))
            n_mismatch = int(np.sum(~both_nan & ~(dd <= 0.51)))
            print(f"  {c:26s} max |shipped-recomputed| = {worst:.4g}   rows differing by >0.51: {n_mismatch:,} "
                  f"  null agreement: {np.sum(np.isnan(a_) == np.isnan(b_)):,}/{len(a_):,}")

        j = (
            shipped.select("MVT_ID_mvt", "ADEP_mvt")
            .join(base.rename({c: f"{c}_b" for c in base.columns if c != "MVT_ID_mvt"}), on="MVT_ID_mvt")
            .join(fixed.rename({c: f"{c}_f" for c in fixed.columns if c != "MVT_ID_mvt"}), on="MVT_ID_mvt")
        )
        for w in (30, 60):
            j = j.with_columns(
                (pl.col(f"arr_taxi_in_mean_{w}m_f") - pl.col(f"arr_taxi_in_mean_{w}m_b")).alias(f"delta_{w}m")
            )
        res[name] = j

        for w in (30, 60):
            print(f"\n--- {name}: delta = mean_{w}m(filled dropped) - mean_{w}m(shipped), per airport ---")
            print("| apt | n dep | n delta!=0 | share | |d|>1s | >10s | >30s | >60s | >300s | new nulls "
                  "| delta p01 | p25 | p50 | p75 | p99 | min | max | mean feat |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for apt in APTS + [None]:
                s = j if apt is None else j.filter(pl.col("ADEP_mvt") == apt)
                dd = s[f"delta_{w}m"].to_numpy()
                bb = s[f"arr_taxi_in_mean_{w}m_b"].to_numpy()
                ff = s[f"arr_taxi_in_mean_{w}m_f"].to_numpy()
                newnull = int(np.sum(~np.isnan(bb) & np.isnan(ff)))
                nz = np.sum(np.abs(np.nan_to_num(dd)) > 1e-9)
                a_ = dd[~np.isnan(dd)]
                lbl = apt or "ALL"
                if not len(a_):
                    print(f"| {lbl} | {s.height:,} | 0 | - | - | - | - | - | - | {newnull:,} | - | - | - | - | - | - | - | - |")
                    continue
                print(f"| {lbl} | {s.height:,} | {nz:,} | {nz / max(s.height,1):.4f} | "
                      f"{np.sum(np.abs(a_) > 1):,} | {np.sum(np.abs(a_) > 10):,} | {np.sum(np.abs(a_) > 30):,} | "
                      f"{np.sum(np.abs(a_) > 60):,} | {np.sum(np.abs(a_) > 300):,} | {newnull:,} | "
                      f"{np.quantile(a_, .01):.1f} | {np.quantile(a_, .25):.1f} | {np.quantile(a_, .5):.1f} | "
                      f"{np.quantile(a_, .75):.1f} | {np.quantile(a_, .99):.1f} | {a_.min():.0f} | {a_.max():.0f} | "
                      f"{np.nanmean(bb):.0f} |")
    return res["TRAIN"], res["RANK"]


# --------------------------------------------------------------------------------------------------
# 4. does it matter for the score?
# --------------------------------------------------------------------------------------------------
def section4(jt: pl.DataFrame) -> None:
    hr("4. DO THE AFFECTED ROWS HAVE UNUSUAL OOF RESIDUALS?")
    lab = (
        pl.scan_parquet(WORK / "feat_train.parquet")
        .select("MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3")
        .collect()
    )
    oof = pl.scan_parquet(OOF).filter(pl.col("oof").is_not_null()).collect()
    d = (
        jt.select("MVT_ID_mvt", "ADEP_mvt", "delta_30m", "delta_60m",
                  "arr_taxi_in_mean_30m_b", "arr_taxi_in_n_30m_b")
        .join(lab, on="MVT_ID_mvt")
        .join(oof, on="MVT_ID_mvt", how="inner")
        .with_columns(
            (pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - pl.col("oof")).alias("resid"),
            pl.col("mvt_minus_aobt3").is_null().alias("unmatched"),
        )
    )
    print(f"scored OOF rows joined: {d.height:,}  (overall RMSE {np.sqrt(np.mean(d['resid'].to_numpy() ** 2)):.2f})")

    def tab(frame: pl.DataFrame, title: str) -> None:
        print(f"\n--- {title} ---")
        print("| bucket | n | share of rows | RMSE | SSE share | mean resid | |resid| p50 | p95 | max |")
        print("|---|---|---|---|---|---|---|---|---|")
        r_all = frame["resid"].to_numpy()
        sse_all = float(np.sum(r_all**2))
        buckets = [
            ("delta == 0", np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) < 1e-9),
            ("0 < |delta| <= 10", (np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) >= 1e-9)
             & (np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) <= 10)),
            ("10 < |delta| <= 60", (np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) > 10)
             & (np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) <= 60)),
            ("60 < |delta| <= 300", (np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) > 60)
             & (np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) <= 300)),
            ("|delta| > 300", np.abs(np.nan_to_num(frame["delta_30m"].to_numpy())) > 300),
        ]
        for lbl, m in buckets:
            if m.sum() == 0:
                print(f"| {lbl} | 0 | - | - | - | - | - | - | - |")
                continue
            r = r_all[m]
            print(f"| {lbl} | {m.sum():,} | {m.mean():.4f} | {np.sqrt(np.mean(r**2)):.1f} | "
                  f"{np.sum(r**2) / sse_all:.4f} | {r.mean():.1f} | {np.quantile(np.abs(r), .5):.0f} | "
                  f"{np.quantile(np.abs(r), .95):.0f} | {np.abs(r).max():.0f} |")

    tab(d, "all scored rows, bucketed by |delta_30m|")
    tab(d.filter(pl.col("ADEP_mvt") == "LIRF"), "LIRF only")
    tab(d.filter(pl.col("unmatched")), "unmatched rows only (model B)")
    tab(d.filter(pl.col("unmatched") & (pl.col("ADEP_mvt") == "LIRF")), "LIRF unmatched only")


# --------------------------------------------------------------------------------------------------
# 5. is the fix safe?
# --------------------------------------------------------------------------------------------------
def section5(jt: pl.DataFrame, jr: pl.DataFrame) -> None:
    hr("5. WOULD THE FIX STARVE ANY CELL OF ROWS?")
    for name, j in (("TRAIN", jt), ("RANK", jr)):
        for w in (30, 60):
            print(f"\n--- {name}: arr_taxi_in_n_{w}m before / after dropping the filled arrivals ---")
            print("| apt | n dep | n==0 before | n==0 after | new nulls | share new null | n p01 before | p01 after "
                  "| p50 before | p50 after | mean before | mean after |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for apt in APTS + [None]:
                s = j if apt is None else j.filter(pl.col("ADEP_mvt") == apt)
                nb = s[f"arr_taxi_in_n_{w}m_b"].to_numpy()
                nf = s[f"arr_taxi_in_n_{w}m_f"].to_numpy()
                newnull = int(np.sum((nb > 0) & (nf == 0)))
                lbl = apt or "ALL"
                print(f"| {lbl} | {s.height:,} | {int(np.sum(nb == 0)):,} | {int(np.sum(nf == 0)):,} | {newnull:,} | "
                      f"{newnull / max(s.height,1):.5f} | {np.quantile(nb, .01):.0f} | {np.quantile(nf, .01):.0f} | "
                      f"{np.quantile(nb, .5):.0f} | {np.quantile(nf, .5):.0f} | {nb.mean():.1f} | {nf.mean():.1f} |")


def main() -> None:
    section1()
    arr_tr, arr_rk = arrivals("train"), arrivals("rank")
    print(f"\narrival rows loaded: train {arr_tr.height:,}  rank {arr_rk.height:,}")
    section2(arr_tr, arr_rk)
    dep_tr, dep_rk = departures("train"), departures("rank")
    print(f"\ndeparture rows loaded: train {dep_tr.height:,}  rank {dep_rk.height:,}")
    jt, jr = section3(dep_tr, arr_tr, dep_rk, arr_rk)
    del dep_tr, dep_rk, arr_tr, arr_rk
    section4(jt)
    section5(jt, jr)


if __name__ == "__main__":
    main()
