"""ADS-B pushback override applied to the submission: EGLL leave-one-day-out check, production bias, candidate file.

Read-only with respect to the repo: nothing is written to `outputs/`, `src/`, `configs/` or `data/`; nothing is fitted
or submitted. The candidate goes to the scratch dir. Pre-registration: `docs/competitor_research_2026-09-21/results/
adsb-full-ingest-candidate.md` section 0, written before this was run.

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    .venv/bin/python scripts/adsb_override_candidate.py [--no-write]

Reads
  labelled 2025 days  `$SCRATCH/<tag>_joined.parquet` of 2025-07-21, 07-13, 01-22, 11-12 (stage-1 `analyse`), with
                      `data/work/feat_train.parquet` (label, lane) and the production OOF `experiments/oof_main_gap-
                      62dd7c6e88.parquet`. BLOCK is read here only (bias and scoring).
  scored 2026 days    `$SCRATCH/<tag>_joined.parquet` of every day `scripts/adsb_ingest_all.py` marked analysed; only
                      the columns MVT_ID_mvt, ADEP_mvt, t_mvt, join, AOBT_3_flt and the stage-1 detection columns are
                      kept, and BLOCK is not among them (asserted).
  base                `outputs/main_gap.parquet` (v14), template `data/raw/submitting.parquet`
Writes `$SCRATCH/adsb_override_candidate.parquet` (template schema and row order) and
`$SCRATCH/adsb_override_rows.parquet` (one row per overridden departure, for audit).

Sections (cited in the note as [C Sx]):
  S0  EGLL leave-one-day-out check (the pre-registered inclusion rule); the same for the four GO airports, for info
  S1  production bias per airport, from all four labelled days
  S2  the candidate: per day and per airport, rows overridden, guard trips, coverage
  S3  overall: rows changed, the distribution of the change, share of scored rows touched
  S4  sanity checks
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_stage1_feasibility as s1

ROOT = s1.REPO_DIR
SCR = s1.SCRATCH
LAB_DAYS = ["2025-07-21", "2025-07-13", "2025-01-22", "2025-11-12"]
GO4 = ["EDDM", "EHAM", "LEBL", "LSZH"]
GUARD = 900.0
B = 2000
EGLL_MIN_GAIN = 5000.0  # pre-registered: include EGLL iff LOO gain >= this and the row-bootstrap CI lower bound > 0
OUT = os.path.join(SCR, "adsb_override_candidate.parquet")
AUDIT = os.path.join(SCR, "adsb_override_rows.parquet")
KEEP26 = ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "join", "AOBT_3_flt", "push", "post_dwell", "t_move", "t_move_lo"]

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(40)
pl.Config.set_float_precision(3)


def tag_of(day: str) -> str:
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def head(s):
    print(f"\n{'=' * 110}\n{s}\n{'=' * 110}", flush=True)


def boot_ci(delta, clusters=None, seed=20260929):
    rng = np.random.default_rng(seed)
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan")
    if clusters is None:
        s = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
    else:
        _, inv = np.unique(clusters, return_inverse=True)
        sums, cnts = np.bincount(inv, weights=delta), np.bincount(inv)
        pick = rng.integers(0, len(sums), size=(B, len(sums)))
        s = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ============================================================================================ labelled days
def load_labelled() -> pl.DataFrame:
    ft = pl.read_parquet(ROOT + "data/work/feat_train.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"]).rename({"TAXITIME_SEC_mvt": "y"})
    oof = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88.parquet")
    frames = []
    for day in LAB_DAYS:
        J = s1.add_estimators(pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet"))
        D = (J.drop("TAXITIME_SEC_mvt").join(ft, on="MVT_ID_mvt", how="left").join(oof, on="MVT_ID_mvt", how="left")
             .with_columns(pl.lit(day).alias("day")))
        frames.append(D)
    D = pl.concat(frames, how="diagonal_relaxed")
    # covered = lane A, joined on CALLSIGN_flt, gated pushback (pre-registration 0.1)
    C = D.filter(pl.col("mvt_minus_aobt3").is_not_null() & (pl.col("join") == "cs") & pl.col("t_push").is_not_null())
    chk = C.select(pl.col("oof").is_null().sum().alias("no_oof"), pl.col("y").is_null().sum().alias("no_y"),
                   (pl.col("t_mvt") - pl.col("t_block") - pl.col("y")).abs().max().alias("max|MVT-BLOCK-y|"))
    print("labelled covered rows, checks:", chk.row(0, named=True))
    return C.with_columns((pl.col("t_push") - pl.col("t_block")).alias("off"),
                          (pl.col("t_mvt") - pl.col("t_push")).alias("y_raw"),
                          pl.col("oof").alias("pred"))


def score(F: pl.DataFrame, bcol: str = "b") -> dict:
    F = F.with_columns((pl.col("y_raw") + pl.col(bcol)).alias("y_a"))
    F = F.with_columns(((pl.col("pred") - pl.col("y_a")).abs() > GUARD).alias("trip"))
    F = F.with_columns(pl.when(pl.col("trip")).then(pl.col("pred")).otherwise(pl.col("y_a")).alias("pred_new"))
    em = (F["pred"] - F["y"]).to_numpy()
    en = (F["pred_new"] - F["y"]).to_numpy()
    d = em ** 2 - en ** 2
    hr = F["MVT_TIME_UTC_mvt"].dt.hour().to_numpy()
    clus = np.array([f"{a}|{h}|{x}" for a, h, x in zip(F["ADEP_mvt"].to_numpy(), hr, F["day"].to_numpy())])
    lo, hi = boot_ci(d)
    clo, chi = boot_ci(d, clus)
    return {"n": len(d), "gain": float(d.mean()) if len(d) else float("nan"), "lo": lo, "hi": hi, "clo": clo,
            "chi": chi, "trips": int(F["trip"].sum()),
            "rmse_m": float(np.sqrt((em ** 2).mean())) if len(d) else float("nan"),
            "rmse_new": float(np.sqrt((en ** 2).mean())) if len(d) else float("nan")}


def fmt(r):
    return (f"n {r['n']:4d}  gain/row {r['gain']:8.0f}  row CI [{r['lo']:8.0f}, {r['hi']:8.0f}]  apt-hour-day CI "
            f"[{r['clo']:8.0f}, {r['chi']:8.0f}]  trips {r['trips']:2d}  RMSE {r['rmse_m']:6.1f} -> {r['rmse_new']:6.1f}")


def loo(C: pl.DataFrame, apt: str) -> tuple[dict, pl.DataFrame, list]:
    A = C.filter(pl.col("ADEP_mvt") == apt)
    parts, per = [], []
    for h in LAB_DAYS:
        train = A.filter(pl.col("day") != h)
        b_h = float(train["off"].median())
        test = A.filter(pl.col("day") == h).with_columns(pl.lit(b_h).alias("b"))
        r = score(test) if test.height else {"n": 0, "gain": float("nan"), "trips": 0}
        per.append((h, b_h, train.height, r))
        parts.append(test)
    P = pl.concat(parts)
    return score(P), P, per


# ============================================================================================ scored days
def analysed_days() -> tuple[list[str], list[str]]:
    days, missing = [], []
    for m in (1, 7):
        for d in range(1, 32):
            day = f"2026-{m:02d}-{d:02d}"
            p = os.path.join(SCR, "ingest", "state", f"{day}.json")
            st = json.load(open(p)) if os.path.exists(p) else {}
            (days if st.get("analysed") else missing).append(day)
    return days, missing


def load_scored(days: list[str]) -> pl.DataFrame:
    frames = []
    for day in days:
        J = pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet").select(KEEP26)
        frames.append(s1.add_estimators(J).with_columns(pl.lit(day).alias("day")))
    D = pl.concat(frames, how="diagonal_relaxed")
    assert not any("BLOCK" in c or c == "t_block" for c in D.columns), D.columns
    return D


def main(write: bool = True) -> None:
    # ------------------------------------------------------------------------------------ S0
    head("S0  EGLL leave-one-day-out (pre-registered rule: include iff gain >= 5,000 and row-CI lower bound > 0)")
    C = load_labelled()
    print("covered rows per airport and day:")
    print(C.group_by("ADEP_mvt").agg([(pl.col("day") == d).sum().alias(d) for d in LAB_DAYS] + [pl.len().alias("all")])
          .sort("ADEP_mvt"))
    decisions = {}
    for apt in ["EGLL"] + GO4:
        r, _, per = loo(C, apt)
        tag = "DECISION" if apt == "EGLL" else "info only"
        print(f"\n{apt} [{tag}] pooled LOO: {fmt(r)}")
        for h, b_h, ntr, rr in per:
            print(f"   held out {h}: b from the other three = {b_h:7.1f} s (n {ntr:4d}); "
                  + (fmt(rr) if rr["n"] else "n    0"))
        decisions[apt] = r
    rg = decisions["EGLL"]
    egll_in = (rg["gain"] >= EGLL_MIN_GAIN) and (rg["lo"] > 0)
    print(f"\nEGLL: LOO gain {rg['gain']:.0f} [{rg['lo']:.0f}, {rg['hi']:.0f}] vs threshold {EGLL_MIN_GAIN:.0f} and "
          f"lower bound > 0 -> {'INCLUDE' if egll_in else 'EXCLUDE'}")
    apts = GO4 + (["EGLL"] if egll_in else [])
    P4 = pl.concat([loo(C, a)[1] for a in GO4])
    print(f"four GO airports pooled LOO (info): {fmt(score(P4))}")
    if egll_in:
        P5 = pl.concat([loo(C, a)[1] for a in GO4 + ['EGLL']])
        print(f"five airports pooled LOO (info): {fmt(score(P5))}")

    # ------------------------------------------------------------------------------------ S1
    head("S1  production bias b = median(t_push - BLOCK) per airport over all four labelled days (covered rows)")
    bt = (C.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("off").median().alias("b"),
                                     *[pl.col("off").filter(pl.col("day") == d).median().alias(f"med_{d}")
                                       for d in LAB_DAYS])
          .sort("ADEP_mvt"))
    print(bt)
    bias = {r["ADEP_mvt"]: float(r["b"]) for r in bt.iter_rows(named=True) if r["ADEP_mvt"] in apts}
    print("USED:", {a: round(v, 2) for a, v in sorted(bias.items())})
    Cb = C.filter(pl.col("ADEP_mvt").is_in(apts)).join(pl.DataFrame({"ADEP_mvt": list(bias), "b": list(bias.values())}),
                                                        on="ADEP_mvt")
    print(f"[in-sample, reference only] production bias on the four labelled days, included airports: {fmt(score(Cb))}")

    # ------------------------------------------------------------------------------------ S2
    head("S2  candidate: per day and airport")
    days, missing = analysed_days()
    print(f"days analysed {len(days)} / 62; MISSING {missing or 'none'}")
    tpl = pl.read_parquet(ROOT + "data/raw/submitting.parquet")
    base = pl.read_parquet(ROOT + "outputs/main_gap.parquet")
    assert base.schema == tpl.schema and base.height == tpl.height and (base["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = (pl.scan_parquet(ROOT + "data/raw/ranking.parquet")
          .select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "AOBT_3_flt").collect())
    scored = tpl.select("MVT_ID_mvt").join(rk, on="MVT_ID_mvt", how="left").with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"),
        (pl.col("MVT_TIME_UTC_mvt").dt.hour() < 2).alias("h0_2"),
        pl.col("CALLSIGN_flt").is_not_null().alias("A"))
    D = load_scored(days)
    assert D["MVT_ID_mvt"].n_unique() == D.height, "a departure appears in two joined files"
    D = D.join(base.rename({"TAXITIME_SEC_mvt": "base"}), on="MVT_ID_mvt", how="inner")
    D = D.join(pl.DataFrame({"ADEP_mvt": list(bias), "b": list(bias.values())}), on="ADEP_mvt", how="left")
    elig = (pl.col("ADEP_mvt").is_in(apts) & pl.col("AOBT_3_flt").is_not_null() & (pl.col("join") == "cs")
            & pl.col("t_push").is_not_null())
    D = D.with_columns(elig.fill_null(False).alias("pushback"))
    D = D.with_columns(pl.when(pl.col("pushback")).then(pl.col("t_mvt") - pl.col("t_push") + pl.col("b")).alias("y_a"))
    D = D.with_columns((pl.col("pushback") & ((pl.col("base").cast(pl.Float64) - pl.col("y_a")).abs() > GUARD))
                       .alias("trip"))
    D = D.with_columns((pl.col("pushback") & ~pl.col("trip")).alias("override"))
    D = D.with_columns(pl.when(pl.col("override")).then(pl.col("y_a").round(0).cast(pl.Int32))
                       .otherwise(pl.col("base")).alias("cand"))
    ov = D.filter(pl.col("override"))
    cand = base.join(ov.select("MVT_ID_mvt", pl.col("cand").alias("new")), on="MVT_ID_mvt", how="left",
                     maintain_order="left").with_columns(
        pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(tpl.columns)

    # per day x airport table on the scored set
    S = scored.join(D.select("MVT_ID_mvt", "pushback", "trip", "override"), on="MVT_ID_mvt", how="left").with_columns(
        pl.col("pushback").fill_null(False), pl.col("trip").fill_null(False), pl.col("override").fill_null(False))
    S = S.join(cand.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").alias("cand")), on="MVT_ID_mvt").join(
        base.select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").alias("base")), on="MVT_ID_mvt").with_columns(
        (pl.col("cand") != pl.col("base")).alias("changed"))
    inc = S.filter(pl.col("ADEP_mvt").is_in(apts))
    agg = [pl.len().alias("dep"), pl.col("A").sum().alias("laneA"), pl.col("pushback").sum().alias("push"),
           pl.col("trip").sum().alias("trips"), pl.col("override").sum().alias("overridden"),
           pl.col("changed").sum().alias("changed"),
           (pl.col("override").sum() / pl.len()).alias("cov_dep"),
           (pl.col("override").sum() / pl.col("A").sum()).alias("cov_laneA")]
    perday = inc.group_by("day").agg(agg).sort("day")
    print("per day, included airports pooled (dep = scored departures; cov_laneA = overridden / lane-A departures):")
    print(perday)
    wide = inc.group_by("day", "ADEP_mvt").agg(pl.col("override").sum().alias("ov"), pl.col("trip").sum().alias("tr"),
                                               (pl.col("override").sum() / pl.col("A").sum()).alias("cov"))
    print("per day x airport: overridden / trips / coverage of lane A")
    print(wide.pivot(on="ADEP_mvt", index="day", values=["ov", "tr", "cov"]).sort("day"))
    perday.write_csv(os.path.join(SCR, "adsb_override_perday.csv"))
    wide.sort("day", "ADEP_mvt").write_csv(os.path.join(SCR, "adsb_override_perday_airport.csv"))
    print("per airport over all days (scored set):")
    per_apt = S.group_by("ADEP_mvt").agg(*agg, pl.col("h0_2").sum().alias("dep_00_02")).sort("ADEP_mvt")
    print(per_apt)
    print("per month, included airports:")
    print(inc.group_by(pl.col("day").str.slice(0, 7).alias("month")).agg(agg).sort("month"))

    # ------------------------------------------------------------------------------------ S3
    head("S3  overall change")
    ch = S.filter(pl.col("changed")).with_columns((pl.col("cand") - pl.col("base")).alias("delta"))
    a = ch["delta"].abs()
    print(f"scored rows {S.height}; overridden {int(S['override'].sum())}; changed (cand != base) {ch.height} "
          f"= {100 * ch.height / S.height:.3f} % of scored rows; overridden but equal to base after rounding "
          f"{int((S['override'] & ~S['changed']).sum())}; guard trips {int(S['trip'].sum())} "
          f"({100 * S['trip'].sum() / max(1, S['pushback'].sum()):.2f} % of pushbacks)")
    print(f"|change| s: median {a.median():.0f}, p90 {a.quantile(0.9):.0f}, p99 {a.quantile(0.99):.0f}, max {a.max()}; "
          f"signed change: median {ch['delta'].median():.0f}, mean {ch['delta'].mean():.1f}, p10 "
          f"{ch['delta'].quantile(0.1):.0f}, p90 {ch['delta'].quantile(0.9):.0f}")
    print("per airport |change|:", ch.group_by("ADEP_mvt").agg(pl.len(), pl.col("delta").abs().median().alias("med"),
                                                             pl.col("delta").abs().quantile(0.9).alias("p90"),
                                                             pl.col("delta").abs().max().alias("max"),
                                                             pl.col("delta").median().alias("signed_med")).sort("ADEP_mvt"))
    print(f"candidate stats: min {cand['TAXITIME_SEC_mvt'].min()}, max {cand['TAXITIME_SEC_mvt'].max()}; "
          f"overridden values min {ov['cand'].min()}, max {ov['cand'].max()}")
    print(f"mean prediction: base {base['TAXITIME_SEC_mvt'].mean():.3f}, candidate {cand['TAXITIME_SEC_mvt'].mean():.3f}")

    # ------------------------------------------------------------------------------------ S4
    head("S4  sanity checks")
    checks = {}
    checks["schema == template"] = cand.schema == tpl.schema
    checks["height == 344,841"] = cand.height == tpl.height == 344841
    checks["row order == template"] = bool((cand["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all())
    checks["no nulls"] = cand["TAXITIME_SEC_mvt"].null_count() == 0
    diff = cand.with_columns(base["TAXITIME_SEC_mvt"].alias("base")).filter(
        pl.col("TAXITIME_SEC_mvt") != pl.col("base"))
    ids_changed = set(diff["MVT_ID_mvt"].to_list())
    ids_ov = set(ov["MVT_ID_mvt"].to_list())
    checks["changed rows subset of overridden"] = ids_changed <= ids_ov
    checks["changed count == S3 count"] = len(ids_changed) == ch.height
    chg = S.filter(pl.col("MVT_ID_mvt").is_in(list(ids_changed)))
    checks["changed rows only at included airports"] = set(chg["ADEP_mvt"].unique().to_list()) <= set(apts)
    checks["no LEMD / EDDF / LFPG / LIRF / LTFM row changed"] = chg.filter(
        ~pl.col("ADEP_mvt").is_in(apts)).height == 0
    checks["every changed row has a pushback, lane A, cs join"] = ov.filter(
        pl.col("MVT_ID_mvt").is_in(list(ids_changed))).select(
        (pl.col("t_push").is_not_null() & pl.col("AOBT_3_flt").is_not_null() & (pl.col("join") == "cs")).all()).item()
    checks["max |cand - base| <= 900"] = int(diff.select((pl.col("TAXITIME_SEC_mvt") - pl.col("base")).abs().max())
                                             .item() or 0) <= GUARD
    checks["no changed row with MVT 00:00-02:00 UTC"] = chg["h0_2"].sum() == 0
    checks["no changed row on a missing day"] = chg.filter(pl.col("day").is_in(missing)).height == 0
    checks["unchanged rows bit-identical to base"] = bool(
        (cand.filter(~pl.col("MVT_ID_mvt").is_in(list(ids_changed)))["TAXITIME_SEC_mvt"]
         == base.filter(~pl.col("MVT_ID_mvt").is_in(list(ids_changed)))["TAXITIME_SEC_mvt"]).all())
    # independent recomputation of every changed value from ranking.parquet MVT_TIME (not the joined file's t_mvt)
    rec = (diff.join(rk.select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"), on="MVT_ID_mvt")
           .join(D.select("MVT_ID_mvt", "t_push"), on="MVT_ID_mvt")
           .join(pl.DataFrame({"ADEP_mvt": list(bias), "b2": list(bias.values())}), on="ADEP_mvt")
           .with_columns(((pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000 - pl.col("t_push") + pl.col("b2"))
                          .round(0).cast(pl.Int32)).alias("recomp")))
    checks["changed values == round(MVT - t_push + b), recomputed from ranking.parquet"] = bool(
        (rec["recomp"] == rec["TAXITIME_SEC_mvt"]).all()) and rec.height == len(ids_changed)
    checks["no trip row changed"] = S.filter(pl.col("trip") & pl.col("changed")).height == 0
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    assert all(checks.values()), "sanity check failed"
    if write:
        cand.write_parquet(OUT)
        ov.select("MVT_ID_mvt", "day", "ADEP_mvt", "t_mvt", "t_push", "b", "base", "y_a", "cand").write_parquet(AUDIT)
        back = pl.read_parquet(OUT)
        n_diff = back.with_columns(base["TAXITIME_SEC_mvt"].alias("b0")).filter(
            pl.col("TAXITIME_SEC_mvt") != pl.col("b0")).height
        print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB): re-read schema {back.schema}, rows {back.height}, "
              f"rows differing from outputs/main_gap.parquet {n_diff}")
        print(f"wrote {AUDIT} ({ov.height} overridden rows)")


if __name__ == "__main__":
    main(write="--no-write" not in sys.argv)
