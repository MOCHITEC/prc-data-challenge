"""ADS-B pushback override: out-of-sample validation on new days, with every rule frozen.

Read-only with respect to the repo: nothing is written to `outputs/`, `src/`, `configs/` or `data/`; nothing is
fitted or submitted. It reads the per-departure `*_joined.parquet` files that
`scripts/adsb_stage1_feasibility.py analyse` writes to the scratch dir, the production OOF
`experiments/oof_main_gap-62dd7c6e88{,_folds}.parquet`, `data/work/feat_train.parquet` (label and lane), and, for
the unlabelled 2026 days, the production submission `outputs/main_gap.parquet` (v14, read only).

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    .venv/bin/python scripts/adsb_oos_validation.py [--days25 D,D,...] [--days26 D,D,...]

Frozen (fixed on 2025-07-21, the design day; nothing here is re-estimated on a scored day):
  detection  stage-1 rules, imported unchanged (s1.add_estimators on the stage-1 joined files)
  bias b     per-airport median of (t_push - BLOCK) on the 2025-07-21 lane-A covered rows, pooled median if an
             airport has < 10 such rows (adsb_gain_estimate.py S4, `b25`)
  guard      y_a = MVT - (t_push - b); use y_a unless |model - y_a| > 900 s
  airports   EDDM, EHAM, LEBL, LSZH, EGLL, LEMD
The label y = TAXITIME_SEC_mvt (= MVT - BLOCK, checked) is read only to score 2025 days. Nothing in the override
reads BLOCK: it reads t_push (ADS-B), MVT_TIME, the airport, b (frozen) and the model prediction.

Sections (cited in the note as [V Sx]):
  S0  the frozen bias table and the reproduction on 2025-07-21 (in-sample, reference only)
  S1  labelled 2025 days: per day, per airport, pooled; gain per covered row with row and airport-hour CIs
  S2  2026 census days: coverage per day and airport, guard trip rates
  S3  the pre-registered decision
  S4  diagnostics recorded but NOT used in the decision (what one might want to change)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_stage1_feasibility as s1

ROOT = s1.REPO_DIR
SCR = s1.SCRATCH
DESIGN_DAY = "2025-07-21"
EVAL = ["EDDM", "EHAM", "LEBL", "LSZH", "EGLL", "LEMD"]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
GUARD = 900.0
MIN_B = 10
B = 2000
RNG = np.random.default_rng(20260929)
DAYS25 = ["2025-07-13", "2025-01-22", "2025-11-12"]
DAYS26 = ["2026-01-19", "2026-01-28", "2026-07-08", "2026-07-29"]
EXTRA26 = ["2026-07-20"]  # already read in stage 1; reported, not part of the pre-registered pooled coverage
FOLD = {"01": "jan", "07": "jul", "11": "forward", "12": "forward"}
# pre-registered thresholds (adsb-gain-estimate.md section 8)
GO_GAIN, KILL_GAIN = 5000.0, 2000.0
GO_COV_JAN, GO_COV_JUL, KILL_COV = 0.10, 0.15, 0.05

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(40)
pl.Config.set_float_precision(3)


def tag_of(day: str) -> str:
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def head(s):
    print(f"\n{'=' * 110}\n{s}\n{'=' * 110}", flush=True)


def boot_ci(delta, clusters=None):
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan")
    if clusters is None:
        idx = RNG.integers(0, n, size=(B, n))
        s = delta[idx].mean(axis=1)
    else:
        _, inv = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inv, weights=delta)
        cnts = np.bincount(inv)
        pick = RNG.integers(0, len(sums), size=(B, len(sums)))
        s = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ============================================================================================ loading
_FT = _OOF = _FOL = None


def labels():
    global _FT, _OOF, _FOL
    if _FT is None:
        _FT = pl.read_parquet(ROOT + "data/work/feat_train.parquet",
                              columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"]).rename(
            {"TAXITIME_SEC_mvt": "y"})
        _OOF = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88.parquet")
        _FOL = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88_folds.parquet").filter(pl.col("pooled"))
    return _FT, _OOF, _FOL


def load25(day: str) -> pl.DataFrame:
    ft, oof, fol = labels()
    J = s1.add_estimators(pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet"))
    D = (J.drop("TAXITIME_SEC_mvt").join(ft, on="MVT_ID_mvt", how="left")
         .join(oof, on="MVT_ID_mvt", how="left").join(fol.select("MVT_ID_mvt", "fold"), on="MVT_ID_mvt", how="left"))
    return D.with_columns(
        pl.col("mvt_minus_aobt3").is_not_null().alias("A"),
        pl.col("oof").alias("pred"),
        (pl.col("t_push") - pl.col("t_block")).alias("off"),
        (pl.col("t_mvt") - pl.col("t_push")).alias("y_raw"),
        pl.lit(day).alias("day"))


def load26(day: str) -> pl.DataFrame:
    J = s1.add_estimators(pl.read_parquet(f"{SCR}/{tag_of(day)}_joined.parquet"))
    sub = pl.read_parquet(ROOT + "outputs/main_gap.parquet").rename({"TAXITIME_SEC_mvt": "pred"})
    return J.join(sub, on="MVT_ID_mvt", how="left").with_columns(
        pl.col("AOBT_3_flt").is_not_null().alias("A"),
        (pl.col("t_mvt") - pl.col("t_push")).alias("y_raw"),
        pl.lit(day).alias("day"))


def frozen_bias() -> tuple[dict, float]:
    """b25 of adsb_gain_estimate.py S4, from the design day only."""
    D = load25(DESIGN_DAY)
    C = D.filter(pl.col("A") & pl.col("t_push").is_not_null())
    pooled = float(np.median(C["off"].to_numpy()))
    t = C.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("off").median().alias("b"))
    b = {r["ADEP_mvt"]: (r["b"] if r["n"] >= MIN_B else pooled) for r in t.iter_rows(named=True)}
    return b, pooled


def apply_override(F: pl.DataFrame, b: dict, pooled_b: float) -> pl.DataFrame:
    """Label-free: y_a from the pushback, the frozen bias and MVT; guard on |pred - y_a|."""
    bmap = pl.DataFrame({"ADEP_mvt": list(b), "b": list(b.values())})
    F = F.join(bmap, on="ADEP_mvt", how="left").with_columns(pl.col("b").fill_null(pooled_b))
    F = F.with_columns((pl.col("y_raw") + pl.col("b")).alias("y_a"))
    F = F.with_columns((pl.col("pred") - pl.col("y_a")).alias("dis"))
    F = F.with_columns((pl.col("t_push").is_not_null() & (pl.col("dis").abs() > GUARD)).alias("trip"))
    return F.with_columns(pl.when(pl.col("t_push").is_not_null() & ~pl.col("trip"))
                          .then(pl.col("y_a")).otherwise(pl.col("pred")).alias("pred_new"))


# ============================================================================================ scoring
def gain_rows(C: pl.DataFrame) -> dict:
    em = (C["pred"] - C["y"]).to_numpy()
    en = (C["pred_new"] - C["y"]).to_numpy()
    ea = (C["y_a"] - C["y"]).to_numpy()
    d = em ** 2 - en ** 2
    hr = C["MVT_TIME_UTC_mvt"].dt.hour().to_numpy()
    clus = np.array([f"{a}|{h}|{dd}" for a, h, dd in zip(C["ADEP_mvt"].to_numpy(), hr, C["day"].to_numpy())])
    lo, hi = boot_ci(d)
    clo, chi = boot_ci(d, clus)
    n = len(d)
    return {"n": n, "rmse_m": float(np.sqrt((em ** 2).mean())) if n else np.nan,
            "rmse_new": float(np.sqrt((en ** 2).mean())) if n else np.nan,
            "mae_m": float(np.abs(em).mean()) if n else np.nan, "mae_new": float(np.abs(en).mean()) if n else np.nan,
            "gain": float(d.mean()) if n else np.nan, "lo": lo, "hi": hi, "clo": clo, "chi": chi,
            "trip": float(C["trip"].mean()) if n else np.nan, "n_trip": int(C["trip"].sum()),
            "fail10": int((np.abs(ea) > 600).sum()), "worse2": int((np.abs(en) > np.abs(em) + 120).sum()),
            "better": float((np.abs(en) < np.abs(em)).mean()) if n else np.nan,
            "d": d}


def fmt(r, label):
    if r["n"] == 0:
        return f"  {label:24s} n    0"
    return (f"  {label:24s} n {r['n']:4d}  model RMSE {r['rmse_m']:6.1f} -> {r['rmse_new']:6.1f}  MAE {r['mae_m']:5.1f} -> "
            f"{r['mae_new']:5.1f}  gain/row {r['gain']:8.0f}  row CI [{r['lo']:8.0f},{r['hi']:8.0f}]  apt-hour CI "
            f"[{r['clo']:8.0f},{r['chi']:8.0f}]  trips {r['n_trip']:2d} ({100 * r['trip']:.2f} %)  |ADS-B err|>10 min "
            f"{r['fail10']:2d}  worse>2 min {r['worse2']:3d}  better {r['better']:.3f}")


def main(days25, days26):
    b, pooled_b = frozen_bias()
    head("S0  frozen bias (from 2025-07-21 only) and the reproduction on the design day (in-sample, reference)")
    print("b (s), per airport; pooled median", round(pooled_b, 3), "used where < 10 rows:",
          {a: round(v, 2) for a, v in sorted(b.items())})
    D0 = apply_override(load25(DESIGN_DAY), b, pooled_b)
    C0 = D0.filter(pl.col("A") & pl.col("t_push").is_not_null())
    print(fmt(gain_rows(C0), "2025-07-21 all apts"), "\n  [gain estimate, cross-fitted bias: 914 rows, +19,400]")
    print(fmt(gain_rows(C0.filter(pl.col("ADEP_mvt").is_in(EVAL))), "2025-07-21 six apts"))

    # ------------------------------------------------------------------------------------ S1 labelled days
    head("S1  labelled 2025 days, rules / bias / guard frozen; covered = lane A, gated pushback, six airports")
    frames = []
    for day in days25:
        if not os.path.exists(f"{SCR}/{tag_of(day)}_joined.parquet"):
            print(f"  {day}: no joined file, skipped")
            continue
        D = apply_override(load25(day), b, pooled_b)
        exp_fold = FOLD[day[5:7]]
        chk = D.select(pl.len().alias("dep"), pl.col("oof").is_null().sum().alias("no_oof"),
                       (pl.col("fold") != exp_fold).sum().alias("wrong_fold"),
                       pl.col("y").is_null().sum().alias("no_y"),
                       (pl.col("t_mvt") - pl.col("t_block") - pl.col("y")).abs().max().alias("max|MVT-BLOCK-y|"),
                       pl.col("A").sum().alias("laneA")).row(0, named=True)
        print(f"\n{day}  expected fold {exp_fold}: {chk}")
        LA = D.filter(pl.col("A"))
        la_rmse = float(np.sqrt(((LA["pred"] - LA["y"]) ** 2).mean()))
        print(f"  lane-A RMSE of the day (all ten airports) {la_rmse:.1f}; lane-A n {LA.height}")
        cov = (LA.group_by("ADEP_mvt").agg(pl.len().alias("nA"), pl.col("t_push").is_not_null().mean().alias("cov"))
               .sort("ADEP_mvt"))
        print("  lane-A pushback coverage per airport:",
              {r[0]: f"{r[2]:.3f} of {r[1]}" for r in cov.iter_rows()})
        C_all = LA.filter(pl.col("t_push").is_not_null())
        C = C_all.filter(pl.col("ADEP_mvt").is_in(EVAL))
        print(f"  covered rows: six airports {C.height} ({C.height / LA.height:.3f} of lane A); all airports "
              f"{C_all.height}; guard trips, all airports {C_all['trip'].sum()} / {C_all.height} "
              f"({100 * C_all['trip'].mean():.2f} %)")
        print(fmt(gain_rows(C), f"{day} six apts"))
        for a in EVAL:
            print(fmt(gain_rows(C.filter(pl.col("ADEP_mvt") == a)), f"   {a}"))
        frames.append(C)
    if frames:
        P = pl.concat(frames, how="diagonal_relaxed")
        print("\npooled over the labelled days")
        rp = gain_rows(P)
        print(fmt(rp, "POOLED six apts"))
        for a in EVAL:
            print(fmt(gain_rows(P.filter(pl.col("ADEP_mvt") == a)), f"   {a}"))
        # day-cluster view: every day must carry its own weight
        per_day = {d: gain_rows(P.filter(pl.col("day") == d))["gain"] for d in P["day"].unique().sort().to_list()}
        print("  per-day point gains:", {k: round(v) for k, v in per_day.items()},
              f" min {min(per_day.values()):.0f}; equal-weight mean of days {np.mean(list(per_day.values())):.0f}")
        # stress: loss of the rows made worse with |ADS-B err| > 120 s, multiplied as in gain estimate 6.2
        d = rp["d"]
        ea = (P["y_a"] - P["y"]).to_numpy()
        f = (np.abs(ea) > 120) & (d < 0)
        print(f"  rows made worse: {(d < 0).sum()} (loss {-d[d < 0].sum() / 1e6:.2f}e6 vs gain {d[d > 0].sum() / 1e6:.2f}e6);"
              f" with |ADS-B err| > 120 s: {f.sum()} (loss {-d[f].sum() / 1e6:.2f}e6); top-10-rows share of the gain "
              f"{100 * np.sort(d)[::-1][:10].sum() / d.sum():.1f} %, gain without the top 10 "
              f"{np.sort(d)[::-1][10:].mean():.0f}")
        worst = P.with_columns(((pl.col("pred_new") - pl.col("y")).abs() - (pl.col("pred") - pl.col("y")).abs()).alias("hurt"))
        print("  the 10 rows the override hurts most:")
        print(worst.sort("hurt", descending=True).head(10).select(
            "day", "ADEP_mvt", "cs", "y", "pred", "y_a", "dis", "off", "hurt"))
    else:
        rp = None

    # ------------------------------------------------------------------------------------ S2 census days
    head("S2  2026 census days: coverage (lane A) and guard trips; the guard uses the production submission "
         "outputs/main_gap.parquet (v14)")
    cens = {}
    for day in days26 + EXTRA26:
        if not os.path.exists(f"{SCR}/{tag_of(day)}_joined.parquet"):
            print(f"  {day}: no joined file, skipped")
            continue
        T = apply_override(load26(day), b, pooled_b)
        LA = T.filter(pl.col("A"))
        no_pred = int(LA["pred"].is_null().sum())
        C_all = LA.filter(pl.col("t_push").is_not_null() & pl.col("pred").is_not_null())
        C = C_all.filter(pl.col("ADEP_mvt").is_in(EVAL))
        touched = C.filter(~pl.col("trip"))
        cens[day] = {"nA": LA.height, "cov": C.height, "touched": touched.height,
                     "nA6": LA.filter(pl.col("ADEP_mvt").is_in(EVAL)).height,
                     "trip_all": (int(C_all["trip"].sum()), C_all.height), "trip6": (int(C["trip"].sum()), C.height)}
        print(f"\n{day}: lane-A departures (ten airports) {LA.height}, no submission prediction {no_pred}")
        print(f"  covered at the six airports {C.height} = {C.height / LA.height:.3f} of lane A (ten airports), "
              f"{C.height / cens[day]['nA6']:.3f} of lane A at the six; after the guard {touched.height} = "
              f"{touched.height / LA.height:.3f}")
        print(f"  guard trips: six airports {C['trip'].sum()} / {C.height} ({100 * C['trip'].mean():.2f} %); all "
              f"airports {C_all['trip'].sum()} / {C_all.height} ({100 * C_all['trip'].mean():.2f} %)")
        print(f"  |AOBT_3 - push| > 20 min on covered rows (six airports): "
              f"{((C['t_aobt3'] - C['t_push']).abs() > 1200).mean():.4f}; RMS disagreement inside the guard "
              f"{np.sqrt((C.filter(~pl.col('trip'))['dis'] ** 2).mean()):.1f} s")
        print(LA.group_by("ADEP_mvt").agg(
            pl.len().alias("nA"), pl.col("seen").mean().alias("seen"), pl.col("ground_fix").mean().alias("ground"),
            pl.col("t_move").is_not_null().mean().alias("stand_exit"),
            pl.col("t_push").is_not_null().mean().alias("pushback"),
            (pl.col("t_push").is_not_null() & ~pl.col("trip")).mean().alias("touched"),
            pl.col("trip").sum().alias("trips"),
            ((pl.col("push").fill_null(0) >= s1.PUSH_MIN) & (pl.col("post_dwell").fill_null(0) >= s1.POST_DWELL_MAX))
            .sum().alias("dropped_by_dwell")).sort("ADEP_mvt"))

    def pool(ds, key="cov"):
        ds = [d for d in ds if d in cens]
        if not ds:
            return float("nan")
        return sum(cens[d][key] for d in ds) / sum(cens[d]["nA"] for d in ds)

    jan = [d for d in days26 if d[5:7] == "01"]
    jul = [d for d in days26 if d[5:7] == "07"]
    cj, cu = pool(jan), pool(jul)
    print(f"\npooled lane-A coverage (pushback at the six / lane A at ten): January {cj:.4f} over {jan}; "
          f"July {cu:.4f} over {jul}; July incl. 2026-07-20 {pool(jul + EXTRA26):.4f}")
    print(f"after the guard: January {pool(jan, 'touched'):.4f}; July {pool(jul, 'touched'):.4f}")

    # ------------------------------------------------------------------------------------ S3 decision
    head("S3  pre-registered decision (adsb-gain-estimate.md section 8)")
    g = rp["gain"] if rp else float("nan")
    print(f"gain per covered row, pooled labelled days: {g:.0f} (row CI [{rp['lo']:.0f}, {rp['hi']:.0f}], "
          f"apt-hour-day CI [{rp['clo']:.0f}, {rp['chi']:.0f}]); GO >= {GO_GAIN:.0f}, KILL < {KILL_GAIN:.0f}")
    print(f"coverage January {cj:.4f} (GO >= {GO_COV_JAN}), July {cu:.4f} (GO >= {GO_COV_JUL}); KILL < {KILL_COV}")
    if g < KILL_GAIN or (cj < KILL_COV and cu < KILL_COV):
        v = "KILL"
    elif g >= GO_GAIN and cj >= GO_COV_JAN and cu >= GO_COV_JUL:
        v = "GO (ingest all 62 scored days)"
    elif g >= GO_GAIN and cu >= GO_COV_JUL:
        v = "GO-JULY-ONLY" + (" (January below the 5 % kill coverage)" if cj < KILL_COV else "")
    else:
        v = "UNDECIDED"
    print("VERDICT:", v)

    # ------------------------------------------------------------------------------------ S4 diagnostics
    head("S4  diagnostics recorded after scoring; NONE of this enters the decision above")
    if frames:
        P = pl.concat(frames, how="diagonal_relaxed")
        print("pushback - BLOCK offset (s) on the new days vs the frozen b, per airport (covered lane-A rows):")
        print(P.group_by("ADEP_mvt", "day").agg(pl.len(), pl.col("off").median().alias("med_off"),
                                                pl.col("b").first().alias("frozen_b"),
                                                (pl.col("off") - pl.col("b")).abs().median().alias("med_abs_after_b"),
                                                (pl.col("off").abs() > 600).sum().alias("gt10min"))
              .sort("ADEP_mvt", "day"))
        print("RMS disagreement inside the guard, 2025 days (compare with the 2026 lines of S2):",
              {d: round(float(np.sqrt((P.filter((pl.col("day") == d) & ~pl.col("trip"))["dis"] ** 2).mean())), 1)
               for d in sorted(P["day"].unique().to_list())})
        # variants that one might want after seeing these days; scored for the record only
        eg = P.filter(pl.col("ADEP_mvt") == "EGLL")
        print(f"EGLL: n {eg.height}, median offset {eg['off'].median():.1f} s against frozen b {eg['b'][0]:.1f}")
        for label, F in (("six airports (frozen)", P),
                         ("five airports, EGLL dropped", P.filter(pl.col("ADEP_mvt") != "EGLL")),
                         ("four GO airports EDDM/EHAM/LEBL/LSZH", P.filter(pl.col("ADEP_mvt").is_in(
                             ["EDDM", "EHAM", "LEBL", "LSZH"])))):
            r = gain_rows(F)
            per = {d: round(gain_rows(F.filter(pl.col("day") == d))["gain"]) for d in sorted(F["day"].unique().to_list())}
            print(f"  [not used] {label:40s} n {r['n']:4d} gain/row {r['gain']:7.0f} [{r['lo']:.0f}, {r['hi']:.0f}] per day {per}")
        # EGLL with a bias re-estimated on the new days themselves: in-sample, record only
        b_eg = float(eg["off"].median())
        eg2 = eg.with_columns((pl.col("y_raw") + b_eg).alias("y_a")).with_columns(
            (pl.col("pred") - pl.col("y_a")).alias("dis")).with_columns(
            (pl.col("dis").abs() > GUARD).alias("trip")).with_columns(
            pl.when(~pl.col("trip")).then(pl.col("y_a")).otherwise(pl.col("pred")).alias("pred_new"))
        r = gain_rows(eg2)
        print(f"  [not used, IN-SAMPLE] EGLL with b = its own median on these days ({b_eg:.0f} s): gain/row "
              f"{r['gain']:.0f} [{r['lo']:.0f}, {r['hi']:.0f}]")
    print("\ndwell guard (post-pushback stop >= 15 min) drops, per labelled day: gated-backwards stand exits removed, "
          "and their error vs BLOCK")
    for day in days25:
        if not os.path.exists(f"{SCR}/{tag_of(day)}_joined.parquet"):
            continue
        D = load25(day).filter(pl.col("A") & pl.col("ADEP_mvt").is_in(EVAL))
        dr = D.filter((pl.col("push").fill_null(0) >= s1.PUSH_MIN) & (pl.col("post_dwell").fill_null(0) >= s1.POST_DWELL_MAX)
                      & pl.col("t_move").is_not_null())
        e = (dr["t_move"] - dr["t_block"])
        print(f"  {day}: dropped {dr.height} (vs {D['t_push'].is_not_null().sum()} kept); per airport "
              f"{dict(dr.group_by('ADEP_mvt').len().iter_rows())}; |t_move - BLOCK| median "
              f"{e.abs().median() if dr.height else float('nan')}, within 5 min {(e.abs() <= 300).sum()}")
    return b


if __name__ == "__main__":
    a = sys.argv[1:]
    d25 = a[a.index("--days25") + 1].split(",") if "--days25" in a else DAYS25
    d26 = a[a.index("--days26") + 1].split(",") if "--days26" in a else DAYS26
    main(d25, d26)
