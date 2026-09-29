"""ADS-B stage 1b: how much would an observed pushback improve the score? Two days only, no new data.

Read-only with respect to the repo: nothing is written to `outputs/`, `src/`, `configs/` or `data/`; no model
is fitted and nothing is submitted. It reads the two per-departure files that `scripts/adsb_stage1_feasibility.py
analyse` left in the scratch dir (and, for section S5, the 2025-07-21 filtered fixes), the production OOF
`experiments/oof_main_gap-62dd7c6e88.parquet` (+ `_folds`), `data/work/feat_train.parquet`, and the v14
submission `outputs/main_gap.parquet` (read only, for the label-free 2026 check).

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    .venv/bin/python scripts/adsb_gain_estimate.py [--skip-sweep]

Sections (the note cites them as [G Sx]):
  S0  reproduce stage 1's pushback set on lane A and the OOF join
  S1  lane-A per-airport baseline: pooled OOF, July fold, the day; coverage on both days
  S2  bias of the pushback vs BLOCK, estimated out of sample (2-fold by row hash; leave-airport-out)
  S3  the ways of using it (replace / blend / disagreement gates), all cross-fitted, per airport, bootstrap CIs
  S4  label-free out-of-sample check on 2026-07-20 against the v14 submission, and a gain proxy validated on 2025
  S5  sensitivity of the gain to the detection rule's thresholds (the in-sample-rules check)
  S6  scale-up to lane-A SSE, pooled OOF and board, as scenarios

Label: y = TAXITIME_SEC_mvt = MVT_TIME - BLOCK_TIME (S0 checks the identity). ADS-B prediction y_a = MVT - (t_push - b) = y_raw + b,
with b the pushback-minus-BLOCK offset estimated on rows other than the ones scored.
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
TAG25 = "v2025.07.21-planes-readsb-prod-0"
TAG26 = "v2026.07.20-planes-readsb-prod-0"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
N_POOLED, RMSE_POOLED = 672_428, 277.5398       # model-a-ceiling.md S1, reproduced in S1 below
N_A, RMSE_A = 664_050, 213.1266
N_BOARD, BOARD_V14 = 344_841, 279.5234
TRANSFER = 0.747                                 # dBoard/dOOF, model-a-ceiling.md 8.2, n = 1
MIN_N_AIRPORT = 30                               # below this many covered rows an airport's own gain is not used
RNG = np.random.default_rng(20260929)
B = 2000

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(40)
pl.Config.set_float_precision(3)


def tag(s):
    print(f"\n{'=' * 110}\n{s}\n{'=' * 110}", flush=True)


def half_of(ids: np.ndarray) -> np.ndarray:
    """Deterministic 2-way split from the row id alone (Knuth multiplicative hash)."""
    k = (ids.astype(np.int64) * 2654435761) % (2 ** 32)
    return (k >> 31).astype(int)


# ============================================================================================ loading
def load_2025() -> pl.DataFrame:
    J = s1.add_estimators(pl.read_parquet(f"{SCR}/{TAG25}_joined.parquet"))
    ft = pl.read_parquet(ROOT + "data/work/feat_train.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"]).rename(
        {"TAXITIME_SEC_mvt": "y"})
    oof = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88.parquet")
    fol = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88_folds.parquet").filter(pl.col("pooled"))
    D = (J.drop("TAXITIME_SEC_mvt").join(ft, on="MVT_ID_mvt", how="left")
         .join(oof, on="MVT_ID_mvt", how="left").join(fol.select("MVT_ID_mvt", "fold"), on="MVT_ID_mvt", how="left"))
    return D.with_columns(
        pl.col("mvt_minus_aobt3").is_not_null().alias("A"),
        (pl.col("oof") - pl.col("y")).alias("e_m"),
        (pl.col("t_push") - pl.col("t_block")).alias("off"),         # pushback - BLOCK, s (+ve = ADS-B later)
        (pl.col("t_mvt") - pl.col("t_push")).alias("y_raw"),         # MVT - pushback, before bias correction
    )


def load_2026() -> pl.DataFrame:
    J = s1.add_estimators(pl.read_parquet(f"{SCR}/{TAG26}_joined.parquet"))
    sub = pl.read_parquet(ROOT + "outputs/main_gap.parquet").rename({"TAXITIME_SEC_mvt": "pred"})
    return J.join(sub, on="MVT_ID_mvt", how="left").with_columns(
        pl.col("AOBT_3_flt").is_not_null().alias("A"),
        (pl.col("t_mvt") - pl.col("t_push")).alias("y_raw"))


def pooled_lane_a() -> pl.DataFrame:
    ft = pl.read_parquet(ROOT + "data/work/feat_train.parquet",
                         columns=["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3"])
    oof = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88.parquet")
    fol = pl.read_parquet(ROOT + "experiments/oof_main_gap-62dd7c6e88_folds.parquet").filter(pl.col("pooled"))
    return (fol.join(oof, on="MVT_ID_mvt").join(ft, on="MVT_ID_mvt")
            .with_columns((pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("e"),
                          pl.col("mvt_minus_aobt3").is_not_null().alias("A")))


# ============================================================================================ S2 bias
def bias_table(C: pl.DataFrame, scheme: str) -> np.ndarray:
    """Out-of-sample bias b (s) for each row of C (the covered lane-A rows): the median pushback-BLOCK offset
    from rows the scored row is not in.
      half      per-airport median from the other hash half (pooled other-half median if < 10 rows there)
      loao      pooled median of all OTHER airports (leave-airport-out)
      pooled_h  pooled median of the other half (ignores airport)
      none      0"""
    ap = C["ADEP_mvt"].to_numpy()
    off = C["off"].to_numpy()
    h = C["half"].to_numpy()
    b = np.zeros(len(C))
    if scheme == "none":
        return b
    for i in range(len(C)):
        if scheme == "half":
            m = (h != h[i]) & (ap == ap[i])
            b[i] = np.median(off[m]) if m.sum() >= 10 else np.median(off[h != h[i]])
        elif scheme == "loao":
            b[i] = np.median(off[ap != ap[i]])
        elif scheme == "pooled_h":
            b[i] = np.median(off[h != h[i]])
    return b


# ============================================================================================ S3 methods
def fit_w(em, ea):
    """Least-squares blend weight on ADS-B for pred = oof - w*(em-ea): minimises sum (em - w (em-ea))^2."""
    d = em - ea
    return float(np.clip((em * d).sum() / (d * d).sum(), 0, 1))


def new_errors(em, ea, method, param):
    """Error of the combined prediction, given model error em and (bias-corrected) ADS-B error ea on the same
    rows. dis = pred_model - pred_adsb = em - ea (label-free)."""
    dis = em - ea
    if method == "model":
        return em
    if method == "replace":
        return ea
    if method == "blend":
        return (1 - param) * em + param * ea
    if method == "gate_gt":        # replace only where |model - ADS-B| > param
        return np.where(np.abs(dis) > param, ea, em)
    if method == "guard_lt":       # replace except where |model - ADS-B| > param (distrust a wild ADS-B)
        return np.where(np.abs(dis) <= param, ea, em)
    raise ValueError(method)


GATE_GRID = [0, 30, 60, 90, 120, 180, 240, 300, 450, 600]
GUARD_GRID = [300, 450, 600, 900, 1200, 1800, 2400, 3600, 1e9]


def crossfit(em, ea, h, method):
    """Parameter chosen on the other half, applied to this half. Returns (new errors, params per half)."""
    out = np.empty_like(em)
    ps = {}
    for k in (0, 1):
        tr, te = h != k, h == k
        if method == "blend_cf":
            p = fit_w(em[tr], ea[tr])
            out[te] = new_errors(em[te], ea[te], "blend", p)
        elif method in ("gate_gt_cf", "guard_lt_cf"):
            base = method[:-3]
            grid = GATE_GRID if base == "gate_gt" else GUARD_GRID
            sse = [(new_errors(em[tr], ea[tr], base, g) ** 2).sum() for g in grid]
            p = grid[int(np.argmin(sse))]
            out[te] = new_errors(em[te], ea[te], base, p)
        ps[k] = p
    return out, ps


def boot_ci(delta, clusters=None):
    """95 % CI of mean(delta) by row bootstrap, or by cluster bootstrap when clusters is given."""
    n = len(delta)
    if clusters is None:
        idx = RNG.integers(0, n, size=(B, n))
        s = delta[idx].mean(axis=1)
    else:
        u, inv = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inv, weights=delta)
        cnts = np.bincount(inv)
        pick = RNG.integers(0, len(u), size=(B, len(u)))
        s = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ============================================================================================ S5 sweep
def rerun_detection(D: pl.DataFrame, stat_r=25.0, gap_s=600.0, push_win=180.0, push_min=0.5, dwell=900.0,
                    opp_deg=120.0):
    """Re-run the frozen stage-1 stand/pushback detection on the 2025-07-21 fixes with changed thresholds, on
    the aircraft the stage-1 join already chose (the join is label-free and the same for every config).
    Returns MVT_ID -> t_push (s) for the rows where a pushback passes the gate."""
    fx = pl.read_parquet(f"{SCR}/{TAG25}.parquet").sort("t")
    by = {(k[0], k[1]): v for k, v in fx.partition_by(["icao", "apt"], as_dict=True).items()}
    old = (s1.STAT_R, s1.GAP_S, s1.pushback_signature)

    def sig(tt, la, lo, trk, e):
        if trk is None:
            return None
        votes = []
        i = e
        while i + 1 < len(tt) and tt[i + 1] - tt[e] <= push_win:
            d = float(s1.dist_m(la[i], lo[i], la[i + 1], lo[i + 1]))
            hh = trk[i + 1]
            if d >= 3.0 and hh is not None and not np.isnan(hh):
                b = s1.bearing(la[i], lo[i], la[i + 1], lo[i + 1])
                votes.append(abs((b - hh + 180.0) % 360.0 - 180.0) > opp_deg)
            i += 1
        return None if not votes else float(np.mean(votes))

    s1.STAT_R, s1.GAP_S, s1.pushback_signature = stat_r, gap_s, sig
    out = {}
    try:
        for r in D.filter(pl.col("icao").is_not_null()).select("MVT_ID_mvt", "icao", "ADEP_mvt", "t_mvt").iter_rows():
            f = by.get((r[1], r[2]))
            if f is None:
                continue
            res = s1.stand_departure(f["t"].to_numpy(), f["lat"].to_numpy(), f["lon"].to_numpy(),
                                     f["ground"].to_numpy(), f["gs"].fill_null(np.nan).to_numpy(), r[3],
                                     f["track"].fill_null(np.nan).to_numpy())
            if (res["t_move"] is not None and (res["push"] or 0) >= push_min
                    and (res["post_dwell"] or 0) < dwell):
                out[r[0]] = res["t_move"]
    finally:
        s1.STAT_R, s1.GAP_S, s1.pushback_signature = old
    return out


# ============================================================================================ main
def main(skip_sweep=False):
    D = load_2025()
    D = D.with_columns(pl.Series("half", half_of(D["MVT_ID_mvt"].to_numpy())))
    T = load_2026()
    P = pooled_lane_a()

    # ---------------------------------------------------------------- S0
    tag("S0  reproduce stage 1 on lane A (AOBT_3 present = mvt_minus_aobt3 not null)")
    print(D.select(pl.len().alias("departures"), pl.col("A").sum().alias("laneA"),
                   (pl.col("A") & pl.col("t_push").is_not_null()).sum().alias("laneA_push"),
                   (~pl.col("A") & pl.col("t_push").is_not_null()).sum().alias("laneB_push"),
                   pl.col("oof").is_null().sum().alias("no_oof"),
                   (pl.col("fold") == "jul").sum().alias("fold_jul"),
                   (pl.col("t_mvt") - pl.col("t_block") - pl.col("y")).abs().max().alias("max|MVT-BLOCK-y|")))
    C = D.filter(pl.col("A") & pl.col("t_push").is_not_null())
    print(f"lane-A covered rows: OOF RMSE {np.sqrt((C['e_m'] ** 2).mean()):.2f}, raw ADS-B (no bias corr.) RMSE "
          f"{np.sqrt(((C['y_raw'] - C['y']) ** 2).mean()):.2f}  [stage 1: 171 / 284]")

    # ---------------------------------------------------------------- S1
    tag("S1  lane-A baseline per airport: pooled OOF (3 folds), July fold, 2025-07-21; pushback coverage both days")
    pa = P.filter(pl.col("A"))
    print(f"pooled OOF n {P.height} RMSE {np.sqrt((P['e'] ** 2).mean()):.4f} | lane A n {pa.height} RMSE "
          f"{np.sqrt((pa['e'] ** 2).mean()):.4f} share {(pa['e'] ** 2).sum() / (P['e'] ** 2).sum():.4f}")
    base = (pa.group_by("ADEP_mvt").agg(pl.len().alias("nA_oof"), (pl.col("e") ** 2).mean().alias("mseA_oof"),
                                        (pl.col("e") ** 2).filter(pl.col("fold") == "jul").mean().alias("mseA_jul"),
                                        (pl.col("fold") == "jan").sum().alias("n_jan"),
                                        (pl.col("fold") == "jul").sum().alias("n_jul")))
    day = (D.filter(pl.col("A")).group_by("ADEP_mvt").agg(
        pl.len().alias("nA_day"), (pl.col("e_m") ** 2).mean().alias("mseA_day"),
        pl.col("t_push").is_not_null().mean().alias("cov25"),
        (pl.col("e_m") ** 2).filter(pl.col("t_push").is_not_null()).mean().alias("mse_cov_day")))
    cov26 = (T.filter(pl.col("A")).group_by("ADEP_mvt").agg(pl.len().alias("nA_26"),
                                                            pl.col("t_push").is_not_null().mean().alias("cov26")))
    base = base.join(day, on="ADEP_mvt", how="left").join(cov26, on="ADEP_mvt", how="left").sort("ADEP_mvt")
    print(base.with_columns([pl.col(c).sqrt().alias(c.replace("mse", "rmse")) for c in
                             ["mseA_oof", "mseA_jul", "mseA_day", "mse_cov_day"]]).select(
        "ADEP_mvt", "nA_oof", "rmseA_oof", "rmseA_jul", "nA_day", "rmseA_day", "cov25", "rmse_cov_day",
        "nA_26", "cov26"))
    print(f"2025-07-21 lane-A RMSE {np.sqrt((D.filter(pl.col('A'))['e_m'] ** 2).mean()):.2f} vs July fold lane-A "
          f"RMSE {np.sqrt((pa.filter(pl.col('fold') == 'jul')['e'] ** 2).mean()):.2f}")

    # ---------------------------------------------------------------- S2
    tag("S2  pushback - BLOCK offset, and the out-of-sample bias correction")
    print(C.group_by("ADEP_mvt").agg(
        pl.len(), pl.col("off").median().alias("med_all"),
        pl.col("off").filter(pl.col("half") == 0).median().alias("med_h0"),
        pl.col("off").filter(pl.col("half") == 1).median().alias("med_h1"),
        pl.col("off").mean().alias("mean_all"),
        (pl.col("off").abs() > 600).sum().alias("n_gt10min")).sort("ADEP_mvt"))
    print("half sizes", C.group_by("half").len().sort("half").rows())
    em = C["e_m"].to_numpy()
    y = C["y"].to_numpy()
    yraw = C["y_raw"].to_numpy()
    hh = C["half"].to_numpy()
    ap = C["ADEP_mvt"].to_numpy()
    EA = {}
    for sch in ["none", "pooled_h", "loao", "half"]:
        b = bias_table(C, sch)
        ea = (yraw + b) - y
        EA[sch] = ea
        print(f"  bias scheme {sch:9s}: ADS-B RMSE {np.sqrt((ea ** 2).mean()):7.2f}  MAE {np.abs(ea).mean():6.2f}"
              f"  mean err {ea.mean():+7.2f}  median err {np.median(ea):+6.1f}")
    # in-sample per-airport median, for reference only (not used anywhere downstream)
    b_in = C.select(pl.col("off").median().over("ADEP_mvt"))["off"].to_numpy()
    print(f"  [reference, in-sample per-airport median, NOT used] ADS-B RMSE {np.sqrt((((yraw + b_in) - y) ** 2).mean()):7.2f}")
    ea = EA["half"]
    print(f"\nmodel RMSE on these rows {np.sqrt((em ** 2).mean()):.2f}")

    # ---------------------------------------------------------------- S3
    tag("S3  ways of using it, on the 914 lane-A covered rows (bias = S2 'half', every parameter from the other half)")
    methods = [("model", None), ("replace", None), ("blend", 0.5), ("blend_cf", None)]
    methods += [("gate_gt", g) for g in (60, 120, 300)] + [("gate_gt_cf", None)]
    methods += [("guard_lt", g) for g in (600, 1200, 1800)] + [("guard_lt_cf", None)]
    hour = C["MVT_TIME_UTC_mvt"].dt.hour().to_numpy()
    clus = np.array([f"{a}{h}" for a, h in zip(ap, hour)])
    rows = []
    NEW = {}
    for m, p in methods:
        if m.endswith("_cf"):
            en, ps = crossfit(em, ea, hh, m)
            label = f"{m}  (half0 uses {ps[0]}, half1 uses {ps[1]})" if m != "blend_cf" else \
                f"{m}  (w half0 {ps[0]:.3f}, half1 {ps[1]:.3f})"
        else:
            en = new_errors(em, ea, m, p)
            label = m if p is None else f"{m} {p}"
        NEW[label.split("  ")[0]] = en
        dlt = em ** 2 - en ** 2
        lo, hi = boot_ci(dlt)
        clo, chi = boot_ci(dlt, clus)
        rows.append((label, len(en), np.sqrt((en ** 2).mean()), dlt.mean(), lo, hi, clo, chi,
                     float((np.abs(en) < np.abs(em)).mean()), float((np.abs(en) > np.abs(em) + 300).sum()),
                     float(np.abs(en).mean())))
    print(f"  {'method':52s} {'n':>4s} {'RMSE':>7s} {'MAE':>6s} {'dMSE/row':>9s} {'row 95% CI':>20s} {'apt-hour CI':>20s}"
          f" {'better':>7s} {'worse>5m':>8s}")
    for r in rows:
        print(f"  {r[0]:52s} {r[1]:4d} {r[2]:7.2f} {r[10]:6.1f} {r[3]:9.0f} [{r[4]:8.0f},{r[5]:8.0f}] [{r[6]:8.0f},{r[7]:8.0f}]"
              f" {r[8]:7.3f} {r[9]:8.0f}")
    print("  dMSE/row = mean(e_model^2 - e_new^2) in s^2 (+ve = better). 'better' = share of rows where |e_new| < |e_model|.")

    PRIMARY = "guard_lt_cf"
    M_GUARD = 900.0  # the cross-fitted guard picked 900 on both halves (S3)
    print("\nper airport, dMSE/row (s^2) by method; model RMSE on covered rows; ADS-B RMSE (bias 'half')")
    show = ["replace", "blend 0.5", "blend_cf", "gate_gt 120", "guard_lt 1200", PRIMARY]
    print("  " + f"{'ADEP':5s} {'n':>4s} {'mdl RMSE':>8s} {'adsb RMSE':>9s} " + " ".join(f"{s:>13s}" for s in show)
          + f" {PRIMARY + ' 95% CI':>24s} {'new RMSE':>8s}")
    PER = {}
    for a in APTS:
        k = ap == a
        if k.sum() == 0:
            continue
        vals = [(em[k] ** 2 - NEW[s][k] ** 2).mean() for s in show]
        lo, hi = boot_ci(em[k] ** 2 - NEW[PRIMARY][k] ** 2)
        PER[a] = dict(n=int(k.sum()), mse_m=float((em[k] ** 2).mean()), **{s: v for s, v in zip(show, vals)},
                      lo=lo, hi=hi)
        print(f"  {a:5s} {k.sum():4d} {np.sqrt((em[k] ** 2).mean()):8.1f} {np.sqrt((ea[k] ** 2).mean()):9.1f} "
              + " ".join(f"{v:13.0f}" for v in vals) + f" [{lo:10.0f},{hi:10.0f}] {np.sqrt((NEW[PRIMARY][k] ** 2).mean()):8.1f}")
    big = np.isin(ap, [a for a in PER if PER[a]["n"] >= MIN_N_AIRPORT])
    dl_big = em[big] ** 2 - NEW[PRIMARY][big] ** 2
    POOL_BIG = dict(point=float(dl_big.mean()), **dict(zip(("lo", "hi"), boot_ci(dl_big))))
    print(f"  pooled over airports with >= {MIN_N_AIRPORT} covered rows ({big.sum()} rows): {PRIMARY} dMSE/row "
          f"{POOL_BIG['point']:.0f} [{POOL_BIG['lo']:.0f}, {POOL_BIG['hi']:.0f}]")

    for meth in (PRIMARY, "replace"):
        dlt = em ** 2 - NEW[meth] ** 2
        order = np.argsort(-dlt)
        tot = dlt.sum()
        print(f"\nconcentration of the {meth} gain: total dSSE {tot:.3e}; top 10 rows carry {100 * dlt[order[:10]].sum() / tot:.1f} %,"
              f" top 50 {100 * dlt[order[:50]].sum() / tot:.1f} %, top 91 (10 %) {100 * dlt[order[:91]].sum() / tot:.1f} %;"
              f" without the top 10 rows dMSE/row {np.delete(dlt, order[:10]).mean():.0f}")
    dlt = em ** 2 - NEW[PRIMARY] ** 2
    print(f"rows with |e_model| > 600 s: {(np.abs(em) > 600).sum()}, their share of the {PRIMARY} dSSE "
          f"{100 * dlt[np.abs(em) > 600].sum() / dlt.sum():.1f} %")
    print("rows where replace hurts by > 5 min: " + ", ".join(
        f"{C['ADEP_mvt'][int(i)]} {C['cs'][int(i)]} e_m {em[i]:+.0f} e_a {ea[i]:+.0f}"
        for i in np.where(np.abs(ea) > np.abs(em) + 300)[0]))
    print(f"rows where {PRIMARY} hurts by > 2 min: " + ", ".join(
        f"{C['ADEP_mvt'][int(i)]} {C['cs'][int(i)]} e_m {em[i]:+.0f} e_new {NEW[PRIMARY][i]:+.0f}"
        for i in np.where(np.abs(NEW[PRIMARY]) > np.abs(em) + 120)[0]))

    # stress for the in-sample rules: rows where the guarded ADS-B is > 2 min off and makes the row worse are the
    # detection failures that pass the guard; multiply their loss by k (as if such failures were k times as common)
    f = (np.abs(ea) > 120) & (dlt < 0)
    print(f"guarded rows made worse: {(dlt < 0).sum()} (loss {-dlt[dlt < 0].sum() / 1e6:.2f}e6 s^2 against a gain of "
          f"{dlt[dlt > 0].sum() / 1e6:.2f}e6); of them with |ADS-B error| > 120 s: {f.sum()} (loss {-dlt[f].sum() / 1e6:.2f}e6)")
    for k in (2, 3, 5):
        print(f"  those failures x{k}: dMSE/row {(dlt.sum() - (k - 1) * (-dlt[f].sum())) / len(dlt):.0f}")
    print(f"  MAE model {np.abs(em).mean():.1f} -> guarded {np.abs(NEW[PRIMARY]).mean():.1f}; per airport "
          + ", ".join(f"{a_} {np.abs(em[ap == a_]).mean():.0f}->{np.abs(NEW[PRIMARY][ap == a_]).mean():.0f}"
                      for a_ in ("EDDM", "EHAM", "LEBL", "LSZH")))

    # lane B on the day (no AOBT_3): the rows with a pushback
    Bn = D.filter(~pl.col("A") & pl.col("t_push").is_not_null())
    if Bn.height:
        eb = Bn["e_m"].to_numpy()
        eab = (Bn["y_raw"].to_numpy() + np.median(C["off"].to_numpy())) - Bn["y"].to_numpy()
        Bl = D.filter(~pl.col("A"))
        gb = (eb ** 2 - eab ** 2).mean()
        print(f"\nlane B (no AOBT_3) on the day: n {Bl.height}, model RMSE {np.sqrt((Bl['e_m'] ** 2).mean()):.1f}; "
              f"pushback on {Bn.height} ({100 * Bn.height / Bl.height:.1f} %): model RMSE {np.sqrt((eb ** 2).mean()):.1f}, "
              f"ADS-B RMSE (lane-A pooled bias) {np.sqrt((eab ** 2).mean()):.1f}, dMSE/row {gb:.0f}")
        dsse_b = 8378 * Bn.height / Bl.height * gb
        print(f"  lane B dSSE if the pooled OOF lane B (8,378 rows) had this coverage and gain: {dsse_b:.3e} s^2 = "
              f"{np.sqrt((N_POOLED * RMSE_POOLED ** 2 - dsse_b) / N_POOLED) - RMSE_POOLED:.3f} s pooled OOF")

    # ---------------------------------------------------------------- S4
    tag("S4  label-free check on 2026-07-20: model-vs-ADS-B disagreement, v14 submission vs 2025 OOF; guarded gain proxy")
    nb = C.group_by("ADEP_mvt").agg(pl.len().alias("nb"), pl.col("off").median().alias("b25"))
    pooled_b = float(np.median(C["off"].to_numpy()))
    nb = nb.with_columns(pl.when(pl.col("nb") >= 10).then(pl.col("b25")).otherwise(pooled_b).alias("b25"))
    C2 = C.with_columns(pl.Series("ea", ea), pl.Series("dis", em - ea),
                        (pl.col("t_aobt3") - pl.col("t_push")).alias("aobt_minus_push"))
    T2 = (T.filter(pl.col("A") & pl.col("t_push").is_not_null() & pl.col("pred").is_not_null())
          .join(nb.select("ADEP_mvt", "b25"), on="ADEP_mvt", how="left")
          .with_columns(pl.col("b25").fill_null(pooled_b))
          .with_columns((pl.col("pred") - (pl.col("y_raw") + pl.col("b25"))).alias("dis"),
                        (pl.col("t_aobt3") - pl.col("t_push")).alias("aobt_minus_push")))

    def dstats(F, yr):
        return F.group_by("ADEP_mvt").agg(
            pl.len().alias(f"n{yr}"), (pl.col("dis") ** 2).mean().sqrt().alias(f"rms_dis{yr}"),
            (pl.col("dis").filter(pl.col("dis").abs() <= M_GUARD) ** 2).mean().sqrt().alias(f"rms_dis_g{yr}"),
            (pl.col("dis").abs() > M_GUARD).mean().alias(f"guarded{yr}"),
            (pl.col("aobt_minus_push").abs() > 1200).mean().alias(f"aobt20m{yr}"))

    print(dstats(C2, 25).join(dstats(T2, 26), on="ADEP_mvt", how="full", coalesce=True).sort("ADEP_mvt"))
    for F, yr in ((C2, 25), (T2, 26)):
        g = F.filter(pl.col("dis").abs() <= M_GUARD)
        print(f"  pooled {yr}: n {F.height}, rms disagreement {np.sqrt((F['dis'] ** 2).mean()):.1f}, inside guard "
              f"{np.sqrt((g['dis'] ** 2).mean()):.1f}; |dis| > {M_GUARD:.0f} s {(F['dis'].abs() > M_GUARD).mean():.4f}; "
              f"|AOBT_3 - push| > 20 min {(F['aobt_minus_push'].abs() > 1200).mean():.4f}")
    # proxy for the guarded gain: on rows inside the guard, if the ADS-B error a is independent of the model error,
    # E[e_m^2 - a^2] = E[dis^2] - 2 E[a^2]; outside the guard the gain is 0 by construction.
    dis25 = C2["dis"].to_numpy()
    inside = np.abs(dis25) <= M_GUARD
    a2_pool = float((ea[inside] ** 2).mean())
    print(f"\nguarded-gain proxy (guard {M_GUARD:.0f} s), validated on 2025; E[a^2 | inside] pooled {a2_pool:.0f} s^2 "
          f"(RMS {np.sqrt(a2_pool):.1f} s)")
    print(f"  {'ADEP':5s} {'n25':>4s} {'actual25':>9s} {'proxy25':>9s} {'E[a2|in]':>9s} {'n26':>4s} {'proxy26':>9s}")
    PROXY = {}
    for a in APTS + ["all"]:
        k25 = np.ones(len(C2), bool) if a == "all" else (ap == a)
        T3 = T2 if a == "all" else T2.filter(pl.col("ADEP_mvt") == a)
        if k25.sum() == 0 and T3.height == 0:
            continue
        kin = k25 & inside
        own = kin.sum() >= MIN_N_AIRPORT
        a2 = float((ea[kin] ** 2).mean()) if own else a2_pool
        act = float((em[k25] ** 2 - NEW[PRIMARY][k25] ** 2).mean()) if k25.sum() else float("nan")
        pr25 = float(np.where(inside[k25], dis25[k25] ** 2 - 2 * a2, 0).mean()) if k25.sum() else float("nan")
        d26 = T3["dis"].to_numpy()
        pr26 = float(np.where(np.abs(d26) <= M_GUARD, d26 ** 2 - 2 * a2, 0).mean()) if T3.height else float("nan")
        PROXY[a] = {"n26": T3.height, "proxy26": pr26, "proxy25": pr25, "actual": act}
        print(f"  {a:5s} {k25.sum():4d} {act:9.0f} {pr25:9.0f} {a2:9.0f} {T3.height:4d} {pr26:9.0f}"
              + ("   (E[a^2] borrowed from pooled 2025)" if not own else ""))

    # ---------------------------------------------------------------- S5
    if not skip_sweep:
        tag("S5  sensitivity of coverage, accuracy and the guarded gain to the detection thresholds (2025-07-21)")
        cfgs = [("frozen", {})]
        cfgs += [(f"stat_r={v:g}", {"stat_r": v}) for v in (15.0, 20.0, 35.0, 50.0)]
        cfgs += [(f"push_min={v:g}", {"push_min": v}) for v in (0.3, 0.4, 0.6, 0.7)]
        cfgs += [(f"push_win={v:g}", {"push_win": v}) for v in (90.0, 120.0, 300.0)]
        cfgs += [(f"opp_deg={v:g}", {"opp_deg": v}) for v in (90.0, 150.0)]
        cfgs += [(f"dwell={v:g}", {"dwell": v}) for v in (600.0, 1800.0, 1e9)]
        cfgs += [(f"gap_s={v:g}", {"gap_s": v}) for v in (300.0, 1200.0)]
        cfgs += [("no_gate", {"push_min": -1.0, "dwell": 1e9})]
        LA = D.filter(pl.col("A")).select("MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_block", "y", "e_m", "half")
        print(f"  {'config':16s} {'n':>5s} {'cov':>6s} {'MAE':>6s} {'>10m':>5s} {'adsb RMSE':>9s} {'mdl RMSE':>8s}"
              f" {'replace':>9s} {'guard900':>9s} {'guard dSSE 1e6':>14s} {'guard dSSE 4 GO apts':>20s} {'n EDDM/EHAM/LEBL/LSZH':>22s}")
        for name, kw in cfgs:
            tp = rerun_detection(D, **kw)
            X = LA.join(pl.DataFrame({"MVT_ID_mvt": list(tp.keys()), "tp": list(tp.values())},
                                     schema={"MVT_ID_mvt": pl.Float64, "tp": pl.Float64}), on="MVT_ID_mvt")
            X = X.with_columns((pl.col("tp") - pl.col("t_block")).alias("off"),
                               (pl.col("t_mvt") - pl.col("tp")).alias("y_raw"))
            b = bias_table(X, "half")
            eax = (X["y_raw"].to_numpy() + b) - X["y"].to_numpy()
            emx = X["e_m"].to_numpy()
            dl = emx ** 2 - eax ** 2
            gx = emx ** 2 - new_errors(emx, eax, "guard_lt", M_GUARD) ** 2
            go = X["ADEP_mvt"].is_in(["EDDM", "EHAM", "LEBL", "LSZH"]).to_numpy()
            n4 = "/".join(str((X["ADEP_mvt"] == a).sum()) for a in ("EDDM", "EHAM", "LEBL", "LSZH"))
            print(f"  {name:16s} {X.height:5d} {X.height / LA.height:6.3f} {X['off'].abs().mean():6.1f}"
                  f" {(X['off'].abs() > 600).sum():5d} {np.sqrt((eax ** 2).mean()):9.1f} {np.sqrt((emx ** 2).mean()):8.1f}"
                  f" {dl.mean():9.0f} {gx.mean():9.0f} {gx.sum() / 1e6:14.2f} {gx[go].sum() / 1e6:20.2f} {n4:>22s}", flush=True)

    # ---------------------------------------------------------------- S6
    tag("S6  scale-up: lane-A SSE -> lane-A RMSE -> pooled OOF -> board (x 0.747)")
    sse_pool = N_POOLED * RMSE_POOLED ** 2
    sse_a = N_A * RMSE_A ** 2
    bt = {r["ADEP_mvt"]: r for r in base.iter_rows(named=True)}
    rk = (pl.read_parquet(ROOT + "data/raw/ranking.parquet", columns=["PHASE_mvt", "ADEP_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"])
          .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("AOBT_3_flt").is_not_null())
          .group_by("ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m")).len())
    nb_board = {(r[0], r[1]): r[2] for r in rk.iter_rows()}
    MJUL, MJAN = 192_122 / N_BOARD, 152_719 / N_BOARD

    def gains(src="measured", level="point", small="pooled", egll="proxy"):
        """Per-airport dMSE per covered row (s^2) of the PRIMARY use.
        src     measured = S3 on 2025-07-21 (level point / lo / hi = row-bootstrap 95 % CI)
                proxy26  = S4 guarded proxy on 2026-07-20 (label-free; biased low on 2025, see S4)
                adj26    = measured 2025 gain + (proxy26 - proxy25): the 2025 gain moved by the change in the
                           guarded disagreement between the days (assumes the ADS-B error structure is stable)
        small   airports with < MIN_N_AIRPORT covered rows in the source: 'pooled' (pooled of the others) or 'zero'
        egll    'proxy' = EGLL from the 2026 proxy (it has 5 rows in 2025), 'small' = treat as the small airports"""
        g = {}
        for a in APTS:
            if src == "measured" and a in PER and PER[a]["n"] >= MIN_N_AIRPORT:
                g[a] = PER[a][PRIMARY] if level == "point" else PER[a][level]
            elif src == "proxy26" and a in PROXY and PROXY[a]["n26"] >= MIN_N_AIRPORT:
                g[a] = PROXY[a]["proxy26"]
            elif (src == "adj26" and a in PER and PER[a]["n"] >= MIN_N_AIRPORT and a in PROXY
                  and PROXY[a]["n26"] >= MIN_N_AIRPORT):
                g[a] = PER[a][PRIMARY] + PROXY[a]["proxy26"] - PROXY[a]["proxy25"]
            else:
                pool = {"measured": POOL_BIG.get(level, 0.0), "proxy26": PROXY["all"]["proxy26"],
                        "adj26": PROXY["all"]["actual"] + PROXY["all"]["proxy26"] - PROXY["all"]["proxy25"]}[src]
                g[a] = pool if small == "pooled" else 0.0
        if egll == "proxy" and PROXY.get("EGLL", {}).get("n26", 0) >= MIN_N_AIRPORT:
            g["EGLL"] = PROXY["EGLL"]["proxy26"] if src != "adj26" else \
                PROXY["all"]["actual"] + PROXY["EGLL"]["proxy26"] - PROXY["all"]["proxy25"]
        return g

    def price(g, cjul, cjan, scale, label, detail=False):
        """cjul/cjan: airport -> coverage in July / January of the scored period. The OOF lane-A rows of each
        airport are split by the board's Jan/Jul share. scale: multiply the gain by (pooled lane-A MSE /
        2025-07-21 lane-A MSE) of the airport, i.e. carry it as a fraction of the error level, not in s^2."""
        dsse = rows_cov = dmse_board = 0.0
        per = {}
        for a in APTS:
            ga = g[a] * (bt[a]["mseA_oof"] / bt[a]["mseA_day"] if scale else 1.0)
            c = cjul[a] * MJUL + cjan[a] * MJAN
            rows_cov += bt[a]["nA_oof"] * c
            dsse += bt[a]["nA_oof"] * c * ga
            per[a] = (c, ga, bt[a]["nA_oof"] * c * ga)
            dmse_board += (nb_board.get((a, 7), 0) * cjul[a] + nb_board.get((a, 1), 0) * cjan[a]) * ga / N_BOARD
        la_new = np.sqrt((sse_a - dsse) / N_A)
        po_new = np.sqrt((sse_pool - dsse) / N_POOLED)
        d_board = TRANSFER * (po_new - RMSE_POOLED)
        d_direct = np.sqrt(BOARD_V14 ** 2 - dmse_board) - BOARD_V14
        print(f"  {label:70s} rows {rows_cov / N_A:5.1%}  SSE removed {dsse / sse_a:6.2%} of lane A | lane A "
              f"{la_new - RMSE_A:+6.2f} | pooled OOF {po_new - RMSE_POOLED:+6.2f} | board x0.747 {d_board:+6.2f} "
              f"| board direct {d_direct:+6.2f}")
        if detail:
            print("      per airport: coverage / gain per covered row (s^2) / share of the SSE removed")
            print("      " + "  ".join(f"{a} {per[a][0]:.3f}/{per[a][1]:.0f}/{per[a][2] / dsse:.0%}" for a in APTS))
        return d_board

    cov25 = {a: float(bt[a]["cov25"] or 0) for a in APTS}
    cov26 = {a: float(bt[a]["cov26"] or 0) for a in APTS}
    covmin = {a: min(cov25[a], cov26[a]) for a in APTS}
    covmax = {a: max(cov25[a], cov26[a]) for a in APTS}
    half = lambda c: {a: 0.5 * v for a, v in c.items()}
    zero = {a: 0.0 for a in APTS}
    for s_, lv in (("measured", "point"), ("measured", "lo"), ("measured", "hi"), ("proxy26", "point"),
                   ("adj26", "point")):
        print(f"gain per covered row, {s_} {lv:5s} (s^2):", {a: round(v) for a, v in gains(s_, lv).items()})
    print("scale factor pooled lane-A MSE / 2025-07-21 lane-A MSE:",
          {a: round(bt[a]["mseA_oof"] / bt[a]["mseA_day"], 2) for a in APTS})
    print("coverage (lane A, pushback detected): 2025-07-21", {a: round(v, 3) for a, v in cov25.items()})
    print("                                      2026-07-20", {a: round(v, 3) for a, v in cov26.items()})

    print("\nA. the two measured days as if each were the whole scored period (same coverage in Jan and Jul), measured point gain:")
    for cv, cl in ((cov25, "2025-07-21"), (cov26, "2026-07-20")):
        for egl in ("proxy", "small"):
            price(gains("measured", "point", egll=egl), cv, cv, False, f"{PRIMARY}, cov {cl}, EGLL gain {egl}")
    print("  (for contrast) unguarded replace, cov 2026-07-20:")
    gr = {a: (PER[a]["replace"] if a in PER and PER[a]["n"] >= MIN_N_AIRPORT else 0.0) for a in APTS}
    price(gr, cov26, cov26, False, "replace (no guard), small airports 0")

    print("\nB. what moves it, one driver at a time from CENTRAL (2026-07-20 cov both months, measured point, absolute):")
    gc = gains("measured", "point")
    price(gc, cov26, cov26, False, "CENTRAL")
    price(gc, cov26, half(cov26), False, "  Jan coverage half of July's [X]")
    price(gc, cov26, zero, False, "  Jan coverage zero")
    price(gc, cov25, cov25, False, "  coverage of 2025-07-21 instead")
    price(gc, covmin, covmin, False, "  coverage min(day) per airport")
    price(gc, covmax, covmax, False, "  coverage max(day) per airport")
    price(gc, cov26, cov26, True, "  gain scaled to pooled lane-A MSE (the day was hard)")
    price(gains("measured", "lo"), cov26, cov26, False, "  gain at row-bootstrap 2.5 %")
    price(gains("measured", "hi"), cov26, cov26, False, "  gain at row-bootstrap 97.5 %")
    price(gains("measured", "point", egll="small"), cov26, cov26, False, "  EGLL gain = pooled measured instead of 2026 proxy")
    price(gains("measured", "point", small="zero", egll="small"), cov26, cov26, False, "  EGLL and small airports gain 0")
    price(gains("proxy26", "point"), cov26, cov26, False, "  all gains from the 2026 label-free proxy")
    price(gains("adj26", "point"), cov26, cov26, False, "  gains 2025 + (proxy26 - proxy25), EGLL by proxy")
    price(gains("adj26", "point", egll="small"), cov26, cov26, False, "  gains 2025 + (proxy26 - proxy25), EGLL pooled")

    print("\nC. the range:")
    lo = price(gains("measured", "lo", small="zero", egll="small"), covmin, half(covmin), True,
               "LOW   2.5 % gain, scaled, min(day) cov, Jan half, EGLL+small 0")
    lo2 = price(gains("proxy26", "point", small="zero"), covmin, half(covmin), False,
                "LOW'  2026 proxy gain (biased low), min(day) cov, Jan half, small 0")
    ce = price(gains("measured", "point", egll="small"), cov26, half(cov26), True,
               "CENTRAL point gain, scaled, EGLL pooled, 2026-07-20 cov Jul, Jan half", detail=True)
    ce2 = price(gains("adj26", "point", egll="small"), cov26, half(cov26), False,
                "CENTRAL' adj26 gain, EGLL pooled, 2026-07-20 cov Jul, Jan half")
    hi = price(gains("measured", "point"), covmax, covmax, False,
               "HIGH  point gain, absolute, max(day) cov, Jan = Jul")
    hi2 = price(gains("measured", "hi"), covmax, covmax, False,
                "HIGH+ 97.5 % gain, absolute, max(day) cov, Jan = Jul")
    return {"lo": lo, "lo2": lo2, "central": ce, "central2": ce2, "hi": hi, "hi2": hi2}


if __name__ == "__main__":
    main(skip_sweep="--skip-sweep" in sys.argv)
