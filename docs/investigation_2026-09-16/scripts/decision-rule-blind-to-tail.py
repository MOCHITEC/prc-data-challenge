"""Item 27 `decision-rule-blind-to-tail` (docs/investigation_2026-09-16/plan.md §1).

No training. POLARS_MAX_THREADS=1, lazy scans, only the columns needed, at most two OOF files in memory.
Uses the project's own functions (prc.compare / prc.cv) so that every number is what `make compare` would print.

Sections
  A  code facts: HEAD line numbers of verdict / robustness_curve / month_bootstrap / paired_bootstrap / load_pair
  B  which rows robustness_curve drops (K = 20 / 200 / 2000) on det_a -> ab_split and main-ab56551b5e -> main-f9d57a661f:
     overlap with rollover rows (y > 20 h), LIRF sched-filled rows, LIRF unmatched, unmatched, label tail, and with the
     rows where the two models disagree most; share of sum((p_a-p_b)^2) removed; alternative drop rules at K = 200
  C  paired-bootstrap CI width on unmatched-only / matched-only rows (H §5: 62 s vs 0.77 s)
  D  paired_bootstrap_rmse_diff arithmetic on a 3-row synthetic example (exhaustive enumeration of the 27 resamples),
     pairing check, NaN handling, verdict boundary cases
  E  load_pair on a `folds: [forward]` OOF vs a 3-fold OOF (which rows / months enter), Jan-only vs forward-only,
     Jan-only vs Jan-only; month bootstrap with 2 / 1 months
  F  re-judge existing OOF pairs under the current rule and four alternative rules; list flips
"""

import inspect
import itertools
import os
import sys
import time
import warnings

os.environ["POLARS_MAX_THREADS"] = "1"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from prc import compare, cv  # noqa: E402
from prc.compare import load_pair, month_bootstrap_rmse_diff, robustness_curve, verdict  # noqa: E402
from prc.cv import paired_bootstrap_rmse_diff, rmse, year_month_key  # noqa: E402
from prc.train import LABEL, PROXY  # noqa: E402

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(40)
warnings.simplefilter("ignore", RuntimeWarning)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
WORK = os.path.join(ROOT, "data", "work")
EXP = os.path.join(ROOT, "experiments")
N_BOOT = 500  # `make compare` default (compare.py main: --n-boot 500)
T0 = time.time()


def hdr(s):
    print(f"\n{'=' * 100}\n{s}   [t={time.time() - T0:6.0f}s]\n{'=' * 100}")


def fmt(r):
    return f"{r['diff']:+8.2f} [{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}] P(B)={r['b_better_prob']:.3f}"


# ------------------------------------------------------------------------------------------------------------------
hdr("A. code facts at HEAD (line numbers from inspect)")
for fn in (compare.robustness_curve, compare.month_bootstrap_rmse_diff, compare.verdict, compare.load_pair, compare.main):
    src, start = inspect.getsourcelines(fn)
    print(f"compare.py: {fn.__name__:28s} lines {start}-{start + len(src) - 1}")
for fn in (cv.paired_bootstrap_rmse_diff, cv.month_block_folds, cv.year_month_key):
    src, start = inspect.getsourcelines(fn)
    print(f"cv.py:      {fn.__name__:28s} lines {start}-{start + len(src) - 1}")
print("DROP_LEVELS =", compare.DROP_LEVELS, " SIGN_CHECK_LEVEL =", compare.SIGN_CHECK_LEVEL)
print("verdict() signature:", inspect.signature(compare.verdict), "-> takes only `curve` (robustness_curve output)")
main_src = inspect.getsource(compare.main)
print("main(): month bootstrap printed:", "month-level bootstrap" in main_src, "; passed to verdict():",
      "verdict(curve" in main_src and "mb" not in main_src.split("verdict(")[1].split(")")[0])
print("fold order from month_block_folds([202501..202512]):", [f.name for f in cv.month_block_folds(list(range(202501, 202513)))])

# ------------------------------------------------------------------------------------------------------------------
# data loading helpers
FEAT_COLS = ["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt", "ADEP_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "mvt_minus_sched", "FLIGHT_ID_mvt"]
feats = (
    pl.scan_parquet(os.path.join(WORK, "feat_train.parquet"))
    .select(FEAT_COLS)
    .with_columns(
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("b_minus_s"),
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"),
    )
    .collect()
)
print(f"\nfeat_train: {feats.height:,} rows, {len(FEAT_COLS)} columns loaded")


def oof(name):
    return pl.scan_parquet(os.path.join(EXP, f"oof_{name}.parquet")).filter(pl.col("oof").is_not_null()).collect()


def pair(a, b):
    """Same join as compare.load_pair (inner on MVT_ID_mvt, non-null label and both predictions), with extra columns."""
    d = feats.join(oof(a).rename({"oof": "p_a"}), on="MVT_ID_mvt", how="inner")
    d = d.join(oof(b).rename({"oof": "p_b"}), on="MVT_ID_mvt", how="inner")
    d = d.filter(pl.col(LABEL).is_not_null() & pl.col("p_a").is_not_null() & pl.col("p_b").is_not_null())
    y = d[LABEL].to_numpy().astype(float)
    return d, y, d["p_a"].to_numpy().astype(float), d["p_b"].to_numpy().astype(float)


def flags(d, y):
    unm = d[PROXY].is_null().to_numpy()
    lirf = (d["ADEP_mvt"] == "LIRF").to_numpy()
    bms = d["b_minus_s"].to_numpy().astype(float)
    ms = d["mvt_minus_sched"].to_numpy().astype(float)
    return {
        "unmatched": unm,
        "LIRF_unmatched": unm & lirf,
        "rollover(y>72000)": y > 72_000,
        "fill10(|B-S|<=10)": np.abs(bms) <= 10,
        "LIRF_unm_fill10": unm & lirf & (np.abs(bms) <= 10),
        "LIRF_unm_y==ms(60s)": unm & lirf & (np.abs(y - ms) <= 60),
        "y>=7200": y >= 7_200,
        "y>=14400": y >= 14_400,
        "y>=43200": y >= 43_200,
        "matched": ~unm,
    }


# ------------------------------------------------------------------------------------------------------------------
hdr("B. which rows robustness_curve drops")
B_PAIRS = [("det_a-cfc9417c52", "ab_split-ca87cb5af7"), ("main-ab56551b5e", "main-f9d57a661f")]
for a, b in B_PAIRS:
    d, y, pa, pb = pair(a, b)
    fl = flags(d, y)
    se_a, se_b = (y - pa) ** 2, (y - pb) ** 2
    gap2 = (pa - pb) ** 2
    worst = np.argsort(np.maximum(se_a, se_b))[::-1]  # identical to compare.robustness_curve line 43
    print(f"\n--- A={a}  B={b}  rows={len(y):,}  RMSE A={rmse(y, pa):.2f} B={rmse(y, pb):.2f}")
    print(f"population: " + ", ".join(f"{k}={int(v.sum())}" for k, v in fl.items()))
    print(f"\n{'K':>5} | " + " | ".join(f"{k[:19]:>19}" for k in fl) + " | share SSE_A | share SSE_B | share sum(pa-pb)^2 | overlap top-K by (pa-pb)^2")
    top_gap = {k: set(np.argsort(gap2)[::-1][:k]) for k in (20, 200, 2000)}
    for k in (20, 200, 2000):
        drop = worst[:k]
        cells = [int(fl[n][drop].sum()) for n in fl]
        ov = len(set(drop) & top_gap[k])
        print(f"{k:>5} | " + " | ".join(f"{c:>19}" for c in cells)
              + f" | {se_a[drop].sum() / se_a.sum():11.3f} | {se_b[drop].sum() / se_b.sum():11.3f} | {gap2[drop].sum() / gap2.sum():18.3f} | {ov:>5}")
    # rank of every rollover row in the drop order
    ro = np.flatnonzero(fl["rollover(y>72000)"])
    rank = np.empty(len(y), int)
    rank[worst] = np.arange(len(y))
    print("\nrollover rows (y>20h) in this OOF: rank in drop order (0 = dropped first), ADEP, y, p_a, p_b:")
    for i in ro[np.argsort(rank[ro])]:
        print(f"   rank {rank[i]:>5}  {d['ADEP_mvt'][int(i)]}  y={y[i]:8.0f}  p_a={pa[i]:8.0f}  p_b={pb[i]:8.0f}  unm={bool(fl['unmatched'][i])}")
    sf = np.flatnonzero(fl["LIRF_unm_fill10"])
    print(f"LIRF unmatched fill10 rows: n={len(sf)}, rank<20: {(rank[sf] < 20).sum()}, rank<200: {(rank[sf] < 200).sum()}, "
          f"rank<2000: {(rank[sf] < 2000).sum()}; of those with y>=7200 (n={int((y[sf] >= 7200).sum())}): "
          f"rank<200: {int(((rank[sf] < 200) & (y[sf] >= 7200)).sum())}, rank<2000: {int(((rank[sf] < 2000) & (y[sf] >= 7200)).sum())}")
    lu = np.flatnonzero(fl["LIRF_unmatched"])
    print(f"LIRF unmatched rows: n={len(lu)}, rank<200: {(rank[lu] < 200).sum()}, rank<2000: {(rank[lu] < 2000).sum()}")
    # top-20 disagreement rows: what are they and are they dropped?
    tg = np.argsort(gap2)[::-1][:20]
    print("\ntop-20 rows by (p_a-p_b)^2: rank in drop order, ADEP, unm, y, p_a, p_b:")
    for i in tg:
        print(f"   rank {rank[i]:>6}  {d['ADEP_mvt'][int(i)]}  unm={bool(fl['unmatched'][i])!s:5}  y={y[i]:8.0f}  p_a={pa[i]:8.0f}  p_b={pb[i]:8.0f}")
    print(f"top-20 disagreement rows dropped at K=20: {(rank[tg] < 20).sum()}, K=200: {(rank[tg] < 200).sum()}, K=2000: {(rank[tg] < 2000).sum()}")

    # the actual `make compare` output for this pair
    print(f"\n`make compare A={a} B={b}` reproduction (n_boot={N_BOOT}):")
    curve = robustness_curve(y, pa, pb, n_boot=N_BOOT)
    for r in curve:
        print(f"   dropped {r['dropped']:>5}  A={r['rmse_a']:8.2f}  B={r['rmse_b']:8.2f}  {fmt(r)}  CI width={r['ci95'][1] - r['ci95'][0]:.2f}")
    ym = d["ym"].to_numpy()
    mb = month_bootstrap_rmse_diff(y, pa, pb, ym, n_boot=N_BOOT)
    print(f"   month bootstrap ({mb['n_months']} months {sorted(set(ym.tolist()))}): {mb['diff_rmse']:+.2f} [{mb['ci95'][0]:+.2f},{mb['ci95'][1]:+.2f}] P(B)={mb['b_better_prob']:.3f}")
    ok, why = verdict(curve)
    print(f"   VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")

    # alternative drop rules at K=200
    print("\nK=200 under alternative choices of the dropped rows (same paired bootstrap, seed 0):")
    rules = {
        "max(se_a,se_b) [current]": np.maximum(se_a, se_b),
        "se_a only (baseline)": se_a,
        "se_b only (candidate)": se_b,
        "largest label y": y,
        "largest (p_a-p_b)^2": gap2,
        "random 200 (seed 1)": np.random.default_rng(1).random(len(y)),
    }
    for name, score in rules.items():
        drop = np.argsort(score)[::-1][:200]
        keep = np.ones(len(y), bool)
        keep[drop] = False
        r = paired_bootstrap_rmse_diff(y[keep], pa[keep], pb[keep], n_boot=N_BOOT, seed=0)
        print(f"   {name:28s} A={rmse(y[keep], pa[keep]):8.2f} B={rmse(y[keep], pb[keep]):8.2f} diff={r['diff_rmse']:+8.2f} "
              f"[{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}]  rollover dropped={int(fl['rollover(y>72000)'][drop].sum())}  "
              f"unmatched dropped={int(fl['unmatched'][drop].sum())}")

# ------------------------------------------------------------------------------------------------------------------
hdr("C. paired-bootstrap CI width on unmatched-only / matched-only rows")
C_PAIRS = [("det_a-cfc9417c52", "ab_split-ca87cb5af7"), ("ab_single-3978ee26d2", "ab_split-ca87cb5af7"),
           ("main-ab56551b5e", "main-f9d57a661f"), ("det_a-cfc9417c52", "det_b-fa2cba5913"),
           ("ff1_noise-47e9566782", "feat_f5_rwyact-33f787c69b")]
for a, b in C_PAIRS:
    d, y, pa, pb = pair(a, b)
    unm = d[PROXY].is_null().to_numpy()
    print(f"\n--- A={a}  B={b}  rows={len(y):,}")
    for name, sel in (("all", np.ones(len(y), bool)), ("unmatched", unm), ("matched", ~unm)):
        r = paired_bootstrap_rmse_diff(y[sel], pa[sel], pb[sel], n_boot=N_BOOT, seed=0)
        print(f"   {name:10s} n={int(sel.sum()):>8,}  A={rmse(y[sel], pa[sel]):8.2f}  B={rmse(y[sel], pb[sel]):8.2f}  "
              f"diff={r['diff_rmse']:+8.2f} CI=[{r['ci95'][0]:+8.2f},{r['ci95'][1]:+8.2f}]  width={r['ci95'][1] - r['ci95'][0]:7.2f}")
    # n_boot sensitivity of the width on unmatched rows (seed 0 vs 1, 500 vs 2000)
    r1 = paired_bootstrap_rmse_diff(y[unm], pa[unm], pb[unm], n_boot=N_BOOT, seed=1)
    r2 = paired_bootstrap_rmse_diff(y[unm], pa[unm], pb[unm], n_boot=2000, seed=0)
    print(f"   unmatched width: seed1/500 -> {r1['ci95'][1] - r1['ci95'][0]:.2f}; seed0/2000 -> {r2['ci95'][1] - r2['ci95'][0]:.2f}")

# ------------------------------------------------------------------------------------------------------------------
hdr("D. paired_bootstrap_rmse_diff arithmetic on a synthetic example (no model)")
y3 = np.array([0.0, 0.0, 0.0])
pa3 = np.array([1.0, 2.0, 3.0])
pb3 = np.array([1.0, 1.0, 1.0])
r = paired_bootstrap_rmse_diff(y3, pa3, pb3, n_boot=200_000, seed=0)
point_hand = np.sqrt(np.mean((y3 - pb3) ** 2)) - np.sqrt(np.mean((y3 - pa3) ** 2))
print(f"y={y3}, p_a={pa3}, p_b={pb3}")
print(f"hand: RMSE_A=sqrt(14/3)={np.sqrt(14 / 3):.6f}, RMSE_B=1, diff={point_hand:.6f}; function diff_rmse={r['diff_rmse']:.6f}  match={np.isclose(point_hand, r['diff_rmse'])}")
# exhaustive enumeration of the 27 equiprobable resample index tuples
vals = []
for idx in itertools.product(range(3), repeat=3):
    idx = np.array(idx)
    vals.append(np.sqrt(((y3 - pb3)[idx] ** 2).mean()) - np.sqrt(((y3 - pa3)[idx] ** 2).mean()))
vals = np.sort(np.array(vals))
lo_ex, hi_ex = np.percentile(vals, [2.5, 97.5])
p_ex = (vals < 0).mean()
print(f"exhaustive 27 resamples: distinct diff values = {np.unique(np.round(vals, 6))}")
print(f"exhaustive percentile CI = [{lo_ex:.4f},{hi_ex:.4f}], P(diff<0)={p_ex:.4f}")
print(f"bootstrap (n_boot=200000): CI = [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}], b_better_prob={r['b_better_prob']:.4f}, n={r['n']}, n_dropped={r['n_dropped']}")
print(f"min/max of exhaustive = [{vals.min():.4f},{vals.max():.4f}]  (the 2.5/97.5 percentile of a 27-point set sits between the 1st and 2nd / 26th and 27th values)")
# pairing check: identical predictions -> every resample diff exactly 0
r_same = paired_bootstrap_rmse_diff(np.random.default_rng(3).normal(900, 300, 1000), *(lambda p: (p, p.copy()))(np.random.default_rng(4).normal(900, 300, 1000)), n_boot=500)
print(f"pairing check (p_b == p_a): diff={r_same['diff_rmse']}, CI={r_same['ci95']}, P(B)={r_same['b_better_prob']}  -> paired resampling confirmed if CI == [0,0]")
# permutation check: same marginal errors, different pairing -> the CI is not [0,0] (pairing matters)
yy = np.zeros(6)
qa = np.array([1.0, 2, 3, 4, 5, 6])
qb = qa[::-1].copy()
r_perm = paired_bootstrap_rmse_diff(yy, qa, qb, n_boot=20_000, seed=0)
print(f"permutation check (p_b = reversed p_a, identical RMSE): diff={r_perm['diff_rmse']:.4f}, CI=[{r_perm['ci95'][0]:+.3f},{r_perm['ci95'][1]:+.3f}]")
# NaN handling
r_nan = paired_bootstrap_rmse_diff(np.array([0.0, 0, 0, np.nan]), np.array([1.0, 2, 3, 1]), np.array([1.0, 1, 1, np.nan]), n_boot=100)
print(f"NaN row: n={r_nan['n']}, n_dropped={r_nan['n_dropped']}, diff={r_nan['diff_rmse']:.6f} (should equal the 3-row diff {point_hand:.6f})")
# seed reuse inside robustness_curve: same seed=0 for every K
print(f"robustness_curve default seed: {inspect.signature(robustness_curve).parameters['seed'].default}; paired_bootstrap called with seed=seed for every K (compare.py line 50)")
# verdict boundary cases
def _c(full_ci, rob_ci, fd=-1.0, rd=-0.5):
    return [{"dropped": 0, "rmse_a": 0, "rmse_b": 0, "diff": fd, "ci95": full_ci, "b_better_prob": 0},
            {"dropped": 200, "rmse_a": 0, "rmse_b": 0, "diff": rd, "ci95": rob_ci, "b_better_prob": 0}]
for name, c in (("full hi == 0 exactly", _c([-2.0, 0.0], [-1.0, -0.1])),
                ("robust hi == 0 exactly", _c([-2.0, -0.5], [-1.0, 0.0])),
                ("full lo > 0 (worse)", _c([0.1, 2.0], [-1.0, -0.1], fd=1.0)),
                ("both improving", _c([-2.0, -0.5], [-1.0, -0.1])),
                ("full improving, robust improving but no K=200 level", _c([-2.0, -0.5], [-1.0, -0.1])[:1]),
                ("month bootstrap not an input", None)):
    if c is None:
        print(f"   {name}: verdict() has no month argument (signature {inspect.signature(verdict)})")
        continue
    ok, why = verdict(c)
    print(f"   {name:55s} -> {'ADOPT' if ok else 'REJECT'}: {why}")

# ------------------------------------------------------------------------------------------------------------------
hdr("E. load_pair on a `folds: [forward]` OOF vs a 3-fold OOF")
from pathlib import Path  # noqa: E402

for a, b in (("det_a-cfc9417c52", "feat_base-4908e7408f"), ("rep_jan_on-9db9f2476e", "feat_base-4908e7408f"),
             ("rep_jan_off-ef82a28b63", "rep_jan_on-9db9f2476e"), ("leak_check-54a21538de", "det_a-cfc9417c52")):
    d = load_pair(a, b, Path(WORK), Path(EXP))
    ym = year_month_key(d).to_numpy() if d.height else np.array([])
    months, counts = np.unique(ym, return_counts=True) if d.height else ([], [])
    print(f"\nload_pair({a}, {b}): rows={d.height:,}; months={dict(zip(months.tolist(), counts.tolist())) if d.height else {}}")
    na = oof(a).height
    nb = oof(b).height
    print(f"   non-null rows: A={na:,}  B={nb:,}  -> comparison uses {d.height:,} rows = {d.height / max(na, 1):.3f} of A, {d.height / max(nb, 1):.3f} of B; no warning is printed by compare.main")
    if d.height:
        y = d[LABEL].to_numpy().astype(float)
        pa, pb = d["p_a"].to_numpy().astype(float), d["p_b"].to_numpy().astype(float)
        mb = month_bootstrap_rmse_diff(y, pa, pb, ym, n_boot=N_BOOT)
        print(f"   month bootstrap: n_months={mb['n_months']}, diff={mb['diff_rmse']:+.2f}, CI=[{mb['ci95'][0]:+.2f},{mb['ci95'][1]:+.2f}]")
        # number of distinct month-multisets the month bootstrap can draw
        m = mb["n_months"]
        if m:
            from math import comb
            print(f"   distinct resample multisets with {m} months: C({m}+{m}-1,{m}) = {comb(2 * m - 1, m)} -> the CI is a percentile of a {comb(2 * m - 1, m)}-valued discrete distribution")
    else:
        curve = robustness_curve(np.array([]), np.array([]), np.array([]), n_boot=10)
        print(f"   robustness_curve on 0 rows -> {curve}; verdict -> {verdict(curve)}")

# ------------------------------------------------------------------------------------------------------------------
hdr("F. re-judge existing OOF pairs: current rule vs alternatives")
F_PAIRS = [
    # 3-fold (jan, jul, forward) pairs; the handoff/README numbers they should reproduce are in the comment
    ("ab_single-3978ee26d2", "ab_split-ca87cb5af7", "H: -74.10 / -6.26 / -6.04"),
    ("ab_split-ca87cb5af7", "abl_vanilla_params-c2251eb080", "H: -0.94(CI spans 0) / +1.41 / +1.58"),
    ("ab_split-ca87cb5af7", "abl_no_encodings-8cf7918aa5", "H: +37.37 / +1.25 / +1.24"),
    ("ab_split-ca87cb5af7", "abl_no_categoricals-7c6f287f34", "H: +7.38 / +7.16 / +4.68"),
    ("main-ab56551b5e", "main-f9d57a661f", "README v2->v6?: -34.82 / -0.06 [-0.61,+0.52] / -0.26"),
    ("det_a-cfc9417c52", "ab_split-ca87cb5af7", "item pair"),
    ("repro_v2-26c5b53b4d", "det_a-cfc9417c52", "determinism"),
    ("det_a-cfc9417c52", "det_b-fa2cba5913", "determinism"),
    ("det_a-cfc9417c52", "main-48c19bc830", "1 seed -> 3 seeds (current code)"),
    ("main-63086f0648", "main-48c19bc830", "main -> main"),
    ("pw_noise-c8f996bf3d", "pw_both-91b711793b", "pw"),
    ("pw_noise2-7672c5b2b7", "f10_plannedflight-b416261a17", "H: planned_flight_s -0.21 [-0.49,+0.08]"),
    # forward-only pairs (328,009 rows, months 11-12)
    ("feat_base-b9e21a829d", "feat_f1_queue-da758e547b", "features_log: +0.49 vs base"),
    ("feat_base-b9e21a829d", "feat_f2_bank-5eb8fdddfd", "features_log: +0.39 vs base"),
    ("feat_base-b9e21a829d", "feat_f3_share-15cc219230", "features_log: +0.50 vs base"),
    ("feat_base-4908e7408f", "feat_f4_noise-dfae91b597", "features_log: noise +0.53 vs base"),
    ("feat_f4_noise-dfae91b597", "feat_f1_queue-da758e547b", "H: -0.16 [-0.40,+0.08]"),
    ("feat_f4_noise-dfae91b597", "feat_f2_bank-5eb8fdddfd", "H: -0.19 [-0.41,+0.07]"),
    ("feat_f4_noise-dfae91b597", "feat_f3_share-15cc219230", "H: -0.10 [-0.38,+0.14]"),
    ("ff1_base-b60b8da07b", "ff1_noise-36511caad3", "noise vs base, ff=1.0"),
    ("ff1_noise-47e9566782", "feat_f5_rwyact-33f787c69b", "H: -0.06 [-0.29,+0.17]"),
    ("ff1_noise-47e9566782", "feat_f6_deparr-e8ed6f23f5", "H: +0.03 [-0.16,+0.22]"),
    ("ff1_noise-47e9566782", "feat_f7_vsusual-613ee07de4", "H: +0.09 [-0.16,+0.31]"),
    ("ff1_noise-47e9566782", "feat_f8_opoffset-c4dc7f0a64", "H: rejected"),
    ("ff1_noise2-856a62b645", "feat_f9_dist-5ed94c00a3", "H: rejected (tail-only)"),
    ("ff1_noise-36511caad3", "ff1_noise-47e9566782", "noise vs noise (different sha)"),
    ("ff1_noise-47e9566782", "ff1_noise2-856a62b645", "noise vs noise2"),
    # mixed: 3-fold baseline vs forward-only candidate (silently restricted to Nov-Dec)
    ("det_a-cfc9417c52", "feat_base-4908e7408f", "3-fold vs forward-only"),
]


def rules(curve, mb):
    full = next(r for r in curve if r["dropped"] == 0)
    rob = next((r for r in curve if r["dropped"] == 200), None)
    r2k = next((r for r in curve if r["dropped"] == 2000), None)
    cur = verdict(curve)[0]
    return {
        "current(full&drop200)": cur,
        "full only": full["ci95"][1] < 0,
        "month only": (mb["ci95"][1] < 0) if mb["n_months"] >= 2 else None,
        "drop200 only": rob is not None and rob["ci95"][1] < 0,
        "full&month": full["ci95"][1] < 0 and mb["n_months"] >= 2 and mb["ci95"][1] < 0,
        "drop2000 only": r2k is not None and r2k["ci95"][1] < 0,
    }


rows_out = []
for a, b, note in F_PAIRS:
    d, y, pa, pb = pair(a, b)
    ym = d["ym"].to_numpy()
    curve = robustness_curve(y, pa, pb, n_boot=N_BOOT)
    mb = month_bootstrap_rmse_diff(y, pa, pb, ym, n_boot=N_BOOT)
    ok, why = verdict(curve)
    rl = rules(curve, mb)
    by = {r["dropped"]: r for r in curve}
    unm = d[PROXY].is_null().to_numpy()
    print(f"\n--- A={a}  B={b}  ({note})  rows={len(y):,} months={sorted(set(ym.tolist()))}")
    for k in (0, 20, 200, 2000):
        r = by[k]
        print(f"   drop {k:>5}: A={r['rmse_a']:8.2f} B={r['rmse_b']:8.2f} {fmt(r)} width={r['ci95'][1] - r['ci95'][0]:.2f}")
    print(f"   month    : {mb['diff_rmse']:+8.2f} [{mb['ci95'][0]:+7.2f},{mb['ci95'][1]:+7.2f}] P(B)={mb['b_better_prob']:.3f} n_months={mb['n_months']}")
    print(f"   matched  : A={rmse(y[~unm], pa[~unm]):8.2f} B={rmse(y[~unm], pb[~unm]):8.2f} | unmatched: A={rmse(y[unm], pa[unm]):8.1f} B={rmse(y[unm], pb[unm]):8.1f}")
    print(f"   VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")
    print("   rules  : " + ", ".join(f"{k}={'ADOPT' if v else ('n/a' if v is None else 'REJECT')}" for k, v in rl.items()))
    rows_out.append((a, b, len(y), by[0]["diff"], by[0]["ci95"], by[200]["diff"], by[200]["ci95"], mb["diff_rmse"], mb["ci95"], rl))

hdr("F summary: verdict flips relative to the current rule")
print(f"{'A -> B':70s} {'full':>22s} {'drop200':>22s} {'month':>22s}  current  full  month  d200  full&mo  d2000")
for a, b, n, fd, fci, rd, rci, md, mci, rl in rows_out:
    def v(x):
        return "ADOPT " if x else ("  n/a " if x is None else "REJECT")
    print(f"{a + ' -> ' + b:70s} {fd:+7.2f}[{fci[0]:+6.2f},{fci[1]:+6.2f}] {rd:+7.2f}[{rci[0]:+6.2f},{rci[1]:+6.2f}] {md:+7.2f}[{mci[0]:+6.2f},{mci[1]:+6.2f}]  "
          + "  ".join(v(rl[k]) for k in rl))
flips = [(a, b, k) for a, b, *_, rl in rows_out for k, val in rl.items() if val is not None and val != rl["current(full&drop200)"]]
print(f"\nflips vs current rule: {len(flips)}")
for a, b, k in flips:
    print(f"   {a} -> {b}: {k}")
print(f"\ndone in {time.time() - T0:.0f}s")
