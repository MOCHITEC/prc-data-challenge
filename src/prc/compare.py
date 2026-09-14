"""Decide whether one experiment beats another, under a metric a handful of rows can dominate.

    python -m prc.compare v6_unmatched_wider-fffd3895cb v8_seed_avg-6b88e350f8

Why this module exists: RMSE squares the error, and in this dataset 10 labelled rows out of 672,428 carry 24%
of the total squared error (docs/08 section 2). A plain paired bootstrap over rows therefore answers "did B beat
A on those few rows", which is a lottery the ranking set will not repeat. v6 looked like a 34.8 s improvement
and is 0.06 s once 200 rows are removed (docs/07). So a comparison here reports three things, and the verdict
needs all of them to agree:

  1. the honest full-data difference, because that is the metric being scored;
  2. the same difference with the worst K rows removed, because an effect that vanishes at K=200 was 200 rows;
  3. a bootstrap that resamples whole months rather than rows, because rows within a month are not independent
     and the tail is seasonal.

Removing "the worst K rows" is deliberately biased against finding a difference: the rows dropped are the ones
where the two models disagree most. That is the point - it asks what is left after the disagreement is gone."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .cv import paired_bootstrap_rmse_diff, rmse, year_month_key
from .train import LABEL, PROXY

DROP_LEVELS = (0, 20, 200, 2000)
SIGN_CHECK_LEVEL = 200  # an effect that reverses once this many rows are gone was those rows


def robustness_curve(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    drop_levels: tuple[int, ...] = DROP_LEVELS,
    n_boot: int = 500,
    seed: int = 0,
) -> list[dict]:
    """RMSE(B) - RMSE(A) after removing the K rows where either model errs most. Negative = B better."""
    worst = np.argsort(np.maximum((y - p_a) ** 2, (y - p_b) ** 2))[::-1]
    out = []
    for k in drop_levels:
        if k >= len(y):
            continue
        keep = np.ones(len(y), bool)
        keep[worst[:k]] = False
        r = paired_bootstrap_rmse_diff(y[keep], p_a[keep], p_b[keep], n_boot=n_boot, seed=seed)
        out.append(
            {
                "dropped": k,
                "rmse_a": rmse(y[keep], p_a[keep]),
                "rmse_b": rmse(y[keep], p_b[keep]),
                "diff": r["diff_rmse"],
                "ci95": r["ci95"],
                "b_better_prob": r["b_better_prob"],
            }
        )
    return out


def month_bootstrap_rmse_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, month: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    """The same difference, resampling whole months with replacement.

    Rows inside a month share weather, schedule and whatever reporting regime was running that month, so a
    row-level bootstrap treats correlated rows as independent evidence and reports a CI that is too narrow.
    Resampling months is the honest unit when the question is 'will this hold next month'."""
    months = np.unique(month)
    if len(months) < 2:
        return {"diff_rmse": float("nan"), "ci95": [float("nan")] * 2, "b_better_prob": float("nan"), "n_months": 0}
    idx_by_month = {m: np.flatnonzero(month == m) for m in months}
    rng = np.random.default_rng(seed)
    se_a, se_b = (y - p_a) ** 2, (y - p_b) ** 2
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([idx_by_month[m] for m in rng.choice(months, len(months), replace=True)])
        diffs[i] = np.sqrt(se_b[pick].mean()) - np.sqrt(se_a[pick].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff_rmse": float(np.sqrt(se_b.mean()) - np.sqrt(se_a.mean())),
        "ci95": [float(lo), float(hi)],
        "b_better_prob": float((diffs < 0).mean()),
        "n_months": len(months),
    }


def verdict(curve: list[dict], sign_check_level: int = SIGN_CHECK_LEVEL) -> tuple[bool, str]:
    """Adopt B only when BOTH the full-data and the tail-removed comparison exclude zero on the improving side.

    Requiring the second one is deliberately conservative, and it does reject changes that genuinely help only
    the tail - which the metric does score. That trade is taken on purpose: the tail of a CV fold is not the
    tail of the ranking set (10 rows carry 24% of the metric here, and they are different 10 rows), and both
    the 2024 and 2025 editions scored the final standings on a set nobody had seen (docs/01 U2). A change that
    is real only on the rows that will not recur is a change that does not transfer. Anything rejected for that
    reason is reported as tail-only rather than as no effect, because it is a judgement call, not a null."""
    full = next((r for r in curve if r["dropped"] == 0), None)
    if full is None:
        return False, "no full-data comparison"
    if full["ci95"][1] >= 0:
        return False, f"full-data CI includes zero [{full['ci95'][0]:+.2f}, {full['ci95'][1]:+.2f}]"
    robust = next((r for r in curve if r["dropped"] == sign_check_level), None)
    if robust is None:
        return True, "adopt (too few rows to run the robustness check)"
    if robust["ci95"][1] >= 0:
        return False, (
            f"tail-only: {full['diff']:+.2f} s on all rows, but {robust['diff']:+.2f} s "
            f"[{robust['ci95'][0]:+.2f}, {robust['ci95'][1]:+.2f}] once the worst {sign_check_level} rows are "
            f"removed - the effect is those rows, and the ranking set has different ones"
        )
    return True, f"adopt ({full['diff']:+.2f} s, still {robust['diff']:+.2f} s at {sign_check_level} rows dropped)"


def load_pair(exp_a: str, exp_b: str, work_dir: Path, exp_dir: Path) -> pl.DataFrame:
    """Join two out-of-fold prediction files onto the labels, keeping only rows both models scored."""
    feats = pl.read_parquet(work_dir / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt"])
    d = feats
    for name, col in ((exp_a, "p_a"), (exp_b, "p_b")):
        oof = pl.read_parquet(exp_dir / f"oof_{name}.parquet").rename({"oof": col})
        d = d.join(oof, on="MVT_ID_mvt", how="inner")
    return d.filter(pl.col(LABEL).is_not_null() & pl.col("p_a").is_not_null() & pl.col("p_b").is_not_null())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp_a", help="baseline experiment id, as in experiments/oof_<id>.parquet")
    ap.add_argument("exp_b", help="candidate experiment id")
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)

    d = load_pair(a.exp_a, a.exp_b, a.work_dir, a.exp_dir)
    y = d[LABEL].to_numpy().astype(float)
    p_a, p_b = d["p_a"].to_numpy(), d["p_b"].to_numpy()
    se = (y - p_b) ** 2
    order = np.sort(se)[::-1]
    print(f"A = {a.exp_a}\nB = {a.exp_b}\nrows = {len(y):,}")
    print(f"top-20 share of B's squared error: {order[:20].sum() / se.sum():.3f}\n")

    curve = robustness_curve(y, p_a, p_b, n_boot=a.n_boot)
    print(f"{'dropped':>8} {'A':>9} {'B':>9} {'diff':>8} {'CI95':>20} {'P(B)':>6}")
    for r in curve:
        ci = f"[{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}]"
        print(
            f"{r['dropped']:>8} {r['rmse_a']:9.2f} {r['rmse_b']:9.2f} {r['diff']:+8.2f} {ci:>20} {r['b_better_prob']:6.3f}"
        )

    mb = month_bootstrap_rmse_diff(y, p_a, p_b, year_month_key(d).to_numpy(), n_boot=a.n_boot)
    print(
        f"\nmonth-level bootstrap ({mb['n_months']} months): {mb['diff_rmse']:+.2f} s "
        f"[{mb['ci95'][0]:+.2f}, {mb['ci95'][1]:+.2f}]  P(B better) = {mb['b_better_prob']:.3f}"
    )

    matched = d[PROXY].is_not_null().to_numpy()
    print()
    for name, sel in (("matched", matched), ("unmatched", ~matched)):
        if sel.any():
            print(f"{name:>10}: n={sel.sum():>7,}  A={rmse(y[sel], p_a[sel]):8.1f}  B={rmse(y[sel], p_b[sel]):8.1f}")

    ok, why = verdict(curve)
    print(f"\nVERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
