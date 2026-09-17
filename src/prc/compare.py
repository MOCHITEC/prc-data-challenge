"""Decide whether one experiment beats another, under a metric a handful of rows can dominate.

    python -m prc.compare v6_unmatched_wider-fffd3895cb v8_seed_avg-6b88e350f8

Why this module exists: RMSE squares the error, and in this dataset 10 labelled rows out of 672,428 carry 24%
of the total squared error on oof_main-f9d57a661f - 34.7% of it on oof_main-48c19bc830, so the share is a
property of the file as well as of the rows (docs/README §4). A plain paired bootstrap over rows therefore
answers "did B beat A on those few rows", which is a lottery the ranking set will not repeat. v6 looked like a
34.8 s improvement and is 0.06 s once 200 rows are removed (docs/README; that exact pair is not reproducible
today, the nearest measured pairs give -34.58 -> -0.27 and -36.60 -> -0.36). So a comparison here reports three things, and the verdict
needs all of them to agree:

  1. the honest full-data difference, because that is the metric being scored;
  2. the same difference with the worst K rows removed, because an effect that vanishes at K=200 was 200 rows;
  3. a bootstrap that resamples whole months rather than rows, because rows within a month are not independent
     and the tail is seasonal.

Removing "the worst K rows" is deliberately biased against finding a difference: the rows dropped are the ones
where the two models disagree most. That is the point - it asks what is left after the disagreement is gone.

The verdict is unchanged by everything below `verdict()`. What the 2026-09-16 investigation measured is that the
K=200 drop set is not a random 200 rows: on det_a -> ab_split it removes 9 of the 10 day-rollover rows, 17 of the
18 rows with y >= 12 h, 129 of the 537 LIRF unmatched rows and 84.5% of the squared disagreement between the two
builds (docs/investigation_2026-09-16/results/decision-rule-blind-to-tail.md §3.2). The rule stays as it is - it
is what rejected the 255-leaf model-B change the leaderboard then did not reward - but a rejection reading
"the effect is those rows" should say which rows. So a comparison also reports, for the same pair and without
touching adopt/reject: the unmatched-row difference with its CI, the same restricted to the airport that
dominates the unmatched disagreement, that difference per airport, and the named list of the rows the K=200 drop
removes with their labels and both predictions. A tail-structural change shows up as many airports or many rows
moving together; a tail lottery shows up as one airport's handful of rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .cv import paired_bootstrap_rmse_diff, rmse, year_month_key
from .train import LABEL, PROXY

DROP_LEVELS = (0, 20, 200, 2000)
SIGN_CHECK_LEVEL = 200  # an effect that reverses once this many rows are gone was those rows


def drop_order(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Row indices ordered by max(se_a, se_b), worst first - the order in which rows leave the curve.

    Shared with `dropped_rows` on purpose: the report of what K=200 removes has to name the same rows the
    verdict stops looking at, or it is describing a different experiment."""
    return np.argsort(np.maximum((y - p_a) ** 2, (y - p_b) ** 2))[::-1]


def robustness_curve(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    drop_levels: tuple[int, ...] = DROP_LEVELS,
    n_boot: int = 500,
    seed: int = 0,
) -> list[dict]:
    """RMSE(B) - RMSE(A) after removing the K rows where either model errs most. Negative = B better."""
    worst = drop_order(y, p_a, p_b)
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
    the 2024 and 2025 editions scored the final standings on a set nobody had seen (docs/README). A change that
    is real only on the rows that will not recur is a change that does not transfer. Anything rejected for that
    reason is reported as tail-only rather than as no effect, because it is a judgement call, not a null."""
    full = next((r for r in curve if r["dropped"] == 0), None)
    if full is None:
        return False, "no full-data comparison"
    if full["ci95"][0] > 0:
        return False, f"B is worse on all rows: {full['diff']:+.2f} s [{full['ci95'][0]:+.2f}, {full['ci95'][1]:+.2f}]"
    if full["ci95"][1] >= 0:
        return (
            False,
            f"no difference on all rows: {full['diff']:+.2f} s [{full['ci95'][0]:+.2f}, {full['ci95'][1]:+.2f}]",
        )
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


def group_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    mask: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """The paired difference on a subset of rows. Negative = B better. Reporting only - `verdict` never sees it.

    The verdict is a whole-fold statistic by design, and the subset that decides the leaderboard (1.5% of rows,
    concentrated at one airport) is never reported on its own. That is how a change that moves the tail as a
    structure and a change that wins the tail lottery look identical in the output."""
    if mask is None:
        mask = np.ones(len(y), bool)
    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0,
            "rmse_a": float("nan"),
            "rmse_b": float("nan"),
            "diff": float("nan"),
            "ci95": [float("nan")] * 2,
            "b_better_prob": float("nan"),
        }
    r = paired_bootstrap_rmse_diff(y[mask], p_a[mask], p_b[mask], n_boot=n_boot, seed=seed)
    return {
        "n": n,
        "rmse_a": rmse(y[mask], p_a[mask]),
        "rmse_b": rmse(y[mask], p_b[mask]),
        "diff": r["diff_rmse"],
        "ci95": r["ci95"],
        "b_better_prob": r["b_better_prob"],
    }


def per_group_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    group: np.ndarray,
    mask: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 0,
) -> list[dict]:
    """`group_diff` inside every group, ordered by that group's share of the squared disagreement.

    Ordering by share of sum (p_a - p_b)^2 rather than by row count answers the question the verdict cannot:
    is the difference between the two builds spread over the airports, or is it one airport's tail?"""
    if mask is None:
        mask = np.ones(len(y), bool)
    gap = (p_a - p_b) ** 2
    total = float(gap[mask].sum())
    out = []
    for g in np.unique(group[mask]):
        sel = mask & (group == g)
        r = group_diff(y, p_a, p_b, sel, n_boot=n_boot, seed=seed)
        r["group"] = str(g)
        r["gap_share"] = float(gap[sel].sum() / total) if total > 0 else 0.0
        out.append(r)
    return sorted(out, key=lambda r: -r["gap_share"])


def dropped_rows(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, k: int = SIGN_CHECK_LEVEL) -> np.ndarray:
    """The row indices `robustness_curve` removes at K, worst first - the rows the verdict stops looking at."""
    return drop_order(y, p_a, p_b)[: min(k, len(y))]


def tail_report(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    matched: np.ndarray,
    group: np.ndarray,
    ids: np.ndarray | None = None,
    k: int = SIGN_CHECK_LEVEL,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Everything the verdict does not look at: the unmatched rows, the airport that dominates them, and the
    named rows the K-drop removes.

    None of this changes adopt/reject. It exists so that "the effect is those rows" can be read further: which
    rows, at which airport, with which labels, and whether the unmatched difference survives being split by
    airport. A tail-structural change moves many airports or many rows; a tail lottery is one airport's handful
    of rows with labels the ranking set has no reason to repeat."""
    unmatched = ~matched
    idx = dropped_rows(y, p_a, p_b, k)
    rows = [
        {
            "rank": int(r),
            "id": None if ids is None else ids[i],
            "group": str(group[i]),
            "matched": bool(matched[i]),
            "y": float(y[i]),
            "p_a": float(p_a[i]),
            "p_b": float(p_b[i]),
        }
        for r, i in enumerate(idx)
    ]
    by_airport = per_group_diff(y, p_a, p_b, group, unmatched, n_boot=n_boot, seed=seed)
    dominant = by_airport[0]["group"] if by_airport else None
    return {
        "k": k,
        "unmatched": group_diff(y, p_a, p_b, unmatched, n_boot=n_boot, seed=seed),
        "dominant_group": dominant,
        "dominant": group_diff(y, p_a, p_b, unmatched & (group == dominant), n_boot=n_boot, seed=seed)
        if dominant is not None
        else None,
        "dominant_gap_share": by_airport[0]["gap_share"] if by_airport else float("nan"),
        "by_airport": by_airport,
        "dropped": rows,
        "n_dropped": len(rows),
        "n_dropped_unmatched": sum(1 for r in rows if not r["matched"]),
    }


def _fmt_id(v) -> str:
    """MVT_ID_mvt is Float64 in the feature frame; print it as the integer id it is."""
    try:
        return f"{int(v)}" if float(v).is_integer() else f"{v}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_diff(r: dict) -> str:
    return f"{r['diff']:+9.2f} [{r['ci95'][0]:+9.2f}, {r['ci95'][1]:+9.2f}]"


def format_tail_report(rep: dict, max_listed: int | None = None) -> list[str]:
    """Render `tail_report` for the console. The heading states that none of it enters the verdict."""
    k = rep["k"]
    unm = (
        f"  unmatched rows            n={rep['unmatched']['n']:>7,}  A={rep['unmatched']['rmse_a']:9.1f}  "
        f"B={rep['unmatched']['rmse_b']:9.1f}  {_fmt_diff(rep['unmatched'])}"
    )
    lines = [
        "",
        "tail structure (reporting only - the VERDICT below is computed by verdict() from the curve alone,",
        "and none of these lines change adopt/reject):",
        "",
        unm,
    ]
    if rep["dominant"] is not None:
        lines.append(
            f"  unmatched @ {rep['dominant_group']:<13} n={rep['dominant']['n']:>7,}  "
            f"A={rep['dominant']['rmse_a']:9.1f}  B={rep['dominant']['rmse_b']:9.1f}  {_fmt_diff(rep['dominant'])}"
            f"   {rep['dominant_gap_share']:.1%} of the unmatched disagreement"
        )
    lines += [
        "",
        "  unmatched difference by airport (ordered by share of sum (p_a - p_b)^2):",
        f"    {'airport':>8} {'n':>7} {'A':>9} {'B':>9} {'diff':>9} {'CI95':>22} {'gap share':>10}",
    ]
    for r in rep["by_airport"]:
        ci = f"[{r['ci95'][0]:+8.2f},{r['ci95'][1]:+8.2f}]"
        lines.append(
            f"    {r['group']:>8} {r['n']:>7,} {r['rmse_a']:9.1f} {r['rmse_b']:9.1f} {r['diff']:+9.2f} {ci:>22} "
            f"{r['gap_share']:9.1%}"
        )
    shown = rep["dropped"] if max_listed is None else rep["dropped"][:max_listed]
    head = (
        f"  the K={k} drop removes {rep['n_dropped']} rows ({rep['n_dropped_unmatched']} unmatched, "
        f"{rep['n_dropped'] - rep['n_dropped_unmatched']} matched) - these are the rows the K={k} CI is blind to:"
    )
    lines += [
        "",
        head,
        f"    {'rank':>4} {'MVT_ID_mvt':>14} {'airport':>8} {'grp':>9} {'y':>10} {'p_a':>10} {'p_b':>10}",
    ]
    for r in shown:
        rid = "" if r["id"] is None else _fmt_id(r["id"])
        lines.append(
            f"    {r['rank']:>4} {rid:>14} {r['group']:>8} {'matched' if r['matched'] else 'unmatched':>9} "
            f"{r['y']:10.1f} {r['p_a']:10.1f} {r['p_b']:10.1f}"
        )
    if max_listed is not None and len(rep["dropped"]) > len(shown):
        lines.append(f"    ... {len(rep['dropped']) - len(shown)} more (raise --max-listed to see them)")
    return lines


def load_pair(exp_a: str, exp_b: str, work_dir: Path, exp_dir: Path) -> pl.DataFrame:
    """Join two out-of-fold prediction files onto the labels, keeping only rows both models scored."""
    cols = ["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt", "ADEP_mvt"]  # ADEP_mvt: the per-airport tail report
    feats = pl.read_parquet(work_dir / "feat_train.parquet", columns=cols)
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
    ap.add_argument(
        "--max-listed",
        type=int,
        default=None,
        help=f"truncate the list of rows the K={SIGN_CHECK_LEVEL} drop removes (default: list all of them)",
    )
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

    rep = tail_report(
        y,
        p_a,
        p_b,
        matched,
        d["ADEP_mvt"].to_numpy(),
        ids=d["MVT_ID_mvt"].to_numpy(),
        n_boot=a.n_boot,
    )
    print("\n".join(format_tail_report(rep, max_listed=a.max_listed)))

    ok, why = verdict(curve)  # unchanged: the curve is still the only input to adopt/reject
    print(f"\nVERDICT (from the curve only, rule unchanged): {'ADOPT' if ok else 'REJECT'} - {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
