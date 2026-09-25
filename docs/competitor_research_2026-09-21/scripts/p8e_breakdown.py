"""P8-E: the three-arm comparison for the dead-column deletion.

Every pairing that matters, each on the full `prc.compare` curve plus the cuts that say *where* a difference
sits: per fold, per lane, and on the rows the deleted columns would have been live on (there are none - that
is the point, and it is asserted rather than assumed).

Usage:
    .venv/bin/python docs/competitor_research_2026-09-21/scripts/p8e_breakdown.py

Reads `experiments/oof_<exp_id>.parquet` and the `_folds.parquet` sidecars; writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from prc.compare import (
    group_diff,
    load_pair,
    month_bootstrap_rmse_diff,
    robustness_curve,
    verdict,
    year_month_key,
)
from prc.train import LABEL, PROXY

EXP = ROOT / "experiments"
WORK = ROOT / "data" / "work"
N_BOOT = 500

ARMS = {
    "base": "p8e_base-798baa7a90",
    "drop": "p8e_drop-30925f6d13",
    "ctrl": "p8e_ctrl-d7e31c32c6",
}
LABELS = {
    "base": "base (90 model-visible numeric)",
    "drop": "drop the 5 dead columns (85)",
    "ctrl": "drop 5 random LIVE columns (85)",
}
# The pairings, in the order the write-up reads them.
PAIRS = [("base", "drop"), ("base", "ctrl"), ("ctrl", "drop")]


def fold_of(exp_id: str) -> pl.DataFrame:
    return pl.read_parquet(EXP / f"oof_{exp_id}_folds.parquet").filter(pl.col("pooled"))


def line(tag: str, r: dict) -> str:
    return (
        f"  {tag:<42s} n={r['n']:>7,} A={r['rmse_a']:9.2f} B={r['rmse_b']:9.2f} "
        f"{r['diff']:+8.2f} [{r['ci95'][0]:+8.2f}, {r['ci95'][1]:+8.2f}]  P(B)={r['b_better_prob']:.3f}"
    )


def main() -> int:
    for a_key, b_key in PAIRS:
        a_id, b_id = ARMS[a_key], ARMS[b_key]
        d = load_pair(a_id, b_id, WORK, EXP)
        # which fold produced each row; both arms ran the same fold list, so either sidecar serves
        d = d.join(fold_of(a_id).select(["MVT_ID_mvt", "fold"]), on="MVT_ID_mvt", how="left")
        y = d[LABEL].to_numpy().astype(float)
        p_a, p_b = d["p_a"].to_numpy(), d["p_b"].to_numpy()

        print("\n" + "=" * 110)
        print(f"A = {LABELS[a_key]}  ({a_id})")
        print(f"B = {LABELS[b_key]}  ({b_id})   rows = {len(y):,}")
        curve = robustness_curve(y, p_a, p_b, n_boot=N_BOOT)
        for r in curve:
            print(
                f"  K={r['dropped']:<5d} A={r['rmse_a']:9.2f} B={r['rmse_b']:9.2f} "
                f"{r['diff']:+8.2f} [{r['ci95'][0]:+7.2f}, {r['ci95'][1]:+7.2f}]  P(B)={r['b_better_prob']:.3f}"
            )
        ok, why = verdict(curve)
        print(f"  VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")

        mb = month_bootstrap_rmse_diff(y, p_a, p_b, year_month_key(d).to_numpy(), n_boot=N_BOOT)
        print(
            f"  month bootstrap ({mb['n_months']} months): {mb['diff_rmse']:+.2f} s "
            f"[{mb['ci95'][0]:+.2f}, {mb['ci95'][1]:+.2f}]  P(B better) = {mb['b_better_prob']:.3f}"
        )

        fold = d["fold"].to_numpy()
        matched = d[PROXY].is_not_null().to_numpy()
        print("  per fold:")
        for f in ("forward", "jul_fwd"):
            print(line(f"fold {f}", group_diff(y, p_a, p_b, fold == f, n_boot=N_BOOT)))
        print("  per fold x lane:")
        for f in ("forward", "jul_fwd"):
            for lane, sel in (("matched", matched), ("unmatched", ~matched)):
                print(line(f"fold {f} {lane}", group_diff(y, p_a, p_b, (fold == f) & sel, n_boot=N_BOOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
