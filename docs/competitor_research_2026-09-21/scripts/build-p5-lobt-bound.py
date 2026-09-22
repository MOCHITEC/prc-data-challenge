"""Build the v12 candidate: the scored v10 file, with every prediction projected onto its LOBT window.

No model is run. The candidate differs from the 284.3584 file only on the rows whose prediction lies
outside `mvt_minus_lobt +- 3606`, so any board delta is those rows and nothing else.

Derivation, the 2026 transfer argument and the enumerated cost are in
docs/competitor_research_2026-09-21/plans/P5-physics-lower-bound.md and P5-REVIEW.md.

Run:  .venv/bin/python docs/competitor_research_2026-09-21/scripts/build-p5-lobt-bound.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from prc.config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL  # noqa: E402
from prc.predict import write_submission  # noqa: E402
from prc.train import LOBT_OFFSET  # noqa: E402

SLACK = 3606.0
# experiments/submissions.jsonl: {"object_name": "resilient-hedgehog_v10.parquet",
#                                 "leaderboard_score": 284.3584, "file": "outputs/v10_probe_reparam.parquet"}
# The exp_id in that entry names the TRAINING RUN (submit_v10_reparam-77f92c802d); outputs/submit_v10_reparam
# .parquet is that run's output and was never uploaded. It differs from the scored file on 336,426 rows.
SCORED = ROOT / "outputs/v10_probe_reparam.parquet"
WRONG = ROOT / "outputs/submit_v10_reparam.parquet"
V11 = ROOT / "outputs/v11_rollover_mixture.parquet"
RANK_FEATS = ROOT / "data/work/feat_rank.parquet"
TEMPLATE = ROOT / "data/raw/submitting.parquet"
OUT = ROOT / "outputs/v12_lobt_bound.parquet"


def main() -> int:
    sub = pl.read_parquet(SCORED).rename({SUBMISSION_TARGET_COL: "pred"})
    feats = pl.read_parquet(RANK_FEATS, columns=[SUBMISSION_ID_COL, LOBT_OFFSET, "mvt_minus_aobt3", "ADEP_mvt"])

    # --- provenance: prove we are holding the file that scored 284.3584, not the training run's output ---
    d11 = sub.join(pl.read_parquet(V11).rename({SUBMISSION_TARGET_COL: "v11"}), on=SUBMISSION_ID_COL)
    n11 = int(((d11["pred"] - d11["v11"]).abs() > 1e-9).sum())
    dw = sub.join(pl.read_parquet(WRONG).rename({SUBMISSION_TARGET_COL: "other"}), on=SUBMISSION_ID_COL)
    nw = int(((dw["pred"] - dw["other"]).abs() > 1e-9).sum())
    assert n11 == 11, f"input differs from v11 on {n11} rows, expected 11 - this is not the scored file"
    assert nw == 336426, f"input differs from the training run's output on {nw} rows, expected 336426"

    df = sub.join(feats, on=SUBMISSION_ID_COL, how="left")
    assert df.height == sub.height == 344841, df.height

    m = df[LOBT_OFFSET].to_numpy().astype(float)
    p = df["pred"].to_numpy().astype(float)
    have = ~np.isnan(m)
    assert np.array_equal(have, ~np.isnan(df["mvt_minus_aobt3"].to_numpy().astype(float))), (
        "LOBT and AOBT nullness disagree; the bound would no longer be confined to model A's rows"
    )
    lo = np.where(have, np.maximum(m - SLACK, 0.0), 0.0)
    hi = np.where(have, m + SLACK, np.inf)
    assert not (lo > hi).any(), "inverted window"
    new = np.where(have, np.clip(p, lo, hi), np.maximum(p, 0.0))

    moved = np.abs(new - p) > 1e-9
    mv = (new - p)[moved]
    sse = float((mv**2).sum())
    n, rmse = 344841, 284.3584
    guaranteed = rmse - float(np.sqrt(rmse * rmse - sse / n))
    worst = float(np.sqrt((rmse * rmse * n + (new[moved] ** 2 - p[moved] ** 2).sum()) / n)) - rmse
    share = (mv**2) / sse

    # --- the label-free gate: every number here is checkable before the board answers ---
    checks = {
        "rows_total": (int(len(new)), 344841),
        "rows_moved": (int(moved.sum()), 34),
        "moved_up": (int((mv > 0).sum()), 30),
        "moved_down": (int((mv < 0).sum()), 4),
        "unmatched_touched": (int(moved[~have].sum()), 0),
        "negative_predictions": (int((new < 0).sum()), 0),
        "non_finite": (int((~np.isfinite(new)).sum()), 0),
        "outside_own_window": (int(((new < lo - 1e-9) | (new > hi + 1e-9)).sum()), 0),
    }
    print(f"{'check':<24} {'value':>10} {'expected':>10}")
    bad = []
    for k, (got, want) in checks.items():
        flag = "" if got == want else "  <-- MISMATCH"
        if got != want:
            bad.append(k)
        print(f"{k:<24} {got:>10} {want:>10}{flag}")
    print(f"\nmax |move|        {np.abs(mv).max():>10.1f}  (expected 4295.0)")
    print(f"median |move|     {np.median(np.abs(mv)):>10.1f}")
    print(f"sum(move^2)       {sse:>10.4e}")
    print(f"guaranteed gain   {guaranteed:>+10.4f} s")
    print(f"worst case (y=0)  {worst:>+10.4f} s")
    print(f"top-1 share       {share.max() * 100:>10.1f} %")
    idx = np.where(moved)[0]
    print("\nby airport:")
    print(df[[int(i) for i in idx]].group_by("ADEP_mvt").len().sort("len", descending=True))
    if bad:
        print(f"\nGATE FAILED on {bad}; nothing written.")
        return 1

    pred = pl.DataFrame({SUBMISSION_ID_COL: df[SUBMISSION_ID_COL], "pred": new})
    write_submission(pred, TEMPLATE, OUT)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
