"""P1 stage 1: sweep an upper cap on the NON-LIRF unmatched predictions, on the stage-0 OOF.

Read-only. Nothing is written except this script's own report on stdout.

The rule under test is `pred := min(pred, c)` applied only to rows the unmatched model served, at
airports other than LIRF. The decision the plan pre-registered:

  * SSE delta negative in EVERY fold AND at EVERY affected airport
  * a month-block CI excluding zero
  * helped/hurt no worse than 3:1

Any one of those failing means record the cap CLOSED with the reason, so it is not reopened a third
time. `tail-postprocessing-cap-shrink.md` already recorded 147 rows hurt at 1,800 on the pre-reparam
model, so the tight end is expected to be a distribution shift rather than a guard.

Run: .venv/bin/python docs/competitor_research_2026-09-21/scripts/p1_stage1_cap_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from prc.train import LABEL, PROXY  # noqa: E402

EXP = "main_reparam-bc60017878"
CAPS = [1500, 1800, 2100, 2400, 3000, 3600]
RNG = np.random.default_rng(0)
N_BOOT = 2000


def main() -> int:
    oof = pl.read_parquet(ROOT / f"experiments/oof_{EXP}.parquet")
    side = pl.read_parquet(ROOT / f"experiments/oof_{EXP}_folds.parquet")
    feats = pl.read_parquet(
        ROOT / "data/work/feat_train.parquet",
        columns=["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt", "MVT_TIME_UTC_mvt"],
    )
    # `overall` keeps one prediction per row; the sidecar says which fold supplied it. Join the fold
    # of the FIRST covering entry, which is the one coalesce kept.
    first = side.group_by("MVT_ID_mvt").agg(pl.col("fold").first())
    df = (
        feats.join(oof, on="MVT_ID_mvt", how="inner")
        .join(first, on="MVT_ID_mvt", how="left")
        .drop_nulls(["oof", LABEL])
    )
    print(f"stage-0 OOF: {df.height:,} rows, folds {sorted(set(df['fold'].drop_nulls().to_list()))}")

    gate = (df[PROXY].is_null() & (df["ADEP_mvt"] != "LIRF")).to_numpy()
    y = df[LABEL].to_numpy().astype(float)
    p = df["oof"].to_numpy().astype(float)
    apt = df["ADEP_mvt"].to_numpy()
    fold = df["fold"].to_numpy()
    day = df["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()
    dayk = df["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy()
    days = np.unique(dayk)
    print(f"gate (unmatched, non-LIRF): {int(gate.sum()):,} rows; label max {y[gate].max():,.0f} s")
    print(f"  predictions above: " + "  ".join(f"{c}: {int((p[gate] > c).sum())}" for c in CAPS))
    print()

    base_sse = float(((p - y) ** 2).sum())
    n = len(y)
    rmse0 = float(np.sqrt(base_sse / n))
    print(f"baseline pooled OOF RMSE {rmse0:.4f} over {n:,} rows\n")

    hdr = f"{'cap':>6} {'moved':>6} {'help':>5} {'hurt':>5} {'ratio':>6} {'dSSE':>12} {'dRMSE':>8} {'folds+':>7} {'apts+':>6} {'CI95 month':>24} {'CI95 day':>24}"
    print(hdr)
    print("-" * len(hdr))
    for c in CAPS:
        q = p.copy()
        q[gate] = np.minimum(q[gate], c)
        moved = np.abs(q - p) > 1e-9
        d = (q - y) ** 2 - (p - y) ** 2
        helped, hurt = int((d[moved] < 0).sum()), int((d[moved] > 0).sum())
        dsse = float(d.sum())
        drmse = float(np.sqrt((base_sse + dsse) / n)) - rmse0
        bad_folds = [f for f in np.unique(fold[moved]) if d[moved & (fold == f)].sum() > 0]
        bad_apts = [a for a in np.unique(apt[moved]) if d[moved & (apt == a)].sum() > 0]
        # The pre-registered arm is a month block, but the OOF spans only 4 months, so a 4-unit
        # resample is very coarse. Report a day block beside it: merry-quicksand measured row
        # bootstraps 1.5-4.6x too narrow on tail-moving arms and moved to date blocks.
        months = np.unique(day)
        per = np.array([d[day == m].sum() for m in months])
        draws = per[RNG.integers(0, len(months), (N_BOOT, len(months)))].sum(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        dperd = np.array([d[dayk == k].sum() for k in days])
        dd = dperd[RNG.integers(0, len(days), (N_BOOT, len(days)))].sum(axis=1)
        dlo, dhi = np.percentile(dd, [2.5, 97.5])
        ratio = (helped / hurt) if hurt else float("inf")
        print(
            f"{c:>6} {int(moved.sum()):>6} {helped:>5} {hurt:>5} {ratio:>6.1f} {dsse:>12.3e} {drmse:>+8.4f}"
            f" {len(bad_folds):>7} {len(bad_apts):>6}  [{lo:>+9.2e},{hi:>+9.2e}] [{dlo:>+9.2e},{dhi:>+9.2e}]"
        )
        if bad_folds or bad_apts:
            print(f"        positive-SSE folds {bad_folds}   airports {bad_apts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
