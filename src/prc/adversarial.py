"""Adversarial validation: can a classifier tell the training rows from the ranking rows?

    python -m prc.adversarial

Why: the local CV and the leaderboard have pointed in opposite directions (docs/README). The standard first
question when that happens is whether the two sets are drawn from the same distribution at all. Label the
training rows 0 and the ranking rows 1, throw away the real target, and try to predict the label. An AUC near
0.5 means the sets are interchangeable and CV can be trusted; an AUC near 1.0 means they are not, and the
model's own feature importances say which columns moved.

Here the two sets are a year apart by construction - training is 2025, the ranking set is January and July
2026 - so any column that encodes the date separates them perfectly and tells us nothing. Those columns are
excluded by default and the interesting question is what remains: a high AUC on non-temporal features is
covariate shift that CV is not reproducing.

The per-row probability is also useful on its own: training rows the classifier believes are ranking rows are
the ones that resemble 2026, and a validation set built from them should be closer to what actually gets scored.

What this actually found (2026-09-14), including the part that did not work:
* AUC 0.807 with the calendar columns removed, so the two sets really are far apart. Restricting the training
  side to January and July 2025 - the ranking set's own months - moves it to 0.819, not down, so the shift is
  year over year and not seasonal.
* It is a shift in proportions, not in vocabulary: only 0.18% of ranking rows carry a category absent from
  training. The heaviest columns are STAND_mvt (0.28 of the gain) and ADES_mvt (0.18).
* Only 5.9% of training rows score above the ranking set's median.
* And it does not explain the CV/leaderboard divergence. Scoring the two candidate models on the most
  ranking-like 1%, 5% or 10% of training rows, or weighting every row by this probability, ranks them the same
  way plain CV does - and the leaderboard disagrees with that ranking. So covariate shift is real here, and
  correcting for it is not the fix. Kept as a diagnostic, not as a validation strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .features import CATEGORICALS, numeric_feature_columns
from .train import LABEL

# Columns that separate 2025 from 2026 by definition rather than by any change in the world.
CALENDAR_COLS = ("month", "dow")


def build_design(
    train: pl.DataFrame, rank: pl.DataFrame, drop_calendar: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack both frames into one design matrix with the origin as the label (0 = training, 1 = ranking)."""
    num = [c for c in numeric_feature_columns(rank) if c != LABEL]
    if drop_calendar:
        num = [c for c in num if c not in CALENDAR_COLS]
    cat = [c for c in CATEGORICALS if c in rank.columns and c in train.columns]
    cols = num + cat
    both = pl.concat([train.select(cols), rank.select(cols)], how="vertical_relaxed")
    for c in cat:
        both = both.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Float32))
    X = both.select(cols).to_numpy().astype(np.float32)
    y = np.concatenate([np.zeros(train.height), np.ones(rank.height)])
    return X, y, cols


def run(X: np.ndarray, y: np.ndarray, cols: list[str], n_cat: int, seed: int = 0) -> tuple[float, lgb.Booster]:
    """Fit on a random half and score the other half. A random split is right here: the question is whether the
    two origins differ, not whether the future can be predicted."""
    rng = np.random.default_rng(seed)
    hold = rng.random(len(y)) < 0.5
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.1,
        "num_leaves": 63,
        "verbose": -1,
        "num_threads": 4,
        "deterministic": True,
        "seed": seed,
    }
    cat_idx = list(range(len(cols) - n_cat, len(cols)))
    dtr = lgb.Dataset(X[~hold], label=y[~hold], categorical_feature=cat_idx, params=params)
    dva = lgb.Dataset(X[hold], label=y[hold], categorical_feature=cat_idx, reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=300,
        valid_sets=[dva],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)],
    )
    return float(booster.best_score["valid_0"]["auc"]), booster


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--keep-calendar", action="store_true", help="leave month/dow in, to show the trivial case")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)

    train = pl.read_parquet(a.work_dir / "feat_train.parquet")
    rank = pl.read_parquet(a.work_dir / "feat_rank.parquet")
    print(f"training rows {train.height:,}   ranking rows {rank.height:,}")

    X, y, cols = build_design(train, rank, drop_calendar=not a.keep_calendar)
    n_cat = sum(1 for c in CATEGORICALS if c in rank.columns and c in train.columns)
    auc, booster = run(X, y, cols, n_cat)
    print(f"\nAUC = {auc:.4f}   (0.5 = indistinguishable, 1.0 = trivially separable)")
    if auc < 0.6:
        print("  the two sets look interchangeable on these features")
    elif auc < 0.8:
        print("  a real shift, but the sets still overlap")
    else:
        print("  the sets are close to separable: CV over the training rows is measuring a different population")

    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)[: a.top]
    print(f"\ntop {a.top} columns driving the separation (share of total gain):")
    for i in order:
        print(f"  {cols[i]:32s} {gain[i] / gain.sum():6.3f}")

    # Where each side sits on the scale the classifier learned. A training row with a high score is one the
    # classifier reads as 2026, and those are the rows a ranking-like validation set should be drawn from.
    p = booster.predict(X)
    p_tr, p_rk = p[: train.height], p[train.height :]
    print("\nP(row is a ranking row):")
    for name, v in (("training", p_tr), ("ranking", p_rk)):
        q = np.quantile(v, [0.1, 0.5, 0.9])
        print(f"  {name:9s} p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f}")
    share = float((p_tr > np.median(p_rk)).mean())
    print(f"\n  {share * 100:.1f}% of training rows score above the ranking set's median")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
