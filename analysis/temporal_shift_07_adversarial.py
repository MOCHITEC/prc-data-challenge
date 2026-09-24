"""Adversarial validation: can DEP rows of Jan/Jul 2025 be told apart from DEP rows of Jan/Jul 2026 using only
ranking-observable features? Which features carry the separation?

Output: markdown tables (docs/analysis/temporal_shift.md §1-3).
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from temporal_shift_features import BASE_CAT, BASE_NUM, DAY_FEATS, features  # noqa: E402

P = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, num_threads=1, verbose=-1, max_cat_threshold=64)


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


f = features().filter(pl.col("month").is_in([1, 7]))
f = f.with_columns(pl.col("APT").cast(pl.Utf8).alias("APT_s"),
                   *[pl.col(c).cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(c)
                     for c in BASE_CAT],
                   (pl.col("year") == 2026).cast(pl.Int8).alias("is26"))
rng = np.random.default_rng(0)


def run(cols: list[str], frame: pl.DataFrame, tag: str, top: int = 12) -> None:
    # split by day to avoid near-duplicate leakage between rows of the same day
    days = frame["date_local"].unique().to_numpy()
    te_days = set(rng.choice(days, size=len(days) // 3, replace=False).tolist())
    is_te = frame["date_local"].is_in(list(te_days)).to_numpy()
    X = frame.select(cols).to_numpy().astype(np.float32)
    y = frame["is26"].to_numpy()
    cats = [cols.index(c) for c in BASE_CAT if c in cols]
    m = lgb.train(P, lgb.Dataset(X[~is_te], y[~is_te], categorical_feature=cats), 200)
    s = m.predict(X[is_te])
    apt = frame["APT_s"].to_numpy()[is_te]
    by = {a: round(auc(y[is_te][apt == a], s[apt == a]), 3) for a in sorted(set(apt))}
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(v for _, v in imp)
    print(f"\n### {tag}: AUC = {auc(y[is_te], s):.3f}")
    print("per airport:", by)
    print("top gain share:", ", ".join(f"{c} {v / tot:.2f}" for c, v in imp[:top]))


base = BASE_NUM + BASE_CAT
for mon in (1, 7):
    fr = f.filter(pl.col("month") == mon)
    run(base, fr, f"month {mon}, base features")
    run([c for c in base if c not in ("STAND_mvt", "airline", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt")], fr,
        f"month {mon}, base without STAND/airline/actype/runway")
    run(DAY_FEATS, fr, f"month {mon}, day-level features only")
    run(["m_aobt3", "m_sched", "m_eobt1", "m_lobt", "aobt3_m_sched", "nm", "APT"], fr, f"month {mon}, clock gaps + APT only")
