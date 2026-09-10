"""Validation utilities: month-blocked folds, metrics, paired bootstrap (docs/04 §5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

CLEAN_LO, CLEAN_HI = 30.0, 7200.0


@dataclass(frozen=True)
class Fold:
    name: str
    valid_months: tuple[int, ...]  # months (1-12) of 2025 held out
    train_months: tuple[int, ...]


def month_block_folds(months_present: list[int], scheme: str = "fast") -> list[Fold]:
    """'fast' = Jan / Jul / forward(Nov+Dec) when available; 'full' = leave-one-month-out over all months present."""
    ms = sorted(set(months_present))
    if scheme == "full":
        return [Fold(f"lomo_{m:02d}", (m,), tuple(x for x in ms if x != m)) for m in ms]
    folds = []
    for m, nm in ((1, "jan"), (7, "jul")):
        if m in ms and len(ms) > 1:
            folds.append(Fold(nm, (m,), tuple(x for x in ms if x != m)))
    fwd = tuple(x for x in ms if x >= 11)
    if fwd and len(fwd) < len(ms):
        folds.append(Fold("forward", fwd, tuple(x for x in ms if x not in fwd)))
    if not folds:  # tiny synthetic sets
        folds.append(Fold(f"lomo_{ms[-1]:02d}", (ms[-1],), tuple(ms[:-1])))
    return folds


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    clean = (y >= CLEAN_LO) & (y <= CLEAN_HI)
    se = (y - p) ** 2
    order = np.argsort(-se)
    top20 = float(se[order[:20]].sum() / se.sum()) if se.sum() > 0 else 0.0
    return {
        "rmse_full": rmse(y, p),
        "rmse_clean": rmse(y[clean], p[clean]) if clean.any() else float("nan"),
        "n": len(y),
        "n_gt_7200": int((y > CLEAN_HI).sum()),
        "top20_share_of_sse": top20,
        "bias": float(np.mean(p - y)),
    }


def paired_bootstrap_rmse_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    """RMSE(B) - RMSE(A) with a bootstrap CI over rows (paired). Negative = B better."""
    rng = np.random.default_rng(seed)
    se_a, se_b = (y - p_a) ** 2, (y - p_b) ** 2
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(se_b[idx].mean()) - np.sqrt(se_a[idx].mean())
    point = float(np.sqrt(se_b.mean()) - np.sqrt(se_a.mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_rmse": point, "ci95": [float(lo), float(hi)], "b_better_prob": float((diffs < 0).mean())}


def per_airport(y: np.ndarray, p: np.ndarray, airport: np.ndarray) -> dict[str, float]:
    out = {}
    for a in np.unique(airport):
        m = airport == a
        out[str(a)] = rmse(y[m], p[m])
    return out


def months_of(df: pl.DataFrame, col: str = "MVT_TIME_UTC_mvt") -> pl.Series:
    return df[col].dt.month()
