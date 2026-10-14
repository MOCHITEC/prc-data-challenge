"""Validation utilities: month-blocked folds, metrics, paired bootstrap (docs/README)."""

from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np
import polars as pl

CLEAN_LO, CLEAN_HI = 30.0, 7200.0


@dataclass(frozen=True)
class Fold:
    name: str
    valid_months: tuple[int, ...]  # year*100+month keys held out
    train_months: tuple[int, ...]


def year_month_key(df: pl.DataFrame, col: str = "MVT_TIME_UTC_mvt") -> pl.Series:
    """Fold key including the year: a month number alone would mix calendar months across years and let a fold
    train on data that comes after its validation period."""
    return (df[col].dt.year() * 100 + df[col].dt.month()).alias("ym")


def month_block_folds(months_present: list[int], named: tuple[str, ...] | list[str] = ()) -> list[Fold]:
    """Folds over year*100+month keys.

    January and July are held out separately because they match the test months' season; the trailing months
    (>= November) form a third fold that measures forward generalisation.

    `named` is the config's `folds` list. The opt-in folds `aso` and `fmj` are built only when it names them, so
    a config without `folds:` gets the same default fold list as before they existed (integration 2026-10-14)."""
    ms = sorted(set(months_present))
    if len(ms) < 2:
        raise ValueError(f"need at least two distinct months to cross-validate, got {ms}")
    folds = []
    for month, name in ((1, "jan"), (7, "jul")):
        sel = tuple(x for x in ms if x % 100 == month)
        if sel and len(sel) < len(ms):
            folds.append(Fold(name, sel, tuple(x for x in ms if x not in sel)))
            # The same month validated the honest way: trained only on what came before it. jan has nothing
            # before it in a single year, so only jul gets a forward twin. Keeping both lets the cost of the
            # leak be measured rather than assumed - the seasonal difference between months is larger than the
            # leak, so comparing jan against forward says nothing about either.
            before = tuple(x for x in ms if x < min(sel))
            if before:
                folds.append(Fold(f"{name}_fwd", sel, before))
    # Final phase (docs/final_phase_2026-10-13): the final set adds Feb and Jun 2026. PRC_FEBJUN_FOLDS=1 adds
    # leave-one-month-out folds for Feb and Jun 2025. Off by default, so every existing fold list is unchanged.
    if os.environ.get("PRC_FEBJUN_FOLDS") == "1":
        for month, name in ((2, "feb"), (6, "jun")):
            sel = tuple(x for x in ms if x % 100 == month)
            if sel and len(sel) < len(ms):
                folds.append(Fold(name, sel, tuple(x for x in ms if x not in sel)))
    # local (finding 2 extension): Aug-Oct held out together, trained on the other nine months; only built when a
    # config names it in `folds`, so production fold lists are unchanged.
    wanted = set(named or ())
    aso = tuple(x for x in ms if x % 100 in (8, 9, 10))
    if "aso" in wanted and aso and len(aso) < len(ms):
        folds.append(Fold("aso", aso, tuple(x for x in ms if x not in aso)))
    # local (finding 2c): Feb-Jun held out together, trained on the other seven months; only built when named.
    fmj = tuple(x for x in ms if x % 100 in (2, 3, 4, 5, 6))
    if "fmj" in wanted and fmj and len(fmj) < len(ms):
        folds.append(Fold("fmj", fmj, tuple(x for x in ms if x not in fmj)))
    fwd = tuple(x for x in ms if x % 100 >= 11)
    if fwd and len(fwd) < len(ms):
        folds.append(Fold("forward", fwd, tuple(x for x in ms if x not in fwd)))
    if not folds:  # tiny sets: hold out the last month
        folds.append(Fold(f"lomo_{ms[-1]}", (ms[-1],), tuple(ms[:-1])))
    return folds


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    finite = np.isfinite(y) & np.isfinite(p)
    n_dropped = int((~finite).sum())
    y, p = y[finite], p[finite]
    if len(y) == 0:
        return {
            "rmse_full": float("nan"),
            "rmse_clean": float("nan"),
            "n": 0,
            "n_dropped": n_dropped,
            "n_gt_7200": 0,
            "top20_share_of_sse": 0.0,
            "bias": float("nan"),
        }
    clean = (y >= CLEAN_LO) & (y <= CLEAN_HI)
    se = (y - p) ** 2
    order = np.argsort(-se)
    top20 = float(se[order[:20]].sum() / se.sum()) if se.sum() > 0 else 0.0
    return {
        "rmse_full": rmse(y, p),
        "rmse_clean": rmse(y[clean], p[clean]) if clean.any() else float("nan"),
        "n": len(y),
        "n_dropped": n_dropped,
        "n_gt_7200": int((y > CLEAN_HI).sum()),
        "top20_share_of_sse": top20,
        "bias": float(np.mean(p - y)),
    }


def paired_bootstrap_rmse_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict:
    """RMSE(B) - RMSE(A) with a bootstrap CI over rows (paired). Negative = B better."""
    rng = np.random.default_rng(seed)
    finite = np.isfinite(y) & np.isfinite(p_a) & np.isfinite(p_b)
    if not finite.all():
        y, p_a, p_b = y[finite], p_a[finite], p_b[finite]
    if len(y) == 0:
        return {
            "diff_rmse": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "b_better_prob": float("nan"),
            "n": 0,
            "n_dropped": int((~finite).sum()),
        }
    se_a, se_b = (y - p_a) ** 2, (y - p_b) ** 2
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(se_b[idx].mean()) - np.sqrt(se_a[idx].mean())
    point = float(np.sqrt(se_b.mean()) - np.sqrt(se_a.mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff_rmse": point,
        "ci95": [float(lo), float(hi)],
        "b_better_prob": float((diffs < 0).mean()),
        "n": int(n),
        "n_dropped": int((~finite).sum()),
    }


def per_airport(y: np.ndarray, p: np.ndarray, airport: np.ndarray) -> dict[str, float]:
    out = {}
    for a in np.unique(airport):
        m = airport == a
        out[str(a)] = rmse(y[m], p[m])
    return out
