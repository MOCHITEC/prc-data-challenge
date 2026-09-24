"""How much of the base arm's leftover error each candidate column can explain on its own.

Read-only: reads data/work/feat_train.parquet and the base arm's OOF, writes nothing outside stdout.

    .venv/bin/python docs/article_fe_2026-09-24/residual_diagnostic.py <base_exp_id> [--meta]

For each candidate column and each lane, on the forward fold's validation rows:

* Spearman correlation with the residual r = y - oof (the part the production model has not explained).
* A held-out binned correction: rows are split by day-of-month parity; the mean residual per 20-quantile bin of
  the column is fitted on one half and applied to the other (both directions, averaged). The reported number is
  the RMSE change that correction buys - an honest, one-column, additive upper-bound-style reading, the same
  construction as model-a-ceiling.md S6 but for one continuous column. Negative = the column carries residual
  signal the model left behind. Reported on all rows and with the 200 largest |r| rows removed, since 10 rows
  can carry a quarter of the squared error here.
* The same for a hashed-id noise column, which prices the construction itself (a pure-noise column cannot
  help out of sample, so anything it "buys" is the bin-mean estimator's own variance).

This does not replace the three-arm gate: a column the tree can use in interaction with others can matter
without an additive one-column signal, and vice versa. It is the second, model-free line of evidence."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from prc.cv import month_block_folds, year_month_key
from prc.train import (
    LABEL,
    PROXY,
    TrainConfig,
    aggdiff_features,
    attach_encodings,
    cyclic_features,
    fit_encodings,
    frequency_features,
    meta_prior_features,
    noise_features,
    visible_numeric_columns,
)

CANDIDATES = {
    "F": ["freq_stand", "freq_stand_rwy", "freq_operator", "freq_actype", "freq_ades", "freq_rwy_hour"],
    "D": [
        "agg_proxy_minus_ref_med",
        "agg_proxy_minus_ref_p10",
        "agg_proxy_minus_arh_med",
        "agg_proxy_cell_dev",
        "agg_sched_minus_ref_med",
        "agg_ref_iqr",
    ],
    "C": ["cyc_sin", "cyc_cos"],
    "noise": ["noise_1"],
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return float("nan")
    ra = pl.Series(a[ok]).rank().to_numpy()
    rb = pl.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def heldout_bin_gain(x: np.ndarray, r: np.ndarray, half: np.ndarray, bins: int = 20) -> float:
    """RMSE of r after a bin-mean correction fitted on the other half, minus RMSE of r. Missing x is its own bin."""
    corrected = r.copy()
    for fit_side in (True, False):
        fit, app = half == fit_side, half != fit_side
        xf = x[fit & np.isfinite(x)]
        if xf.size < 200:
            return float("nan")
        edges = np.unique(np.quantile(xf, np.linspace(0, 1, bins + 1)[1:-1]))

        def code(v: np.ndarray) -> np.ndarray:
            c = np.searchsorted(edges, v, side="right")
            return np.where(np.isfinite(v), c, -1)

        cf, ca = code(x[fit]), code(x[app])
        means = {k: float(r[fit][cf == k].mean()) for k in np.unique(cf)}
        corrected[app] = r[app] - np.array([means.get(k, 0.0) for k in ca])
    return float(np.sqrt(np.mean(corrected**2)) - np.sqrt(np.mean(r**2)))


def main(argv: list[str]) -> int:
    base_id = argv[0]
    want_meta = "--meta" in argv
    feats = pl.read_parquet("data/work/feat_train.parquet")
    months = year_month_key(feats)
    fold = next(f for f in month_block_folds(months.unique().to_list()) if f.name == "forward")
    cfg = TrainConfig(reference_min_count=30)
    tr = feats.filter(months.is_in(list(fold.train_months)))
    tr = tr.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    va = feats.filter(months.is_in(list(fold.valid_months)))
    enc = fit_encodings(tr, cfg)
    tr, va = attach_encodings(tr, enc), attach_encodings(va, enc)
    tr, va = frequency_features(tr, va)
    tr, va = aggdiff_features(tr, va, cfg.reference_min_count)
    tr, va = cyclic_features(tr), cyclic_features(va)
    tr, va = noise_features(tr, 1), noise_features(va, 1)
    if want_meta:
        num = visible_numeric_columns(tr, cfg)
        cats = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt"]
        cats = [c for c in cats if c in tr.columns]
        tr, va = meta_prior_features(tr, va, cfg, num, cats, {})
        CANDIDATES["M"] = ["meta_prior", "meta_prior_minus_proxy"]
    del tr
    oof = pl.read_parquet(f"experiments/oof_{base_id}.parquet").drop_nulls("oof")
    va = va.join(oof, on="MVT_ID_mvt", how="inner").drop_nulls([LABEL])
    y = va[LABEL].to_numpy().astype(float)
    r = y - va["oof"].to_numpy().astype(float)
    half = (va["MVT_TIME_UTC_mvt"].dt.day().to_numpy() % 2) == 0
    lanes = {"A (matched)": va[PROXY].is_not_null().to_numpy(), "B (unmatched)": va[PROXY].is_null().to_numpy()}
    print(f"base {base_id}: forward validation rows {va.height:,}, rmse {np.sqrt(np.mean(r**2)):.3f}")
    for lane, m in lanes.items():
        rr = r[m]
        keep = np.ones(m.sum(), dtype=bool)
        keep[np.argsort(-np.abs(rr))[:200]] = False
        print(
            f"\n## lane {lane}: n={m.sum():,}  rmse {np.sqrt(np.mean(rr**2)):.3f}  "
            f"(200 largest |r| removed: {np.sqrt(np.mean(rr[keep] ** 2)):.3f})"
        )
        print("| family | column | non-null | spearman(x, r) | held-out bin gain, all rows | ... 200 removed |")
        print("|---|---|---:|---:|---:|---:|")
        for fam, cols in CANDIDATES.items():
            for c in cols:
                x = va[c].cast(pl.Float64).to_numpy()[m]
                nn = float(np.isfinite(x).mean())
                if nn == 0:
                    print(f"| {fam} | `{c}` | 0 % | — | — | — |")
                    continue
                g_all = heldout_bin_gain(x, rr, half[m])
                g_rob = heldout_bin_gain(x[keep], rr[keep], half[m][keep])
                print(f"| {fam} | `{c}` | {nn:.1%} | {spearman(x, rr):+.4f} | {g_all:+.3f} | {g_rob:+.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
