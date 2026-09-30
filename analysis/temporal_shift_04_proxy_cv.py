"""Proxy-model experiments for the temporal_shift analysis (see temporal_shift_features.py for what the proxy is).

Usage:
  .venv/bin/python analysis/temporal_shift_04_proxy_cv.py base        # jan/jul folds, base features, OOF saved
  .venv/bin/python analysis/temporal_shift_04_proxy_cv.py day         # + day_/cw_ features
  .venv/bin/python analysis/temporal_shift_04_proxy_cv.py weight      # neighbour-month weighting grid
  .venv/bin/python analysis/temporal_shift_04_proxy_cv.py decay       # single-month models: RMSE vs month distance
Results are appended to data/interim/temporal_shift/results.jsonl and printed.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from temporal_shift_common import CACHE  # noqa: E402
from temporal_shift_features import BASE_CAT, BASE_NUM, DAY_FEATS, features  # noqa: E402

RES = CACHE / "results_clean.jsonl"
PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=255, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, max_bin=255,
              cat_smooth=10, max_cat_threshold=64)
ROUNDS = 400


def prep() -> pl.DataFrame:
    f = features().filter(pl.col("set") == "train")
    return f.with_columns([pl.col(c).cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(c)
                           for c in BASE_CAT])


def fit_predict(tr: pl.DataFrame, va: pl.DataFrame, cols: list[str], w: np.ndarray | None = None,
                rounds: int = ROUNDS) -> np.ndarray:
    # Train on the clean band only (30 s <= y <= 2 h). The production model routes the copied-clock tail through
    # the clock mixture (docs/06); temporal effects act on the regression part, and an L2 fit on the raw tail makes
    # the proxy's RMSE a function of a few dozen LIRF rows (measured: jul full 556 vs clean 342).
    keep = tr["y"].is_between(30, 7200).to_numpy()
    tr = tr.filter(pl.Series(keep))
    if w is not None:
        w = w[keep]
    X = tr.select(cols).to_numpy().astype(np.float32)
    ds = lgb.Dataset(X, tr["y"].to_numpy(), weight=w, categorical_feature=[cols.index(c) for c in BASE_CAT],
                     free_raw_data=True)
    m = lgb.train(PARAMS, ds, rounds)
    return m.predict(va.select(cols).to_numpy().astype(np.float32))


def rmse(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(np.sqrt(np.mean((y - p) ** 2)))


def boot_diff(y, pa, pb, n=300, seed=0):
    rng = np.random.default_rng(seed)
    sa, sb = (y - pa) ** 2, (y - pb) ** 2
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ds.append(np.sqrt(sb[i].mean()) - np.sqrt(sa[i].mean()))
    return float(np.sqrt(sb.mean()) - np.sqrt(sa.mean())), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def summarize(tag: str, va: pl.DataFrame, p: np.ndarray, extra: dict | None = None) -> dict:
    y = va["y"].to_numpy()
    clean = (y >= 30) & (y <= 7200)
    apt = va["APT_s"].to_numpy()
    r = {"tag": tag, "n": len(y), "rmse": rmse(y, p), "rmse_clean": rmse(y, p, clean),
         "bias_clean": float(np.mean(p[clean] - y[clean])),
         "by_apt": {a: round(rmse(y, p, apt == a), 1) for a in sorted(set(apt))},
         "bias_by_apt_clean": {a: round(float(np.mean((p - y)[(apt == a) & clean])), 1) for a in sorted(set(apt))}}
    if extra:
        r.update(extra)
    with open(RES, "a") as fh:
        fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    return r


def folds(f: pl.DataFrame):
    for m in (1, 7):
        yield m, f.filter(pl.col("month") != m), f.filter(pl.col("month") == m)


def main(mode: str) -> None:
    t0 = time.time()
    f = prep()
    f = f.with_columns(features().filter(pl.col("set") == "train")["APT"].cast(pl.Utf8).alias("APT_s"))
    base = BASE_NUM + BASE_CAT
    # day2: only features without a year signature (adversarial AUC, script 07): airport-normalised ratios, the
    # centred window and the disruption markers; no absolute day volume or day mean level.
    norm = f.group_by("APT").agg(pl.col("m_aobt3").filter(pl.col("m_aobt3").is_between(0, 7200)).median().alias("_n"))
    f = f.join(norm, on="APT", how="left").with_columns(
        (pl.col("cw_nbr_m_aobt3_60m") / pl.col("_n")).alias("cw_ratio"),
        (pl.col("bw_nbr_m_aobt3_60m") / pl.col("_n")).alias("bw_ratio"),
        (pl.col("day_m_aobt3_mean") / pl.col("_n")).alias("day_ratio"))
    DAY2 = ["cw_nbr_m_aobt3_60m", "cw_arr_taxi_in_60m", "cw_ratio", "bw_ratio", "day_ratio", "day_msched_gt3h",
            "day_nm_unmatched"]
    if mode in ("base", "day", "day2"):
        cols = base + {"base": [], "day": DAY_FEATS, "day2": DAY2}[mode]
        for m, tr, va in folds(f):
            p = fit_predict(tr, va, cols)
            extra = None
            if mode in ("day", "day2"):
                ref = pl.read_parquet(CACHE / f"oofc_base_m{m}.parquet")["pred"].to_numpy()
                y = va["y"].to_numpy()
                cl = (y >= 30) & (y <= 7200)
                extra = {"diff_vs_base_clean": boot_diff(y[cl], ref[cl], p[cl])}
            summarize(f"{mode}_m{m}", va, p, extra)
            out = CACHE / f"oofc_{mode}_m{m}.parquet"
            va.select("MVT_ID_mvt").with_columns(pl.Series("pred", p)).write_parquet(out)
    elif mode == "weight":
        # neighbour months of the held-out month get weight w (Jan: Dec+Feb, Jul: Jun+Aug), everything else 1
        for m, tr, va in folds(f):
            nb = {1: (12, 2), 7: (6, 8)}[m]
            ref = pl.read_parquet(CACHE / f"oofc_base_m{m}.parquet")["pred"].to_numpy()
            y = va["y"].to_numpy()
            cl = (y >= 30) & (y <= 7200)
            for w in (2.0, 4.0):
                wt = np.where(tr["month"].is_in(list(nb)).to_numpy(), w, 1.0)
                p = fit_predict(tr, va, base, wt)
                summarize(f"weight{w}_m{m}", va, p, {"diff_vs_base_clean": boot_diff(y[cl], ref[cl], p[cl])})
            # "season only": same-season months only
            seas = {1: (11, 12, 2, 3), 7: (5, 6, 8, 9)}[m]
            trs = tr.filter(pl.col("month").is_in(list(seas)))
            p = fit_predict(trs, va, base)
            summarize(f"season_only_m{m}", va, p, {"diff_vs_base_clean": boot_diff(y[cl], ref[cl], p[cl])})
    elif mode == "decay":
        for m, _, va in folds(f):
            for k in range(1, 13):
                if k == m:
                    continue
                tr = f.filter(pl.col("month") == k)
                p = fit_predict(tr, va, base, rounds=200)
                dist = min(abs(k - m), 12 - abs(k - m))
                summarize(f"decay_m{m}_from{k}", va, p, {"train_month": k, "dist": dist})
    print(f"done {mode} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
