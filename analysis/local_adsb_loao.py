"""Leave-one-airport-out (LOAO) transfer test of the learned ADS-B residual model (Jan+Jul 2025, labelled).

Question: when the model is applied to an airport whose ADS-B regime it has not been trained on (2026 brings ground
coverage to LEMD/LFPG/LIRF in January, to EDDM/EGLL in July, and some air-only rows to LTFM), does the correction
still help? For each airport A: fit on the ADS-B rows of the other nine airports (all 62 days; early stopping on a
day split of those rows, same rule as adsb_learn_cv.py), predict A's rows, and compare A's gain with its
leave-days-out gain from cv_rows (the in-airport CV). Same features, parameters and clipping as adsb_learn_cv.py.

Read-only; writes $SCRATCH/learn/loao_<base>.parquet (per row c_loao) and prints a table.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsb_loao.py --base h2
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import adsb_learn_features as L  # noqa: E402

_g = L.mem_gate
L.mem_gate = lambda limit_mb=12500.0: _g(12500.0)  # local: 15 GB box
import adsb_learn_cv as CV  # noqa: E402

OUT = L.OUT
cvrows = os.path.join(OUT, f"cv_rows{CV.SFX}.parquet")


def main() -> None:
    D = CV.load_labelled()
    F = pl.read_parquet(os.path.join(OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    days = sorted(D["day"].unique().to_list())
    X, feats, cats = L.design(D, "b")
    Xn = X.select(feats).to_numpy().astype(np.float64)
    has = D["has"].to_numpy()
    yv, bv = D["y"].to_numpy(), D["b"].to_numpy()
    r = np.clip(yv - bv, -CV.CLIP_Y, CV.CLIP_Y)
    apt = D["ADEP_mvt"].to_numpy()
    day_arr = D["day"].to_numpy()
    val_days = CV.inner_split(days, CV.SEED + 101)
    c = np.zeros(D.height)
    for a in sorted(set(apt)):
        if (has & (apt == a)).sum() < 1000:
            print(f"{a}: skipped ({(has & (apt == a)).sum()} ADS-B rows)", flush=True)
            continue
        L.mem_gate()
        tr = has & (apt != a)
        is_val = tr & np.isin(day_arr, list(val_days))
        is_fit = tr & ~is_val
        te = has & (apt == a)
        dtr = lgb.Dataset(Xn[is_fit], r[is_fit], feature_name=feats, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(Xn[is_val], r[is_val], reference=dtr)
        m = lgb.train(CV.PARAMS, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        c[te] = np.clip(m.predict(Xn[te], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
        print(f"{a}: fit {is_fit.sum()} / es {is_val.sum()} / test {te.sum()}; best_iter {m.best_iteration}", flush=True)

    R = pl.read_parquet(cvrows).select("MVT_ID_mvt", "c_learned")
    O = D.select("MVT_ID_mvt", "ADEP_mvt", "y", "b", "has").with_columns(pl.Series("c_loao", c)).join(
        R, on="MVT_ID_mvt", how="left")
    O.write_parquet(os.path.join(OUT, f"loao{CV.SFX or '_h2'}.parquet"))
    T = (O.group_by("ADEP_mvt").agg(
        pl.len().alias("n"), pl.col("has").sum().alias("adsb_rows"),
        ((pl.col("y") - pl.col("b")) ** 2).mean().sqrt().round(1).alias("rmse_base"),
        ((pl.col("y") - pl.col("b") - pl.col("c_learned")) ** 2).mean().sqrt().round(1).alias("rmse_in_airport_cv"),
        ((pl.col("y") - pl.col("b") - pl.col("c_loao")) ** 2).mean().sqrt().round(1).alias("rmse_loao"),
        (((pl.col("y") - pl.col("b")) ** 2 - (pl.col("y") - pl.col("b") - pl.col("c_learned")) ** 2).mean())
        .round(0).alias("gain_in_airport"),
        (((pl.col("y") - pl.col("b")) ** 2 - (pl.col("y") - pl.col("b") - pl.col("c_loao")) ** 2).mean())
        .round(0).alias("gain_loao"),
    ).with_columns((pl.col("gain_loao") / pl.col("gain_in_airport")).round(2).alias("transfer_ratio"))
        .sort("ADEP_mvt"))
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(200)
    print(T)


if __name__ == "__main__":
    main()
