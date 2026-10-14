"""adsbaudit: multivariate adversarial AUC (LightGBM, 5-fold) 2025 vs 2026 on the residual's design matrix, with and
without the coverage-count family, ground-fix rows non-LTFM/LIRF, 150k-row samples per side. Local, read-only.

    POLARS_MAX_THREADS=1 .venv/bin/python analysis/local_adsbaudit_adv.py
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "analysis")
import local_adsbaudit_dist as Dm  # noqa: E402

X, feats = Dm.load()
X = X.filter(pl.col("lane") == "A")
COUNT = ["n_ground", "n_win", "n_win_ground", "post_n", "stand_nfix", "post_n_stops", "post_max_gap"]
POS = ["x_a", "y_a", "x_l", "y_l"]
rng = np.random.default_rng(0)
for lab, sel1, sel2 in (("2025 12m vs 2026", ~pl.col("per").str.starts_with("2026"), pl.col("per").str.starts_with("2026")),
                        ("ND25 vs Jan26", pl.col("per").is_in(["2025-11", "2025-12"]), pl.col("per") == "2026-01"),
                        ("Jul25 vs Jul26", pl.col("per") == "2025-07", pl.col("per") == "2026-07")):
    A, B = X.filter(sel1), X.filter(sel2)
    A = A.sample(min(150000, A.height), seed=1)
    B = B.sample(min(150000, B.height), seed=1)
    Z = pl.concat([A, B])
    yv = np.r_[np.zeros(A.height), np.ones(B.height)]
    fold = rng.integers(0, 5, Z.height)
    out = {}
    for name, fs in (("all", feats), ("no base/res", [f for f in feats if f != "base" and not f.startswith("res_")]),
                     ("no counts+pos", [f for f in feats if f not in COUNT + POS]),
                     ("base+res only", ["base"] + [f for f in feats if f.startswith("res_")] + ["cat_ADEP_mvt"])):
        M = Z.select(fs).to_numpy().astype(np.float32)
        p = np.zeros(Z.height)
        for k in range(5):
            m = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.1, num_threads=1, verbose=-1),
                          lgb.Dataset(M[fold != k], yv[fold != k]), 150)
            p[fold == k] = m.predict(M[fold == k])
        out[name] = round(float(roc_auc_score(yv, p)), 3)
    print(lab, out, flush=True)
