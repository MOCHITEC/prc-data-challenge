"""adsbaudit: does the production fixed point b = file - c(b) (3 iterations, local_adsb_apply_swap.py) recover b?

Emulation on Jan+Jul 2025: fit the residual (V2 params, 450 rounds) on all Jan+Jul ADS-B rows with b = cb hybrid,
build file = b + c(b) on ground-fix rows, recover b_hat by the production iteration, and report |b_hat - b| and
|c(b_hat) - c(b)| (the swap error is of that order). Local, read-only.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsbaudit_fixpt.py
"""
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "analysis")
import local_adsbaudit_bswap as B  # noqa: E402

L, CV = B.L, B.CV
D = B.jj()
X, feats, cats = L.design(D, "b_cb")
r = np.clip((D["y"] - D["b_cb"]).to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
m = lgb.train(B.P, lgb.Dataset(X.select(feats).to_numpy().astype(np.float32), r, feature_name=feats, categorical_feature=cats), 450)
G = D.filter(pl.col("ground_fix").fill_null(False))


def c(bv):
    Xg, _, _ = L.design(G.with_columns(pl.Series("bb", bv)), "bb")
    return np.clip(m.predict(Xg.select(feats).to_numpy().astype(np.float32)), -CV.CLIP_C, CV.CLIP_C)


b = G["b_cb"].to_numpy()
c0 = c(b)
f = b + c0
bh = f.copy()
for it in range(6):
    bh = f - c(bh)
    e = np.abs(bh - b)
    print(f"iter {it + 1}: |b_hat - b| mean {e.mean():.2f} p99 {np.quantile(e, 0.99):.1f} max {e.max():.0f}; "
          f"|c(b_hat)-c(b)| mean {np.abs(c(bh) - c0).mean():.2f}", flush=True)
