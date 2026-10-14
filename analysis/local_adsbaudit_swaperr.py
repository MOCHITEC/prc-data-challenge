"""adsbaudit: cost of the production b-recovery (b_hat = file - c_old(b_hat), 3 iterations) in a residual swap.

Jan+Jul 2025, b = cb hybrid. Two residuals fitted in-sample on all Jan+Jul ADS-B rows with production V2 params:
old = production features (450 rounds), new = + cat_apron (441 rounds, rescat A3). On ground-fix rows (non-LTFM/LIRF):
file = b + c_old(b); production swap = file + c_new(b_hat) - c_old(b_hat); exact swap = b + c_new(b).
Reports the recovery error, the swap error, the in-sample MSE gain of the exact vs the production swap, and how the
recovered c_hat = file - b_hat distorts the slope of c on (proxy - b) used in the 2025-vs-2026 comparison.

    POLARS_MAX_THREADS=1 .venv/bin/python analysis/local_adsbaudit_swaperr.py
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "analysis")
import local_adsbaudit_bswap as B  # noqa: E402
import local_adsb_rescat as R  # noqa: E402

L, CV = B.L, B.CV
D = B.jj()
C, V = R.code_tables()
D = R.add_codes(D, C, V)
X, feats, cats = L.design(D, "b_cb")
allf = feats + ["cat_apron"]
r = np.clip((D["y"] - D["b_cb"]).to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
Xn = X.select(allf).to_numpy().astype(np.float32)
del X
mods = {}
for tag, fn, n in (("old", feats, 450), ("new", allf, 441)):
    idx = [allf.index(f) for f in fn]
    mods[tag] = (lgb.train(B.P, lgb.Dataset(Xn[:, idx], r, feature_name=fn, categorical_feature=[c for c in cats + ["cat_apron"] if c in fn]), n), fn)
    print(tag, "fitted", flush=True)
del Xn
G = D.filter(pl.col("ground_fix").fill_null(False) & ~pl.col("ADEP_mvt").is_in(["LTFM", "LIRF"]))


def c(tag, bv):
    m, fn = mods[tag]
    Xg, _, _ = L.design(G.with_columns(pl.Series("bb", bv)), "bb")
    return np.clip(m.predict(Xg.select(fn).to_numpy().astype(np.float32)), -CV.CLIP_C, CV.CLIP_C)


b, y = G["b_cb"].to_numpy(), G["y"].to_numpy()
p = (G["t_mvt"] - G["t_move"]).to_numpy()
co = c("old", b)
f = b + co
bh = f.copy()
for _ in range(3):
    bh = f - c("old", bh)
exact = b + c("new", b)
swap = f + c("new", bh) - c("old", bh)
out = {"n": len(b), "recovery |b_hat-b| mean": float(np.abs(bh - b).mean()), "p99": float(np.quantile(np.abs(bh - b), 0.99)),
       "mean(b_hat-b)": float((bh - b).mean()),
       "RMS delta exact (new-old)": float(np.sqrt(((exact - f) ** 2).mean())),
       "RMS delta production swap": float(np.sqrt(((swap - f) ** 2).mean())),
       "RMS(swap - exact)": float(np.sqrt(((swap - exact) ** 2).mean())),
       "corr(delta_swap, delta_exact)": float(np.corrcoef(swap - f, exact - f)[0, 1]),
       "in-sample MSE gain exact (per ground row)": float(((y - f) ** 2 - (y - exact) ** 2).mean()),
       "in-sample MSE gain production swap (per ground row)": float(((y - f) ** 2 - (y - swap) ** 2).mean())}
ok = np.isfinite(p) & (np.abs(b - p) < 900)
okh = np.isfinite(p) & (np.abs(bh - p) < 900)
out["slope c on (p-b), true b"] = float(np.polyfit((p - b)[ok], co[ok], 1)[0])
out["slope c_hat on (p-b_hat), recovered"] = float(np.polyfit((p - bh)[okh], (f - bh)[okh], 1)[0])
out["sd c true / c_hat"] = (float(co.std()), float((f - bh).std()))
out["share |c|>120 true / c_hat"] = (float((np.abs(co) > 120).mean()), float((np.abs(f - bh) > 120).mean()))
for k, v in out.items():
    print(f"{k:55s} {v}")
json.dump(out, open(os.path.join(L.SCR, "agents", "adsbaudit", "swaperr.json"), "w"), indent=1)
