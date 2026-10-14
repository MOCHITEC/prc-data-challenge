"""2025 check: on rows reached only by the positional join (never in any residual's training set), compare the
learned residual used in v36 (Jan+Jul, current settings: final_model_cb_local.txt) with the 7-month V2 residual."""
import json, os, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "analysis")
import local_adsb_apply_swap as W
L, CV, OUT = W.L, W.CV, W.OUT
D = W.data()
X, feats, cats = L.design(D, "b")
ym = D["day"].str.slice(0, 7).to_numpy()
r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
m7 = lgb.train({**CV.PARAMS, **W.V2}, lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats, categorical_feature=cats),
               num_boost_round=W.rounds("7:v2"))
m7.save_model(os.path.join(OUT, "final_model_7v2.txt"))
mA = lgb.Booster(model_file=os.path.join(OUT, "final_model_cb_local.txt"))
Lab = W.M.CV.load_labelled()
Fp = pl.read_parquet(os.path.join(OUT, "feat_pos_2025.parquet")).drop("ADEP_mvt", "day")
E_ = Lab.join(Fp, on="MVT_ID_mvt", how="inner").filter(pl.col("ADEP_mvt") != "LIRF")
Xp, f2, _ = L.design(E_, "b"); assert f2 == feats
Xn = Xp.select(feats).to_numpy().astype(np.float64)
cA = np.clip(mA.predict(Xn), -CV.CLIP_C, CV.CLIP_C); c7 = np.clip(m7.predict(Xn), -CV.CLIP_C, CV.CLIP_C)
y, b = E_["y"].to_numpy(), E_["b"].to_numpy(); N = 344419
unm = E_["mvt_minus_aobt3"].is_null().to_numpy()
for lab, s in (("unmatched", unm), ("matched", ~unm), ("all", np.ones(len(y), bool))):
    g = lambda c: float(((b[s] - y[s]) ** 2 - (b[s] + c[s] - y[s]) ** 2).sum() / N)
    print(f"{lab:10s} n={s.sum():4d} RMSE base {np.sqrt(np.mean((b[s]-y[s])**2)):.0f} | A {np.sqrt(np.mean((b[s]+cA[s]-y[s])**2)):.0f} ({g(cA):+.1f}) | 7v2 {np.sqrt(np.mean((b[s]+c7[s]-y[s])**2)):.0f} ({g(c7):+.1f})")
