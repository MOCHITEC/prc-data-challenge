"""2025 check: apply the learned ADS-B residual (final fit on the Jan+Jul callsign rows, cb base, as adsb_learn_apply)
to departures reached only by the positional join (never in its training set) and compare with the raw-ADS-B overlay."""
import json, os, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "scripts")
import adsb_learn_features as L
_g = L.mem_gate; L.mem_gate = lambda limit_mb=12500.0: _g(12500.0)
sys.argv = [sys.argv[0], "--base", "cb"]
import adsb_learn_cv as CV
import adsb_stage1_feasibility as s1
OUT = L.OUT
cvj = json.load(open(os.path.join(OUT, "cv_cb.json")))
D = CV.load_labelled()
F = pl.read_parquet(os.path.join(OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
Dt = D.join(F, on="MVT_ID_mvt", how="inner")
X, feats, cats = L.design(Dt, "b")
assert feats == cvj["features"]
r = np.clip(Dt["y"].to_numpy() - Dt["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
m = lgb.train(CV.PARAMS, lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats, categorical_feature=cats),
              num_boost_round=cvj["final_rounds"])
m.save_model(os.path.join(OUT, "final_model_cb_local.txt"))
Fp = pl.read_parquet(os.path.join(OUT, "feat_pos_2025.parquet")).drop("ADEP_mvt", "day")
E_ = D.join(Fp, on="MVT_ID_mvt", how="inner").filter(pl.col("mvt_minus_aobt3").is_null())
Xp, f2, _ = L.design(E_, "b"); assert f2 == feats
c = np.clip(m.predict(Xp.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
E_ = E_.with_columns(pl.Series("c", c), (pl.col("t_mvt") - pl.col("t_move")).alias("a"),
                     ((pl.col("push").fill_null(0) >= s1.PUSH_MIN) & (pl.col("post_dwell").fill_null(0) < s1.POST_DWELL_MAX) & pl.col("t_move").is_not_null()).alias("overlay"))
E_ = E_.with_columns(pl.when(pl.col("overlay")).then(pl.col("a")).otherwise(pl.col("b")).alias("p_overlay"), (pl.col("b") + pl.col("c")).alias("p_learned"),
                     pl.when(pl.col("overlay")).then(pl.col("a")).otherwise(pl.col("b") + pl.col("c")).alias("p_both"))
N = 344419
for lab, X_ in [("all pos rows (non-LIRF)", E_.filter(pl.col("ADEP_mvt") != "LIRF")), ("overlay rows", E_.filter("overlay")),
                ("non-overlay rows (non-LIRF)", E_.filter(~pl.col("overlay") & (pl.col("ADEP_mvt") != "LIRF"))), ("LIRF", E_.filter(pl.col("ADEP_mvt") == "LIRF"))]:
    y, b = X_["y"].to_numpy(), X_["b"].to_numpy()
    s = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    g = lambda p: float(((b - y) ** 2 - (p - y) ** 2).sum() / N)
    print(f"{lab:30s} n={X_.height:4d} RMSE base {s(b):6.0f} | learned {s(X_['p_learned'].to_numpy()):6.0f} ({g(X_['p_learned'].to_numpy()):+6.1f}) | "
          f"overlay {s(X_['p_overlay'].to_numpy()):6.0f} ({g(X_['p_overlay'].to_numpy()):+6.1f}) | both {s(X_['p_both'].to_numpy()):6.0f} ({g(X_['p_both'].to_numpy()):+6.1f})")
