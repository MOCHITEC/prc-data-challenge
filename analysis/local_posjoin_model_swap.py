"""On positional-join rows (the ones local_posjoin_learned_apply.py corrected with final_model_cb_local), swap that
residual for the 7-month V2 one: b = file - c_A(b) (fixed point), new = b + c_7v2(b). Same row selection as the
apply script (non-LIRF, non-LTFM, not one of the 430 raw-ADS-B overlay rows).
2025 check (local_posjoin_model_compare.py): +256.4 vs +218.5 MSE on 1,049 such rows.

    .venv/bin/python analysis/local_posjoin_model_swap.py <in.parquet> <out.parquet>
"""
import json, os, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "scripts")
import adsb_learn_features as L
src, dst = sys.argv[1:3]
sys.argv = [sys.argv[0], "--base", "cb"]
import adsb_learn_cv as CV
OUT = L.OUT
mA = lgb.Booster(model_file=os.path.join(OUT, "final_model_cb_local.txt"))
m7 = lgb.Booster(model_file=os.path.join(OUT, "final_model_7v2.txt"))
feats = json.load(open(os.path.join(OUT, "cv_cb.json")))["features"]
sub = pl.read_parquet(src)
rep = pl.read_csv("/home/user/prc-data-challenge/docs/posjoin_2026-10-05/replacements_v33.csv").select("MVT_ID_mvt")
G = (pl.read_parquet(os.path.join(OUT, "feat_pos_2026.parquet")).filter(~pl.col("ADEP_mvt").is_in(["LIRF", "LTFM"]))
     .join(rep, on="MVT_ID_mvt", how="anti").join(sub.rename({"TAXITIME_SEC_mvt": "file"}), on="MVT_ID_mvt")
     .with_columns(pl.col("file").cast(pl.Float64)))
def corr(m, b):
    X, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b"); assert f2 == feats
    return np.clip(m.predict(X.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
f = G["file"].to_numpy(); b = f.copy()
for _ in range(3):
    b = f - corr(mA, b)
new = np.round(b + corr(m7, b)); new = np.where(new < 0, f, new)
G = G.with_columns(pl.Series("new", new).cast(pl.Int32))
out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
       .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
out.write_parquet(dst)
d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
print(f"{dst}: {G.height} positional rows, {int((d != 0).sum())} changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.1f}")
