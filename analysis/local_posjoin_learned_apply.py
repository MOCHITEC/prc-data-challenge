"""Candidate on top of an existing file: add the learned ADS-B residual (final_model_cb_local.txt, Jan+Jul callsign
rows) to 2026 departures reached only by the positional join, with b = the file's value (these rows carry no
learned correction yet: no callsign join). Excludes LIRF (2025: -260 MSE on 19 rows), LTFM (production c = 0) and
rows already replaced by the raw-ADS-B overlay (docs/posjoin_2026-10-05/replacements_v33.csv).
2025 check (rows never in training): unmatched non-overlay +151.5 MSE (RMSE 569 -> 501, 724 rows); matched +6.9.

    .venv/bin/python analysis/local_posjoin_learned_apply.py <in.parquet> <out.parquet>
"""
import json, os, sys
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "scripts")
import adsb_learn_features as L
sys.argv, (src, dst) = [sys.argv[0], "--base", "cb"], sys.argv[1:3]
import adsb_learn_cv as CV
OUT = L.OUT
m = lgb.Booster(model_file=os.path.join(OUT, "final_model_cb_local.txt"))
feats = json.load(open(os.path.join(OUT, "cv_cb.json")))["features"]
sub = pl.read_parquet(src)
rep = pl.read_csv("/home/user/prc-data-challenge/docs/posjoin_2026-10-05/replacements_v33.csv").select("MVT_ID_mvt")
F = pl.read_parquet(os.path.join(OUT, "feat_pos_2026.parquet"))
G = (F.filter(~pl.col("ADEP_mvt").is_in(["LIRF", "LTFM"])).join(rep, on="MVT_ID_mvt", how="anti")
     .join(sub.rename({"TAXITIME_SEC_mvt": "file"}), on="MVT_ID_mvt").with_columns(pl.col("file").cast(pl.Float64).alias("b")))
X, f2, _ = L.design(G, "b"); assert f2 == feats
c = np.clip(m.predict(X.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)
new = np.round(G["b"].to_numpy() + c); new = np.where(new < 0, G["b"].to_numpy(), new)
G = G.with_columns(pl.Series("new", new).cast(pl.Int32), pl.Series("c", c))
out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
       .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
assert out.height == sub.height and (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
out.write_parquet(dst)
d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
print(f"{dst}: {G.height} target rows, {int((d != 0).sum())} changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.0f}, max |delta| {float(d.abs().max()):.0f}")
print(G.group_by("ADEP_mvt", pl.col("t_aobt3").is_not_null().alias("matched")).agg(pl.len(), pl.col("c").median().round(0), pl.col("c").abs().mean().round(0).alias("mean|c|")).sort("ADEP_mvt", "matched"))
