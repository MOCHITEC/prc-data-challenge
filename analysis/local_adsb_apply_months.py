"""Apply finding 2 to a submission: swap the learned ADS-B residual trained on Jan+Jul 2025 (arm A) for the one
trained on Jan+Jul+Nov+Dec 2025 (arm B), as a delta on top of an existing submission file (local).

The production file holds b + c_old (plus the overlays); b alone is not stored here. So: final models A and B are fit
on all ADS-B rows of their months (rounds = median CV best iteration, cv_months.json); b is recovered from the file by
b = file - c_A(b) (two fixed-point steps; c depends on b only through the residual features); the new value is
file + c_B(b) - c_A(b). c = 0 at LTFM; rows without a callsign ADS-B feature row are untouched (this includes the
430 rows of the positional overlay, which have no callsign join).

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsb_apply_months.py <in.parquet> <out.parquet>
"""
from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402  (sets up L, CV with --base cb)

L, CV = M.L, M.CV


def main(src: str, dst: str) -> None:
    cfg = json.load(open(os.path.join(L.OUT, "cv_months.json")))
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    D = pl.concat([CV.load_labelled().select(keep), M.load_novdec().select(keep)])
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")),
                   pl.read_parquet(os.path.join(L.OUT, "feat_2025novdec.parquet"))], how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="inner")
    X, feats, cats = L.design(D, "b")
    assert feats == cfg["features"]
    Xn = X.select(feats).to_numpy().astype(np.float64)
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    jj = D["day"].str.slice(0, 7).is_in(["2025-01", "2025-07"]).to_numpy()
    models = {}
    for arm, sel in (("A", jj), ("B", np.ones(D.height, bool))):
        ds = lgb.Dataset(Xn[sel], r[sel], feature_name=feats, categorical_feature=cats)
        models[arm] = lgb.train(CV.PARAMS, ds, num_boost_round=cfg["final_rounds"][arm])
        models[arm].save_model(os.path.join(L.OUT, f"final_model_months_{arm}.txt"))
        print(f"arm {arm}: {int(sel.sum())} rows, {cfg['final_rounds'][arm]} rounds", flush=True)

    tpl = pl.read_parquet("data/raw/submitting.parquet")
    sub = pl.read_parquet(src)
    assert sub.schema == tpl.schema and (sub["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = pl.scan_parquet("data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet"))
    G = F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
    G = G.filter(pl.col("ADEP_mvt") != "LTFM").with_columns(pl.col("file").cast(pl.Float64))

    def corr(arm, b):
        Xg, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        assert f2 == feats
        return np.clip(models[arm].predict(Xg.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f = G["file"].to_numpy()
    b = f.copy()
    for it in range(3):
        cA = corr("A", b)
        b_new = f - cA
        print(f"fixed point step {it}: max |db| {np.abs(b_new - b).max():.1f}", flush=True)
        b = b_new
    cA, cB = corr("A", b), corr("B", b)
    new = np.round(f + cB - cA)
    new = np.where(new < 0, f, new)
    G = G.with_columns(pl.Series("new", new), pl.Series("cA", cA), pl.Series("cB", cB))
    out = (sub.join(G.select("MVT_ID_mvt", pl.col("new").cast(pl.Int32)), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(tpl.columns))
    assert out.height == tpl.height and (out["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    out.write_parquet(dst)
    d = (out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64))
    print(f"wrote {dst}: rows changed {int((d != 0).sum())}, sum delta^2 / N = {float((d ** 2).sum()) / out.height:.0f}, "
          f"max |delta| {float(d.abs().max()):.0f}")
    S = G.with_columns((pl.col("new") - pl.col("file")).alias("delta"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    print(S.group_by("ADEP_mvt", "m").agg(pl.len(), pl.col("cA").median().round(0), pl.col("cB").median().round(0),
                                          pl.col("delta").abs().mean().round(1).alias("mean|delta|"),
                                          ((pl.col("delta") ** 2).sum() / out.height).round(0).alias("delta^2/N"))
          .sort("ADEP_mvt", "m"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
