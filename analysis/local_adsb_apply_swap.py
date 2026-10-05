"""Swap one learned ADS-B residual for another on top of a submission file, as a delta on ground-fix rows (local).

OLD and NEW are each "<months>:<params>":
  months  4 = Jan+Jul+Nov+Dec 2025, 7 = those plus Aug+Sep+Oct
  params  cur = adsb_learn_cv.PARAMS, v2 = 255 leaves / min 300 / ff 0.7 (local_adsb_tune_months.py V2)
Rounds = median CV best iteration from the matching json (cv_months.json for 4:cur, cv_months7.json for 7:cur,
cv_<tag>.json if present), else 600. b is recovered from the file by b = file - c_OLD(b) (fixed point), the new value
is file + c_NEW(b) - c_OLD(b) on rows with a ground fix and not at LTFM; other rows are untouched.

    .venv/bin/python analysis/local_adsb_apply_swap.py <old> <new> <in.parquet> <out.parquet>
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

ARGS = sys.argv[1:5]
sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402

L, CV = M.L, M.CV
OUT = L.OUT
V2 = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)


ROUNDS_SRC = {"4:cur": ("cv_months.json", "B"), "7:cur": ("cv_months7.json", "C"),
              "4:v2": ("cv_months_v2.json", "V2"), "7:v2": ("cv_months7_v2.json", "C")}


def rounds(tag):
    f, k = ROUNDS_SRC[tag]
    p = os.path.join(OUT, f)
    return int(json.load(open(p))["final_rounds"][k]) if os.path.exists(p) else 600


def data():
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    parts = [CV.load_labelled().select(keep), M.load_novdec().select(keep)]
    feats = ["feat_2025.parquet", "feat_2025novdec.parquet"]
    if os.path.exists(os.path.join(OUT, "feat_2025aso.parquet")):
        parts.append(M.load_novdec("aso", "aso").select(keep))
        feats.append("feat_2025aso.parquet")
    D = pl.concat(parts)
    F = pl.concat([pl.read_parquet(os.path.join(OUT, f)) for f in feats], how="diagonal_relaxed").drop("ADEP_mvt", "day")
    return D.join(F, on="MVT_ID_mvt", how="inner")


def main(old, new, src, dst):
    D = data()
    X, feats, cats = L.design(D, "b")
    Xn = X.select(feats).to_numpy().astype(np.float64)
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    ym = D["day"].str.slice(0, 7).to_numpy()
    models = {}
    for tag in (old, new):
        mo, pa = tag.split(":")
        sel = np.isin(ym, ["2025-01", "2025-07", "2025-11", "2025-12"] + (["2025-08", "2025-09", "2025-10"] if mo == "7" else []))
        P = {**CV.PARAMS, **(V2 if pa == "v2" else {})}
        n = rounds(tag)
        models[tag] = lgb.train(P, lgb.Dataset(Xn[sel], r[sel], feature_name=feats, categorical_feature=cats), num_boost_round=n)
        print(f"{tag}: {int(sel.sum())} rows, {n} rounds", flush=True)
    sub = pl.read_parquet(src)
    rk = pl.scan_parquet("data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"))
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter((pl.col("ADEP_mvt") != "LTFM") & pl.col("ground_fix").fill_null(False)).with_columns(pl.col("file").cast(pl.Float64)))

    def corr(tag, b):
        Xg, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        assert f2 == feats
        return np.clip(models[tag].predict(Xg.select(feats).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f = G["file"].to_numpy()
    b = f.copy()
    for _ in range(3):
        b = f - corr(old, b)
    new_v = np.round(f + corr(new, b) - corr(old, b))
    new_v = np.where(new_v < 0, f, new_v)
    G = G.with_columns(pl.Series("new", new_v).cast(pl.Int32))
    out = (sub.join(G.select("MVT_ID_mvt", "new"), on="MVT_ID_mvt", how="left", maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(sub.columns))
    assert (out["MVT_ID_mvt"] == sub["MVT_ID_mvt"]).all() and out.null_count().sum_horizontal().item() == 0
    out.write_parquet(dst)
    d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    print(f"{dst}: {G.height} ground rows, {int((d != 0).sum())} changed, sum delta^2/N {float((d ** 2).sum()) / sub.height:.0f}")


if __name__ == "__main__":
    main(*ARGS)
