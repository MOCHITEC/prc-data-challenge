"""Put the appear tier on top of an existing submission as a delta (the method of v37's months swap,
analysis/local_adsb_apply_months.py on claude/laughing-cray-vvwh0k).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/apply_delta.py <in.parquet> <out.parquet>
    ... apply_delta.py <in> <out> --months   # P/A fit on Jan+Jul+Nov+Dec 2025 (plan.md 6.2), rounds from cv_months_a2.json

The submission holds b + (production's ADS-B corrections); b alone is not stored. So:
1. final models P (plain) and A (plain + appear columns) are fit on all ADS-B rows of Jan+Jul 2025 on the `--base a2`
   OOF, with the rounds of their CV runs (cv_a2.json, cv_a2_appear.json);
2. b is recovered from the file by the fixed point b = file - c_P(b) (three steps; c depends on b only through the
   residual features);
3. new = file + c_A(b) - c_P(b) on the rows that have appear columns (a ground phase with a callsign join), else the
   file. In the CV that restriction keeps the gain (-1.842 s vs -1.858 s on all rows); on rows without appear columns
   the two models differ by noise only (-0.030 [-0.090, +0.019]). c = 0 at LTFM. The 430 positional-overlay rows
   have no callsign join and are untouched. Negative results revert to the file.
Nothing is uploaded."""

import json
import os
import sys

ARGS = sys.argv[1:3]
MONTHS = "--months" in sys.argv
sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import adsb_appear as AP  # noqa: E402
import adsb_learn_cv as CV  # noqa: E402
import adsb_learn_features as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months as CM  # noqa: E402


def fit(D: pl.DataFrame, appear: bool, rounds: int, feats_ref: list[str]):
    X, feats, cats = L.design(D, "b")
    if appear:
        X, feats = AP.design(X, feats, "b")
    assert feats == feats_ref, "feature list differs from the CV run"
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    ds = lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats, categorical_feature=cats)
    return lgb.train(CV.PARAMS, ds, num_boost_round=rounds), feats


def main(src: str, dst: str) -> None:
    if MONTHS:
        m = json.load(open(os.path.join(L.OUT, "cv_months_a2.json")))
        assert m["verdict"] == "ADOPT", "plan.md 6.2: the 4-month re-measure did not pass; no 4-month delta"
        cj = {k: {"final_rounds": m["final_rounds"][a], "features": m["features"][a]}
              for k, a in (("P", "plain4"), ("A", "appear4"))}
        D = CM.load().filter("has")
    else:
        cj = {k: json.load(open(os.path.join(L.OUT, f"cv_a2{s}.json"))) for k, s in (("P", ""), ("A", "_appear"))}
        D = CV.load_labelled()
        F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).drop("ADEP_mvt", "day")
        D = D.join(F, on="MVT_ID_mvt", how="inner")
        D = AP.add(D, "2025")
    models, feats = {}, {}
    for k in ("P", "A"):
        models[k], feats[k] = fit(D, k == "A", cj[k]["final_rounds"], cj[k]["features"])
        models[k].save_model(os.path.join(L.OUT, f"final_model_a2_delta{'_months' if MONTHS else ''}_{k}.txt"))
        print(f"model {k}: {D.height} ADS-B rows, {cj[k]['final_rounds']} rounds, {len(feats[k])} features", flush=True)

    tpl = pl.read_parquet(os.path.join(CV.ROOT, "data/raw/submitting.parquet"))
    sub = pl.read_parquet(src)
    assert sub.schema == tpl.schema and sub.height == tpl.height and (sub["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = pl.scan_parquet(os.path.join(CV.ROOT, "data/raw/ranking.parquet")).select(
        "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet"))
    assert not (set(F26.columns) & L.LABEL_COLS)
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter(pl.col("ADEP_mvt") != "LTFM").with_columns(pl.col("file").cast(pl.Float64)))
    G = AP.add(G, "2026")
    has = G["own_ok"].is_not_null().to_numpy()
    print(f"2026 ADS-B feature rows (not LTFM) {G.height:,}; with appear columns {int(has.sum()):,}", flush=True)
    AP.parity(D, G)

    def corr(k, b):
        Xg, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        if k == "A":
            Xg, f2 = AP.design(Xg, f2, "b")
        assert f2 == feats[k]
        return np.clip(models[k].predict(Xg.select(f2).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f = G["file"].to_numpy()
    b = f.copy()
    for it in range(3):
        b_new = f - corr("P", b)
        print(f"fixed point step {it}: max |db| {np.abs(b_new - b).max():.1f}", flush=True)
        b = b_new
    cP, cA = corr("P", b), corr("A", b)
    new = np.where(has, np.round(f + cA - cP), f)
    new = np.where(new < 0, f, new)
    G = G.with_columns(pl.Series("new", new), pl.Series("delta", new - f))
    out = (sub.join(G.select("MVT_ID_mvt", pl.col("new").cast(pl.Int32)), on="MVT_ID_mvt", how="left",
                    maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(tpl.columns))
    assert out.schema == tpl.schema and out.height == tpl.height and (out["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    assert out.null_count().sum_horizontal().item() == 0 and int((out["TAXITIME_SEC_mvt"] < 0).sum()) == 0
    out.write_parquet(dst)
    d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    print(f"wrote {dst}: rows changed {int((d != 0).sum()):,}, sum delta^2 / N = {float((d ** 2).sum()) / out.height:.0f}, "
          f"max |delta| {float(d.abs().max()):.0f}, mean delta {float(d.mean()):+.2f}")
    S = G.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    print(S.group_by("ADEP_mvt", "m").agg(pl.len(), (pl.col("delta") != 0).sum().alias("changed"),
                                          pl.col("delta").mean().round(1).alias("mean"),
                                          pl.col("delta").abs().mean().round(1).alias("mean|d|"),
                                          ((pl.col("delta") ** 2).sum() / out.height).round(0).alias("d^2/N"))
          .sort("ADEP_mvt", "m"))


if __name__ == "__main__":
    main(*ARGS)
