"""plan_months12.md section 4: replace v39's 4-month appear delta by the 12-month one, as a per-row delta.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/apply_delta12.py <in> <out> <delta_out>

A submission carrying v39's appear tier holds, on the rows with appear columns, b + c_plain(b) + (c_A4 - c_P4)(b), where
c_A4 / c_P4 are v39's models (4 months, V2, fill: final_model_a2_delta_months_fill_v2_{A,P}.txt) and c_plain is whatever
plain residual the file carries (4:v2 in v39, 12:v2 or 7:v2 in v40). The change is

    delta = (c_A12 - c_P12)(b) - (c_A4 - c_P4)(b)        on rows with appear columns, else 0,

with P12 / A12 fit on all ADS-B rows of the 12 months of 2025 with the rounds of cv_months12_a2.json. b enters only
through the residual features, but the delta is sensitive to it (on v39: RMS 15.7 s between the two recoveries below,
against an RMS delta of 25.9 s), so b must be recovered with the plain residual the input file really carries:
b = file - c_plain(b) - (c_A4 - c_P4)(b), with c_plain = c_P4 for v39 (v38's 4:v2). The base b under v39 and v40 is the
same, so the delta found on v39 is also the one for v40 and later files that kept v39's appear delta. The recovery with
c_P12 is printed for comparison only.
<delta_out> holds MVT_ID_mvt and delta (whole seconds, as applied to <in>) for the rows that change, so it can be added
to any later file that still carries v39's appear delta. Nothing is uploaded."""

import json
import os
import sys

SRC, DST, DELTA_OUT = sys.argv[1:4]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months12 as M  # noqa: E402  (sets --base a2)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CV, L, AP = M.CV, M.L, M.AP
CV.PARAMS = {**CV.PARAMS, **M.V2}


def fit(D: pl.DataFrame, appear: bool, rounds: int, feats_ref: list[str]) -> lgb.Booster:
    X, feats, cats = L.design(D, "b")
    if appear:
        X, feats = AP.design(X, feats, "b")
    assert feats == feats_ref, "feature list differs from the CV run"
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    ds = lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats, categorical_feature=cats)
    return lgb.train(CV.PARAMS, ds, num_boost_round=rounds)


def main() -> None:
    cj = json.load(open(os.path.join(L.OUT, "cv_months12_a2.json")))
    assert cj["verdict"] == "ADOPT", "plan_months12.md: the 12-month re-measure did not pass"
    feats = {"P": cj["features"]["P12"], "A": cj["features"]["A12"]}
    models = {}
    for k in ("P4", "A4"):
        models[k] = lgb.Booster(model_file=os.path.join(L.OUT, f"final_model_a2_delta_months_fill_v2_{k[0]}.txt"))
        assert models[k].feature_name() == feats[k[0]], f"{k}: v39's model has other features"
    D = M.load().filter("has")
    for k in ("P12", "A12"):
        path = os.path.join(L.OUT, f"final_model_a2_delta_m12_v2_{k[0]}.txt")
        if os.path.exists(path):
            models[k] = lgb.Booster(model_file=path)
        else:
            models[k] = fit(D, k[0] == "A", cj["final_rounds"][k], feats[k[0]])
            models[k].save_model(path)
        print(f"model {k}: {D.height:,} ADS-B rows (12 months), {cj['final_rounds'][k]} rounds", flush=True)

    tpl = pl.read_parquet(os.path.join(CV.ROOT, "data/raw/submitting.parquet"))
    sub = pl.read_parquet(SRC)
    assert sub.schema == tpl.schema and sub.height == tpl.height and (sub["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    rk = pl.scan_parquet(os.path.join(CV.ROOT, "data/raw/ranking.parquet")).select(
        "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet"))
    assert not (set(F26.columns) & L.LABEL_COLS)
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter(pl.col("ADEP_mvt") != "LTFM").with_columns(pl.col("file").cast(pl.Float64)))
    G = AP.add(G, "fill_2026")
    has = G["own_ok"].is_not_null().to_numpy()
    print(f"2026 ADS-B feature rows (not LTFM) {G.height:,}; with appear columns {int(has.sum()):,}", flush=True)
    AP.parity(D, G)

    def corr(k, b):
        Xg, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        if k[0] == "A":
            Xg, f2 = AP.design(Xg, f2, "b")
        assert f2 == feats[k[0]]
        return np.clip(models[k].predict(Xg.select(f2).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f = G["file"].to_numpy()
    deltas = {}
    for plain in ("P4", "P12"):
        b = f.copy()
        for it in range(3):
            app4 = np.where(has, corr("A4", b) - corr("P4", b), 0.0)
            b_new = f - corr(plain, b) - app4
            print(f"[{plain}] fixed point step {it}: max |db| {np.abs(b_new - b).max():.1f}", flush=True)
            b = b_new
        deltas[plain] = np.where(has, corr("A12", b) - corr("P12", b) - corr("A4", b) + corr("P4", b), 0.0)
    dd = deltas["P12"] - deltas["P4"]
    print(f"delta with b from c_P4 vs c_P12: RMS {np.sqrt(np.mean(dd[has] ** 2)):.2f} s, max |.| {np.abs(dd).max():.1f} s; "
          f"RMS delta {np.sqrt(np.mean(deltas['P12'][has] ** 2)):.2f} s", flush=True)
    delta = deltas["P4"]  # the input is v39, whose plain residual is v38's 4-month one
    new = np.round(f + delta)
    new = np.where(new < 0, f, new)
    G = G.with_columns(pl.Series("new", new), pl.Series("delta", new - f))
    G.filter(pl.col("delta") != 0).select("MVT_ID_mvt", pl.col("delta").cast(pl.Int32)).write_parquet(DELTA_OUT)
    out = (sub.join(G.select("MVT_ID_mvt", pl.col("new").cast(pl.Int32)), on="MVT_ID_mvt", how="left",
                    maintain_order="left")
           .with_columns(pl.coalesce("new", "TAXITIME_SEC_mvt").cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select(tpl.columns))
    assert out.schema == tpl.schema and out.height == tpl.height and (out["MVT_ID_mvt"] == tpl["MVT_ID_mvt"]).all()
    assert out.null_count().sum_horizontal().item() == 0 and int((out["TAXITIME_SEC_mvt"] < 0).sum()) == 0
    out.write_parquet(DST)
    d = out["TAXITIME_SEC_mvt"].cast(pl.Float64) - sub["TAXITIME_SEC_mvt"].cast(pl.Float64)
    print(f"wrote {DST}: rows changed {int((d != 0).sum()):,}, sum delta^2 / N = {float((d ** 2).sum()) / out.height:.0f}, "
          f"max |delta| {float(d.abs().max()):.0f}, mean delta {float(d.mean()):+.2f}; delta table {DELTA_OUT}")
    S = G.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"))
    print(S.group_by("ADEP_mvt", "m").agg(pl.len(), (pl.col("delta") != 0).sum().alias("changed"),
                                          pl.col("delta").mean().round(1).alias("mean"),
                                          pl.col("delta").abs().mean().round(1).alias("mean|d|"),
                                          ((pl.col("delta") ** 2).sum() / out.height).round(0).alias("d^2/N"))
          .sort("ADEP_mvt", "m"))


if __name__ == "__main__":
    main()
