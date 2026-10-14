"""plan_cov26.md section 4: the EDDF-thinning-trained 12-month appear residual as a delta on v47, EDDF rows only.

    THIN=<thin dir> SCRATCH=<normal adsb dir> .venv/bin/python docs/appear_tier_2026-10-05/apply_cov26.py <dir with v38 v47> <delta_out>

Final A12thin: fit on all ADS-B rows of the 12 months of 2025 with EDDF's features rebuilt on thinned fixes (cov_thin.py),
660 rounds as the final A12. On the 2026 EDDF rows with appear columns (their real 2026 features):
delta = c_A12thin(b) - c_A12(b), with b recovered from v38 as for candidate A (3 plain fixed-point steps with P4).
Writes <delta_out> (MVT_ID_mvt, delta in whole seconds); c12_apply.py then adds it to v47."""

import os
import sys

SRC, OUT = sys.argv[1:3]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_cov26b as CB  # noqa: E402  (sets --base a2)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

M, CV, L, AP = CB.M, CB.CV, CB.L, CB.AP


def main() -> None:
    params = {**CV.PARAMS, **M.V2}
    path = os.path.join(L.OUT, "final_model_a2_delta_m12_v2_A_thinEDDF.txt")
    if os.path.exists(path):
        thin = lgb.Booster(model_file=path)
    else:
        Dt = CB.with_thin_eddf(M.load("fill")).filter("has")
        X, feats, cats = L.design(Dt, "b")
        X, feats = AP.design(X, feats, "b")
        r = np.clip(Dt["y"].to_numpy() - Dt["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
        ds = lgb.Dataset(X.select(feats).to_numpy().astype(np.float64), r, feature_name=feats, categorical_feature=cats)
        thin = lgb.train(params, ds, num_boost_round=660)
        thin.save_model(path)
        print(f"A12thin: {Dt.height:,} rows, 660 rounds", flush=True)
    cur = lgb.Booster(model_file=os.path.join(L.OUT, "final_model_a2_delta_m12_v2_A.txt"))
    p4 = lgb.Booster(model_file=os.path.join(L.OUT, "final_model_a2_delta_months_fill_v2_P.txt"))
    assert thin.feature_name() == cur.feature_name()
    rk = pl.scan_parquet(os.path.join(CV.ROOT, "data/raw/ranking.parquet")).select("MVT_ID_mvt", "ADEP_mvt").collect()
    G = pl.read_parquet(os.path.join(L.OUT, "feat_2026.parquet")).drop("ADEP_mvt").join(rk, on="MVT_ID_mvt")
    G = G.join(pl.read_parquet(os.path.join(SRC, "v38.parquet")).select(
        "MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v38")), on="MVT_ID_mvt")
    G = AP.add(G.filter(pl.col("ADEP_mvt") == "EDDF"), "fill_2026")
    has = G["own_ok"].is_not_null().to_numpy()

    def corr(model, b, appear):
        Xg, f2, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        if appear:
            Xg, f2 = AP.design(Xg, f2, "b")
        return np.clip(model.predict(Xg.select(f2).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f = G["v38"].to_numpy()
    b = f.copy()
    for _ in range(3):
        b = f - corr(p4, b, False)
    d = np.where(has, corr(thin, b, True) - corr(cur, b, True), 0.0)
    print(f"EDDF rows {G.height:,}, with appear columns {int(has.sum()):,}; delta RMS {np.sqrt(np.mean(d[has] ** 2)):.2f} s, "
          f"mean {d[has].mean():+.2f}, sum d^2/N(344,841) {float((d ** 2).sum()) / 344841:.1f}", flush=True)
    out = G.select("MVT_ID_mvt").with_columns(pl.Series("delta", np.round(d)).cast(pl.Int32)).filter(pl.col("delta") != 0)
    out.write_parquet(OUT)
    print(f"wrote {OUT}: {out.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
