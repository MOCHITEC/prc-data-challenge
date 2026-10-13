"""Final phase (handoff/2026-10-11/FINAL_PHASE.md): the ADS-B residual on a pre-ADS-B base, for the rows of the final
file (Jan, Feb, Jun, Jul 2026).

    SCRATCH=<adsb dir> POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/final_adsb.py \
        <base.parquet> <out.parquet> [--months 02,06] [--increment]

<base.parquet>: MVT_ID_mvt, TAXITIME_SEC_mvt, the prediction BEFORE any ADS-B layer (any subset of final_submitting).
On the base rows in --months (default Feb and Jun) that have ADS-B features (feat/appear_fill of 2026fj for Feb/Jun,
of 2026 for Jan/Jul):
  - c = c_A12thin(b) at EDDF (plan_cov26.md, results.md section 18), c_A12(b) elsewhere (the 12-month appear residual,
    final_model_a2_delta_m12_v2_A*.txt), clipped to +-1800 s as in training;
  - c = 0 at LTFM (no ground coverage) and on unmatched LIRF rows (AOBT_3 null: overlay territory, copy band /
    time-zone class / ucmix), and on rows sitting on the LOBT bound (their pre-clip value is unknown);
  - new = b + c, then the LOBT identity clip |y - (MVT - LOBT)| <= 3606 s, and a negative value reverts to b.
Writes every base row (unchanged ones copied) as int32 seconds.

--increment: c = c_A12thin(b) - c_P12(b) at EDDF, c_A12(b) - c_P12(b) elsewhere (same rows and exclusions), i.e. only
what the 12-month appear tier and the EDDF coverage model add on top of a plain 12-month ADS-B residual. This is the
v48 recipe (v48 = v47 + (c_A12 - c_P12)(b) + (c_A12thin - c_A12)(b) on EDDF; LB 234.4831 -> 234.4099) for a file whose
ADS-B layer is the integrator's 12:v2+apron one (laughing-cray docs/final_2026-10-11/adsb_layer.py). b must still be the
pre-residual base of that file, not the file itself."""

import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import adsb_learn_features as L  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


# The appear-tier design of scripts/adsb_appear.py, copied so that this script needs only adsb_learn_features (identical
# on every working branch).
AP_TIMES = ("t_leave_own_lo", "t_leave_own", "t_appear_adj")
AP_PLAIN = ("own_ok", "d_dep1_own", "d_dep1_near", "d_stand_own", "n_in_own")
AP_FEATS = [f"{p}_{e}" for p in ("mvt_m", "res") for e in AP_TIMES] + [f"{e}_m_aobt3" for e in AP_TIMES] + list(AP_PLAIN)


class AP:
    @staticmethod
    def design(X: pl.DataFrame, feats: list[str], base_col: str = "b") -> tuple[pl.DataFrame, list[str]]:
        ex = []
        for e in AP_TIMES:
            ex += [(pl.col("t_mvt") - pl.col(e)).alias(f"mvt_m_{e}"),
                   (pl.col("t_mvt") - pl.col(e) - pl.col(base_col).cast(pl.Float64)).alias(f"res_{e}"),
                   (pl.col(e) - pl.col("t_aobt3")).alias(f"{e}_m_aobt3")]
        X = X.with_columns(ex).with_columns([pl.col(c).cast(pl.Float64) for c in AP_PLAIN])
        return X, feats + AP_FEATS

CLIP_C = 1800.0
BASE, OUT = ARGS[0], ARGS[1]
MONTHS = ARGS[ARGS.index("--months") + 1].split(",") if "--months" in ARGS else ["02", "06"]
INCREMENT = "--increment" in ARGS


def features(months: list[str]) -> pl.DataFrame:
    keys = sorted({"2026fj" if m in ("02", "06") else "2026" for m in months})
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"feat_{k}.parquet")) for k in keys], how="diagonal_relaxed")
    T = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"appear_fill_{k}.parquet")) for k in keys], how="diagonal_relaxed")
    return F.drop("ADEP_mvt").join(T, on="MVT_ID_mvt", how="left")


def main() -> None:
    base = pl.read_parquet(BASE).select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("b"))
    raw = (pl.scan_parquet("data/raw/final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
           .select("MVT_ID_mvt", "ADEP_mvt", pl.col("AOBT_3_flt").is_null().alias("unm"),
                   pl.col("MVT_TIME_UTC_mvt").dt.strftime("%m").alias("mon"),
                   ((pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds()).cast(pl.Float64).alias("mml"))
           .collect())
    G = base.join(raw, on="MVT_ID_mvt", how="left")
    assert G["ADEP_mvt"].null_count() == 0, "base rows not in final_ranking"
    X = G.filter(pl.col("mon").is_in(MONTHS)).join(features(MONTHS), on="MVT_ID_mvt", how="inner")
    X = X.filter(pl.col("own_ok").is_not_null())  # rows with the appear columns = rows the residual was trained on
    lo, hi = (pl.col("mml") - 3606).ceil(), (pl.col("mml") + 3606).ceil()
    on_bound = pl.col("mml").is_not_null() & ((pl.col("b") == lo) | (pl.col("b") == hi))
    X = X.filter((pl.col("ADEP_mvt") != "LTFM") & ~((pl.col("ADEP_mvt") == "LIRF") & pl.col("unm")) & ~on_bound)
    cur = lgb.Booster(model_file=os.path.join(L.OUT, "final_model_a2_delta_m12_v2_A.txt"))
    thin = lgb.Booster(model_file=os.path.join(L.OUT, "final_model_a2_delta_m12_v2_A_thinEDDF.txt"))
    assert cur.feature_name() == thin.feature_name()
    Xg, f2, _ = L.design(X, "b")
    Xg, f2 = AP.design(Xg, f2, "b")
    M = Xg.select(f2).to_numpy().astype(np.float64)
    eddf = (X["ADEP_mvt"] == "EDDF").to_numpy()
    c = np.clip(np.where(eddf, thin.predict(M), cur.predict(M)), -CLIP_C, CLIP_C)
    if INCREMENT:
        p12 = lgb.Booster(model_file=os.path.join(L.OUT, "final_model_a2_delta_m12_v2_P.txt"))
        Xp, fp, _ = L.design(X, "b")
        assert fp == p12.feature_name()
        c = c - np.clip(p12.predict(Xp.select(fp).to_numpy().astype(np.float64)), -CLIP_C, CLIP_C)
    C = X.select("MVT_ID_mvt").with_columns(pl.Series("c", c))
    G = G.join(C, on="MVT_ID_mvt", how="left").with_columns(pl.col("c").fill_null(0.0))
    G = G.with_columns(raw=pl.col("b") + pl.col("c"))
    G = G.with_columns(clipd=pl.when(pl.col("mml").is_null()).then(pl.col("raw"))
                       .otherwise(pl.max_horizontal(pl.min_horizontal(pl.col("raw"), hi), lo)))
    G = G.with_columns(new=pl.when(pl.col("clipd") < 0).then(pl.col("b").clip(lower_bound=0)).otherwise(pl.col("clipd"))
                       .round(0).cast(pl.Int32))
    out = G.select("MVT_ID_mvt", pl.col("new").alias("TAXITIME_SEC_mvt"))
    assert out.null_count().sum_horizontal().item() == 0 and int((out["TAXITIME_SEC_mvt"] < 0).sum()) == 0
    out.write_parquet(OUT)
    d = (G["new"].cast(pl.Float64) - G["b"]).to_numpy()
    print(f"base rows {G.height:,}; corrected {C.height:,} (months {MONTHS}); RMS c {np.sqrt(np.mean(c ** 2)):.1f} s, "
          f"mean c {c.mean():+.1f}; rows changed {int((d != 0).sum()):,}; sum d^2 / 670,790 {float((d ** 2).sum()) / 670790:.0f}")
    print(G.filter(pl.col("c") != 0).group_by("ADEP_mvt", "mon").agg(
        pl.len().alias("n"), (pl.col("c") ** 2).mean().sqrt().round(1).alias("rms_c"), pl.col("c").mean().round(1).alias("mean_c"))
        .sort("ADEP_mvt", "mon"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
