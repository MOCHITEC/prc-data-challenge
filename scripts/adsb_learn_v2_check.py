"""Coverage-class check of the current learned model (section 9.0 of adsb-learned-residual.md).

Reads `$SCRATCH/learn/cv_rows_pa.parquet` (section 8 CV) and `$SCRATCH/learn/feat_2025.parquet`. Prints per class
K0-K3 the rows, base and learned MSE, relative reduction and share of the total gain; for K2 also the stage-1
reason and the first-fix speed and position.

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_v2_check.py
"""

from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_learn_features as L

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)


def classes(D: pl.DataFrame) -> pl.DataFrame:
    return D.with_columns(
        pl.when(~pl.col("has")).then(pl.lit("K0 no ADS-B"))
        .when(~pl.col("ground_fix").fill_null(False)).then(pl.lit("K1 air only"))
        .when(pl.col("t_move").is_null()).then(pl.lit("K2 mid-taxi first"))
        .otherwise(pl.lit("K3 stand seen")).alias("cls"))


def main() -> None:
    L.mem_gate()
    R = pl.read_parquet(os.path.join(L.OUT, "cv_rows_pa.parquet"))
    F = pl.read_parquet(os.path.join(L.OUT, "feat_2025.parquet")).select(
        "MVT_ID_mvt", "ground_fix", "t_move", "stand_rule", "dep1_gs", "t_dep1", "t_mvt", "post_n")
    D = classes(R.join(F, on="MVT_ID_mvt", how="left"))
    D = D.with_columns(((pl.col("y") - pl.col("b")) ** 2).alias("se0"),
                       ((pl.col("y") - pl.col("b") - pl.col("c_learned")) ** 2).alias("se1"))
    tot = float((D["se0"] - D["se1"]).sum())
    T = D.group_by("cls").agg(pl.len().alias("n"), (pl.len() / D.height).alias("share_rows"),
                              pl.col("se0").mean().alias("mse_base"), pl.col("se1").mean().alias("mse_learned"),
                              ((pl.col("se0") - pl.col("se1")).sum() / tot).alias("share_gain")).with_columns(
        (1 - pl.col("mse_learned") / pl.col("mse_base")).alias("rel_red")).sort("cls")
    print(T)
    print("per airport, K2 vs K3 relative reduction:")
    print(D.filter(pl.col("cls").str.starts_with("K2") | pl.col("cls").str.starts_with("K3"))
          .group_by("ADEP_mvt", "cls").agg(pl.len(), (1 - pl.col("se1").mean() / pl.col("se0").mean()).alias("rel_red"))
          .pivot(on="cls", index="ADEP_mvt", values=["len", "rel_red"]).sort("ADEP_mvt"))
    K2 = D.filter(pl.col("cls").str.starts_with("K2"))
    print("K2 stage-1 stand_rule:", dict(K2.group_by("stand_rule").len().iter_rows()))
    print("K2 first-departure-fix gs (kt) quantiles:",
          {q: K2["dep1_gs"].quantile(q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
          "| null", K2["dep1_gs"].null_count())
    print("K2 MVT - t_dep1 (s) quantiles:",
          {q: (K2["t_mvt"] - K2["t_dep1"]).quantile(q) for q in (0.1, 0.5, 0.9)},
          "| post-anchor fixes non-null", int(K2["post_n"].is_not_null().sum()), "of", K2.height)
    r = {row["cls"]: row for row in T.iter_rows(named=True)}
    k2, k3 = r["K2 mid-taxi first"], r["K3 stand seen"]
    under = k2["rel_red"] < 0.5 * k3["rel_red"] and k2["share_rows"] >= 0.05
    print(f"\nCRITERION (9.0): K2 rel_red {k2['rel_red']:.3f} vs 0.5 x K3 {0.5 * k3['rel_red']:.3f}; "
          f"K2 rows {k2['share_rows']:.3f} -> {'UNDER-SERVED: V3 runs' if under else 'not under-served: V3 not run'}")


if __name__ == "__main__":
    main()
