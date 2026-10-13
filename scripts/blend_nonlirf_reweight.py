"""Matched-lane blend with the non-LIRF inner split re-weighted (docs/research_2026-10-08_blend).

Production:  ((0.5*a2o + 0.5*pao) + p2o + c2 + m4) / 4 on matched rows.
This script: same, except on matched rows OUTSIDE LIRF the inner split a2o:pao is INNER_A2O:(1-INNER_A2O), default
0.2:0.8. LIRF matched rows and all unmatched rows are returned unchanged from the production formula.
Evidence (2025 OOF, leave-one-fold-out, without c2): inner split alone -0.12 s whole frame, every fold negative.

Usage:
  python scripts/blend_nonlirf_reweight.py FRAME a2o.parquet pao.parquet p2o.parquet c2.parquet m4.parquet OUT [INNER_A2O]
  FRAME: a parquet with MVT_ID_mvt, ADEP_mvt, mvt_minus_aobt3 (e.g. the final feat_rank).
  Members: parquet files with MVT_ID_mvt and one prediction column (the first non-id column is used).
  OUT gets MVT_ID_mvt, pred_prod, pred (matched rows only; merge into the full pipeline before the ADS-B layer).
"""
import sys

import numpy as np
import polars as pl


def _member(path: str, name: str) -> pl.DataFrame:
    d = pl.read_parquet(path)
    col = next(c for c in d.columns if c != "MVT_ID_mvt")
    return d.select("MVT_ID_mvt", pl.col(col).cast(pl.Float64).alias(name))


def main(argv: list[str]) -> None:
    frame, a2o, pao, p2o, c2, m4, out = argv[:7]
    inner = float(argv[7]) if len(argv) > 7 else 0.2
    if not 0.0 <= inner <= 1.0:
        raise ValueError(f"INNER_A2O must be in [0, 1], got {inner}")
    f = pl.read_parquet(frame, columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_minus_aobt3"]).filter(
        pl.col("mvt_minus_aobt3").is_not_null()
    )
    d = f
    for path, name in ((a2o, "a2o"), (pao, "pao"), (p2o, "p2o"), (c2, "c2"), (m4, "m4")):
        d = d.join(_member(path, name), on="MVT_ID_mvt", how="left")
    miss = {c: int(d[c].is_null().sum()) for c in ("a2o", "pao", "p2o", "c2", "m4")}
    if any(miss.values()):
        raise AssertionError(f"members missing matched rows: {miss}")
    rest = pl.col("p2o") + pl.col("c2") + pl.col("m4")
    prod = ((0.5 * pl.col("a2o") + 0.5 * pl.col("pao")) + rest) / 4.0
    new = ((inner * pl.col("a2o") + (1.0 - inner) * pl.col("pao")) + rest) / 4.0
    d = d.with_columns(prod.alias("pred_prod"))
    d = d.with_columns(pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.col("pred_prod")).otherwise(new).alias("pred"))
    moved = (d["pred"] - d["pred_prod"]).to_numpy()
    print(f"matched rows {d.height}, moved {(np.abs(moved) > 1e-9).sum()}, RMS change {np.sqrt(np.mean(moved**2)):.2f} s")
    d.select("MVT_ID_mvt", "pred_prod", "pred").write_parquet(out)


if __name__ == "__main__":
    main(sys.argv[1:])
