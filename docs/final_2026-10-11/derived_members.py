"""Derived member files used by the final build (v4): run from /home/user/pb-work/outputs/final or pass --dir.

  pao_l6_cattight_nolirf.parquet = pao_l6_cattight everywhere except LIRF rows, which keep the production pao
                                   (main_flseq_osm_nolirf_l2_pa_opt_l6).
  mlp_l6_5seeds.parquet          = round((3 * mlp_l6 [seeds 42-44] + 2 * mlp_l6_s45 [seeds 45-46]) / 5) as int32.
  CatBoost 5-seed mean is taken inside assemble_step.py: --cb cb_l2 cb_l2_s43 cb_l2_s44 cb_l2_s45 cb_l2_s45
  (cb_l2_s45.parquet is already the mean of seeds 45 and 46, so listing it twice gives each seed weight 1/5).
"""
import argparse

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/home/user/pb-work/outputs/final")
ap.add_argument("--rank", default="/home/user/prc-work/data/final/final_ranking.parquet")
a = ap.parse_args()
D = a.dir.rstrip("/") + "/"
adep = pl.read_parquet(a.rank, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "ADEP_mvt")

cat = pl.read_parquet(D + "pao_l6_cattight.parquet")
prod = pl.read_parquet(D + "main_flseq_osm_nolirf_l2_pa_opt_l6.parquet").rename({"TAXITIME_SEC_mvt": "prod"})
j = cat.join(prod, on="MVT_ID_mvt", how="left").join(adep, on="MVT_ID_mvt", how="left")
out = j.with_columns(pl.when(pl.col("ADEP_mvt") == "LIRF").then(pl.col("prod")).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt")).select(cat.columns)
assert out.null_count().sum_horizontal().item() == 0
out.write_parquet(D + "pao_l6_cattight_nolirf.parquet")

m3 = pl.read_parquet(D + "main_flseq_osm_nolirf_l2_mlp_l6.parquet").select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("m3"))
m45 = pl.read_parquet(D + "mlp_l6_s45.parquet").select("MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("m45"))
o = m3.join(m45, on="MVT_ID_mvt", how="left").with_columns(
    pl.coalesce((3 * pl.col("m3") + 2 * pl.col("m45")) / 5, pl.col("m3")).round(0).cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
o.write_parquet(D + "mlp_l6_5seeds.parquet")

# a2o (cat_tight) 5-seed flfix file: pred_fix and pred_orig = (3 * seeds 42-44 + 2 * seeds 45-46) / 5 (float, not rounded).
a3 = pl.read_parquet(D + "a2o_cattight_flfix.parquet")
a2 = pl.read_parquet(D + "a2o_cattight_s45_flfix.parquet").select("MVT_ID_mvt", pl.col("pred_fix").alias("fix45"), pl.col("pred_orig").alias("orig45"))
j = a3.join(a2, on="MVT_ID_mvt", how="left")
assert j["fix45"].null_count() == 0
j.with_columns(((3 * pl.col("pred_fix") + 2 * pl.col("fix45")) / 5).alias("pred_fix"),
               ((3 * pl.col("pred_orig") + 2 * pl.col("orig45")) / 5).alias("pred_orig")).select(a3.columns).write_parquet(D + "a2o_cattight_5seeds_flfix.parquet")
print("written pao_l6_cattight_nolirf.parquet, mlp_l6_5seeds.parquet and a2o_cattight_5seeds_flfix.parquet in", D)
