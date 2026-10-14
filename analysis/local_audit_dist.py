"""Audit worker, step 2: per airport x lane distribution of every production feature, 2025 Jan+Jul vs 2026.

Sets compared, per (ADEP, lane A = NM proxy present / B = absent, feature):
  25_12m  feat_train rows of Jan+Jul 2025 (built in the 12-month frame) - what the model was trained on
  25_2m   the same rows rebuilt in a Jan+Jul-only frame (local_audit_frame.py) - the ranking-style computation
  26      feat_rank (2026 Jan+Jul)
Encodings (ref_*/arh_*/apt_med, fitted on ALL labelled 2025 departures as in production, self-inclusive) and the
OSM distance are attached to all three; for 2025 rows also an out-of-sample variant (fitted on the 10 other
months) to show what self-inclusion does to the training rows' values.
Statistic: null share, ==0 share, ==1 share, mean, p10/p50/p90. Flags where 25_12m -> 26 moves by more than
0.03 (null), 0.05 (0/1 shares) or 0.25 pooled SD (mean), and whether the 25_12m -> 25_2m move explains it.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_audit_dist.py
Outputs: $SCRATCH/agents/audit/dist_stats.parquet, dist_report.txt
"""
import os
import sys

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from local_audit_frame import visible  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import ENCODING_COLS, attach_encodings, fit_encodings, osm_distance_features  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "audit")
LABEL = "TAXITIME_SEC_mvt"
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def stats(df: pl.DataFrame, cols: list[str], tag: str) -> pl.DataFrame:
    df = df.with_columns(pl.when(pl.col("mvt_minus_aobt3").is_null()).then(pl.lit("B")).otherwise(pl.lit("A"))
                         .alias("lane"))
    out = []
    for c in cols:
        x = pl.col(c).cast(pl.Float64).fill_nan(None)
        out.append(df.group_by("ADEP_mvt", "lane").agg(
            pl.lit(c).alias("feat"), pl.len().alias("n"), x.is_null().mean().alias("null"),
            (x == 0).mean().alias("eq0"), (x == 1).mean().alias("eq1"), x.mean().alias("mean"), x.std().alias("sd"),
            x.quantile(0.1).alias("p10"), x.median().alias("p50"), x.quantile(0.9).alias("p90")))
    return pl.concat(out).with_columns(pl.lit(tag).alias("set"))


def main():
    spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf.yaml"))
    for k in ("name", "use_weather"):
        spec.pop(k, None)
    cfg = train_config_from_spec(spec)
    num = visible()
    keys = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local", "AIRCRAFT_OPERATOR_flt",
            "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", LABEL]
    lab = (pl.scan_parquet("data/work_flseq/feat_train.parquet").select(keys)
           .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite()).collect())
    enc_all = fit_encodings(lab, cfg)
    enc_10 = fit_encodings(lab.filter(~pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])), cfg)
    del lab
    cols = list(dict.fromkeys(keys + num))
    t12 = (pl.scan_parquet("data/work_flseq/feat_train.parquet").select(cols)
           .filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])).collect())
    t2 = t12.select(keys).join(pl.read_parquet(os.path.join(OUT, "feat_jj2m.parquet")).select(
        ["MVT_ID_mvt", *[c for c in num if c not in keys]]), on="MVT_ID_mvt", how="left")
    rk = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=[c for c in cols if c != LABEL])
    na = cfg.fe_osm_null_airports
    fe = num + ENCODING_COLS + ["osm_stand_rwy_km"]
    S = [stats(osm_distance_features(attach_encodings(t12, enc_all), na), fe, "25_12m"),
         stats(osm_distance_features(attach_encodings(t2, enc_all), na), fe, "25_2m"),
         stats(attach_encodings(t12, enc_10), ENCODING_COLS, "25_oos"),
         stats(osm_distance_features(attach_encodings(rk, enc_all), na), fe, "26")]
    S = pl.concat(S, how="diagonal")
    S.write_parquet(os.path.join(OUT, "dist_stats.parquet"))
    idx = ["ADEP_mvt", "lane", "feat"]
    w = None
    for tag in ("25_12m", "25_2m", "25_oos", "26"):
        s = S.filter(pl.col("set") == tag).select(idx + [pl.col(c).alias(f"{c}_{tag}") for c in
                                                         ("n", "null", "eq0", "eq1", "mean", "sd", "p50")])
        w = s if w is None else w.join(s, on=idx, how="left")
    sd = ((pl.col("sd_25_12m") + pl.col("sd_26")) / 2).replace(0, None)
    w = w.with_columns(
        (pl.col("null_26") - pl.col("null_25_12m")).alias("d_null"),
        (pl.col("eq0_26") - pl.col("eq0_25_12m")).alias("d_eq0"),
        (pl.col("eq1_26") - pl.col("eq1_25_12m")).alias("d_eq1"),
        ((pl.col("mean_26") - pl.col("mean_25_12m")) / sd).alias("d_mean_sd"),
        (pl.col("null_25_2m") - pl.col("null_25_12m")).alias("a_null"),
        (pl.col("eq0_25_2m") - pl.col("eq0_25_12m")).alias("a_eq0"),
        (pl.col("eq1_25_2m") - pl.col("eq1_25_12m")).alias("a_eq1"),
        ((pl.col("mean_25_2m") - pl.col("mean_25_12m")) / sd).alias("a_mean_sd"),
    )
    flag = ((pl.col("d_null").abs() > 0.03) | (pl.col("d_eq0").abs() > 0.05) | (pl.col("d_eq1").abs() > 0.05)
            | (pl.col("d_mean_sd").abs() > 0.25))
    big = w.filter(flag & (pl.col("n_26") >= 200)).sort("feat", "lane", "ADEP_mvt")
    show = ["feat", "lane", "ADEP_mvt", "n_26", "null_25_12m", "null_26", "eq0_25_12m", "eq0_26", "eq1_25_12m",
            "eq1_26", "mean_25_12m", "mean_25_2m", "mean_26", "d_mean_sd", "a_mean_sd"]
    log(f"flagged (feature, lane, airport) cells: {big.height} of {w.height}")
    log("flag counts per feature:")
    with pl.Config(tbl_rows=200, tbl_width_chars=250, tbl_cols=20, float_precision=3):
        log(big.group_by("feat").agg(pl.len().alias("cells"), pl.col("ADEP_mvt").str.join(",").alias("where"),
                                     pl.col("d_mean_sd").abs().max().alias("max|d_mean_sd|"),
                                     pl.col("a_mean_sd").abs().max().alias("max|a_mean_sd|"))
            .sort("cells", descending=True))
        log(big.select(show))
        log("\nencodings: in-sample (as trained) vs out-of-sample (10 other months) for 2025 JJ rows, vs 2026:")
        e = w.filter(pl.col("feat").is_in(ENCODING_COLS)).group_by("feat", "lane").agg(
            *[(pl.col(f"{c}_{t}") * pl.col("n_25_12m")).sum().truediv(pl.col("n_25_12m").sum()).alias(f"{c}_{t}")
              for c in ("null", "mean") for t in ("25_12m", "25_oos")],
            *[(pl.col(f"{c}_26") * pl.col("n_26")).sum().truediv(pl.col("n_26").sum()).alias(f"{c}_26")
              for c in ("null", "mean")]).sort("lane", "feat")
        log(e)
    with open(os.path.join(OUT, "dist_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
