"""Shift worker, task 3c: which pooled feature carries the 2-month-pooling cost, and is it stable across seeds.

Follow-up of local_shift_pool.py (same proxy models, same emulation: Jan+Jul 2025 rows predicted with EXTRA_COLS
pooled over the Jan+Jul 2025 frame instead of the 12-month frame the model was trained with). Arms:
  all  every pooled EXTRA col from the 2-month frame
  fl   only fl_nm_rate_other
  seq  only the seq_* columns (pairq, excess, burst_*, hw means)
Two model seeds (different 45 % lane-A subsample and bagging seed). Also excludes LIRF in a summary line, because
v40 handles LIRF with its own overlays and the proxy is least faithful there.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_shift_pool_ablate.py
Outputs: $SCRATCH/agents/shift/pool_extra_2m.parquet, pool_ablate.parquet, pool_ablate_report.txt
"""
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from local_shift_pool import catcode, mat  # noqa: E402
from prc.features import CATEGORICALS, mask_departures  # noqa: E402
from prc.features_extra import EXTRA_COLS, extra_candidate_features  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import unmatched_columns, visible_numeric_columns  # noqa: E402

SCR = os.environ["SCRATCH"]
OUT = os.path.join(SCR, "agents", "shift")
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def main() -> None:
    spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf.yaml"))
    for k in ("name", "use_weather", "wx_codes", "wind_dir", "gust_repair", "coverage_gate", "coverage_allow", "note"):
        spec.pop(k, None)
    cfg = train_config_from_spec(spec)
    head = pl.read_parquet("data/work_flseq/feat_rank.parquet", n_rows=10)
    num = [c for c in visible_numeric_columns(head, cfg) if not c.startswith("noise_control")]
    numB = unmatched_columns(num)
    cats = [c for c in CATEGORICALS if c in head.columns]
    pooled = [c for c in EXTRA_COLS if c in num]
    arms = {"all": pooled, "fl": ["fl_nm_rate_other"], "seq": [c for c in pooled if c.startswith("seq_")]}

    need = list(dict.fromkeys(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"] + num + cats))
    f32 = [pl.col(c).cast(pl.Float32) for c in num]
    tr = pl.scan_parquet("data/work_flseq/feat_train.parquet").select(need).with_columns(
        *f32, pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect()
    jj = tr.filter(pl.col("mon").is_in([1, 7]))
    fit = tr.filter(~pl.col("mon").is_in([1, 7]))
    del tr
    cache = os.path.join(OUT, "pool_extra_2m.parquet")
    if not os.path.exists(cache):
        raw = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_2025-0[17]-01_*.parquet"))],
                        how="vertical_relaxed")
        extra_candidate_features(mask_departures(raw), jj["MVT_ID_mvt"]).select(
            "MVT_ID_mvt", *[pl.col(c).cast(pl.Float32) for c in pooled]).write_parquet(cache)
        del raw
    ex = pl.read_parquet(cache)
    variants = {"base": jj}
    for a, cs in arms.items():
        variants[a] = jj.drop(cs).join(ex.select("MVT_ID_mvt", *cs), on="MVT_ID_mvt", how="left").select(jj.columns)

    res = []
    for seed in (0, 1):
        for lane in ("A", "B"):
            cols = num if lane == "A" else numB
            sel = pl.col("mvt_minus_aobt3").is_not_null() if lane == "A" else pl.col("mvt_minus_aobt3").is_null()
            f = fit.filter(sel)
            if lane == "A":
                f = f.filter((pl.col("MVT_ID_mvt").hash(11 + seed) % 100) < 45)
            vs = {k: v.filter(sel) for k, v in variants.items()}
            keys = list(vs)
            Cs = catcode([f] + [vs[k] for k in keys], cats)
            Xf = mat(f, cols, Cs[0])
            off_f = f["mvt_minus_aobt3"].to_numpy() if lane == "A" else 0.0
            p = dict(objective="regression", learning_rate=0.08 if lane == "A" else 0.05,
                     num_leaves=255 if lane == "A" else 127, min_data_in_leaf=100 if lane == "A" else 50,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                     num_threads=2, seed=seed, max_bin=255)
            cif = list(range(len(cols), len(cols) + len(cats)))
            bst = lgb.train(p, lgb.Dataset(Xf, f["TAXITIME_SEC_mvt"].to_numpy() - off_f, categorical_feature=cif),
                            700 if lane == "A" else 1000)
            del Xf
            v0 = vs["base"]
            off_v = v0["mvt_minus_aobt3"].to_numpy() if lane == "A" else 0.0
            d = {"MVT_ID_mvt": v0["MVT_ID_mvt"], "ADEP_mvt": v0["ADEP_mvt"], "mon": v0["mon"],
                 "y": v0["TAXITIME_SEC_mvt"].cast(pl.Float64)}
            for i, k in enumerate(keys):
                d[f"p_{k}"] = bst.predict(mat(vs[k], cols, Cs[i + 1])) + off_v
            res.append(pl.DataFrame(d).with_columns(pl.lit(lane).alias("lane"), pl.lit(seed).alias("seed")))
            log(f"seed {seed} lane {lane} done")
    R = pl.concat(res)
    R.write_parquet(os.path.join(OUT, "pool_ablate.parquet"))
    for seed in (0, 1):
        S = R.filter(pl.col("seed") == seed)
        N = S.height
        for a in arms:
            dd = ((S["y"] - S[f"p_{a}"]) ** 2 - (S["y"] - S["p_base"]) ** 2)
            S2 = S.with_columns(dd.alias("dd"), (S[f"p_{a}"] - S["p_base"]).alias("sh"))
            parts = []
            for lane in ("A", "B"):
                for m in (1, 7):
                    x = S2.filter((pl.col("lane") == lane) & (pl.col("mon") == m))
                    parts.append(f"{lane}{m}:{x['dd'].sum() / N:+.0f}")
            xa = S2.filter((pl.col("lane") == "A") & (pl.col("ADEP_mvt") != "LIRF"))
            apt = xa.group_by("ADEP_mvt").agg((pl.col("dd").sum() / N).alias("g")).sort("ADEP_mvt")
            log(f"seed {seed} arm {a:4s} dMSE all {S2['dd'].sum() / N:+7.1f} | " + " ".join(parts)
                + f" | laneA exLIRF {xa['dd'].sum() / N:+.1f} (airports >0: {int((apt['g'] > 0).sum())}/9)"
                + f" | laneA mean shift {S2.filter(pl.col('lane') == 'A')['sh'].mean():+.2f}")
    with open(os.path.join(OUT, "pool_ablate_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
