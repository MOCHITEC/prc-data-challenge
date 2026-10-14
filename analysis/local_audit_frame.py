"""Audit worker, step 1: which production features depend on the frame they are computed in.

Rebuilds the full departure feature table (prc.features.build_departure_features + weather join) for the 2025
Jan+Jul departures from a frame holding ONLY the Jan+Jul 2025 raw files - the ranking.parquet layout: two
months, a Feb-Jun gap, departure BLOCK/TAXITIME masked (as for every build). Then compares every column the
production configs see, row by row, against feat_train.parquet (built inside the 12-month frame).
A column that is not identical is frame-dependent; its 2026 values are computed the 2-month way.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_audit_frame.py
Outputs: $SCRATCH/agents/audit/feat_jj2m.parquet (rebuilt rows), frame_diff.txt
"""
import glob
import os
import sys

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, "src")
from prc.features import CATEGORICALS, build_departure_features  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import visible_numeric_columns  # noqa: E402
from prc.weather import join_weather  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "audit")
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def visible():
    spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf.yaml"))
    for k in ("name", "use_weather"):
        spec.pop(k, None)
    cfg = train_config_from_spec(spec)
    h = pl.read_parquet("data/work_flseq/feat_train.parquet", n_rows=5)
    return visible_numeric_columns(h, cfg)


def main():
    num = visible()
    cache = os.path.join(OUT, "feat_jj2m.parquet")
    if not os.path.exists(cache):
        raw = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_2025-0[17]-01_*.parquet"))],
                        how="diagonal_relaxed")
        f = build_departure_features(raw, proxy_mode=None)
        f = join_weather(f, pl.read_parquet("data/ref/metar_hourly.parquet"))
        f.select(["MVT_ID_mvt", *num, *CATEGORICALS]).write_parquet(cache)
        del raw, f
    new = pl.read_parquet(cache)
    old = (pl.scan_parquet("data/work_flseq/feat_train.parquet")
           .filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
           .select(["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", *num, *[c for c in CATEGORICALS if c != "ADEP_mvt"]])
           .collect())
    log(f"rows: 12m-frame {old.height:,}  2m-frame {new.height:,}")
    j = old.join(new, on="MVT_ID_mvt", how="inner", suffix="__2m")
    log(f"joined {j.height:,}")
    res = []
    for c in num + [c for c in CATEGORICALS if c != "ADEP_mvt"]:
        a, b = j[c], j[c + "__2m"]
        if a.dtype.is_numeric():
            a, b = a.cast(pl.Float64).fill_nan(None), b.cast(pl.Float64).fill_nan(None)
            same = ((a - b).abs() <= 1e-4 * (1 + a.abs())) | (a.is_null() & b.is_null())
        else:
            same = (a == b) | (a.is_null() & b.is_null())
        same = same.fill_null(False)
        nd = int((~same).sum())
        if nd:
            d = j.filter(~same)
            ex = d.select("ADEP_mvt", "MVT_TIME_UTC_mvt", c, c + "__2m").head(3).rows()
            res.append((c, nd, nd / j.height, ex))
    log(f"\ncolumns not identical between the 12-month and the 2-month frame ({len(res)} of {len(num) + 9}):")
    for c, nd, sh, ex in sorted(res, key=lambda r: -r[1]):
        log(f"  {c:30s} differs on {nd:8,d} rows ({sh:.4f})  e.g. {ex}")
    # where in time do the non-pooled differences sit (edge effects)?
    for c, nd, sh, ex in res:
        if c in ("fl_nm_rate_other", "seq_hw_prev_pairq", "seq_hw_excess") or nd < 5:
            continue
        a, b = j[c].cast(pl.Float64).fill_nan(None), j[c + "__2m"].cast(pl.Float64).fill_nan(None)
        same = (((a - b).abs() <= 1e-4 * (1 + a.abs())) | (a.is_null() & b.is_null())).fill_null(False)
        d = j.filter(~same).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%m-%d").alias("md"))
        log(f"  {c}: days {d['md'].value_counts().sort('count', descending=True).head(6).rows()}")
    with open(os.path.join(OUT, "frame_diff.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
