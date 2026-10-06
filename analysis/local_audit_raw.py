"""Audit worker, step 0: raw-file parity between 2025 Jan+Jul (training files) and 2026 (ranking.parquet).

Per PHASE x airport: row counts, null share of every raw column, uniqueness of MVT_ID, whether ranking rows are
all in the submission template, and arrival label availability (BLOCK/TAXITIME of arrivals). Read-only.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_audit_raw.py
Output: $SCRATCH/agents/audit/raw_parity.txt
"""
import glob
import os

import polars as pl

OUT = os.path.join(os.environ["SCRATCH"], "agents", "audit")
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_2025-0[17]-01_*.parquet"))],
               how="diagonal_relaxed").with_columns(pl.lit("2025").alias("yr"))
rk = pl.read_parquet("data/raw/ranking.parquet").with_columns(pl.lit("2026").alias("yr"))
sub = pl.read_parquet("data/raw/submitting.parquet")
log(f"train JJ rows {tr.height:,}  rank rows {rk.height:,}  sub rows {sub.height:,}")
log(f"columns only in train: {sorted(set(tr.columns) - set(rk.columns))}; only in rank: {sorted(set(rk.columns) - set(tr.columns))}")
log(f"sub columns {sub.columns}")
for nm, f in (("train", tr), ("rank", rk)):
    log(f"{nm}: MVT_ID unique {f['MVT_ID_mvt'].n_unique() == f.height}; phases {f['PHASE_mvt'].value_counts().sort('PHASE_mvt').rows()}")
    log(f"{nm}: MVT_TIME range {f['MVT_TIME_UTC_mvt'].min()} .. {f['MVT_TIME_UTC_mvt'].max()}")
rk_dep = rk.filter(pl.col("PHASE_mvt") == "DEP")
ids = set(sub["MVT_ID_mvt"].cast(pl.Int64).to_list())
log(f"rank DEP rows {rk_dep.height:,}; in sub {rk_dep['MVT_ID_mvt'].cast(pl.Int64).is_in(list(ids)).sum():,}; "
    f"sub ids that are rank DEP {len(ids & set(rk_dep['MVT_ID_mvt'].cast(pl.Int64).to_list())):,}")
both = pl.concat([tr, rk], how="diagonal_relaxed")
cols = [c for c in rk.columns if c not in ("yr",)]
for ph in ("DEP", "ARR"):
    log(f"\n=== {ph}: null share per column, 2025 Jan+Jul vs 2026 (all airports); flag |diff| > 0.02")
    b = both.filter(pl.col("PHASE_mvt") == ph)
    t = b.group_by("yr").agg([pl.col(c).is_null().mean().alias(c) for c in cols]).sort("yr")
    for c in cols:
        a, r = t[c][0], t[c][1]
        flag = " <<<" if abs(a - r) > 0.02 else ""
        log(f"  {c:24s} {a:.4f} {r:.4f}{flag}")
    key = "ADEP_mvt" if ph == "DEP" else "ADES_mvt"
    log(f"--- {ph} per airport counts and selected null shares")
    sel = ["FLIGHT_ID_mvt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
           "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "LOBT_flt", "EOBT_1_flt"]
    t = (b.group_by(key, "yr").agg(pl.len().alias("n"), *[pl.col(c).is_null().mean().round(4).alias(c) for c in sel])
         .sort(key, "yr"))
    with pl.Config(tbl_rows=40, tbl_width_chars=250, tbl_cols=20):
        log(t)
# arrivals: block == sched copy share, taxitime distribution
a = both.filter(pl.col("PHASE_mvt") == "ARR")
t = (a.group_by("ADES_mvt", "yr").agg(
    pl.col("TAXITIME_SEC_mvt").median().alias("taxi_in_med"),
    pl.col("TAXITIME_SEC_mvt").mean().alias("taxi_in_mean"),
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 6).mean().alias("blk_copy_sched"))
    .sort("ADES_mvt", "yr"))
with pl.Config(tbl_rows=40):
    log(t)
with open(os.path.join(OUT, "raw_parity.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
