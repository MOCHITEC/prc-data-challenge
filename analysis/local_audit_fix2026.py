"""Audit worker, step 4: corrected 2026 values for the frame-pooled EXTRA columns (union pooling).

The production frames compute fl_nm_rate_other and the seq_* runway-sequence columns (pairq = P5 headway per
airport x runway x WTC pair, and excess/burst_* built on it) over the frame they are given: 12 months for
feat_train, Jan+Jul 2026 for feat_rank. The fix for ranking rows: compute them over the UNION of the 2025
training movements and the 2026 ranking movements (parity-masked, labels never read), so 2026 rows get pools of
~14 months, like the 12 months the model was trained on. Everything in prc.features_extra is per departure
airport, so it is run airport by airport on the masked union and only the 2026 rows are kept.

Columns written: every EXTRA column visible to main_flseq_osm_nolirf / main_flseq (fl_*, dup_*, seq_*). Non-pooled
columns come out identical to feat_rank except at Jan 1 2026 00:xx, where Dec 31 2025 now precedes them (as in
training). A check against feat_rank is printed per column.

    POLARS_MAX_THREADS=1 .venv/bin/python analysis/local_audit_fix2026.py
Output: $SCRATCH/agents/audit/feat_rank_extra_union.parquet (MVT_ID_mvt + columns), fix2026.txt
"""
import glob
import os
import sys

import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from local_audit_frame import visible  # noqa: E402
from prc.features import mask_departures  # noqa: E402
from prc.features_extra import EXTRA_COLS, extra_candidate_features  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "audit")
cols = [c for c in EXTRA_COLS if c in visible()]
files = sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]
rk_ids = pl.read_parquet("data/raw/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt"]).filter(
    pl.col("PHASE_mvt") == "DEP")["MVT_ID_mvt"]
parts = []
for apt in ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]:
    u = pl.concat([pl.scan_parquet(f).filter((pl.col("ADEP_mvt") == apt) | (pl.col("ADES_mvt") == apt)).collect()
                   for f in files], how="diagonal_relaxed")
    # keep only this airport's departures and arrivals (extra features are computed per airport anyway)
    u = u.filter(((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt))
                 | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt)))
    ids = u.filter((pl.col("PHASE_mvt") == "DEP") & pl.col("MVT_ID_mvt").is_in(rk_ids.implode()))["MVT_ID_mvt"]
    parts.append(extra_candidate_features(mask_departures(u), ids).select("MVT_ID_mvt", *cols))
    print(apt, u.height, ids.len(), flush=True)
    del u
fx = pl.concat(parts)
fx.write_parquet(os.path.join(OUT, "feat_rank_extra_union.parquet"))
rk = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "mvt_minus_aobt3", *cols])
j = rk.join(fx, on="MVT_ID_mvt", how="left", suffix="__u")
lines = [f"2026 rows {rk.height:,}, union rows matched {j['fl_len__u'].is_not_null().sum():,}"]
for c in cols:
    a, b = j[c].cast(pl.Float64).fill_nan(None), j[c + "__u"].cast(pl.Float64).fill_nan(None)
    same = (((a - b).abs() <= 1e-4 * (1 + a.abs())) | (a.is_null() & b.is_null())).fill_null(False)
    lines.append(f"  {c:24s} differs on {int((~same).sum()):7,d} rows; mean feat_rank {a.mean():.4f} union {b.mean():.4f}")
t = j.with_columns(pl.when(pl.col("mvt_minus_aobt3").is_null()).then(pl.lit("B")).otherwise(pl.lit("A")).alias("lane")) \
    .group_by("ADEP_mvt", "lane").agg((pl.col("fl_nm_rate_other") == 1).mean().alias("eq1_rank"),
                                      (pl.col("fl_nm_rate_other__u") == 1).mean().alias("eq1_union")).sort("lane", "ADEP_mvt")
with pl.Config(tbl_rows=30):
    lines.append(str(t))
msg = "\n".join(lines)
print(msg)
open(os.path.join(OUT, "fix2026.txt"), "w").write(msg + "\n")
