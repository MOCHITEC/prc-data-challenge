"""Learned-residual features for departures that only the positional join reached (local).

For each day: copy the *_joined file, fill the stage-1 columns of rows without a callsign join from pos/<tag>_pos.parquet
(icao_pos and its stand_departure output), mark them join = "cs" so adsb_learn_features.day_features accepts them,
and run day_features against a temp dir that holds the modified joined file plus a symlink to the fixes file.
Keeps only the positional rows. Writes $SCRATCH/learn/feat_pos_<year>.parquet.

    POLARS_MAX_THREADS=1 .venv/bin/python analysis/local_posjoin_learnfeat.py 2025|2026
"""
import os
import sys

import polars as pl

sys.path.insert(0, "scripts")
import adsb_learn_features as L  # noqa: E402

yr = sys.argv[1]
S = L.SCR
TMP = os.path.join(S, "pos", "tmp_scr")
os.makedirs(TMP, exist_ok=True)
FILL = ["ground_fix", "t_first", "t_dep1", "t_move", "t_move_lo", "t_to", "n_ground", "stand_dur", "dep1_gs",
        "gap_before_dep1", "opens_with_landing", "stand_rule", "stand_nfix", "push", "post_dwell"]
out = []
for day in L.DAYS[yr]:
    tag = L.tag_of(day)
    pf = os.path.join(S, "pos", f"{tag}_pos.parquet")
    if not os.path.exists(pf):
        continue
    P = pl.read_parquet(pf).filter(pl.col("icao_pos").is_not_null())
    if P.height == 0:
        continue
    J = pl.read_parquet(os.path.join(S, f"{tag}_joined.parquet"))
    sub = P.select(["MVT_ID_mvt", pl.col("icao_pos").alias("icao")] + [c for c in FILL if c in P.columns])
    keep_cols = [c for c in sub.columns if c != "MVT_ID_mvt"]
    J2 = (J.join(sub.rename({c: c + "__p" for c in keep_cols}), on="MVT_ID_mvt", how="left")
          .with_columns([pl.when(pl.col("icao__p").is_not_null()).then(pl.col(c + "__p")).otherwise(pl.col(c)).alias(c)
                         for c in keep_cols if c in J.columns])
          .with_columns(pl.when(pl.col("icao__p").is_not_null()).then(pl.lit("cs")).otherwise(pl.col("join")).alias("join"),
                        pl.when(pl.col("icao__p").is_not_null()).then(True).otherwise(pl.col("seen")).alias("seen"))
          .drop([c + "__p" for c in keep_cols]))
    J2.write_parquet(os.path.join(TMP, f"{tag}_joined.parquet"))
    lk = os.path.join(TMP, f"{tag}.parquet")
    if not os.path.exists(lk):
        os.symlink(os.path.join(S, f"{tag}.parquet"), lk)
    L.SCR = TMP
    try:
        f = L.day_features(day)
    finally:
        L.SCR = S
    out.append(f.filter(pl.col("MVT_ID_mvt").is_in(P["MVT_ID_mvt"].implode())))
    os.remove(os.path.join(TMP, f"{tag}_joined.parquet"))
R = pl.concat(out, how="diagonal_relaxed")
R.write_parquet(os.path.join(L.OUT, f"feat_pos_{yr}.parquet"))
print(yr, R.shape, dict(R.group_by("ADEP_mvt").len().sort("ADEP_mvt").iter_rows()))
