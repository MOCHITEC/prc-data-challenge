"""For departures with a pushback signature and a later stop >= 900 s, find the start / end of that longest later stop
(fixes within STAT_R of the run's first fix, between t_move and take-off). Writes $SCRATCH/pos/dwell_<year>.parquet.
Rows: callsign-joined departures (matched + unmatched) from the *_joined files, plus positional joins from pos/.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_dwell_end.py 2025|2026
"""
import glob, os, sys
import numpy as np
import polars as pl
sys.path.insert(0, "scripts")
import adsb_stage1_feasibility as s1

S = os.environ["SCRATCH"]
yr = sys.argv[1]
out = []
for jf in sorted(glob.glob(f"{S}/v{yr}.*_joined.parquet")):
    tag = os.path.basename(jf).replace("_joined.parquet", "")
    cols = ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_aobt3", "t_block", "t_sched", "icao", "t_move", "t_to", "push", "post_dwell"]
    J = pl.read_parquet(jf, columns=cols).with_columns(pl.lit("cs").alias("src"))
    pf = f"{S}/pos/{tag}_pos.parquet"
    if os.path.exists(pf):
        Pp = pl.read_parquet(pf).filter(pl.col("icao_pos").is_not_null()).select(
            [c for c in cols if c != "icao"] + [pl.col("icao_pos").alias("icao")]).with_columns(pl.lit("pos").alias("src"))
        J = pl.concat([J.filter(pl.col("icao").is_not_null()), Pp.select(J.columns)], how="vertical_relaxed")
    J = J.filter(pl.col("t_move").is_not_null() & (pl.col("push").fill_null(0) >= s1.PUSH_MIN)
                 & (pl.col("post_dwell").fill_null(0) >= s1.POST_DWELL_MAX) & (pl.col("ADEP_mvt") != "LIRF"))
    if J.height == 0:
        continue
    fx = pl.read_parquet(f"{S}/{tag}.parquet").filter(pl.col("icao").is_in(J["icao"].unique().to_list()))
    g = {k: v.sort("t") for k, v in fx.partition_by(["icao", "apt"], as_dict=True).items()}
    for r in J.iter_rows(named=True):
        f = g.get((r["icao"], r["ADEP_mvt"]))
        if f is None:
            continue
        end = r["t_to"] if r["t_to"] is not None else r["t_mvt"] + 120
        f = f.filter(pl.col("ground") & (pl.col("t") > r["t_move"]) & (pl.col("t") < end))
        t, la, lo = f["t"].to_numpy(), f["lat"].to_numpy(), f["lon"].to_numpy()
        best, s0 = None, 0
        while s0 < len(t):
            e = s0
            while e + 1 < len(t) and s1.dist_m(la[s0], lo[s0], la[e + 1], lo[e + 1]) <= s1.STAT_R:
                e += 1
            if best is None or t[e] - t[s0] > best[1] - best[0]:
                best = (t[s0], t[e], float(s1.dist_m(la[0], lo[0], la[s0], lo[s0])) if len(t) else None)
            s0 = e + 1
        if best:
            out.append({**r, "dw_start": float(best[0]), "dw_end": float(best[1]), "dw_dist_from_push_m": best[2]})
R = pl.DataFrame(out, infer_schema_length=None)
R.write_parquet(f"{S}/pos/dwell_{yr}.parquet")
print(yr, R.height, "rows")
