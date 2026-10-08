"""covw step 1: ADS-B coverage descriptors per airport x month (2025 all 12 months, 2026 Jan/Jul) and, per airport,
which 2025 months look most like Jan 2026 / Jul 2026 (label-free, read-only).

Descriptors (over departures in the ADS-B feature files, i.e. icao matched): share with ground_fix, share with stand
detected (t_push non-null, stand_dur non-null), medians among ground-fix rows of post_n, n_ground, n_win,
post_max_gap and fixes per minute of the post-anchor window (post_n / (t_glast - t_anchor)). Also the ADS-B match
share (feature rows / departures in the raw files). Similarity: standardised Euclidean distance (per descriptor,
z-scored over the 14 airport-months of that airport... no: scaled by the pooled across-airport-month SD) between each
2025 month and each 2026 month, on the descriptor vector.

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage .venv/bin/python analysis/local_covw_coverage.py
Writes $SCRATCH/agents/covw/coverage.parquet, similarity.parquet.
"""
import glob
import os

import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
LE = os.path.join(S, "learn")
OUTD = os.path.join(S, "agents", "covw")
os.makedirs(OUTD, exist_ok=True)
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)
FILES = ["feat_2025.parquet", "feat_2025fmj.parquet", "feat_2025aso.parquet", "feat_2025novdec.parquet", "feat_2026.parquet"]
COLS = ["MVT_ID_mvt", "ADEP_mvt", "day", "ground_fix", "post_n", "n_ground", "n_win", "stand_dur", "t_push",
        "post_max_gap", "t_anchor", "t_glast"]
DESC = ["gf", "stand", "push", "l_post_n", "l_n_ground", "l_n_win", "l_gap", "l_rate"]


def coverage_rows(files=FILES) -> pl.DataFrame:
    F = pl.concat([pl.read_parquet(os.path.join(LE, f), columns=COLS) for f in files], how="diagonal_relaxed")
    return F.with_columns(pl.col("day").str.slice(0, 7).alias("ym"),
                          (pl.col("post_n") / ((pl.col("t_glast") - pl.col("t_anchor")) / 60.0).clip(lower_bound=1.0)).alias("rate"))


def main():
    F = coverage_rows()
    gf = pl.col("ground_fix").fill_null(False)
    A = F.group_by("ADEP_mvt", "ym").agg(
        pl.len().alias("n_adsb"), gf.mean().alias("gf"), pl.col("stand_dur").is_not_null().mean().alias("stand"),
        pl.col("t_push").is_not_null().mean().alias("push"),
        pl.col("post_n").filter(gf).median().alias("post_n"), pl.col("n_ground").filter(gf).median().alias("n_ground"),
        pl.col("n_win").median().alias("n_win"), pl.col("post_max_gap").filter(gf).median().alias("gap"),
        pl.col("rate").filter(gf).median().alias("rate"))
    # departures per airport-month (denominator for the match share)
    raws = sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]
    R = pl.concat([pl.scan_parquet(f).select("ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt").filter(pl.col("PHASE_mvt") == "DEP")
                   .select("ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym")).collect() for f in raws])
    R = R.group_by("ADEP_mvt", "ym").len("n_dep")
    A = A.join(R, on=["ADEP_mvt", "ym"], how="left").with_columns((pl.col("n_adsb") / pl.col("n_dep")).alias("match"))
    A = A.with_columns([pl.col(c).log1p().alias("l_" + c) for c in ("post_n", "n_ground", "n_win", "gap", "rate")]).sort("ADEP_mvt", "ym")
    A.write_parquet(os.path.join(OUTD, "coverage.parquet"))
    show = A.select("ADEP_mvt", "ym", "n_dep", pl.col("match").round(2), pl.col("gf").round(2), pl.col("stand").round(2),
                    pl.col("push").round(2), "post_n", "n_ground", "n_win", "gap", pl.col("rate").round(1))
    print(show.filter(~pl.col("ADEP_mvt").is_in(["EDDF", "EDDM", "EGLL", "LIRF", "LSZH", "LTFM"])))
    sd = {c: float(A[c].std()) for c in DESC}
    rows = []
    for ap in sorted(A["ADEP_mvt"].unique()):
        a = A.filter(pl.col("ADEP_mvt") == ap)
        v = {ym: np.array([np.nan if r[c] is None else r[c] / sd[c] for c in DESC]) for ym, r in zip(a["ym"], a.iter_rows(named=True))}
        for t in ("2026-01", "2026-07"):
            if t not in v:
                continue
            for ym in sorted(k for k in v if k.startswith("2025")):
                rows.append(dict(airport=ap, target=t, ym=ym, dist=float(np.sqrt(np.nansum((v[ym] - v[t]) ** 2)))))
    Sim = pl.DataFrame(rows).with_columns(pl.col("dist").rank("ordinal").over("airport", "target").alias("rank"))
    Sim.write_parquet(os.path.join(OUTD, "similarity.parquet"))
    top = (Sim.filter(pl.col("rank") <= 3).sort("airport", "target", "rank")
           .group_by("airport", "target", maintain_order=True)
           .agg(pl.format("{} ({})", pl.col("ym").str.slice(5, 2), pl.col("dist").round(2)).str.join(", ").alias("closest 3"),
                pl.col("dist").min().alias("dmin")))
    jan25 = Sim.filter(((pl.col("ym") == "2025-01") & (pl.col("target") == "2026-01")) | ((pl.col("ym") == "2025-07") & (pl.col("target") == "2026-07"))
                       ).select("airport", "target", pl.col("dist").round(2).alias("same-month-2025 dist"), pl.col("rank").alias("same-month rank"))
    print(top.join(jan25, on=["airport", "target"]))


if __name__ == "__main__":
    main()
