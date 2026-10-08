"""plan_cov26.md section 1: thin the 2025 EDDF ground fixes to the 2026 per-cell density and rebuild stage 1,
the residual features and the appear columns (fill) on them. No labels are used.

    SRC=<normal adsb dir> SCRATCH=<thin dir> .venv/bin/python docs/appear_tier_2026-10-05/cov_thin.py density
    SRC=<normal adsb dir> SCRATCH=<thin dir> .venv/bin/python docs/appear_tier_2026-10-05/cov_thin.py build [day,...]

density: ground fixes at EDDF per 100 m cell and day, 2025 Jan+Jul vs 2026 Jan+Jul (scaled by the EDDF departures per
day), keep probability = min(1, 2026 / 2025) -> <thin dir>/keep_EDDF.parquet.
build: for each 2025 day (default Jan+Jul), the fixes with EDDF ground fixes kept at that probability (seeded per day);
other airports and airborne fixes unchanged; then stage-1 analyse, residual features and appear (fill) into
<thin dir> and <thin dir>/learn. The normal files are never touched."""

import os
import shutil
import sys
import zlib
from contextlib import redirect_stdout

import numpy as np
import polars as pl

SRC = os.environ["SRC"]
assert os.path.abspath(SRC) != os.path.abspath(os.environ["SCRATCH"]), "SCRATCH must be the thin dir"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_appear as A  # noqa: E402
import adsb_learn_features as L  # noqa: E402
import adsb_stage1_feasibility as s1  # noqa: E402

APT, CELL = "EDDF", 100.0
from build_features import DAYS as _BD  # noqa: E402
DAYS_ALL = _BD["2025"] + _BD["2025nd"] + _BD["2025m8"]
LAT0, LON0 = s1.AIRPORTS[APT]
KEEP = os.path.join(s1.SCRATCH, f"keep_{APT}.parquet")


def cells(df: pl.DataFrame) -> pl.DataFrame:
    k = np.cos(np.radians(LAT0))
    return df.with_columns(((pl.col("lon") - LON0) * k * 111320.0 / CELL).floor().cast(pl.Int32).alias("cx"),
                           ((pl.col("lat") - LAT0) * 110540.0 / CELL).floor().cast(pl.Int32).alias("cy"))


def ground(day: str) -> pl.DataFrame:
    fx = pl.read_parquet(os.path.join(SRC, f"{L.tag_of(day)}.parquet"), columns=["apt", "lat", "lon", "ground"])
    return cells(fx.filter((pl.col("apt") == APT) & pl.col("ground")))


def n_dep(day: str) -> int:
    return pl.read_parquet(os.path.join(SRC, "learn", f"feat_{day}.parquet"), columns=["ADEP_mvt"]).filter(
        pl.col("ADEP_mvt") == APT).height


def density() -> None:
    per = {}
    for y in ("2025", "2026"):
        days = [d for d in L.DAYS[y] if os.path.exists(os.path.join(SRC, f"{L.tag_of(d)}.parquet"))]
        G = pl.concat([ground(d).group_by("cx", "cy").len() for d in days]).group_by("cx", "cy").agg(pl.col("len").sum())
        dep = sum(n_dep(d) for d in days)
        per[y] = G.with_columns((pl.col("len") / dep).alias(f"r{y}")).select("cx", "cy", f"r{y}")
        print(f"{y}: {len(days)} days, {dep} EDDF departures, {int(G['len'].sum()):,} ground fixes, "
              f"{G['len'].sum() / dep:.1f} per departure, {G.height} cells", flush=True)
    K = per["2025"].join(per["2026"], on=["cx", "cy"], how="left").with_columns(
        pl.min_horizontal(pl.col("r2026").fill_null(0.0) / pl.col("r2025"), pl.lit(1.0)).alias("keep"))
    K.write_parquet(KEEP)
    w = K["r2025"].to_numpy()
    print(f"keep: fix-weighted mean {float((K['keep'].to_numpy() * w).sum() / w.sum()):.3f}; cells with keep < 0.2: "
          f"{int((K['keep'] < 0.2).sum())} of {K.height} -> {KEEP}", flush=True)


def keep_for(month: str) -> pl.DataFrame:
    """Jan / Jul: the pooled table of `density` (stage 1). Other 2025 months: that month's own density against the same
    2026 density (plan_cov26.md section 4), cached as keep_EDDF_<month>.parquet."""
    if month in ("2025-01", "2025-07"):
        return pl.read_parquet(KEEP).select("cx", "cy", "keep")
    path = os.path.join(s1.SCRATCH, f"keep_{APT}_{month}.parquet")
    if not os.path.exists(path):
        r26 = pl.read_parquet(KEEP).select("cx", "cy", "r2026").drop_nulls()
        days = [d for d in DAYS_ALL if d.startswith(month) and os.path.exists(os.path.join(SRC, f"{L.tag_of(d)}.parquet"))]
        G = pl.concat([ground(d).group_by("cx", "cy").len() for d in days]).group_by("cx", "cy").agg(pl.col("len").sum())
        dep = sum(n_dep(d) for d in days)
        K = G.with_columns((pl.col("len") / dep).alias("rm")).join(r26, on=["cx", "cy"], how="left").with_columns(
            pl.min_horizontal(pl.col("r2026").fill_null(0.0) / pl.col("rm"), pl.lit(1.0)).alias("keep"))
        K.select("cx", "cy", "keep").write_parquet(path)
        w = K["rm"].to_numpy()
        print(f"{month}: {len(days)} days, {G['len'].sum() / dep:.1f} ground fixes per departure, keep (fix-weighted) "
              f"{float((K['keep'].to_numpy() * w).sum() / w.sum()):.3f}", flush=True)
    return pl.read_parquet(path)


def build(days: list[str]) -> None:
    os.makedirs(L.OUT, exist_ok=True)
    m = os.path.join(L.OUT, "stand_map_2026.parquet")
    if not os.path.exists(m):
        shutil.copy(os.path.join(SRC, "learn", "stand_map_2026.parquet"), m)
    import stand_map as SM
    ST = SM.stand_table("fill")
    for d in days:
        tag = L.tag_of(d)
        fp, ap = os.path.join(L.OUT, f"feat_{d}.parquet"), os.path.join(L.OUT, f"appear_fill_{d}.parquet")
        if os.path.exists(fp) and os.path.exists(ap):
            continue
        K = keep_for(d[:7])
        fx = cells(pl.read_parquet(os.path.join(SRC, f"{tag}.parquet")))
        rng = np.random.default_rng(zlib.crc32(d.encode()))
        is_g = ((fx["apt"] == APT) & fx["ground"]).to_numpy()
        kp = fx.select("cx", "cy").join(K, on=["cx", "cy"], how="left")["keep"].fill_null(1.0).to_numpy()
        keep = ~is_g | (rng.random(fx.height) < kp)
        fx.filter(pl.Series(keep)).drop("cx", "cy").write_parquet(os.path.join(s1.SCRATCH, f"{tag}.parquet"))
        with open(os.devnull, "w") as dn, redirect_stdout(dn):
            s1.analyse(tag, d)
        L.day_features(d).write_parquet(fp)
        A.day_appear(d, ST).write_parquet(ap)
        print(f"{d}: EDDF ground fixes kept {int((keep & is_g).sum()):,} of {int(is_g.sum()):,}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "density":
        density()
    else:
        arg = sys.argv[2] if len(sys.argv) > 2 else "2025"
        build(_BD[arg] if arg in _BD else arg.split(","))
