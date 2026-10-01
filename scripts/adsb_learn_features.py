"""Per-departure ADS-B features for the learned residual model (label-free).

Pre-registration: `docs/competitor_research_2026-09-21/results/adsb-learned-residual.md` section 0.
Written from scratch for this repo. It reads the stage-1 outputs in $SCRATCH (`<tag>_joined.parquet` for the
per-departure join and the stage-1 detection columns, and `<tag>.parquet` for the filtered fixes). Nothing is written
to `src/`, `configs/`, `outputs/` or `data/`.

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_features.py build   # all 124 days, resumable
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_features.py concat  # -> learn/feat_{2025,2026}.parquet

No column of the output reads BLOCK or a label: the joined files' `t_block`, `g`, `fill`, `e_aobt3` and `e_to` are
never loaded (asserted). `design()` turns the absolute times into the model's relative features, given a base
prediction.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_stage1_feasibility as s1

SCR = s1.SCRATCH
OUT = os.path.join(SCR, "learn")
DAYS = {y: [f"{y}-{m:02d}-{d:02d}" for m in (1, 7) for d in range(1, 32)] for y in ("2025", "2026")}
LABEL_COLS = {"t_block", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "g", "fill", "e_aobt3", "e_to"}
KEEP = ["MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_aobt3", "join", "icao", "ground_fix", "t_first", "t_dep1", "t_move",
        "t_move_lo", "t_to", "n_ground", "stand_dur", "dep1_gs", "gap_before_dep1", "opens_with_landing", "stand_rule",
        "stand_nfix", "push", "post_dwell", "seen"]
STOP_KT = 2.0
GS_THR = (3, 10, 25)
R_E = 6371000.0


def tag_of(day: str) -> str:
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def mem_mb() -> float:
    """Largest total_rss (MB) among the prescribed per-process cgroup path and the memory-cgroup root."""
    vals = []
    rel = ""
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                p = line.strip().split(":")
                if len(p) == 3 and p[1] == "memory":
                    rel = p[2]
    except OSError:
        pass
    for p in {f"/sys/fs/cgroup/memory{rel}/memory.stat", "/sys/fs/cgroup/memory/memory.stat"}:
        try:
            with open(p) as f:
                for line in f:
                    if line.startswith("total_rss "):
                        vals.append(int(line.split()[1]) / 2 ** 20)
        except OSError:
            pass
    return max(vals) if vals else 0.0


def mem_gate(limit_mb: float = 9500.0) -> None:
    while (m := mem_mb()) > limit_mb:
        print(f"  memory gate: total_rss {m:.0f} MB > {limit_mb:.0f}; waiting", flush=True)
        time.sleep(60)


def day_features(day: str) -> pl.DataFrame:
    tag = tag_of(day)
    jp = os.path.join(SCR, f"{tag}_joined.parquet")
    have = pl.read_parquet_schema(jp)
    J = pl.read_parquet(jp, columns=[c for c in KEEP if c in have])
    assert not (set(J.columns) & LABEL_COLS)
    J = s1.add_estimators(J)  # t_push / t_push_lo: stage-1 gated pushback (label-free)
    J = J.filter(pl.col("icao").is_not_null() & pl.col("join").is_in(["cs", "fn"]))
    J = J.with_columns(pl.coalesce("t_move_lo", "t_dep1").alias("t_anchor"),
                       pl.coalesce("t_to", pl.col("t_mvt") + 120.0).alias("t_end"),
                       pl.lit(day).alias("day"))
    fx = pl.read_parquet(os.path.join(SCR, f"{tag}.parquet"), columns=["icao", "apt", "t", "lat", "lon", "ground", "gs"])
    K = J.select("MVT_ID_mvt", "icao", pl.col("ADEP_mvt").alias("apt"), "t_mvt", "t_anchor", "t_end")
    F = (fx.join(K, on=["icao", "apt"], how="inner")
         .filter((pl.col("t") >= pl.col("t_mvt") - 4 * 3600) & (pl.col("t") <= pl.col("t_mvt") + 1800)))
    del fx
    ref = pl.DataFrame({"apt": list(s1.AIRPORTS), "lat0": [v[0] for v in s1.AIRPORTS.values()],
                        "lon0": [v[1] for v in s1.AIRPORTS.values()]})
    k = math.pi / 180.0
    F = F.join(ref, on="apt").with_columns(
        ((pl.col("lon") - pl.col("lon0")) * k * (pl.col("lat0") * k).cos() * R_E).alias("x"),
        ((pl.col("lat") - pl.col("lat0")) * k * R_E).alias("y"))
    win = F.group_by("MVT_ID_mvt").agg(pl.len().alias("n_win"), pl.col("ground").sum().alias("n_win_ground"))
    air = (F.filter(~pl.col("ground") & pl.col("t_anchor").is_not_null() & (pl.col("t") > pl.col("t_anchor")))
           .group_by("MVT_ID_mvt").agg(pl.col("t").min().alias("t_air1")))
    G = (F.filter(pl.col("ground") & (pl.col("t") >= pl.col("t_anchor")) & (pl.col("t") <= pl.col("t_end")))
         .sort("MVT_ID_mvt", "t"))
    G = G.with_columns(
        (pl.col("t") - pl.col("t").shift(1).over("MVT_ID_mvt")).alias("dt"),
        ((pl.col("x") - pl.col("x").shift(1).over("MVT_ID_mvt")) ** 2
         + (pl.col("y") - pl.col("y").shift(1).over("MVT_ID_mvt")) ** 2).sqrt().alias("step"),
        (pl.col("gs") < STOP_KT).alias("slow"),
        (pl.col("gs").shift(1).over("MVT_ID_mvt") < STOP_KT).alias("prev_slow"))
    agg = [pl.len().alias("post_n"), pl.col("step").sum().alias("post_path_m"),
           pl.col("dt").max().alias("post_max_gap"),
           pl.col("dt").filter(pl.col("slow") & pl.col("prev_slow") & (pl.col("dt") <= 120)).sum().alias("post_stop_s"),
           (pl.col("slow") & (pl.col("prev_slow") == False)).sum().alias("post_n_stops"),  # noqa: E712
           pl.col("gs").filter(pl.col("gs") >= 3).mean().alias("post_mov_gs"),
           pl.col("t").max().alias("t_glast"),
           pl.col("x").first().alias("x_a"), pl.col("y").first().alias("y_a"),
           pl.col("x").last().alias("x_l"), pl.col("y").last().alias("y_l")]
    agg += [pl.col("t").filter(pl.col("gs") >= g).min().alias(f"t_gs{g}") for g in GS_THR]
    P = G.group_by("MVT_ID_mvt").agg(agg).with_columns(
        ((pl.col("x_l") - pl.col("x_a")) ** 2 + (pl.col("y_l") - pl.col("y_a")) ** 2).sqrt().alias("post_disp_m"))
    out = (J.join(win, on="MVT_ID_mvt", how="left").join(air, on="MVT_ID_mvt", how="left")
           .join(P, on="MVT_ID_mvt", how="left"))
    assert out["MVT_ID_mvt"].n_unique() == out.height
    return out


EVENTS = ["t_first", "t_dep1", "t_move_lo", "t_move", "t_to", "t_gs3", "t_gs10", "t_gs25", "t_glast", "t_air1"]
PLAIN = ["ground_fix", "n_ground", "stand_dur", "stand_nfix", "dep1_gs", "gap_before_dep1", "opens_with_landing",
         "push", "post_dwell", "n_win", "n_win_ground", "post_n", "post_path_m", "post_max_gap", "post_stop_s",
         "post_n_stops", "post_mov_gs", "x_a", "y_a", "x_l", "y_l", "post_disp_m"]
CATS = {"ADEP_mvt": list(s1.AIRPORTS), "join": ["cs", "fn"],
        "stand_rule": ["run_at_dep1", "longest_run", "moving_when_first_seen"]}


def design(D: pl.DataFrame, base_col: str = "b") -> tuple[pl.DataFrame, list[str], list[str]]:
    """Model matrix: relative event times (to MVT, to the base prediction, to AOBT_3), plain columns, categoricals
    as fixed-code integers, and the base itself. Returns (frame, feature names, categorical feature names)."""
    ex = []
    for e in EVENTS:
        ex += [(pl.col("t_mvt") - pl.col(e)).alias(f"mvt_m_{e}"),
               (pl.col("t_mvt") - pl.col(e) - pl.col(base_col)).alias(f"res_{e}"),
               (pl.col(e) - pl.col("t_aobt3")).alias(f"{e}_m_aobt3")]
    ex += [(pl.col("t_move") - pl.col("t_move_lo")).alias("exit_res"),
           pl.col("t_push").is_not_null().cast(pl.Int8).alias("push_ok"),
           pl.col(base_col).cast(pl.Float64).alias("base")]
    for c, levels in CATS.items():
        ex.append(pl.col(c).replace_strict(levels, list(range(len(levels))), default=-1).cast(pl.Int32)
                  .alias(f"cat_{c}"))
    X = D.with_columns(ex).with_columns([pl.col(c).cast(pl.Float64) for c in PLAIN])
    names = [f"mvt_m_{e}" for e in EVENTS] + [f"res_{e}" for e in EVENTS] + [f"{e}_m_aobt3" for e in EVENTS]
    names += ["exit_res", "push_ok", "base"] + PLAIN
    cats = [f"cat_{c}" for c in CATS]
    return X, names + cats, cats


def build() -> None:
    os.makedirs(OUT, exist_ok=True)
    for y in ("2025", "2026"):
        for day in DAYS[y]:
            dst = os.path.join(OUT, f"feat_{day}.parquet")
            if os.path.exists(dst):
                continue
            mem_gate()
            t0 = time.time()
            f = day_features(day)
            f.write_parquet(dst)
            print(f"{day}: {f.height} ADS-B rows, post_n non-null {f['post_n'].is_not_null().sum()}, "
                  f"{time.time() - t0:.1f} s, total_rss {mem_mb():.0f} MB", flush=True)


def concat() -> None:
    for y in ("2025", "2026"):
        fr = [pl.read_parquet(os.path.join(OUT, f"feat_{d}.parquet")) for d in DAYS[y]]
        D = pl.concat(fr, how="diagonal_relaxed")
        assert D["MVT_ID_mvt"].n_unique() == D.height, "a departure in two day files"
        assert not (set(D.columns) & LABEL_COLS)
        D.write_parquet(os.path.join(OUT, f"feat_{y}.parquet"))
        print(y, D.shape, "days", D["day"].n_unique(), "per airport:",
              dict(D.group_by("ADEP_mvt").len().sort("ADEP_mvt").iter_rows()))


if __name__ == "__main__":
    {"build": build, "concat": concat}[sys.argv[1]]()
