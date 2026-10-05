"""Appear tier for the learned ADS-B residual: where a departure's aircraft is first seen, relative to its own stand
(docs/appear_tier_2026-10-05/, gap study recommendation A in docs/research_2026-10-01/gap/HANDOFF_PROMPT.md).

    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_appear.py build    # per day, resumable -> learn/appear_<day>
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_appear.py concat   # -> learn/appear_{2025,2026}.parquet
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base <b> --appear-shuffle   # noise control
    POLARS_MAX_THREADS=2 .venv/bin/python scripts/adsb_learn_cv.py --base <b> --appear           # the arm

Why. Stage 1 finds the off-block only through a stationary stand run (`t_move`, `t_move_lo`). An aircraft whose
transponder comes on while it is already moving (`moving_when_first_seen`, 18-26 % of scored departures at the five
pushback airports, adsb-coverage-breakdown.md A9a) gets no stand time; the residual sees only `t_dep1` (first fix
after the last silence) and its ground speed, not WHERE that fix is. Pushback usually starts with the transponder,
so a first fix at the aircraft's own stand is close to the off-block, and one 600 m down the taxiway is not.

Columns (label-free; the joined files' BLOCK / label columns are never read):
  own_ok           1 if STAND_mvt matches an OpenStreetMap parking_position (normalised ref), else 0
  d_dep1_own       metres from the t_dep1 fix to the own stand (null without own_ok)
  d_dep1_near      metres from the t_dep1 fix to the nearest OSM parking_position of the airport (any stand)
  d_stand_own      metres from the stand run's last fix (t_move_lo) to the own stand: is the run the real stand?
  n_in_own         ground fixes of the departure phase within R_IN of the own stand
  t_leave_own_lo   last ground fix within R_IN of the own stand before take-off
  t_leave_own      first ground fix after it (outside R_IN): a geometric stand exit, stationary or not
  t_appear_adj     t_dep1 - min(d_dep1_own, D_CAP) / V_TAXI: the first fix projected back to the stand
The design adds, for the three times, MVT - t, MVT - t - base and t - AOBT_3, as for the stage-1 events.

`--appear-shuffle` adds the same columns with the block permuted across the rows that have one, within each airport;
the times are permuted as offsets from each row's own MVT, so every relative column keeps its marginal distribution
and only the per-flight link is broken. Licence: adsb.lol ODbL 1.0; OpenStreetMap ODbL 1.0 (both open)."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_learn_features as L
import adsb_stage1_feasibility as s1
from adsb_opdi import _paired

from prc.osm import normalise_stand, stand_table

MODE = "appear" if "--appear" in sys.argv else "appearshuf" if "--appear-shuffle" in sys.argv else ""
SFX = f"_{MODE}" if MODE else ""
R_IN = 50.0  # m: "at the stand" (OSM stop-line point vs the transponder's antenna position)
V_TAXI = 6.0  # m/s: back-projection speed for a first fix away from the stand
D_CAP = 3000.0  # m
R_E = 6371000.0
SEED = 20261005
TIMES = ("t_leave_own_lo", "t_leave_own", "t_appear_adj")
PLAIN = ("own_ok", "d_dep1_own", "d_dep1_near", "d_stand_own", "n_in_own")
COLS = (*PLAIN, *TIMES)
FEATS = [f"{p}_{e}" for p in ("mvt_m", "res") for e in TIMES] + [f"{e}_m_aobt3" for e in TIMES] + list(PLAIN)


def stands() -> pl.DataFrame:
    osm = {a: json.load(open(os.path.join(s1.REPO_DIR, f"data/external/osm/parking_{a}.json"))) for a in s1.AIRPORTS}
    return stand_table(osm).select(pl.col("icao").alias("ADEP_mvt"), pl.col("stand").alias("sn"),
                                   pl.col("lat").alias("slat"), pl.col("lon").alias("slon"))


def _dist(lat1, lon1, lat2, lon2):
    k = math.pi / 180.0
    x = (lon2 - lon1) * k * np.cos((lat1 + lat2) * 0.5 * k) * R_E
    y = (lat2 - lat1) * k * R_E
    return np.sqrt(x * x + y * y)


def day_appear(day: str, ST: pl.DataFrame) -> pl.DataFrame:
    tag = L.tag_of(day)
    J = pl.read_parquet(os.path.join(s1.SCRATCH, f"{tag}_joined.parquet"),
                        columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "icao", "join", "t_mvt", "t_first", "t_dep1",
                                 "t_move_lo", "t_to"])
    J = J.filter(pl.col("icao").is_not_null() & pl.col("join").is_in(["cs", "fn"]) & pl.col("t_first").is_not_null())
    J = J.with_columns(pl.coalesce("t_to", pl.col("t_mvt") + 120.0).alias("t_end"),
                       pl.col("STAND_mvt").cast(pl.Utf8).map_elements(normalise_stand, return_dtype=pl.Utf8).alias("sn"))
    J = J.join(ST, on=["ADEP_mvt", "sn"], how="left")
    fx = pl.read_parquet(os.path.join(s1.SCRATCH, f"{tag}.parquet"), columns=["icao", "apt", "t", "lat", "lon", "ground"])
    F = (fx.filter("ground").join(J.select("MVT_ID_mvt", "icao", pl.col("ADEP_mvt").alias("apt"), "t_first", "t_end",
                                           "slat", "slon"), on=["icao", "apt"], how="inner")
         .filter((pl.col("t") >= pl.col("t_first")) & (pl.col("t") <= pl.col("t_end"))).sort("MVT_ID_mvt", "t"))
    del fx
    F = F.with_columns(pl.Series("d_own", _dist(F["lat"].to_numpy(), F["lon"].to_numpy(),
                                                F["slat"].to_numpy(), F["slon"].to_numpy())).fill_nan(None))
    # the stand exit: the last fix within R_IN of the own stand, and the fix after it
    F = F.with_columns((pl.col("d_own") <= R_IN).fill_null(False).alias("inn"))
    last_in = F.filter("inn").group_by("MVT_ID_mvt").agg(pl.col("t").max().alias("t_leave_own_lo"),
                                                         pl.len().cast(pl.Float64).alias("n_in_own"))
    leave = (F.join(last_in.select("MVT_ID_mvt", "t_leave_own_lo"), on="MVT_ID_mvt")
             .filter(pl.col("t") > pl.col("t_leave_own_lo")).group_by("MVT_ID_mvt")
             .agg(pl.col("t").min().alias("t_leave_own")))
    # the t_dep1 fix and the stand run's last fix: positions -> distance to own stand and to the nearest stand
    pos = F.select("MVT_ID_mvt", "t", "lat", "lon", "d_own")
    d1 = J.select("MVT_ID_mvt", "ADEP_mvt", pl.col("t_dep1").alias("t")).join(pos, on=["MVT_ID_mvt", "t"], how="left")
    near = np.full(d1.height, np.nan)
    for a in d1["ADEP_mvt"].unique().to_list():
        m = (d1["ADEP_mvt"] == a).to_numpy() & d1["lat"].is_not_null().to_numpy()
        sa = ST.filter(pl.col("ADEP_mvt") == a)
        if not m.any() or sa.height == 0:
            continue
        la, lo = d1["lat"].to_numpy()[m], d1["lon"].to_numpy()[m]
        dd = _dist(la[:, None], lo[:, None], sa["slat"].to_numpy()[None, :], sa["slon"].to_numpy()[None, :])
        near[m] = dd.min(axis=1)
    d1 = d1.select("MVT_ID_mvt", pl.col("d_own").alias("d_dep1_own"), pl.Series("d_dep1_near", near).fill_nan(None))
    ds = (J.select("MVT_ID_mvt", pl.col("t_move_lo").alias("t")).join(pos, on=["MVT_ID_mvt", "t"], how="inner")
          .select("MVT_ID_mvt", pl.col("d_own").alias("d_stand_own")))
    out = (J.select("MVT_ID_mvt", "t_dep1", pl.col("slat").is_not_null().cast(pl.Float64).alias("own_ok"))
           .join(d1, on="MVT_ID_mvt", how="left").join(ds, on="MVT_ID_mvt", how="left")
           .join(last_in, on="MVT_ID_mvt", how="left").join(leave, on="MVT_ID_mvt", how="left")
           .with_columns((pl.col("t_dep1") - pl.min_horizontal("d_dep1_own", pl.lit(D_CAP)) / V_TAXI)
                         .alias("t_appear_adj"))
           .with_columns(pl.col("n_in_own").fill_null(pl.when(pl.col("own_ok") > 0).then(0.0)))
           .drop("t_dep1"))
    assert out["MVT_ID_mvt"].n_unique() == out.height
    return out.select("MVT_ID_mvt", *COLS)


def build() -> None:
    os.makedirs(L.OUT, exist_ok=True)
    ST = stands()
    for y in ("2025", "2026"):
        for day in L.DAYS[y]:
            dst = os.path.join(L.OUT, f"appear_{day}.parquet")
            if os.path.exists(dst) or not os.path.exists(os.path.join(s1.SCRATCH, f"{L.tag_of(day)}_joined.parquet")):
                continue
            a = day_appear(day, ST)
            a.write_parquet(dst)
            print(f"{day}: {a.height} rows; own stand {a['own_ok'].mean():.3f}; d_dep1_own <= {R_IN:.0f} m "
                  f"{(a['d_dep1_own'] <= R_IN).mean():.3f}; stand exit {a['t_leave_own'].is_not_null().mean():.3f}",
                  flush=True)


def concat() -> None:
    for y in ("2025", "2026"):
        fr = [pl.read_parquet(os.path.join(L.OUT, f"appear_{d}.parquet")) for d in L.DAYS[y]
              if os.path.exists(os.path.join(L.OUT, f"appear_{d}.parquet"))]
        D = pl.concat(fr, how="vertical_relaxed")
        assert D["MVT_ID_mvt"].n_unique() == D.height, "a departure in two day files"
        D.write_parquet(os.path.join(L.OUT, f"appear_{y}.parquet"))
        print(y, D.shape, "days", len(fr))


def add(D: pl.DataFrame, year: str) -> pl.DataFrame:
    """Left-join the appear columns by MVT_ID_mvt; under --appear-shuffle permute the block within airport."""
    T = pl.read_parquet(os.path.join(L.OUT, f"appear_{year}.parquet"))
    D = D.drop([c for c in COLS if c in D.columns]).join(T, on="MVT_ID_mvt", how="left")
    if MODE != "appearshuf":
        return D
    rng = np.random.default_rng(SEED)
    t_mvt = D["t_mvt"].to_numpy().astype(np.float64)
    vals = {c: D[c].cast(pl.Float64).to_numpy().copy() for c in COLS}
    for c in TIMES:
        vals[c] = t_mvt - vals[c]  # offsets from the row's own MVT
    filled = D["own_ok"].is_not_null().to_numpy()
    apt = D["ADEP_mvt"].to_numpy()
    for a in np.unique(apt[filled]):
        idx = np.flatnonzero(filled & (apt == a))
        src = idx[rng.permutation(len(idx))]
        for c in COLS:
            vals[c][idx] = vals[c][src]
    for c in TIMES:
        vals[c] = t_mvt - vals[c]
    return D.with_columns([pl.Series(c, vals[c]).fill_nan(None) for c in COLS])


def design(X: pl.DataFrame, feats: list[str], base_col: str = "b") -> tuple[pl.DataFrame, list[str]]:
    ex = []
    for e in TIMES:
        ex += [(pl.col("t_mvt") - pl.col(e)).alias(f"mvt_m_{e}"),
               (pl.col("t_mvt") - pl.col(e) - pl.col(base_col).cast(pl.Float64)).alias(f"res_{e}"),
               (pl.col(e) - pl.col("t_aobt3")).alias(f"{e}_m_aobt3")]
    X = X.with_columns(ex).with_columns([pl.col(c).cast(pl.Float64) for c in PLAIN])
    return X, feats + FEATS


def report(D: pl.DataFrame, p_new: np.ndarray, out_dir: str, base_sfx: str) -> dict:
    """Paired comparisons with the same base's run without the columns and, for --appear, with --appear-shuffle:
    all rows, rows first seen moving (stand_rule moving_when_first_seen), rows first seen at the own stand.
    Pre-registered decision (docs/appear_tier_2026-10-05/plan.md): ADOPT when the all-row CI excludes zero on the
    improving side against BOTH references."""
    refs = {"plain": os.path.join(out_dir, f"cv_rows{base_sfx}.parquet")}
    if MODE == "appear":
        refs["shuffle"] = os.path.join(out_dir, f"cv_rows{base_sfx}_appearshuf.parquet")
    y, day_idx = D["y"].to_numpy(), D["day_idx"].to_numpy()
    n_days = int(day_idx.max()) + 1
    moving = (D["stand_rule"] == "moving_when_first_seen").fill_null(False).to_numpy()
    at_own = (D["d_dep1_own"] <= R_IN).fill_null(False).to_numpy()
    print(f"\nAPPEAR ({MODE}): rows with the columns {int(D['own_ok'].is_not_null().sum()):,} of {D.height:,}; "
          f"own stand {int((D['own_ok'] == 1).sum()):,}; first seen moving {int(moving.sum()):,}; "
          f"first seen at own stand {int(at_own.sum()):,}")
    out, ok = {}, {}
    for name, path in refs.items():
        if not os.path.exists(path):
            print(f"  vs {name}: {path} missing - run it first")
            ok[name] = None
            continue
        R = pl.read_parquet(path).select("MVT_ID_mvt", (pl.col("b") + pl.col("c_learned")).alias("p_ref"))
        p_ref = D.select("MVT_ID_mvt").join(R, on="MVT_ID_mvt", how="left", maintain_order="left")["p_ref"].to_numpy()
        assert np.isfinite(p_ref).all(), f"{path} does not cover this run's rows"
        out[name] = {}
        for label, m in (("all rows", np.ones(len(y), bool)), ("first seen moving", moving),
                         ("first seen at own stand", at_own)):
            dd, lo, hi = _paired(y[m], p_ref[m], p_new[m], day_idx[m], n_days)
            out[name][label] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  vs {name:7s} {label:>24s} n={int(m.sum()):>7,}  RMSE diff {dd:+.3f} s  day-CI95 [{lo:+.3f}, {hi:+.3f}]")
        apt = D["ADEP_mvt"].to_numpy()
        for a in sorted(set(apt)):
            m = apt == a
            dd, lo, hi = _paired(y[m], p_ref[m], p_new[m], day_idx[m], n_days)
            out[name][a] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  vs {name:7s} {a:>24s} n={int(m.sum()):>7,}  RMSE diff {dd:+.3f} s  day-CI95 [{lo:+.3f}, {hi:+.3f}]")
        ok[name] = out[name]["all rows"]["hi"] < 0
    if MODE == "appear":
        verdict = "ADOPT" if ok.get("plain") and ok.get("shuffle") else "INCOMPLETE" if None in ok.values() else "REJECT"
        print(f"DECISION (docs/appear_tier_2026-10-05/plan.md): vs plain {ok.get('plain')}, vs shuffle "
              f"{ok.get('shuffle')} -> {verdict}")
        out["verdict"] = verdict
    return out


def parity(D: pl.DataFrame, G: pl.DataFrame) -> None:
    """Per airport: share of ADS-B rows first seen within R_IN of the own stand, and with a stand exit, 2025 vs 2026."""
    f = lambda x: (round(float((x["d_dep1_own"] <= R_IN).fill_null(False).mean()), 3),  # noqa: E731
                   round(float(x["t_leave_own"].is_not_null().mean()), 3))
    for a in sorted(set(G["ADEP_mvt"].to_list())):
        print(f"APPEAR parity {a}: 2025 fit (at own stand, exit) {f(D.filter(pl.col('ADEP_mvt') == a))}; "
              f"2026 {f(G.filter(pl.col('ADEP_mvt') == a))}")


if __name__ == "__main__":
    {"build": build, "concat": concat}[sys.argv[1]]()
