"""Travel-time field from own-stand exit, built from 2026 ADS-B only (plan_field.md).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/field.py field   # -> field_A / field_B
    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/field.py feats   # -> field_<key>

Field: for every 2026 departure whose own-stand exit is seen (`t_leave_own_lo` in the fill appear table), the ground
fixes from that exit to take-off are put on a 50 m grid (local metres from the airport reference point); per cell the
first entry time minus `t_leave_own_lo` is the elapsed time. A = median per (airport, cell), B = per (airport, 400 m
area of the own stand, cell); cells with >= 5 departures. Label-free and 2026 only.

Columns per departure: t_appear_fA / t_appear_fB = t_dep1 - field(cell of the t_dep1 fix); t_run_fA = start of the
stage-1 stand run (t_move_lo - stand_dur) - field A(cell of the run), blank when the run is at the own stand
(d_stand_own <= 50 m, a real stand); fA_n / fB_n = departures behind the cell's value."""

import math
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_appear as A  # noqa: E402
import adsb_learn_features as L  # noqa: E402
import adsb_stage1_feasibility as s1  # noqa: E402
import stand_map as SM  # noqa: E402
from build_features import DAYS  # noqa: E402

from prc.osm import normalise_stand  # noqa: E402

CELL, AREA, MIN_N = 50.0, 400.0, 5
TIMES = ("t_appear_fA", "t_appear_fB", "t_run_fA")
PLAIN = ("fA_n", "fB_n")
COLS = (*TIMES, *PLAIN)
FEATS = [f"{p}_{e}" for p in ("mvt_m", "res") for e in TIMES] + [f"{e}_m_aobt3" for e in TIMES] + list(PLAIN)
REF = pl.DataFrame({"ADEP_mvt": list(s1.AIRPORTS), "lat0": [v[0] for v in s1.AIRPORTS.values()],
                    "lon0": [v[1] for v in s1.AIRPORTS.values()]})


def with_xy(df: pl.DataFrame, lat: str = "lat", lon: str = "lon", px: str = "") -> pl.DataFrame:
    k = math.pi / 180.0
    df = df.join(REF, on="ADEP_mvt", how="left")
    return df.with_columns(
        ((pl.col(lon) - pl.col("lon0")) * k * (pl.col("lat0") * k).cos() * A.R_E).alias(f"{px}x"),
        ((pl.col(lat) - pl.col("lat0")) * k * A.R_E).alias(f"{px}y")).drop("lat0", "lon0")


def own_stands(J: pl.DataFrame, ST: pl.DataFrame) -> pl.DataFrame:
    J = J.with_columns(pl.col("STAND_mvt").cast(pl.Utf8).map_elements(normalise_stand, return_dtype=pl.Utf8).alias("sn"))
    J = J.join(ST, on=["ADEP_mvt", "sn"], how="left")
    J = with_xy(J, "slat", "slon", "s")
    return J.with_columns((pl.col("sx") // AREA).cast(pl.Int32).alias("ax"), (pl.col("sy") // AREA).cast(pl.Int32).alias("ay"))


def day_exits(day: str, ST: pl.DataFrame) -> pl.DataFrame:
    tag = L.tag_of(day)
    J = pl.read_parquet(os.path.join(s1.SCRATCH, f"{tag}_joined.parquet"),
                        columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "icao", "join", "t_mvt", "t_to"])
    J = J.filter(pl.col("icao").is_not_null() & pl.col("join").is_in(["cs", "fn"]))
    ap = pl.read_parquet(os.path.join(L.OUT, f"appear_fill_{day}.parquet"), columns=["MVT_ID_mvt", "t_leave_own_lo"])
    J = J.join(ap, on="MVT_ID_mvt", how="inner").filter(pl.col("t_leave_own_lo").is_not_null())
    J = own_stands(J, ST).with_columns(pl.coalesce("t_to", pl.col("t_mvt") + 120.0).alias("t_end"))
    fx = pl.read_parquet(os.path.join(s1.SCRATCH, f"{tag}.parquet"), columns=["icao", "apt", "t", "lat", "lon", "ground"])
    F = (fx.filter("ground").rename({"apt": "ADEP_mvt"})
         .join(J.select("MVT_ID_mvt", "icao", "ADEP_mvt", "t_leave_own_lo", "t_end", "ax", "ay"), on=["icao", "ADEP_mvt"])
         .filter((pl.col("t") >= pl.col("t_leave_own_lo")) & (pl.col("t") <= pl.col("t_end"))))
    F = with_xy(F).with_columns((pl.col("x") // CELL).cast(pl.Int16).alias("cx"), (pl.col("y") // CELL).cast(pl.Int16).alias("cy"))
    return (F.group_by("MVT_ID_mvt", "ADEP_mvt", "ax", "ay", "cx", "cy")
            .agg((pl.col("t").min() - pl.col("t_leave_own_lo").first()).cast(pl.Float32).alias("el")))


def build_field() -> None:
    ST = SM.stand_table("fill")
    fr = [day_exits(d, ST) for d in L.DAYS["2026"]
          if os.path.exists(os.path.join(L.OUT, f"appear_fill_{d}.parquet"))]
    E = pl.concat(fr)
    print(f"field rows {E.height:,} from {E['MVT_ID_mvt'].n_unique():,} departures", flush=True)
    FA = (E.group_by("ADEP_mvt", "cx", "cy").agg(pl.col("el").median().alias("fA"), pl.len().alias("fA_n"))
          .filter(pl.col("fA_n") >= MIN_N))
    FB = (E.drop_nulls(["ax", "ay"]).group_by("ADEP_mvt", "ax", "ay", "cx", "cy")
          .agg(pl.col("el").median().alias("fB"), pl.len().alias("fB_n")).filter(pl.col("fB_n") >= MIN_N))
    FA.write_parquet(os.path.join(L.OUT, "field_A.parquet"))
    FB.write_parquet(os.path.join(L.OUT, "field_B.parquet"))
    print(f"field A cells {FA.height:,}; field B cells {FB.height:,}", flush=True)
    print(FA.group_by("ADEP_mvt").agg(pl.len().alias("cells"), pl.col("fA").median().round(0).alias("med_s"),
                                       pl.col("fA_n").sum().alias("visits")).sort("ADEP_mvt"))


def day_feats(day: str, ST: pl.DataFrame, FA: pl.DataFrame, FB: pl.DataFrame) -> pl.DataFrame:
    tag = L.tag_of(day)
    J = pl.read_parquet(os.path.join(s1.SCRATCH, f"{tag}_joined.parquet"),
                        columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "icao", "join", "t_dep1", "t_move_lo", "stand_dur"])
    J = J.filter(pl.col("icao").is_not_null() & pl.col("join").is_in(["cs", "fn"]) & pl.col("t_dep1").is_not_null())
    J = J.join(pl.read_parquet(os.path.join(L.OUT, f"appear_fill_{day}.parquet"), columns=["MVT_ID_mvt", "d_stand_own"]),
               on="MVT_ID_mvt", how="left")
    J = own_stands(J, ST)
    fx = (pl.read_parquet(os.path.join(s1.SCRATCH, f"{tag}.parquet"), columns=["icao", "apt", "t", "lat", "lon"])
          .rename({"apt": "ADEP_mvt"}).unique(["icao", "ADEP_mvt", "t"], keep="first"))

    def cell_at(tcol: str, px: str) -> pl.DataFrame:
        P = J.select("MVT_ID_mvt", "icao", "ADEP_mvt", pl.col(tcol).alias("t")).join(fx, on=["icao", "ADEP_mvt", "t"])
        P = with_xy(P)
        return P.select("MVT_ID_mvt", (pl.col("x") // CELL).cast(pl.Int16).alias(f"{px}cx"),
                        (pl.col("y") // CELL).cast(pl.Int16).alias(f"{px}cy"))

    J = J.join(cell_at("t_dep1", "d"), on="MVT_ID_mvt", how="left").join(cell_at("t_move_lo", "r"), on="MVT_ID_mvt", how="left")
    J = (J.join(FA.rename({"cx": "dcx", "cy": "dcy"}), on=["ADEP_mvt", "dcx", "dcy"], how="left")
         .join(FB.rename({"cx": "dcx", "cy": "dcy"}), on=["ADEP_mvt", "ax", "ay", "dcx", "dcy"], how="left")
         .join(FA.rename({"cx": "rcx", "cy": "rcy", "fA": "fR", "fA_n": "fR_n"}), on=["ADEP_mvt", "rcx", "rcy"], how="left"))
    real_stand = (pl.col("d_stand_own") <= A.R_IN).fill_null(False)
    out = J.select(
        "MVT_ID_mvt",
        (pl.col("t_dep1") - pl.col("fA")).alias("t_appear_fA"),
        (pl.col("t_dep1") - pl.col("fB")).alias("t_appear_fB"),
        pl.when(real_stand).then(None).otherwise(pl.col("t_move_lo") - pl.col("stand_dur") - pl.col("fR")).alias("t_run_fA"),
        pl.col("fA_n").cast(pl.Float64), pl.col("fB_n").cast(pl.Float64))
    out = out.unique("MVT_ID_mvt", keep="first")
    return out


def build_feats() -> None:
    ST = SM.stand_table("fill")
    FA = pl.read_parquet(os.path.join(L.OUT, "field_A.parquet"))
    FB = pl.read_parquet(os.path.join(L.OUT, "field_B.parquet"))
    for key in ("2025", "2025nd", "2026"):
        fr = [day_feats(d, ST, FA, FB) for d in DAYS[key]
              if os.path.exists(os.path.join(L.OUT, f"appear_fill_{d}.parquet"))]
        D = pl.concat(fr)
        assert D["MVT_ID_mvt"].n_unique() == D.height
        D.write_parquet(os.path.join(L.OUT, f"field_{key}.parquet"))
        print(f"{key}: {D.height} rows; t_appear_fA {D['t_appear_fA'].is_not_null().mean():.3f}, fB "
              f"{D['t_appear_fB'].is_not_null().mean():.3f}, t_run_fA {D['t_run_fA'].is_not_null().mean():.3f}", flush=True)


def design(X: pl.DataFrame, feats: list[str], base_col: str = "b") -> tuple[pl.DataFrame, list[str]]:
    ex = []
    for e in TIMES:
        ex += [(pl.col("t_mvt") - pl.col(e)).alias(f"mvt_m_{e}"),
               (pl.col("t_mvt") - pl.col(e) - pl.col(base_col).cast(pl.Float64)).alias(f"res_{e}"),
               (pl.col(e) - pl.col("t_aobt3")).alias(f"{e}_m_aobt3")]
    return X.with_columns(ex).with_columns([pl.col(c).cast(pl.Float64) for c in PLAIN]), feats + FEATS


def shuffled(D: pl.DataFrame) -> pl.DataFrame:
    """The field block permuted across the rows that have one, within airport; times as offsets from each row's MVT."""
    rng = np.random.default_rng(A.SEED + 7)
    t_mvt = D["t_mvt"].to_numpy().astype(np.float64)
    vals = {c: D[c].cast(pl.Float64).to_numpy().copy() for c in COLS}
    for c in TIMES:
        vals[c] = t_mvt - vals[c]
    filled = np.zeros(D.height, bool)
    for c in COLS:
        filled |= ~np.isnan(vals[c])
    apt = D["ADEP_mvt"].to_numpy()
    for a in np.unique(apt[filled]):
        idx = np.flatnonzero(filled & (apt == a))
        src = idx[rng.permutation(len(idx))]
        for c in COLS:
            vals[c][idx] = vals[c][src]
    for c in TIMES:
        vals[c] = t_mvt - vals[c]
    return D.with_columns([pl.Series(c, vals[c]).fill_nan(None) for c in COLS])


if __name__ == "__main__":
    {"field": build_field, "feats": build_feats}[sys.argv[1]]()
