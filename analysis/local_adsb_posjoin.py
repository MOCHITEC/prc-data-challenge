"""Positional ADS-B join for departures the callsign join cannot reach (local analysis, read-only).

Unmatched departures (no NM record, so no CALLSIGN_flt) carry only FLIGHT_mvt in IATA form, which almost never
equals the transponder callsign. Instead: find the aircraft whose ground->air transition at the same airport lies
within WIN s of MVT_TIME (after the airport's median take-off offset), on a heading that matches RUNWAY_mvt, and
prefer a type match. Take-offs already claimed by the callsign join are excluded. The stand / push estimators of
adsb_stage1_feasibility.stand_departure are then run unchanged on that aircraft.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsb_posjoin.py <tag> [<tag> ...]
writes $SCRATCH/pos/<tag>_pos.parquet (one row per departure that had no callsign join).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import adsb_stage1_feasibility as s1  # noqa: E402

S = os.environ["SCRATCH"]
OUT = os.path.join(S, "pos")
WIN = 90.0
# median ADS-B take-off minus MVT_TIME per airport (stage-1 time check, 2025-01-15)
OFF = {"EDDF": -9, "EDDM": 13, "EGLL": 31, "EHAM": -19, "LEBL": -7, "LEMD": 14, "LFPG": 0, "LIRF": 0,
       "LSZH": 0, "LTFM": 0}


def rwy_heading(r):
    m = re.match(r"\s*(\d{1,2})", r or "")
    return None if not m else (int(m.group(1)) * 10) % 360


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def run(tag: str) -> None:
    fx = pl.read_parquet(os.path.join(S, f"{tag}.parquet"))
    J = pl.read_parquet(os.path.join(S, f"{tag}_joined.parquet"))
    used = set(J.filter(pl.col("t_to").is_not_null()).select("icao", "ADEP_mvt").iter_rows())
    todo = J.filter(pl.col("join").is_null() | pl.col("icao").is_null())
    typ = dict(fx.drop_nulls("typ").group_by("icao").agg(pl.col("typ").mode().first()).iter_rows())
    arrs, tos = {}, {}
    for (icao, apt), f in fx.sort("t").partition_by(["icao", "apt"], as_dict=True).items():
        t, gr = f["t"].to_numpy(), f["ground"].to_numpy()
        trk = f["track"].fill_null(np.nan).to_numpy()
        arrs[(icao, apt)] = (t, f["lat"].to_numpy(), f["lon"].to_numpy(), gr, f["gs"].fill_null(np.nan).to_numpy(), trk)
        idx = np.where(~gr[1:] & gr[:-1])[0] + 1
        for i in idx:
            h = trk[i] if not np.isnan(trk[i]) else (trk[i + 1] if i + 1 < len(trk) else np.nan)
            tos.setdefault(apt, []).append((float(t[i]), icao, h))
    for apt in tos:
        tos[apt].sort()
    recs = []
    for r in todo.iter_rows(named=True):
        apt, tm = r["ADEP_mvt"], r["t_mvt"] + OFF.get(r["ADEP_mvt"], 0)
        hd = rwy_heading(r["RUNWAY_mvt"])
        cands = []
        for t_to, icao, h in tos.get(apt, []):
            if abs(t_to - tm) > WIN or (icao, apt) in used:
                continue
            rwy_ok = None if (hd is None or np.isnan(h)) else angdiff(h, hd) <= 25
            ty = typ.get(icao)
            type_ok = None if (ty is None or r["AIRCRAFT_TYPE_mvt"] is None) else ty == r["AIRCRAFT_TYPE_mvt"]
            cands.append((rwy_ok is not False, type_ok is True, type_ok is not False, -abs(t_to - tm), icao, t_to,
                          rwy_ok, type_ok))
        rec = {"MVT_ID_mvt": r["MVT_ID_mvt"], "n_cand": len(cands)}
        cands = [c for c in cands if c[0] and c[2]]  # drop heading or type contradictions
        rec["n_ok"] = len(cands)
        if cands:
            cands.sort(reverse=True)
            best = cands[0]
            rec.update({"icao_pos": best[4], "dt_to": best[5] - tm, "rwy_ok": best[6], "type_ok": best[7],
                        "runner_up_dt": (-cands[1][3]) if len(cands) > 1 else None})
            t, la, lo, gr, gs, trk = arrs[(best[4], apt)]
            s = s1.stand_departure(t, la, lo, gr, gs, r["t_mvt"], trk)
            rec.update(s)
        recs.append(rec)
    R = pl.DataFrame(recs, infer_schema_length=None)
    for c in ("push", "post_dwell", "t_move", "t_move_lo"):
        if c not in R.columns:
            R = R.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    R = s1.add_estimators(R)
    keep = todo.select("MVT_ID_mvt", "ADEP_mvt", "t_mvt", "t_aobt3", "t_sched", "t_block", "TAXITIME_SEC_mvt",
                       "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt")
    os.makedirs(OUT, exist_ok=True)
    keep.join(R, on="MVT_ID_mvt", how="left").write_parquet(os.path.join(OUT, f"{tag}_pos.parquet"))
    n_j = R["icao_pos"].is_not_null().sum() if "icao_pos" in R.columns else 0
    print(f"{tag}: {todo.height} without callsign join, positional join {n_j}", flush=True)


if __name__ == "__main__":
    for tg in sys.argv[1:]:
        run(tg)
