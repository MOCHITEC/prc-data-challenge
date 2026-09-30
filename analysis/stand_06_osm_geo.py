"""Stand -> runway geometry from OpenStreetMap (ODbL) and its relation to taxi-out.

Inputs: SCRATCH/osm_<APT>.json from stand_04_osm_fetch.py.
  * stand position = centre of aeroway=parking_position (way or node) whose `ref` matches STAND_mvt after
    normalisation (upper case, spaces/dashes removed, leading zeros of the number stripped on both sides);
    aeroway=gate nodes are used as a fallback.
  * runway threshold for departure on designator XX = the runway end from which the vector along the runway has
    bearing ~ XX*10 degrees.
Features per departure (all static, ranking-safe): geo_dist_thr_m (stand -> departure threshold, great circle),
geo_stand_lat/lon (airport-local metres east/north of the aerodrome mean), geo_matched.
Writes SCRATCH/stand_geo_feats.parquet and SCRATCH/stand_geo_table.parquet; prints match rates and the
relation between distance and the stand x runway median taxi-out.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import polars as pl

from stand_common import SCRATCH, load_all

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
R = 6371000.0


def norm(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"[\s\-_/.]", "", s.upper())
    m = re.match(r"^([A-Z]*)0*(\d+)([A-Z]*)$", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return s


def centre(e: dict) -> tuple[float, float] | None:
    if e["type"] == "node":
        return e["lat"], e["lon"]
    if "center" in e:
        return e["center"]["lat"], e["center"]["lon"]
    g = e.get("geometry")
    if g:
        return float(np.mean([p["lat"] for p in g])), float(np.mean([p["lon"] for p in g]))
    return None


def bearing(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    x = math.sin(lo2 - lo1) * math.cos(la2)
    y = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(lo2 - lo1)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def hav(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def parse(apt: str):
    d = json.load(open(f"{SCRATCH}/osm_{apt}.json"))["elements"]
    stands: dict[str, tuple[float, float]] = {}
    gates: dict[str, tuple[float, float]] = {}
    thr: dict[str, tuple[float, float]] = {}
    for e in d:
        t = e.get("tags", {})
        a = t.get("aeroway")
        if a in ("parking_position", "gate"):
            c = centre(e)
            if c is None:
                continue
            for key in ("ref", "name"):
                r = norm(t.get(key))
                if r:
                    tgt = stands if a == "parking_position" else gates
                    tgt.setdefault(r, c)
        elif a == "runway" and t.get("ref") and e.get("geometry"):
            g = e["geometry"]
            p0, p1 = (g[0]["lat"], g[0]["lon"]), (g[-1]["lat"], g[-1]["lon"])
            b01 = bearing(p0, p1)
            for des in t["ref"].split("/"):
                des = des.strip().upper()
                m = re.match(r"^(\d{1,2})([LCR]?)$", des)
                if not m:
                    continue
                hd = int(m.group(1)) * 10
                diff = min(abs(hd - b01), 360 - abs(hd - b01))
                # take-off on XX starts at the end from which the runway points towards XX*10 deg
                thr[des] = p0 if diff < 90 else p1
    return stands, gates, thr


def main() -> None:
    dep = load_all(keep_stand_str=True).filter(pl.col("PHASE_mvt") == "DEP").select(
        ["MVT_ID_mvt", pl.col("APT").cast(pl.String), "STAND_ok", pl.col("RUNWAY_mvt").cast(pl.String), "y", "split",
         "t_mvt", "t_aobt"]
    )
    rows, match = [], []
    for apt in APTS:
        try:
            stands, gates, thr = parse(apt)
        except Exception as ex:  # noqa: BLE001
            print(apt, "no OSM data", ex)
            continue
        sub = dep.filter(pl.col("APT") == apt)
        used = sub.group_by(["STAND_ok", "RUNWAY_mvt"]).len()
        lat0 = np.mean([v[0] for v in stands.values()]) if stands else 0
        lon0 = np.mean([v[1] for v in stands.values()]) if stands else 0
        n_ok = n_all = 0
        for s, r, n in used.iter_rows():
            k = norm(s)
            c = stands.get(k) if k else None
            src = "stand" if c else None
            if c is None and k:
                c = gates.get(k)
                src = "gate" if c else None
            if c is None and k:
                # sub-stand variants: data '111' vs OSM '111A'/'111B', or data '151A' vs OSM '151'
                kb = re.sub(r"[A-Z]+$", "", k)
                cand = [v for r_, v in stands.items() if re.sub(r"[A-Z]+$", "", r_) == kb and kb]
                if cand:
                    c = (float(np.mean([v[0] for v in cand])), float(np.mean([v[1] for v in cand])))
                    src = "stand_base"
            th = thr.get((r or "").upper())
            dist = hav(c, th) if (c and th) else None
            n_all += n
            n_ok += n if c else 0
            rows.append((apt, s, r, src, c[0] if c else None, c[1] if c else None,
                         (c[1] - lon0) * 111320 * math.cos(math.radians(lat0)) if c else None,
                         (c[0] - lat0) * 110540 if c else None, dist))
        match.append((apt, len(stands), len(gates), sorted(thr), n_ok / max(n_all, 1)))
    print(pl.DataFrame(match, schema=["apt", "osm_stands", "osm_gates", "thresholds", "dep_rows_matched"], orient="row"))
    tab = pl.DataFrame(rows, schema=["APT", "STAND_ok", "RUNWAY_mvt", "geo_src", "lat", "lon", "geo_x_m", "geo_y_m",
                                     "geo_dist_thr_m"], orient="row")
    tab.write_parquet(f"{SCRATCH}/stand_geo_table.parquet")
    f = dep.join(tab, on=["APT", "STAND_ok", "RUNWAY_mvt"], how="left")
    print("ranking rows matched:", f.filter(pl.col("split") == "rank").select(pl.col("geo_x_m").is_not_null().mean()).item())
    f.select(["MVT_ID_mvt", "geo_x_m", "geo_y_m", "geo_dist_thr_m",
              pl.col("geo_x_m").is_not_null().cast(pl.Int8).alias("geo_matched")]).write_parquet(f"{SCRATCH}/stand_geo_feats.parquet")

    # relation: stand x runway median (clean training) vs distance
    tr = f.filter((pl.col("split") == "train") & (pl.col("y") >= 30) & (pl.col("y") <= 7200))
    cell = tr.group_by(["APT", "STAND_ok", "RUNWAY_mvt"]).agg(
        pl.len().alias("n"), pl.col("y").median().alias("y_med"), pl.col("geo_dist_thr_m").first(),
        (pl.col("y") - (pl.col("t_mvt") - pl.col("t_aobt"))).median().alias("resid_med"),
    ).filter((pl.col("n") >= 30) & pl.col("geo_dist_thr_m").is_not_null())
    out = []
    for apt in APTS:
        c = cell.filter(pl.col("APT") == apt)
        if c.height < 10:
            continue
        x, y = c["geo_dist_thr_m"].to_numpy(), c["y_med"].to_numpy()
        w = c["n"].to_numpy().astype(float)
        # pooled over runways and within-runway (demeaned) correlation, slope s per km
        cc = np.corrcoef(x, y)[0, 1]
        dm = c.with_columns((pl.col("geo_dist_thr_m") - pl.col("geo_dist_thr_m").mean().over("RUNWAY_mvt")).alias("xd"),
                            (pl.col("y_med") - pl.col("y_med").mean().over("RUNWAY_mvt")).alias("yd"))
        xd, yd = dm["xd"].to_numpy(), dm["yd"].to_numpy()
        slope = float(np.sum(w * xd * yd) / np.sum(w * xd * xd)) * 1000
        cw = np.corrcoef(xd, yd)[0, 1]
        out.append((apt, c.height, float(np.median(x)), cc, cw, slope, float(np.sqrt(np.average((yd - slope / 1000 * xd) ** 2, weights=w))),
                    float(np.sqrt(np.average(yd ** 2, weights=w)))))
    print("## distance to departure threshold vs stand x runway median y (cells n>=30)\n",
          pl.DataFrame(out, schema=["apt", "cells", "dist_med_m", "corr_pooled", "corr_within_rwy", "sec_per_km",
                                    "resid_sd_after_dist", "sd_within_rwy"], orient="row"))


if __name__ == "__main__":
    main()
