"""Per-departure physical off-block reference from adsb.lol fixes (ODbL 1.0), for 2025 labelled days.

Detection is label-free (reads only ADS-B fixes, CALLSIGN_flt/FLIGHT_mvt, ADEP, MVT_TIME) and was fixed
before looking at BLOCK. It reuses the team's stage-1 stand finder (stand_departure: stand run = consecutive
ground fixes within 25 m; t_move_lo = last fix in the stand run, t_move = first fix outside it; push = share
of steps in the 180 s after the stand moving opposite to the reported track) and adds:

  t_onset  earliest fix in the stand run's tail from which gs >= 1 kt holds up to t_move (first motion,
           tighter than the 25 m test); falls back to t_move.
  t_taxi   first fix after t_move_lo with gs >= 5 kt, moving forward (displacement bearing within 45 deg of
           track), and gs >= 3 kt / forward for the next >= 20 s (start of sustained forward taxi).
  res_s    t_move - t_move_lo (sampling resolution of the stand exit).

Output: <day>_det.parquet in this folder (row level, local only; contains competition columns -> not shared).
"""
import os
import sys

import numpy as np
import polars as pl

A = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, A)
os.environ.setdefault("SCRATCH", os.path.join(A, "raw"))
import adsb_stage1_feasibility as s1  # noqa: E402

REPO = "/home/user/prc-data-challenge/"


def departures(day):
    y, m, _ = day.split("-")
    nm = int(m) % 12 + 1
    src = f"{REPO}data/raw/training_{y}-{m}-01_{y if m != '12' else int(y) + 1}-{nm:02d}-01.parquet"
    d0 = pl.lit(day).str.to_datetime(time_zone="UTC")
    ep = lambda c: (pl.col(c).dt.epoch("ms") / 1000)  # noqa: E731
    return (pl.scan_parquet(src)
            .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt") >= d0 + pl.duration(hours=2))
                    & (pl.col("MVT_TIME_UTC_mvt") < d0 + pl.duration(days=1)))
            .select("MVT_ID_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "RUNWAY_mvt", "MARKET_SEGMENT_flt",
                    pl.col("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("cs"),
                    pl.col("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("fn"),
                    ep("MVT_TIME_UTC_mvt").alias("t_mvt"), ep("BLOCK_TIME_UTC_mvt").alias("t_block"),
                    ep("AOBT_3_flt").alias("t_aobt3"), ep("SCHED_TIME_UTC_mvt").alias("t_sched"),
                    ep("EOBT_1_flt").alias("t_eobt1"), ep("LOBT_flt").alias("t_lobt"))
            .collect())


def bearing_arr(la1, lo1, la2, lo2):
    k = np.pi / 180
    x = (lo2 - lo1) * k * np.cos((la1 + la2) * 0.5 * k)
    y = (la2 - la1) * k
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def extra(t, la, lo, gs, trk, gr, res):
    """t_onset and t_taxi on the ground phase that stand_departure found (needs t_move_lo)."""
    out = {"t_onset": None, "t_taxi": None}
    if res.get("t_move_lo") is None:
        return out
    t_lo, t_mv, t_to = res["t_move_lo"], res["t_move"], res["t_to"] or (res["t_move"] + 3600)
    m = gr & (t >= t_lo - 600) & (t <= t_to)
    tt, a, o, g, h = t[m], la[m], lo[m], gs[m], trk[m]
    # onset: walk back from the first fix outside the stand while gs >= 1 kt
    i_mv = int(np.searchsorted(tt, t_mv))
    j = i_mv
    while j - 1 >= 0 and tt[j - 1] >= t_lo - 600 and not np.isnan(g[j - 1]) and g[j - 1] >= 1.0:
        j -= 1
    out["t_onset"] = float(tt[j]) if j < i_mv else float(t_mv)
    # taxi: forward and >= 5 kt, sustained 20 s
    i0 = int(np.searchsorted(tt, t_lo))
    n = len(tt)
    if n - i0 < 3:
        return out
    b = np.full(n, np.nan)
    b[:-1] = bearing_arr(a[:-1], o[:-1], a[1:], o[1:])
    d = s1.dist_m(a[:-1], o[:-1], a[1:], o[1:])
    step_ok = np.zeros(n, bool)
    step_ok[:-1] = d >= 2.0
    diff = np.abs((b - h + 180) % 360 - 180)
    fwd = step_ok & (diff <= 45)
    for i in range(i0, n - 1):
        if not (fwd[i] and g[i] >= 5):
            continue
        k = i + 1
        ok = True
        while k < n and tt[k] - tt[i] < 20:
            if np.isnan(g[k]) or g[k] < 3 or (step_ok[k] and diff[k] > 90):
                ok = False
                break
            k += 1
        if ok and k < n:
            out["t_taxi"] = float(tt[i])
            break
    return out


def run(day):
    tag = f"v{day.replace('-', '.')}-planes-readsb-prod-0"
    fx = pl.read_parquet(os.path.join(A, "raw", f"{tag}.parquet")).sort("t")
    dep = departures(day)
    keyed = pl.concat([fx.select("apt", "icao", pl.col("cs_ff").alias("k")),
                       fx.select("apt", "icao", pl.col("cs_bf").alias("k"))]).drop_nulls("k").unique()
    kd = {}
    for apt, icao, k in keyed.iter_rows():
        kd.setdefault((apt, k), set()).add(icao)
    parts = {k: v for k, v in fx.partition_by(["icao", "apt"], as_dict=True).items()}
    cache = {}

    def arr(icao, apt):
        if (icao, apt) not in cache:
            f = parts[(icao, apt)]
            cache[(icao, apt)] = (f["t"].to_numpy(), f["lat"].to_numpy(), f["lon"].to_numpy(),
                                  f["ground"].to_numpy(), f["gs"].fill_null(np.nan).to_numpy(),
                                  f["track"].fill_null(np.nan).to_numpy())
        return cache[(icao, apt)]

    recs = []
    for r in dep.iter_rows(named=True):
        rec = {"MVT_ID_mvt": r["MVT_ID_mvt"], "joined": False}
        cands = set()
        for key in ("cs", "fn"):
            if r[key] and (r["ADEP_mvt"], r[key]) in kd:
                cands = kd[(r["ADEP_mvt"], r[key])]
                break
        best = None
        for icao in sorted(cands):
            if (icao, r["ADEP_mvt"]) not in parts:
                continue
            t, la, lo, gr, gs, trk = arr(icao, r["ADEP_mvt"])
            if not np.any(np.abs(t - r["t_mvt"]) <= 3 * 3600):
                continue
            s = s1.stand_departure(t, la, lo, gr, gs, r["t_mvt"], trk)
            if s["t_to"] is None or abs(s["t_to"] - r["t_mvt"]) > 120:   # take-off must match MVT within 2 min
                continue
            sc = (s["ground_fix"], s["t_move"] is not None, -abs(s["t_to"] - r["t_mvt"]))
            if best is None or sc > best[0]:
                best = (sc, icao, s, (t, la, lo, gs, trk, gr))
        if best is not None:
            _, icao, s, (t, la, lo, gs, trk, gr) = best
            rec.update({"joined": True, "icao": icao, **{k: s[k] for k in (
                "ground_fix", "t_first", "t_dep1", "t_move", "t_move_lo", "t_to", "stand_rule", "stand_dur",
                "stand_nfix", "push", "post_dwell", "dep1_gs")}})
            rec.update(extra(t, la, lo, gs, trk, gr, s))
            for k in ("t_taxi", "t_dep1"):   # continuity: largest silence between the fix and take-off
                v = rec.get(k)
                if v is not None:
                    w = t[(t >= v) & (t <= s["t_to"])]
                    rec[f"gap_{k}"] = float(np.diff(w).max()) if len(w) > 1 else None
        recs.append(rec)
    R = pl.DataFrame(recs, infer_schema_length=None)
    J = dep.join(R, on="MVT_ID_mvt", how="left").with_columns(pl.lit(day).alias("day"))
    J.write_parquet(os.path.join(A, f"{day}_det.parquet"))
    print(day, J.height, "deps;", J["joined"].sum(), "joined;", J["t_move"].is_not_null().sum(), "stand exits")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
