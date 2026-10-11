"""ADS-B stage 1: can open adsb.lol data see departures leave the stand at our 10 airports, and how well?

Feasibility check on two days only. Read-only with respect to the repo: nothing is written to `outputs/`,
`src/`, `configs/` or `data/`; no model is fitted. Written from scratch for this repo (no third-party
extractor code was consulted or copied).

Data source: adsb.lol `globe_history_2025` / `globe_history_2026` GitHub releases (readsb `trace_full_*.json`
files, one tarball per UTC day split into 2 GB parts). Licence: Open Database License (ODbL) 1.0, stated in
each release's README.txt and LICENSE-ODbL.txt ("This database is made available under the Open Database
License"). Attribution: "Contains data from adsb.lol, made available under the ODbL 1.0".

Sub-commands (run with the repo venv; SCRATCH defaults to the session scratchpad):

  sizes   <repo> <tag>        HEAD every part of one day's release, print sizes (no download)
  fetch   <repo> <tag>        download the parts of one day into $SCRATCH (refuses if disk is short)
  extract <tag>               stream the local parts as one tar, keep only fixes near the 10 airports
                              (<= 8 km of the ARP, on ground or <= 4,000 ft), write $SCRATCH/<tag>.parquet
  purge   <tag>               delete the raw parts of that day
  analyse <tag> <YYYY-MM-DD>  join to our departures of that day and print coverage / accuracy tables

Time: every timestamp is UTC. readsb `timestamp` is a Unix epoch (UTC midnight of the archive day); each
trace point's first element is seconds after it. Our parquet datetimes are tz-aware UTC.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from multiprocessing import Pool

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
SCRATCH = os.environ.get(
    "SCRATCH",
    "/tmp/claude-0/-home-user-prc-data-challenge/ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/adsb",
)
BASE = "https://github.com/adsblol"
SUFFIXES = ["aa", "ab", "ac", "ad", "ae", "af"]

# Airport reference points: data/external/airports.parquet (read at import so the filter is self-contained).
AIRPORTS = {
    "EDDF": (50.026706, 8.55835), "EDDM": (48.353802, 11.7861), "EGLL": (51.470748, -0.459909),
    "EHAM": (52.308601, 4.76389), "LEBL": (41.2971, 2.07846), "LEMD": (40.493407, -3.572249),
    "LFPG": (49.00896, 2.554117), "LIRF": (41.804532, 12.251998), "LSZH": (47.458056, 8.548056),
    "LTFM": (41.274874, 28.732136),
}
RADIUS_M = 8000.0   # every runway end of the 10 airports lies within ~7 km of its ARP [X]
ALT_MAX_FT = 4000   # keep the initial climb so a take-off time can be read off the trace


def dist_m(lat1, lon1, lat2, lon2):
    """Equirectangular distance in metres (fine at airport scale). Works on scalars or numpy arrays."""
    k = math.pi / 180.0
    x = (np.asarray(lon2) - lon1) * k * np.cos((np.asarray(lat2) + lat1) * 0.5 * k)
    y = (np.asarray(lat2) - lat1) * k
    return 6371000.0 * np.sqrt(x * x + y * y)


# ================================================================================ size / fetch / purge
def part_url(repo: str, tag: str, sfx: str) -> str:
    return f"{BASE}/{repo}/releases/download/{tag}/{tag}.tar.{sfx}"


def sizes(repo: str, tag: str) -> list[tuple[str, int]]:
    out = []
    for s in SUFFIXES:
        r = subprocess.run(["curl", "-sS", "-m", "60", "-o", "/dev/null", "-I", "-L", "-w",
                            "%{http_code} %header{content-length}", part_url(repo, tag, s)],
                           capture_output=True, text=True, check=False)
        code, *cl = r.stdout.split()
        if code != "200":
            break
        out.append((f"{tag}.tar.{s}", int(cl[-1])))
    return out


def fetch(repo: str, tag: str) -> None:
    parts = sizes(repo, tag)
    need = sum(n for _, n in parts)
    free = shutil.disk_usage(SCRATCH).free
    print(f"{tag}: {len(parts)} parts, {need / 1e9:.3f} GB; free {free / 1e9:.2f} GB")
    if need + 3e9 > free:  # keep >= 3 GB margin for the filtered output and everything else
        sys.exit("REFUSE: not enough disk for one raw day plus a 3 GB margin")
    for name, n in parts:
        dst = os.path.join(SCRATCH, name)
        subprocess.run(["curl", "-sS", "-L", "--retry", "3", "-o", dst, part_url(repo, tag, name[-2:])],
                       check=True)
        assert os.path.getsize(dst) == n, (name, os.path.getsize(dst), n)
    print("fetched", [p for p, _ in parts])


def local_parts(tag: str) -> list[str]:
    return sorted(os.path.join(SCRATCH, f) for f in os.listdir(SCRATCH) if f.startswith(tag + ".tar."))


def purge(tag: str) -> None:
    for p in local_parts(tag):
        os.remove(p)
        print("deleted", p)


# ============================================================================================ extract
class Concat:
    """File-like object that reads the split parts back-to-back (so the tar is never re-joined on disk)."""

    def __init__(self, paths):
        self.paths, self.i, self.f = list(paths), 0, None

    def read(self, n=-1):
        buf = b""
        while n < 0 or len(buf) < n:
            if self.f is None:
                if self.i >= len(self.paths):
                    break
                self.f = open(self.paths[self.i], "rb")  # noqa: SIM115 (closed when exhausted)
                self.i += 1
            chunk = self.f.read(-1 if n < 0 else n - len(buf))
            if not chunk:
                self.f.close()
                self.f = None
                continue
            buf += chunk
        return buf


def parse_trace(raw: bytes):
    """One readsb trace_full file -> list of kept fixes, or None.

    Trace point layout (readsb globe_history): [dt, lat, lon, alt_baro | "ground" | null, gs, track, flags,
    vrate, aircraft_dict | null, source, alt_geom, geom_rate, ias, roll]. flags bit 0 = stale, bit 1 = start
    of a new leg. The callsign ('flight') only appears in the aircraft dict, so it is forward- and
    back-filled within a leg.
    """
    try:
        d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    except Exception:  # noqa: BLE001
        return None
    tr = d.get("trace") or []
    if not tr:
        return None
    ts0 = float(d["timestamp"])
    lat = np.array([p[1] if p[1] is not None else np.nan for p in tr], dtype=float)
    lon = np.array([p[2] if p[2] is not None else np.nan for p in tr], dtype=float)
    near = np.full(len(tr), "", dtype=object)
    anyhit = False
    for apt, (a_lat, a_lon) in AIRPORTS.items():
        box = (np.abs(lat - a_lat) < 0.08) & (np.abs(lon - a_lon) < 0.12)
        if not box.any():
            continue
        m = box & (dist_m(a_lat, a_lon, lat, lon) <= RADIUS_M)
        if m.any():
            near[m] = apt
            anyhit = True
    if not anyhit:
        return None
    # callsign fill within legs
    n = len(tr)
    leg = np.cumsum([1 if (p[6] or 0) & 2 else 0 for p in tr])
    cs = [None] * n
    for i, p in enumerate(tr):
        ac = p[8]
        if isinstance(ac, dict) and ac.get("flight"):
            cs[i] = ac["flight"].strip().upper() or None
    ff, bf = [None] * n, [None] * n
    cur, cl = None, -1
    for i in range(n):
        if leg[i] != cl:
            cur, cl = None, leg[i]
        if cs[i]:
            cur = cs[i]
        ff[i] = cur
    cur, cl = None, -1
    for i in range(n - 1, -1, -1):
        if leg[i] != cl:
            cur, cl = None, leg[i]
        if cs[i]:
            cur = cs[i]
        bf[i] = cur
    rows = []
    icao = d.get("icao")
    typ = d.get("t")
    reg = d.get("r")
    for i, p in enumerate(tr):
        if not near[i]:
            continue
        alt = p[3]
        ground = alt == "ground"
        if not ground and alt is not None and alt > ALT_MAX_FT:
            continue
        rows.append((icao, near[i], ts0 + p[0], p[1], p[2], ground,
                     None if ground or alt is None else int(alt),
                     None if p[4] is None else float(p[4]), None if p[5] is None else float(p[5]),
                     int(p[6] or 0), None if p[9] is None else str(p[9]), ff[i], bf[i], int(leg[i]), typ, reg))
    return rows or None


def _gen(tag):
    with tarfile.open(fileobj=Concat(local_parts(tag)), mode="r|") as tf:
        for m in tf:
            if m.isfile() and "/traces/" in m.name and m.name.endswith(".json"):
                yield tf.extractfile(m).read()


def extract(tag: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["icao", "apt", "t", "lat", "lon", "ground", "alt_ft", "gs", "track", "flags", "src", "cs_ff",
            "cs_bf", "leg", "typ", "reg"]
    out = {c: [] for c in cols}
    chunks = []
    n_files = n_kept = n_fix = 0
    with Pool(3) as pool:
        for rows in pool.imap_unordered(parse_trace, _gen(tag), chunksize=64):
            n_files += 1
            if rows:
                n_kept += 1
                for r in rows:
                    for c, v in zip(cols, r):
                        out[c].append(v)
                n_fix += len(rows)
                if len(out["t"]) > 500_000:
                    chunks.append(pa.table(out))
                    out = {c: [] for c in cols}
            if n_files % 10000 == 0:
                print(f"  {n_files} traces, {n_kept} kept, {n_fix} fixes", flush=True)
    if out["t"] or not chunks:  # an empty final chunk has all-null column types and breaks concat_tables
        chunks.append(pa.table(out))
    tbl = pa.concat_tables(chunks)
    dst = os.path.join(SCRATCH, f"{tag}.parquet")
    pq.write_table(tbl, dst, compression="zstd")
    print(f"{tag}: {n_files} trace files, {n_kept} near an airport, {tbl.num_rows} fixes -> {dst} "
          f"({os.path.getsize(dst) / 1e6:.1f} MB)")


# ============================================================================================ analyse
STAT_R = 25.0       # m: fixes within this radius of the stand anchor count as "not yet moved"
STAT_MIN_S = 60.0   # s: the anchor must be held for at least this long to count as a stationary period
MATCH_WIN = 1800.0  # s: ADS-B take-off (or last ground fix) must lie within +-30 min of MVT_TIME
LATE_G = 1200       # s: late NM off-block stamp, g = AOBT_3 - BLOCK
FILL_TOL = 12       # s: |BLOCK - SCHED| <= 12 s is a schedule fill (lirf-lfpg-mechanisms.md)
GAP_S = 600.0       # s: a silence this long on the ground = transponder off / not received


def our_departures(day: str):
    import polars as pl

    y, m, _ = day.split("-")
    # Feb and Jun 2026 exist only in the final-phase file (Jan+Feb+Jun+Jul 2026, released 2026-10-08)
    src = REPO_DIR + ("data/raw/final_ranking.parquet" if y == "2026" and m in ("02", "06") else
                      "data/raw/ranking.parquet" if y == "2026" else
                      f"data/raw/training_{y}-{m}-01_{y if m != '12' else int(y) + 1}-"
                      f"{int(m) % 12 + 1:02d}-01.parquet")
    d0 = pl.lit(day).str.to_datetime(time_zone="UTC")
    return (pl.scan_parquet(src)
            .filter((pl.col("PHASE_mvt") == "DEP")
                    & (pl.col("MVT_TIME_UTC_mvt") >= d0 + pl.duration(hours=2))  # stand phase inside the day
                    & (pl.col("MVT_TIME_UTC_mvt") < d0 + pl.duration(days=1)))
            .collect())


def stand_departure(t, lat, lon, ground, gs, t_mvt, trk=None):
    """One aircraft's fixes at the origin (sorted by t, UTC epoch s) -> the departure ground phase that ends
    with the take-off nearest MVT_TIME, and the candidate off-block observations in it.

    take-off   first airborne fix whose predecessor is a ground fix, within MATCH_WIN of MVT_TIME. If none,
               the airborne fix nearest MVT_TIME is kept as t_to and the ground phase ends at MVT_TIME+2 min.
    phase      the ground fixes before take-off, back to the previous airborne fix (the arrival leg, if
               seen) and never more than 4 h.
    t_dep1     first fix after the last silence of >= GAP_S (10 min) in the phase, i.e. when the
               transponder came back on for the departure; the first fix of the phase if there is no
               silence.
    runs       maximal runs of consecutive ground fixes that all stay within STAT_R m of the run's first
               fix. A run may span a silence when the fixes either side of it are in the same place.
    stand      if the departure fixes start after a silence, or the phase does not open with a landing:
               the run that holds the t_dep1 fix (it may reach back across the silence). It counts only if
               it has >= 2 fixes, or its single fix reports gs <= 1 kt (otherwise the aircraft was already
               moving when first seen).
               if the aircraft stayed visible from landing to take-off: the longest run, if >= 10 min.
    t_move     first fix after the stand run (movement had begun); t_move_lo = last fix of the stand run
               (movement had not begun). The truth lies between; the gap is the resolution.
    """
    res = {"ground_fix": False, "t_first": None, "t_dep1": None, "t_move": None, "t_move_lo": None,
           "t_to": None, "n_ground": 0, "stand_dur": None, "dep1_gs": None,
           "gap_before_dep1": None, "opens_with_landing": None, "stand_rule": None, "stand_nfix": None, "push": None,
           "post_dwell": None}
    air_idx = np.where(~ground)[0]
    to_idx = [i for i in air_idx if i > 0 and ground[i - 1] and abs(t[i] - t_mvt) <= MATCH_WIN]
    if to_idx:
        i_to = min(to_idx, key=lambda i: abs(t[i] - t_mvt))
        res["t_to"] = float(t[i_to])
        end = i_to
    else:
        a = [i for i in air_idx if abs(t[i] - t_mvt) <= MATCH_WIN]
        if a:
            res["t_to"] = float(t[min(a, key=lambda i: abs(t[i] - t_mvt))])
        g = np.where(ground & (t <= t_mvt + 120) & (t >= t_mvt - 4 * 3600))[0]
        if len(g) == 0:
            return res
        end = g[-1] + 1
    t_end = t[end - 1]
    j = end - 1
    while j >= 0 and ground[j] and t_end - t[j] <= 4 * 3600:
        j -= 1
    idx = np.arange(j + 1, end)
    if len(idx) == 0:
        return res
    res["ground_fix"] = True
    res["n_ground"] = len(idx)
    tt, la, lo = t[idx], lat[idx], lon[idx]
    res["t_first"] = float(tt[0])
    dt = np.diff(tt)
    big = np.where(dt >= GAP_S)[0]
    k1 = int(big[-1] + 1) if len(big) else 0
    res["t_dep1"] = float(tt[k1])
    res["gap_before_dep1"] = float(dt[k1 - 1]) if k1 > 0 else None
    g1 = gs[idx[k1]]
    res["dep1_gs"] = None if np.isnan(g1) else float(g1)
    # episodes: runs of consecutive fixes within STAT_R of the run's first fix (a run may span a silence)
    n = len(idx)
    runs = []
    s0 = 0
    while s0 < n:
        e = s0
        while e + 1 < n and dist_m(la[s0], lo[s0], la[e + 1], lo[e + 1]) <= STAT_R:
            e += 1
        runs.append((s0, e))
        s0 = e + 1
    landed = j >= 0 and not ground[j] and tt[0] - t[j] < 300  # the phase opens with the arrival's landing
    res["opens_with_landing"] = bool(landed)
    stand = None
    if k1 > 0 or not landed:
        # the transponder came (back) on for the departure at k1: the stand is the run holding k1, which
        # may reach back across the silence if the aircraft was in the same place before it
        stand = next(r for r in runs if r[0] <= k1 <= r[1])
        res["stand_rule"] = "run_at_dep1"
    else:
        # no silence: the aircraft stayed visible from landing to take-off; the stand is the longest run,
        # and it must look like a turnaround (>= 10 min)
        cand = [r for r in runs if tt[r[1]] - tt[r[0]] >= GAP_S]
        if cand:
            stand = max(cand, key=lambda r: tt[r[1]] - tt[r[0]])
            res["stand_rule"] = "longest_run"
    if stand is not None and res["stand_rule"] == "run_at_dep1" and stand[1] == stand[0]:
        g0 = gs[idx[stand[0]]]
        if np.isnan(g0) or g0 > 1.0:
            res["stand_rule"] = "moving_when_first_seen"
            stand = None
    if stand is not None:
        s0, e = stand
        if e + 1 < n and tt[e + 1] <= t_end - 60:
            res["stand_dur"] = float(tt[e] - tt[s0])
            res["stand_nfix"] = int(e - s0 + 1)
            res["t_move_lo"] = float(tt[e])
            res["t_move"] = float(tt[e + 1])
            res["push"] = pushback_signature(tt, la, lo, None if trk is None else trk[idx], e)
            later = [tt[r[1]] - tt[r[0]] for r in runs if r[0] > e]
            res["post_dwell"] = float(max(later)) if later else 0.0
    return res


def bearing(lat1, lon1, lat2, lon2):
    k = math.pi / 180.0
    x = (lon2 - lon1) * k * math.cos((lat1 + lat2) * 0.5 * k)
    y = (lat2 - lat1) * k
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def pushback_signature(tt, la, lo, trk, e):
    """Share of the displacement steps in the 180 s after the stand run whose direction is opposite
    (> 120 deg off) to the reported track/heading, i.e. the aircraft moving backwards. None if no usable step.
    Label-free: reads positions and the transponder's own track only."""
    if trk is None:
        return None
    votes = []
    i = e
    while i + 1 < len(tt) and tt[i + 1] - tt[e] <= 180:
        d = float(dist_m(la[i], lo[i], la[i + 1], lo[i + 1]))
        h = trk[i + 1]
        if d >= 3.0 and h is not None and not np.isnan(h):
            b = bearing(la[i], lo[i], la[i + 1], lo[i + 1])
            diff = abs((b - h + 180.0) % 360.0 - 180.0)
            votes.append(diff > 120.0)
        i += 1
    return None if not votes else float(np.mean(votes))


def analyse(tag: str, day: str) -> None:
    import polars as pl

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(30)
    fx = pl.read_parquet(os.path.join(SCRATCH, f"{tag}.parquet"))
    print(f"[fixes] {fx.height} fixes, {fx['icao'].n_unique()} aircraft, "
          f"t range {pl.from_epoch(pl.Series([fx['t'].min(), fx['t'].max()]), time_unit='s').to_list()} UTC")
    print(fx.group_by("apt").agg(pl.len().alias("fixes"), pl.col("ground").sum().alias("ground_fixes"),
                                 pl.col("icao").n_unique().alias("aircraft")).sort("apt"))
    dep = our_departures(day)
    y = day[:4]
    dep = dep.with_columns(
        pl.col("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("cs"),
        pl.col("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("fn"),
        (pl.col("MVT_TIME_UTC_mvt").dt.epoch("ms") / 1000).alias("t_mvt"),
        (pl.col("AOBT_3_flt").dt.epoch("ms") / 1000).alias("t_aobt3"),
        (pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("ms") / 1000).alias("t_block"),
        (pl.col("SCHED_TIME_UTC_mvt").dt.epoch("ms") / 1000).alias("t_sched"),
    )
    print(f"[dep] {dep.height} departures with MVT_TIME in [{day} 02:00, {day} 24:00) UTC")

    # callsign -> icao candidates per airport (any fix with that callsign, forward or back filled)
    keyed = pl.concat([fx.select("apt", "icao", pl.col("cs_ff").alias("k")),
                       fx.select("apt", "icao", pl.col("cs_bf").alias("k"))]).drop_nulls("k").unique()
    by_icao = {k[0]: v for k, v in fx.sort("t").partition_by(["icao"], as_dict=True).items()}
    arrs = {}

    def arrays(icao, apt):
        key = (icao, apt)
        if key not in arrs:
            f = by_icao[icao].filter(pl.col("apt") == apt)
            arrs[key] = (f["t"].to_numpy(), f["lat"].to_numpy(), f["lon"].to_numpy(),
                         f["ground"].to_numpy(), f["gs"].fill_null(np.nan).to_numpy(),
                         f["track"].fill_null(np.nan).to_numpy())
        return arrs[key]

    kd = {}
    for apt, icao, k in keyed.iter_rows():
        kd.setdefault((apt, k), set()).add(icao)

    recs = []
    for r in dep.iter_rows(named=True):
        rec = {"MVT_ID_mvt": r["MVT_ID_mvt"], "join": None}
        cands = []
        for keyname in ("cs", "fn"):
            if r[keyname] and (r["ADEP_mvt"], r[keyname]) in kd:
                cands = sorted(kd[(r["ADEP_mvt"], r[keyname])])
                rec["join"] = keyname
                break
        best = None
        for icao in cands:
            t, la, lo, gr, gs, trk = arrays(icao, r["ADEP_mvt"])
            # the callsign must be carried near this departure's time (guards against a reused callsign)
            if not np.any(np.abs(t - r["t_mvt"]) <= 3 * 3600):
                continue
            s = stand_departure(t, la, lo, gr, gs, r["t_mvt"], trk)
            s["seen"] = True
            if s["t_to"] is None and not s["ground_fix"]:
                continue
            score = (s["ground_fix"], s["t_move"] is not None, -abs((s["t_to"] or 1e12) - r["t_mvt"]))
            if best is None or score > best[0]:
                best = (score, icao, s)
        if best is None:
            rec["join"] = None if not cands else rec["join"] + "_notime"
            recs.append(rec)
            continue
        _, icao, s = best
        rec.update({"icao": icao, **s})
        recs.append(rec)
    R = pl.DataFrame(recs, infer_schema_length=None)
    for c in ["seen", "ground_fix"]:
        if c not in R.columns:
            R = R.with_columns(pl.lit(None).alias(c))
    J = dep.join(R, on="MVT_ID_mvt", how="left").with_columns(
        pl.col("seen").fill_null(False), pl.col("ground_fix").fill_null(False),
    )
    if y == "2025":
        J = J.with_columns(
            (pl.col("t_aobt3") - pl.col("t_block")).alias("g"),
            ((pl.col("t_block") - pl.col("t_sched")).abs() <= FILL_TOL).alias("fill"),
            (pl.col("t_aobt3") - pl.col("t_block")).alias("e_aobt3"),
            (pl.col("t_to") - pl.col("t_mvt")).alias("e_to"),
        )
    out_path = os.path.join(SCRATCH, f"{tag}_joined.parquet")
    J.drop([c for c in J.columns if J[c].dtype == pl.Object]).write_parquet(out_path)
    print(f"[joined] -> {out_path}")
    report(J, y)


def q(s, p):
    return None if s.len() == 0 else float(s.quantile(p))


def acc(frame, col):
    e = frame[col].drop_nulls()
    if e.len() == 0:
        return {"n": 0}
    a = e.abs()
    return {"n": e.len(), "med": q(e, .5), "q25": q(e, .25), "q75": q(e, .75), "iqr": q(e, .75) - q(e, .25),
            "w2": float((a <= 120).mean()), "w5": float((a <= 300).mean()),
            "rmse": float((e ** 2).mean() ** .5), "mae": float(a.mean())}


EST = {  # estimator name -> column holding the estimated off-block (UTC epoch s)
    "move": "t_move",        # first fix outside the stand run, whatever the direction of motion
    "push": "t_push",        # t_move, only when the first motion is backwards (pushback signature >= 0.5)
                             # and the aircraft does not stop again for >= 15 min before take-off
    "push_lo": "t_push_lo",  # last fix inside the stand run, same flights as "push"
    "dep1": "t_dep1",        # first fix after the transponder came back on (no stand test at all)
}
PUSH_MIN = 0.5
POST_DWELL_MAX = 900.0


def add_estimators(J):
    import polars as pl

    # label-free guard: backwards first motion, and no later stop of >= POST_DWELL_MAX before take-off
    # (a second long stop means the push seen may not be the one that started this departure)
    ok = (pl.col("push").fill_null(0) >= PUSH_MIN) & (pl.col("post_dwell").fill_null(0) < POST_DWELL_MAX)
    return J.with_columns(pl.when(ok).then(pl.col("t_move")).alias("t_push"),
                          pl.when(ok).then(pl.col("t_move_lo")).alias("t_push_lo"))


def report(J, y):
    import polars as pl

    J = add_estimators(J)
    def frac(c):
        return pl.col(c).is_not_null().mean()
    aggs = [pl.len().alias("n"), pl.col("cs").is_not_null().mean().alias("cs_present"),
            (pl.col("join") == "cs").fill_null(False).mean().alias("join_cs"), (pl.col("join") == "fn").fill_null(False).mean().alias("join_fn"),
            pl.col("seen").mean().alias("seen_any"), pl.col("ground_fix").mean().alias("ground_before_to"),
            frac("t_move").alias("stand_episode"), frac("t_push").alias("push_detected")]
    print("\n[coverage] all departures in scope, per airport")
    cov = J.group_by("ADEP_mvt").agg(aggs).sort("ADEP_mvt")
    print(pl.concat([cov, J.select([pl.lit("ALL").alias("ADEP_mvt")] + aggs).cast(cov.schema)]))
    Jm = J.filter(pl.col("t_aobt3").is_not_null())
    aggs2 = [pl.len().alias("n"), (pl.col("join") == "cs").fill_null(False).mean().alias("join_cs"),
             pl.col("seen").mean().alias("seen_any"),
             pl.col("ground_fix").mean().alias("ground_before_to"), frac("t_move").alias("stand_episode"),
             frac("t_push").alias("push_detected")]
    print("\n[coverage] matched lane only (CALLSIGN_flt and AOBT_3 present)")
    cm = Jm.group_by("ADEP_mvt").agg(aggs2).sort("ADEP_mvt")
    print(pl.concat([cm, Jm.select([pl.lit("ALL").alias("ADEP_mvt")] + aggs2).cast(cm.schema)]))
    print("\n[sampling] on flights with ground fixes: fixes per phase, stand-exit resolution, silence")
    print(J.filter(pl.col("ground_fix")).group_by("ADEP_mvt").agg(
        pl.col("n_ground").median().alias("med_ground_fixes"),
        (pl.col("t_move") - pl.col("t_move_lo")).median().alias("med_exit_gap_s"),
        (pl.col("t_move") - pl.col("t_move_lo")).quantile(0.9).alias("p90_exit_gap_s"),
        pl.col("stand_dur").median().alias("med_stand_dur_s"),
        pl.col("gap_before_dep1").is_not_null().mean().alias("share_silence_before_dep1"),
        (pl.col("dep1_gs").fill_null(99) <= 5).mean().alias("share_dep1_slow"),
        (pl.col("t_mvt") - pl.col("t_dep1")).median().alias("med_mvt_minus_dep1_s"),
    ).sort("ADEP_mvt"))
    if y == "2026":
        print("\n[2026] AOBT_3 - t_est (s) where an estimate exists: same statistic as 2025, for stability")
        for name in ["move", "push", "dep1"]:
            x = J.filter(pl.col(EST[name]).is_not_null() & pl.col("t_aobt3").is_not_null()).with_columns(
                (pl.col("t_aobt3") - pl.col(EST[name])).alias("d"))
            t = x.group_by("ADEP_mvt").agg(pl.len(), pl.col("d").median().alias("med"),
                                           pl.col("d").quantile(0.25).alias("q25"),
                                           pl.col("d").quantile(0.75).alias("q75"),
                                           (pl.col("d") > 1200).mean().alias("share_gt_1200")).sort("ADEP_mvt")
            print(name, "pooled (n, median, share > 1200 s)",
                  x.select(pl.len(), pl.col("d").median().alias("m"), (pl.col("d") > 1200).mean().alias("s")).row(0))
            print(t)
        return
    print("\n[time check] ADS-B take-off minus MVT_TIME (s), joined flights with a take-off fix")
    print(J.filter(pl.col("e_to").is_not_null()).group_by("ADEP_mvt").agg(
        pl.len(), pl.col("e_to").median().alias("med"), pl.col("e_to").quantile(.1).alias("p10"),
        pl.col("e_to").quantile(.9).alias("p90"), (pl.col("e_to").abs() <= 120).mean().alias("w2")).sort("ADEP_mvt"))
    J = J.with_columns([(pl.col(c) - pl.col("t_block")).alias("e_" + k) for k, c in EST.items()])
    for name in ["move", "push", "push_lo", "dep1"]:
        M = J.filter(pl.col(EST[name]).is_not_null() & pl.col("t_aobt3").is_not_null())
        print(f"\n[accuracy:{name}] error vs BLOCK (s, +ve = later than BLOCK), same flights for ADS-B and AOBT_3")
        rows = []
        for apt in sorted(M["ADEP_mvt"].unique().to_list()) + ["ALL"]:
            f = M if apt == "ALL" else M.filter(pl.col("ADEP_mvt") == apt)
            a1, a2 = acc(f, "e_" + name), acc(f, "e_aobt3")
            rows.append({"apt": apt, "n": a1["n"], "adsb_med": a1["med"], "adsb_iqr": a1["iqr"],
                         "adsb_w2": a1["w2"], "adsb_w5": a1["w5"], "adsb_mae": a1["mae"],
                         "aobt3_med": a2["med"], "aobt3_iqr": a2["iqr"], "aobt3_w2": a2["w2"],
                         "aobt3_w5": a2["w5"], "aobt3_mae": a2["mae"],
                         "adsb_closer": float((f["e_" + name].abs() < f["e_aobt3"].abs()).mean())})
        print(pl.DataFrame(rows, infer_schema_length=None))
    # ------------------------------------------------------------- late NM stamp population
    L = J.filter(pl.col("t_aobt3").is_not_null() & (pl.col("g") > LATE_G) & ~pl.col("fill"))
    print(f"\n[late NM] g = AOBT_3 - BLOCK > {LATE_G} s, not a schedule fill: {L.height} rows this day")
    print(L.group_by("ADEP_mvt").agg(pl.len().alias("n"), pl.col("seen").mean().alias("seen"),
                                     pl.col("ground_fix").mean().alias("ground"),
                                     frac("t_move").alias("stand"), frac("t_push").alias("push"),
                                     pl.col("e_push").abs().median().alias("push_abs_med"),
                                     (pl.col("e_push").abs() <= 300).sum().alias("push_within5"),
                                     pl.col("e_aobt3").median().alias("aobt3_med")).sort("ADEP_mvt"))
    for name in ["move", "push", "dep1"]:
        f = L.filter(pl.col(EST[name]).is_not_null())
        print(f"late NM [{name}]", {"adsb": acc(f, "e_" + name), "aobt3": acc(f, "e_aobt3")})
    # ------------------------------------------------------------- what it would do for the target
    oof_path = REPO_DIR + "experiments/oof_main_gap-62dd7c6e88.parquet"
    if not os.path.exists(oof_path):  # informational only; the joined file is already written
        print(f"\n[target] skipped: {oof_path} not on this machine")
        return
    oof = pl.read_parquet(oof_path)
    for name in ["move", "push"]:
        Y = J.filter(pl.col(EST[name]).is_not_null() & pl.col("t_aobt3").is_not_null()).join(
            oof, on="MVT_ID_mvt", how="inner").with_columns(
            (pl.col("t_mvt") - pl.col("t_block")).alias("y"), (pl.col("t_mvt") - pl.col(EST[name])).alias("y_adsb"))
        Y = Y.with_columns((pl.col("oof") - pl.col("y")).alias("r_oof"), (pl.col("y_adsb") - pl.col("y")).alias("r_adsb"))
        print(f"\n[target:{name}] rows with an estimate and an OOF prediction: RMSE of y (s); raw ADS-B y = MVT - t_est")
        print(Y.group_by("ADEP_mvt").agg(pl.len(), (pl.col("r_oof") ** 2).mean().sqrt().alias("rmse_oof"),
                                         (pl.col("r_adsb") ** 2).mean().sqrt().alias("rmse_adsb_raw"),
                                         pl.col("r_adsb").median().alias("med_r_adsb"),
                                         (pl.col("r_adsb").abs() < pl.col("r_oof").abs()).mean().alias("adsb_closer")
                                         ).sort("ADEP_mvt"))
        print("pooled (n, rmse_oof, rmse_adsb_raw)",
              Y.select(pl.len(), (pl.col("r_oof") ** 2).mean().sqrt(), (pl.col("r_adsb") ** 2).mean().sqrt()).row(0))
        YL = Y.filter((pl.col("g") > LATE_G) & ~pl.col("fill"))
        print("late NM (n, rmse_oof, rmse_adsb_raw)", YL.select(pl.len(), (pl.col("r_oof") ** 2).mean().sqrt(),
                                                                (pl.col("r_adsb") ** 2).mean().sqrt()).row(0))


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    cmd, *a = sys.argv[1:]
    if cmd == "sizes":
        s = sizes(*a)
        for n, b in s:
            print(n, b)
        print("TOTAL", sum(b for _, b in s))
    elif cmd == "fetch":
        fetch(*a)
    elif cmd == "extract":
        extract(*a)
    elif cmd == "purge":
        purge(*a)
    elif cmd == "analyse":
        analyse(*a)
    else:
        sys.exit(__doc__)
