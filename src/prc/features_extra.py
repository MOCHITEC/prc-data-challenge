"""v14 context features from the multi-perspective analysis (docs/analysis/README.md).

Every group reads only what survives in ranking.parquet: departures' take-off, schedule, stand, runway, NM clocks
and flight number, and arrivals' full rows (their in-block and taxi-in are not blanked). No departure's
BLOCK_TIME or TAXITIME is read; the input is the parity-masked frame, as for prc.features.context_features.

Groups are named by column prefix so a config can drop any of them from one cached feature build
(main's candidate mechanism: every column is computed always and hidden unless named in include_candidates):

  occ_   stand occupancy: the latest same-stand in-block before my take-off bounds my off-block from above,
         so take-off minus that in-block bounds the taxi-out from below (docs/analysis/stand.md section 2)
  ta_    turnaround pair: the arrival that last in-blocked at my stand, and whether *its* in-block was a
         copy of its scheduled time - the reporting chain's fingerprint (reporting_fingerprint.md section 3)
  fl_/dup_  flight-number format and duplicate records (reporting_fingerprint.md sections 4-5)
  seq_   runway take-off sequence: headway over the wake-pair minimum, saturated bursts, following density
  bk_    schedule backlog and arrivals taxiing in (queue_runway.md)
"""

from __future__ import annotations

import numpy as np
import polars as pl

COPY_TOL_S = 6  # 7 of 10 airports report BLOCK as minute + uniform +-6 s jitter (reporting_fingerprint.md 1)
OCC_LB_MAX_S = 1800
TA_OFFSET_S = 120
TA_MAX_GAP_S = 18 * 3600
HEADWAY_MAX_S = 3 * 3600  # beyond this the "neighbour" is in another data block (ranking: January -> July)
WTC_CODE = {"L": 0, "M": 1, "H": 2, "J": 3}
_KEY = 1e10  # stand code * _KEY + epoch seconds stays exact in float64 (codes < 9e5)

FL_FMT_CODE = {"null": 0, "dup_prefix": 1, "icao": 2, "iata": 3, "icao_alpha": 4, "other": 5}


def _ep(s: pl.Series) -> np.ndarray:
    return s.dt.epoch("s").cast(pl.Float64).to_numpy()


def _codes(values: pl.Series, vocab: dict) -> np.ndarray:
    out = np.full(len(values), np.nan)
    arr = values.to_list()
    for i, v in enumerate(arr):
        if v is not None:
            out[i] = vocab.setdefault(v, len(vocab))
    return out


def _prev_next(key_sorted: np.ndarray, q: np.ndarray, code_q: np.ndarray, side_prev: str = "right"):
    """Index of the latest key <= q (or < q with side_prev='left') and of the next key > q, within the same code."""
    n = len(key_sorted)
    ip = np.searchsorted(key_sorted, q, side=side_prev) - 1
    inx = np.searchsorted(key_sorted, q, side="right")
    code_k = np.floor(key_sorted / _KEY) if n else np.array([])
    ok_p = (ip >= 0) & np.isfinite(q)
    ok_p[ok_p] = code_k[ip[ok_p]] == code_q[ok_p]
    ok_n = (inx < n) & np.isfinite(q)
    ok_n[ok_n] = code_k[inx[ok_n]] == code_q[ok_n]
    return ip, ok_p, inx, ok_n


def _stand_features(d: pl.DataFrame, a: pl.DataFrame) -> dict[str, np.ndarray]:
    n = d.height
    vocab: dict = {}
    dc = _codes(d["STAND_mvt"], vocab)
    ac = _codes(a["STAND_mvt"], vocab)
    t = _ep(d["MVT_TIME_UTC_mvt"])
    aobt = _ep(d["AOBT_3_flt"])
    sched = _ep(d["SCHED_TIME_UTC_mvt"])
    tin = _ep(a["BLOCK_TIME_UTC_mvt"])
    a_sched = _ep(a["SCHED_TIME_UTC_mvt"])
    a_taxi = a["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    a_op = a["AIRCRAFT_OPERATOR_flt"].to_numpy()
    d_op = d["AIRCRAFT_OPERATOR_flt"].to_numpy()
    out = {}

    ok_a = np.isfinite(ac) & np.isfinite(tin)
    order = np.argsort(ac[ok_a] * _KEY + tin[ok_a], kind="stable")
    ka = (ac[ok_a] * _KEY + tin[ok_a])[order]
    tin_s, asch_s, ataxi_s = tin[ok_a][order], a_sched[ok_a][order], a_taxi[ok_a][order]
    aop_s = a_op[ok_a][order]
    acopy_s = (np.abs(tin_s - asch_s) <= COPY_TOL_S).astype(float)
    code_d = np.where(np.isfinite(dc), dc, -1.0)
    q = code_d * _KEY + t

    # --- occupancy -----------------------------------------------------------------------------------------
    ip, okp, inx, okn = _prev_next(ka, q, code_d)
    ipc = np.clip(ip, 0, max(len(ka) - 1, 0))
    inxc = np.clip(inx, 0, max(len(ka) - 1, 0))
    nan = np.full(n, np.nan)
    g1 = np.where(okp, t - tin_s[ipc] if len(ka) else nan, np.nan)
    ip2 = ipc - 1
    okp2 = okp & (ip >= 1)
    okp2[okp2] = np.floor(ka[ip2[okp2]] / _KEY) == code_d[okp2]
    g2 = np.where(okp2, t - tin_s[np.clip(ip2, 0, None)] if len(ka) else nan, np.nan)
    out["occ_g_arr_prev1"] = g1
    out["occ_g_arr_prev2"] = g2
    out["occ_g_arr_next1"] = np.where(okn, tin_s[inxc] - t if len(ka) else nan, np.nan)
    out["occ_aobt_minus_arr_prev1"] = aobt - (t - g1)
    out["occ_aobt_minus_arr_prev2"] = aobt - (t - g2)
    lb = np.where(g1 < OCC_LB_MAX_S, g1, np.nan)
    out["occ_lb"] = lb
    out["occ_lb_minus_aobt"] = lb - (t - aobt)
    out["occ_lb_minus_sched"] = lb - (t - sched)
    # rank of my take-off among this stand's arrivals, for "arrivals since the previous departure"
    n_arr_upto = np.searchsorted(ka, q, side="right").astype(float)

    # same-stand departures in take-off order (self excluded by construction)
    ok_d = np.isfinite(dc) & np.isfinite(t)
    idx = np.where(ok_d)[0]
    o = idx[np.argsort(dc[idx] * _KEY + t[idx], kind="stable")]
    same_prev = np.zeros(len(o), bool)
    same_prev[1:] = dc[o][1:] == dc[o][:-1]
    same_next = np.zeros(len(o), bool)
    same_next[:-1] = same_prev[1:]
    to = t[o]
    gp = np.full(len(o), np.nan)
    gn = np.full(len(o), np.nan)
    gp[1:] = to[1:] - to[:-1]
    gn[:-1] = to[1:] - to[:-1]
    nxt_aobt = np.full(len(o), np.nan)
    nxt_aobt[:-1] = aobt[o][1:]
    nup = n_arr_upto[o]
    since = np.full(len(o), np.nan)
    since[1:] = nup[1:] - nup[:-1]
    for name, v, m in (
        ("occ_g_dep_prev", gp, same_prev),
        ("occ_g_dep_next", gn, same_next),
        ("occ_n_arr_since_prev_dep", since, same_prev),
        ("occ_g_next_dep_aobt", nxt_aobt - to, same_next),
    ):
        col = np.full(n, np.nan)
        col[o] = np.where(m, v, np.nan)
        out[name] = col

    # --- turnaround fingerprint -----------------------------------------------------------------------------
    q2 = code_d * _KEY + (t - TA_OFFSET_S)
    jp, okj, _, _ = _prev_next(ka, q2, code_d)
    jpc = np.clip(jp, 0, max(len(ka) - 1, 0))
    if len(ka):
        gap = t - tin_s[jpc]
        okj &= gap < TA_MAX_GAP_S
        out["ta_copy"] = np.where(okj, acopy_s[jpc], np.nan)
        out["ta_gap_sched"] = np.where(okj, sched - tin_s[jpc], np.nan)
        out["ta_gap_mvt"] = np.where(okj, gap, np.nan)
        out["ta_delay"] = np.where(okj, tin_s[jpc] - asch_s[jpc], np.nan)
        out["ta_taxi"] = np.where(okj, ataxi_s[jpc], np.nan)
        same_op = np.array([x is not None and x == y for x, y in zip(aop_s[jpc], d_op)], dtype=float)
        out["ta_same_op"] = np.where(okj, same_op, np.nan)
        qs = code_d * _KEY + sched
        _, _, kn, okn2 = _prev_next(ka, qs, code_d)
        out["ta_next_inb_minus_sched"] = np.where(okn2, tin_s[np.clip(kn, 0, len(ka) - 1)] - sched, np.nan)
    else:
        for c in (
            "ta_copy",
            "ta_gap_sched",
            "ta_gap_mvt",
            "ta_delay",
            "ta_taxi",
            "ta_same_op",
            "ta_next_inb_minus_sched",
        ):
            out[c] = np.full(n, np.nan)
    return out


def _count_between(sorted_ev: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """# events in (lo, hi)."""
    return (np.searchsorted(sorted_ev, hi, side="left") - np.searchsorted(sorted_ev, lo, side="right")).astype(float)


SEQ_COLS = (
    "seq_hw_prev_pairq",
    "seq_hw_excess",
    "seq_lead_wtc",
    "seq_burst_pos",
    "seq_burst_len",
    "seq_burst_elapsed",
    "seq_burst_remaining",
    "seq_burst_pos_frac",
    "seq_hw_mean_prev5",
    "seq_hw_mean_next5",
    "seq_hw_min_prev3",
    "seq_arr_since_prevdep",
    "seq_arr_prev_s",
    "seq_arr_next_s",
    "seq_arr_pm10m",
    "seq_rwy_share_pm30",
)


def _runway_sequence(d: pl.DataFrame, a: pl.DataFrame) -> dict[str, np.ndarray]:
    """Port of analysis/queue_runway_prep.runway_sequence_features (the `seq` group), with headways beyond
    HEADWAY_MAX_S treated as missing."""
    n = d.height
    out = {k: np.full(n, np.nan) for k in SEQ_COLS}
    t_all = _ep(d["MVT_TIME_UTC_mvt"])
    wtc_all = d["WK_TBL_CAT_flt"].replace_strict(WTC_CODE, default=None).cast(pl.Float64).fill_null(1.0).to_numpy()
    rwy = d["RUNWAY_mvt"].fill_null("__NA__").to_numpy()
    a_rwy = a["RUNWAY_mvt"].fill_null("__NA__").to_numpy()
    a_t = _ep(a["MVT_TIME_UTC_mvt"])
    t_sorted_all = np.sort(t_all[np.isfinite(t_all)])
    for r in np.unique(rwy):
        idx = np.where((rwy == r) & np.isfinite(t_all))[0]
        if len(idx) < 3 or r == "__NA__":
            continue
        o = idx[np.argsort(t_all[idx], kind="stable")]
        t = t_all[o]
        hw = np.diff(t, prepend=np.nan)
        hw = np.where(hw <= HEADWAY_MAX_S, hw, np.nan)
        wtc = wtc_all[o]
        lead = np.concatenate([[np.nan], wtc[:-1]])
        pairq = np.full(len(o), np.nan)
        for lw in (0, 1, 2, 3):
            for fw in (0, 1, 2, 3):
                m = (lead == lw) & (wtc == fw) & (hw < 600)
                if m.sum() >= 50:
                    pairq[(lead == lw) & (wtc == fw)] = np.nanpercentile(hw[m], 5)
        short = hw[hw < 600]
        base_q = np.nanpercentile(short, 5) if np.isfinite(short).sum() > 50 else 60.0
        pairq = np.where(np.isfinite(pairq), pairq, base_q)
        excess = hw - pairq
        sat = np.isfinite(excess) & (excess <= 45)
        bid = np.cumsum(~sat)
        _, first_idx, counts = np.unique(bid, return_index=True, return_counts=True)
        blen = counts[bid - 1].astype(float)
        bstart = t[first_idx][bid - 1]
        bend = t[first_idx + counts - 1][bid - 1]
        i = np.arange(len(o))
        pos = (i - first_idx[bid - 1] + 1).astype(float)
        hw0 = np.nan_to_num(hw, nan=0.0)
        okh = np.isfinite(hw).astype(float)
        cs, cc = np.concatenate([[0.0], np.cumsum(hw0)]), np.concatenate([[0.0], np.cumsum(okh)])
        lo = np.clip(i - 4, 0, None)
        c_prev = cc[i + 1] - cc[lo]
        with np.errstate(all="ignore"):
            mprev = np.where(c_prev > 0, (cs[i + 1] - cs[lo]) / c_prev, np.nan)
        hi = np.clip(i + 5, None, len(o) - 1)
        with np.errstate(all="ignore"):
            mnext = np.where(i < len(o) - 1, (t[hi] - t) / np.maximum(hi - i, 1), np.nan)
        mnext = np.where(mnext <= HEADWAY_MAX_S, mnext, np.nan)
        hwp = np.lib.stride_tricks.sliding_window_view(np.concatenate([[np.nan] * 2, hw]), 3)
        with np.errstate(all="ignore"):
            minp3 = np.nanmin(np.where(np.isfinite(hwp), hwp, np.inf), axis=1)
        minp3 = np.where(np.isfinite(minp3), minp3, np.nan)
        at = np.sort(a_t[(a_rwy == r) & np.isfinite(a_t)])
        if len(at):
            tprev = np.concatenate([[np.nan], t[:-1]])
            since = np.where(np.isfinite(hw), _count_between(at, tprev, t), np.nan)
            ip = np.searchsorted(at, t, side="left") - 1
            inx = np.searchsorted(at, t, side="right")
            aprev = np.where(ip >= 0, t - at[np.clip(ip, 0, len(at) - 1)], np.nan)
            anext = np.where(inx < len(at), at[np.clip(inx, 0, len(at) - 1)] - t, np.nan)
            aprev = np.where(aprev <= HEADWAY_MAX_S, aprev, np.nan)
            anext = np.where(anext <= HEADWAY_MAX_S, anext, np.nan)
            pm10 = _count_between(at, t - 600, t + 600)
        else:
            since = np.zeros(len(o))
            aprev = anext = np.full(len(o), np.nan)
            pm10 = np.zeros(len(o))
        tot = _count_between(t_sorted_all, t - 1800, t + 1800)
        mine = _count_between(t, t - 1800, t + 1800)
        share = np.where(tot > 0, mine / np.maximum(tot, 1), np.nan)
        for k, v in zip(
            SEQ_COLS,
            (
                pairq,
                excess,
                lead,
                pos,
                blen,
                t - bstart,
                bend - t,
                pos / blen,
                mprev,
                mnext,
                minp3,
                since,
                aprev,
                anext,
                pm10,
                share,
            ),
        ):
            out[k][o] = v
    return out


def _backlog(d: pl.DataFrame, a: pl.DataFrame) -> dict[str, np.ndarray]:
    """Scheduled-but-not-yet-departed count at take-off, scheduled demand, and arrivals taxiing in."""
    t = _ep(d["MVT_TIME_UTC_mvt"])
    s = _ep(d["SCHED_TIME_UTC_mvt"])
    ok = np.isfinite(t) & np.isfinite(s)
    s_ok, t_ok = s[ok], t[ok]
    s_sorted = np.sort(s_ok)
    # gone(t) = #{j: s_j in [t-3600, t], t_j < t}: offline sweep in take-off order with a Fenwick tree on sched rank
    n_ev = len(s_sorted)
    tree = np.zeros(n_ev + 1)
    ev_order = np.argsort(t_ok, kind="stable")
    t_ev = t_ok[ev_order]
    s_rank = np.searchsorted(s_sorted, s_ok[ev_order], side="left")
    anchors = np.argsort(np.where(np.isfinite(t), t, np.inf), kind="stable")
    hi_all = np.searchsorted(s_sorted, t + 1e-6, side="right")
    lo_all = np.searchsorted(s_sorted, t - 3600 - 1e-6, side="left")
    gone = np.full(len(t), np.nan)
    ptr = 0
    for a_ in anchors:
        ta = t[a_]
        if not np.isfinite(ta):
            break
        while ptr < len(t_ev) and t_ev[ptr] < ta:
            i = int(s_rank[ptr]) + 1
            while i <= n_ev:
                tree[i] += 1
                i += i & -i
            ptr += 1
        acc = 0.0
        i = int(hi_all[a_])
        while i > 0:
            acc += tree[i]
            i -= i & -i
        i = int(lo_all[a_])
        while i > 0:
            acc -= tree[i]
            i -= i & -i
        gone[a_] = acc
    n_sched = _count_between(s_sorted, t - 3600 - 1e-6, t + 1e-6)
    own = (np.isfinite(s) & (s >= t - 3600) & (s <= t)).astype(float)
    out = {
        "bk_sched_backlog_60m": np.where(np.isfinite(t), n_sched - gone - own, np.nan),
        "bk_sched_demand_next30m": _count_between(s_sorted, t, t + 1800),
        "bk_sched_demand_prev30m": _count_between(s_sorted, t - 1800, t),
    }
    ao = _ep(d["AOBT_3_flt"])
    al = _ep(a["MVT_TIME_UTC_mvt"])
    ab = _ep(a["BLOCK_TIME_UTC_mvt"])
    ti = a["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    oka = np.isfinite(al) & np.isfinite(ab) & (ab > al) & (ab - al < 7200)
    sl, sb = np.sort(al[oka]), np.sort(ab[oka])

    def insys(x: np.ndarray) -> np.ndarray:
        return (np.searchsorted(sl, x, side="right") - np.searchsorted(sb, x, side="right")).astype(float)

    out["bk_arr_taxiing_at_tko"] = np.where(np.isfinite(t), insys(t), np.nan)
    out["bk_arr_taxiing_at_aobt"] = np.where(np.isfinite(ao), insys(ao), np.nan)
    order = np.argsort(al[oka])
    tl, tv = al[oka][order], ti[oka][order]
    cs = np.concatenate([[0.0], np.cumsum(tv)])
    hi = np.searchsorted(tl, t, side="left")
    lo = np.searchsorted(tl, t - 1200, side="left")
    cnt = hi - lo
    with np.errstate(all="ignore"):
        out["bk_arr_taxi_in_landed_20m"] = np.where(cnt > 0, (cs[hi] - cs[lo]) / cnt, np.nan)
    return out


def _flight_format() -> pl.Expr:
    f = pl.col("FLIGHT_mvt")
    p1, p2, p3 = f.str.slice(0, 1), f.str.slice(1, 1), f.str.slice(2, 1)
    rest = f.str.slice(3)
    dup = f.str.contains(r"^[A-Z]{3}[A-Z]") & (
        rest.str.starts_with(p1 + p2 + p3) | rest.str.starts_with(p2 + p3) | rest.str.starts_with(p3)
    )
    return (
        pl.when(f.is_null())
        .then(pl.lit("null"))
        .when(dup)
        .then(pl.lit("dup_prefix"))
        .when(f.str.contains(r"^[A-Z]{3}\d"))
        .then(pl.lit("icao"))
        .when(f.str.contains(r"^[A-Z0-9]{2}\d{1,4}[A-Z]?$"))
        .then(pl.lit("iata"))
        .when(f.str.contains(r"^[A-Z]{3}[A-Z]"))
        .then(pl.lit("icao_alpha"))
        .otherwise(pl.lit("other"))
    )


def flight_features(dep: pl.DataFrame) -> pl.DataFrame:
    """Flight-number fingerprints and duplicate records, per MVT_ID. Rates only (no counts that scale with the
    length of the frame: training is a year, ranking is two months)."""
    m = pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Float64)
    df = dep.select("MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt")
    df = df.with_columns(
        _flight_format().replace_strict(FL_FMT_CODE, return_dtype=pl.Float32).alias("fl_fmt_code"),
        pl.col("FLIGHT_mvt").str.len_chars().cast(pl.Float32).alias("fl_len"),
        m.alias("_m"),
    )
    grp = ["ADEP_mvt", "FLIGHT_mvt"]
    n_other = pl.len().over(grp) - 1
    df = df.with_columns(
        pl.when(pl.col("FLIGHT_mvt").is_not_null() & (n_other > 0))
        .then((pl.col("_m").sum().over(grp) - pl.col("_m")) / n_other)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("fl_nm_rate_other")
    )
    key = ["ADEP_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"]
    has = pl.col("FLIGHT_mvt").is_not_null()
    unm = 1.0 - pl.col("_m")
    df = df.with_columns(
        pl.when(has).then(pl.len().over(key)).otherwise(1).cast(pl.Float32).alias("dup_n"),
        pl.when(has).then(unm.sum().over(key) - unm).otherwise(0).cast(pl.Float32).alias("dup_n_unm_twins"),
        pl.when(has)
        .then(pl.col("_m").sum().over(key) - pl.col("_m"))
        .otherwise(0)
        .cast(pl.Float32)
        .alias("dup_n_m_twins"),
        pl.when(has)
        .then(
            (pl.col("MVT_TIME_UTC_mvt").max().over(key) - pl.col("MVT_TIME_UTC_mvt").min().over(key)).dt.total_seconds()
        )
        .otherwise(None)
        .cast(pl.Float32)
        .alias("dup_mvt_span"),
    )
    return df.select(
        "MVT_ID_mvt",
        "fl_fmt_code",
        "fl_len",
        "fl_nm_rate_other",
        "dup_n",
        "dup_n_unm_twins",
        "dup_n_m_twins",
        "dup_mvt_span",
    )


def extra_context_features(masked: pl.DataFrame, dep_ids: pl.Series) -> pl.DataFrame:
    """occ_/ta_/seq_/bk_ columns for the departures in dep_ids, computed per airport from the masked frame."""
    cols_d = [
        "MVT_ID_mvt",
        "ADEP_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "MVT_TIME_UTC_mvt",
        "AOBT_3_flt",
        "SCHED_TIME_UTC_mvt",
        "WK_TBL_CAT_flt",
        "AIRCRAFT_OPERATOR_flt",
    ]
    cols_a = [
        "ADES_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "AIRCRAFT_OPERATOR_flt",
    ]
    dep_all = masked.filter(pl.col("PHASE_mvt") == "DEP").select(cols_d)
    arr_all = masked.filter(pl.col("PHASE_mvt") == "ARR").select(cols_a)
    frames = []
    for apt in dep_all["ADEP_mvt"].drop_nulls().unique().sort().to_list():
        d = dep_all.filter(pl.col("ADEP_mvt") == apt)
        a = arr_all.filter(pl.col("ADES_mvt") == apt)
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        cols.update(_stand_features(d, a))
        cols.update(_runway_sequence(d, a))
        cols.update(_backlog(d, a))
        frames.append(pl.DataFrame(cols))
    out = pl.concat(frames, how="vertical_relaxed")
    out = out.filter(pl.col("MVT_ID_mvt").is_in(dep_ids.implode()))
    num = [c for c in out.columns if c != "MVT_ID_mvt"]
    return out.with_columns([pl.col(c).cast(pl.Float32).fill_nan(None) for c in num])


OCC_COLS = (
    "occ_g_arr_prev1",
    "occ_g_arr_prev2",
    "occ_g_arr_next1",
    "occ_aobt_minus_arr_prev1",
    "occ_aobt_minus_arr_prev2",
    "occ_lb",
    "occ_lb_minus_aobt",
    "occ_lb_minus_sched",
    "occ_g_dep_prev",
    "occ_g_dep_next",
    "occ_n_arr_since_prev_dep",
    "occ_g_next_dep_aobt",
)
TA_COLS = ("ta_copy", "ta_gap_sched", "ta_gap_mvt", "ta_delay", "ta_taxi", "ta_same_op", "ta_next_inb_minus_sched")
FL_COLS = ("fl_fmt_code", "fl_len", "fl_nm_rate_other", "dup_n", "dup_n_unm_twins", "dup_n_m_twins", "dup_mvt_span")
BK_COLS = (
    "bk_sched_backlog_60m",
    "bk_sched_demand_next30m",
    "bk_sched_demand_prev30m",
    "bk_arr_taxiing_at_tko",
    "bk_arr_taxiing_at_aobt",
    "bk_arr_taxi_in_landed_20m",
)
EXTRA_GROUPS = {"occ": OCC_COLS, "ta": TA_COLS, "fl": FL_COLS, "seq": SEQ_COLS, "bk": BK_COLS}
EXTRA_COLS = tuple(c for cols in EXTRA_GROUPS.values() for c in cols)


def extra_candidate_features(ctx_src: pl.DataFrame, dep_ids: pl.Series) -> pl.DataFrame:
    """All v14-analysis groups for the departures in dep_ids, one row per MVT_ID, columns EXTRA_COLS."""
    ctx = extra_context_features(ctx_src, dep_ids)
    fl = flight_features(ctx_src.filter(pl.col("PHASE_mvt") == "DEP")).filter(
        pl.col("MVT_ID_mvt").is_in(dep_ids.implode())
    )
    return (
        pl.DataFrame({"MVT_ID_mvt": dep_ids})
        .join(ctx, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(fl, on="MVT_ID_mvt", how="left", maintain_order="left")
        .select(["MVT_ID_mvt", *EXTRA_COLS])
    )
