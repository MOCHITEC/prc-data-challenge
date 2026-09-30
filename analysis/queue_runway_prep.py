"""queue_runway: build the analysis frame (existing features + new label-free runway/queue features).

Run:  POLARS_MAX_THREADS=2 .venv/bin/python analysis/queue_runway_prep.py
Out:  data/interim/queue_runway/{base,new}.parquet

Parity: every new feature is computed from the *masked* view (departure BLOCK/TAXITIME nulled for training and
ranking alike), i.e. only from take-off times, runway, stand, NM clocks, schedule, and full ARR rows. Features
that need label statistics (reference taxi times) are built per fold in queue_runway_eval.py.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prc.clocks import with_clock_source  # noqa: E402
from prc.features import clock_and_static_features, context_features, mask_departures  # noqa: E402

OUT = os.path.join(ROOT, "data", "interim", "queue_runway")
COLS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "IOBT_flt",
    "EOBT_1_flt", "AOBT_3_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "ADES_flt", "ADES_FILED_flt",
    "AIRCRAFT_TYPE_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
]
WTC_CODE = {"L": 0, "M": 1, "H": 2, "J": 3}


def load(apt: str | None = None) -> pl.LazyFrame:
    files = sorted(glob.glob(os.path.join(ROOT, "data/raw/training_*.parquet")))
    tr = pl.scan_parquet(files).select(COLS).with_columns(pl.lit(0, pl.Int8).alias("is_rank"))
    rk = pl.scan_parquet(os.path.join(ROOT, "data/raw/ranking.parquet")).select(COLS).with_columns(
        pl.lit(1, pl.Int8).alias("is_rank")
    )
    lf = pl.concat([tr, rk], how="vertical_relaxed")
    if apt is not None:
        lf = lf.filter(
            ((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == apt))
            | ((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt))
        )
    return lf


def ep(s: pl.Series) -> np.ndarray:
    return s.dt.epoch("s").cast(pl.Float64).to_numpy()


def _count_between(sorted_ev: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """# events in (lo, hi)."""
    return (np.searchsorted(sorted_ev, hi, side="left") - np.searchsorted(sorted_ev, lo, side="right")).astype(
        np.float64
    )


def runway_sequence_features(d: pl.DataFrame, a: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per departure, features of its runway's take-off sequence. d: departures of one airport (any order)."""
    n = d.height
    out: dict[str, np.ndarray] = {}
    names = [
        "hw_prev_pairq", "hw_excess", "lead_wtc", "burst_pos", "burst_len", "burst_elapsed", "burst_remaining",
        "burst_pos_frac", "hw_mean_prev5", "hw_mean_next5", "hw_min_prev3", "arr_same_rwy_since_prevdep",
        "arr_same_rwy_prev_s", "arr_same_rwy_next_s", "arr_same_rwy_pm10m", "nbr_seq_mvt_aobt_prev5",
        "nbr_seq_mvt_aobt_next5", "nbr_seq_mvt_aobt_pm5", "nbr_seq_mvt_sched_pm5", "overtaken_by_aobt",
        "overtook_aobt", "rank_shift_aobt", "rwy_share_pm30",
    ]
    for k in names:
        out[k] = np.full(n, np.nan)
    t_all = ep(d["MVT_TIME_UTC_mvt"])
    aobt_all = ep(d["AOBT_3_flt"])
    sched_all = ep(d["SCHED_TIME_UTC_mvt"])
    wtc_all = d["WK_TBL_CAT_flt"].replace_strict(WTC_CODE, default=None).cast(pl.Float64).fill_null(1.0).to_numpy()
    rwy = d["RUNWAY_mvt"].fill_null("__NA__").to_numpy()
    a_rwy = a["RUNWAY_mvt"].fill_null("__NA__").to_numpy()
    a_t = ep(a["MVT_TIME_UTC_mvt"])
    # all-runway dep times for share
    t_sorted_all = np.sort(t_all[np.isfinite(t_all)])
    for r in np.unique(rwy):
        idx = np.where((rwy == r) & np.isfinite(t_all))[0]
        if len(idx) < 3:
            continue
        o = idx[np.argsort(t_all[idx], kind="stable")]
        t = t_all[o]
        hw = np.diff(t, prepend=np.nan)  # gap to previous dep on the same runway
        hw_n = np.append(t[1:] - t[:-1], np.nan)
        wtc = wtc_all[o]
        lead = np.concatenate([[np.nan], wtc[:-1]])
        # empirical minimum separation per (lead, follow) pair on this runway: 5th pct of headways < 10 min
        pairq = np.full(len(o), np.nan)
        for lw in (0, 1, 2, 3):
            for fw in (0, 1, 2, 3):
                m = (lead == lw) & (wtc == fw) & (hw < 600)
                if m.sum() >= 50:
                    pairq[m | ((lead == lw) & (wtc == fw))] = np.nanpercentile(hw[m], 5)
        base_q = np.nanpercentile(hw[hw < 600], 5) if np.isfinite(hw).sum() > 50 else 60.0
        pairq = np.where(np.isfinite(pairq), pairq, base_q)
        excess = hw - pairq
        # saturated burst: consecutive departures whose headway is within 45 s of the pair minimum
        sat = np.isfinite(excess) & (excess <= 45)
        start_new = ~sat
        bid = np.cumsum(start_new)
        # burst stats
        _, first_idx, counts = np.unique(bid, return_index=True, return_counts=True)
        blen = counts[bid - 1]
        bstart = t[first_idx][bid - 1]
        bend = t[first_idx + counts - 1][bid - 1]
        pos = np.arange(len(o)) - first_idx[bid - 1] + 1
        # rolling headway means (prev 5 / next 5)
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(hw, nan=0.0))])
        k = 5
        i = np.arange(len(o))
        lo = np.clip(i - k + 1, 1, None)
        cnt = (i - lo + 1).astype(float)
        mprev = np.where(i >= 1, (cs[i + 1] - cs[lo]) / np.maximum(cnt, 1), np.nan)
        hi = np.clip(i + k, None, len(o) - 1)
        mnext = np.where(i < len(o) - 1, (t[hi] - t) / np.maximum(hi - i, 1), np.nan)
        hwp = np.lib.stride_tricks.sliding_window_view(np.concatenate([[np.nan] * 2, hw]), 3)
        with np.errstate(all="ignore"):
            minp3 = np.nanmin(np.where(np.isfinite(hwp), hwp, np.inf), axis=1)
        minp3 = np.where(np.isfinite(minp3), minp3, np.nan)
        # arrivals on the same runway (mixed mode)
        at = np.sort(a_t[(a_rwy == r) & np.isfinite(a_t)])
        if len(at):
            tprev = np.concatenate([[np.nan], t[:-1]])
            since = np.where(np.isfinite(tprev), _count_between(at, tprev, t), np.nan)
            ip = np.searchsorted(at, t, side="left") - 1
            inx = np.searchsorted(at, t, side="right")
            aprev = np.where(ip >= 0, t - at[np.clip(ip, 0, len(at) - 1)], np.nan)
            anext = np.where(inx < len(at), at[np.clip(inx, 0, len(at) - 1)] - t, np.nan)
            pm10 = _count_between(at, t - 600, t + 600)
        else:
            since = np.zeros(len(o))
            aprev = anext = np.full(len(o), np.nan)
            pm10 = np.zeros(len(o))
        # sequence neighbours' observable taxi proxies (their own labels are NOT used)
        ma = t - aobt_all[o]
        ma = np.where((ma > 0) & (ma < 7200), ma, np.nan)
        ms = t - sched_all[o]
        ms = np.where((ms > -3600) & (ms < 14400), ms, np.nan)

        def nb_mean(v: np.ndarray, back: int, fwd: int) -> np.ndarray:
            vv = np.nan_to_num(v, nan=0.0)
            ok = np.isfinite(v).astype(float)
            c1 = np.concatenate([[0.0], np.cumsum(vv)])
            c2 = np.concatenate([[0.0], np.cumsum(ok)])
            L = len(v)
            lo_ = np.clip(i - back, 0, L)
            hi_ = np.clip(i + fwd + 1, 0, L)
            s = c1[hi_] - c1[lo_]
            c = c2[hi_] - c2[lo_]
            # exclude self
            s = s - vv
            c = c - ok
            with np.errstate(all="ignore"):
                return np.where(c > 0, s / c, np.nan)

        p5 = nb_mean(ma, 5, 0)
        n5 = nb_mean(ma, 0, 5)
        pm5 = nb_mean(ma, 5, 5)
        pms = nb_mean(ms, 5, 5)
        # AOBT order vs take-off order on this runway
        ao = aobt_all[o]
        okA = np.isfinite(ao)
        overtaken = np.full(len(o), np.nan)
        overtook = np.full(len(o), np.nan)
        shift = np.full(len(o), np.nan)
        if okA.sum() > 10:
            jj = np.where(okA)[0]
            aoj = ao[jj]
            # rank by aobt among okA rows vs rank by take-off
            r_t = np.arange(len(jj))
            r_a = np.empty(len(jj))
            r_a[np.argsort(aoj, kind="stable")] = np.arange(len(jj))
            shift[jj] = r_t - r_a  # >0: took off later in sequence than its pushback order
            # overtaken: pushed after me (within 60 min) but took off before me -> count via window on local order
            W = min(40, len(jj) - 1)
            for s_ in range(1, W + 1):
                # j = i - s_ took off before i; overtaken if aobt_j > aobt_i
                a_prev = np.concatenate([np.full(s_, np.nan), aoj[:-s_]])
                a_next = np.concatenate([aoj[s_:], np.full(s_, np.nan)])
                if s_ == 1:
                    ov = np.zeros(len(jj))
                    ot = np.zeros(len(jj))
                ov += (a_prev > aoj + 0).astype(float)
                ot += (a_next < aoj).astype(float)
            overtaken[jj] = ov
            overtook[jj] = ot
        # share of airport departures on this runway within +-30 min
        tot = _count_between(t_sorted_all, t - 1800, t + 1800)
        mine = _count_between(t, t - 1800, t + 1800)
        share = np.where(tot > 0, (mine) / np.maximum(tot, 1), np.nan)
        for k_, v in (
            ("hw_prev_pairq", pairq), ("hw_excess", excess), ("lead_wtc", lead), ("burst_pos", pos),
            ("burst_len", blen), ("burst_elapsed", t - bstart), ("burst_remaining", bend - t),
            ("burst_pos_frac", pos / blen), ("hw_mean_prev5", mprev), ("hw_mean_next5", mnext),
            ("hw_min_prev3", minp3), ("arr_same_rwy_since_prevdep", since), ("arr_same_rwy_prev_s", aprev),
            ("arr_same_rwy_next_s", anext), ("arr_same_rwy_pm10m", pm10), ("nbr_seq_mvt_aobt_prev5", p5),
            ("nbr_seq_mvt_aobt_next5", n5), ("nbr_seq_mvt_aobt_pm5", pm5), ("nbr_seq_mvt_sched_pm5", pms),
            ("overtaken_by_aobt", overtaken), ("overtook_aobt", overtook), ("rank_shift_aobt", shift),
            ("rwy_share_pm30", share),
        ):
            out[k_][o] = v
    return out


def config_features(d: pl.DataFrame, a: pl.DataFrame) -> dict[str, np.ndarray]:
    """Runway configuration in 15-min bins: dominant arrival runway, #active dep/arr runways, time since change."""
    t_d = ep(d["MVT_TIME_UTC_mvt"])
    bins_d = np.floor(t_d / 900)
    dd = pl.DataFrame({"b": bins_d, "r": d["RUNWAY_mvt"].fill_null("NA"), "ph": "D"})
    aa = pl.DataFrame({"b": np.floor(ep(a["MVT_TIME_UTC_mvt"]) / 900), "r": a["RUNWAY_mvt"].fill_null("NA"), "ph": "A"})
    x = pl.concat([dd, aa]).filter(pl.col("b").is_not_nan())
    # use a +-1 bin (45 min) neighbourhood so sparse bins are stable
    x3 = pl.concat([x.with_columns(pl.col("b") + s) for s in (-1, 0, 1)])
    cnt = x3.group_by("b", "ph", "r").len()
    tot = cnt.group_by("b", "ph").agg(pl.col("len").sum().alias("tot"))
    cnt = cnt.join(tot, on=["b", "ph"]).with_columns((pl.col("len") / pl.col("tot")).alias("sh"))
    act = cnt.filter(pl.col("sh") >= 0.1)
    cfg = (
        act.sort("r")
        .group_by("b", "ph")
        .agg(pl.col("r").str.join("+").alias("set"), pl.len().alias("n_active"))
    )
    wide = cfg.pivot(on="ph", index="b", values=["set", "n_active"]).sort("b")
    for c in ("set_A", "set_D", "n_active_A", "n_active_D"):
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None).alias(c))
    wide = wide.with_columns((pl.col("set_D").fill_null("-") + "|" + pl.col("set_A").fill_null("-")).alias("config"))
    # time since config change (in bins), contiguous bins only
    b = wide["b"].to_numpy()
    c = wide["config"].to_numpy()
    since = np.zeros(len(b))
    until = np.zeros(len(b))
    for i in range(1, len(b)):
        since[i] = since[i - 1] + 1 if (c[i] == c[i - 1] and b[i] - b[i - 1] == 1) else 0
    for i in range(len(b) - 2, -1, -1):
        until[i] = until[i + 1] + 1 if (c[i] == c[i + 1] and b[i + 1] - b[i] == 1) else 0
    wide = wide.with_columns(pl.Series("cfg_since_change_bins", since), pl.Series("cfg_until_change_bins", until))
    j = pl.DataFrame({"b": bins_d}).join(wide, on="b", how="left", maintain_order="left")
    # dominant arrival runway (top share) at t
    top = (
        cnt.filter(pl.col("ph") == "A").sort("sh", descending=True).group_by("b").agg(pl.col("r").first().alias("arr_top"))
    )
    j = j.join(top, on="b", how="left", maintain_order="left")
    return {
        "cfg_config": j["config"].to_numpy(),
        "cfg_arr_top": j["arr_top"].to_numpy(),
        "cfg_n_dep_rwy": j["n_active_D"].cast(pl.Float64).to_numpy(),
        "cfg_n_arr_rwy": j["n_active_A"].cast(pl.Float64).to_numpy(),
        "cfg_since_change_bins": j["cfg_since_change_bins"].to_numpy(),
        "cfg_until_change_bins": j["cfg_until_change_bins"].to_numpy(),
    }


def schedule_backlog(d: pl.DataFrame) -> dict[str, np.ndarray]:
    """Schedule-based backlog at take-off: departures scheduled in the last 60 min that have not yet taken off
    (airport-wide and on my runway), and scheduled demand in the next 30 min. Uses SCHED + take-off times only."""
    t = ep(d["MVT_TIME_UTC_mvt"])
    s = ep(d["SCHED_TIME_UTC_mvt"])
    out = {}
    ok = np.isfinite(t) & np.isfinite(s)
    # backlog(t) = #{j: s_j in [t-3600, t], t_j > t}; computed as sched-in-window minus those already gone
    # (#{s_j in W and t_j <= t}) via sort on s and a 2-pass over a sparse grid is overkill: exact O(n log n) below.
    order = np.argsort(s[ok])
    s_ok, t_ok = s[ok][order], t[ok][order]
    # number scheduled in [t-3600, t]
    n_sched = _count_between(s_ok, t - 3600 - 1e-6, t + 1e-6)
    # of those, how many already took off (t_j < t): use a 2D count via sorting on sched and BIT over t ranks
    # approximate with offline sweep: for anchors sorted by t, process events by t_j.
    anchors = np.argsort(t)
    ev_order = np.argsort(t_ok)
    t_ev = t_ok[ev_order]
    s_ev = s_ok[ev_order]
    # Fenwick over sched ranks
    s_rank_vals = s_ok  # already sorted
    N = len(s_rank_vals)
    tree = np.zeros(N + 1)
    res_gone = np.zeros(len(t))
    ptr = 0
    s_ev_rank = np.searchsorted(s_rank_vals, s_ev, side="left")

    def add(i: int) -> None:
        i += 1
        while i <= N:
            tree[i] += 1
            i += i & -i

    def q(i: int) -> float:  # sum over ranks < i
        r = 0.0
        while i > 0:
            r += tree[i]
            i -= i & -i
        return r

    for a_ in anchors:
        ta = t[a_]
        if not np.isfinite(ta):
            res_gone[a_] = np.nan
            continue
        while ptr < len(t_ev) and t_ev[ptr] < ta:
            add(int(s_ev_rank[ptr]))
            ptr += 1
        hi = np.searchsorted(s_rank_vals, ta + 1e-6, side="right")
        lo = np.searchsorted(s_rank_vals, ta - 3600 - 1e-6, side="left")
        res_gone[a_] = q(hi) - q(lo)
    # self: if own sched in window, own t is not < own t so not in gone; subtract own from n_sched
    own = (np.isfinite(s) & (s >= t - 3600) & (s <= t)).astype(float)
    out["sched_backlog_60m"] = n_sched - res_gone - own
    out["sched_demand_next30m"] = _count_between(s_ok, t, t + 1800)
    out["sched_demand_prev30m"] = _count_between(s_ok, t - 1800, t)
    return out


def arrival_state(d: pl.DataFrame, a: pl.DataFrame) -> dict[str, np.ndarray]:
    """Arrival ground state: arrivals landed but not yet in-block at my take-off (taxiing-in count), and at the
    AOBT_3 pushback time; plus mean taxi-in of arrivals that *landed* in [t-20m, t) (landing-anchored, unlike the
    existing in-block anchored mean)."""
    t = ep(d["MVT_TIME_UTC_mvt"])
    ao = ep(d["AOBT_3_flt"])
    al = ep(a["MVT_TIME_UTC_mvt"])
    ab = ep(a["BLOCK_TIME_UTC_mvt"])
    ti = a["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    ok = np.isfinite(al) & np.isfinite(ab) & (ab > al) & (ab - al < 7200)
    sl, sb = np.sort(al[ok]), np.sort(ab[ok])

    def insys(x):
        return (np.searchsorted(sl, x, side="right") - np.searchsorted(sb, x, side="right")).astype(float)

    out = {"arr_taxiing_at_tko": insys(t), "arr_taxiing_at_aobt": np.where(np.isfinite(ao), insys(ao), np.nan)}
    order = np.argsort(al[ok])
    tl, tv = al[ok][order], ti[ok][order]
    cs = np.concatenate([[0.0], np.cumsum(tv)])
    hi = np.searchsorted(tl, t, side="left")
    lo = np.searchsorted(tl, t - 1200, side="left")
    n = hi - lo
    with np.errstate(all="ignore"):
        out["arr_taxi_in_landed_20m"] = np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)
    return out


def main() -> None:
    """Per-airport loop keeps peak RSS low (the whole-frame context_features call peaked at 6 GB)."""
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    apts = sorted(
        load().filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()["ADEP_mvt"].drop_nulls()
    )
    tmp = os.path.join(OUT, "_parts")
    os.makedirs(tmp, exist_ok=True)
    for apt in apts:
        sub = load(apt).collect()
        masked = mask_departures(sub)
        dep = sub.filter(pl.col("PHASE_mvt") == "DEP")
        # ---------- existing features (exact src/prc code) ----------
        row = clock_and_static_features(dep)
        ctx = context_features(masked, dep_ids=dep["MVT_ID_mvt"])
        lab = with_clock_source(dep.filter(pl.col("is_rank") == 0)).select("MVT_ID_mvt", "clock_src")
        base = row.join(ctx, on="MVT_ID_mvt", how="left").join(lab, on="MVT_ID_mvt", how="left")
        drop = [c for c in base.columns if c.endswith("_flt") and c not in ("WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt")]
        base.drop(drop, strict=False).write_parquet(os.path.join(tmp, f"base_{apt}.parquet"))
        # ---------- new features ----------
        d = masked.filter(pl.col("PHASE_mvt") == "DEP")
        a = masked.filter(pl.col("PHASE_mvt") == "ARR")
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        cols.update(runway_sequence_features(d, a))
        cols.update(config_features(d, a))
        cols.update(schedule_backlog(d))
        cols.update(arrival_state(d, a))
        pl.DataFrame(cols).write_parquet(os.path.join(tmp, f"new_{apt}.parquet"))
        print(apt, d.height, f"{time.time()-t0:.0f}s", flush=True)
        del sub, masked, dep, row, ctx, base, d, a, cols
    base = pl.concat(
        [pl.read_parquet(os.path.join(tmp, f"base_{a}.parquet")) for a in apts], how="diagonal_relaxed"
    ).with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym")
    )
    base.write_parquet(os.path.join(OUT, "base.parquet"))
    print("base", base.shape, f"{time.time()-t0:.0f}s", flush=True)
    del base
    new = pl.concat([pl.read_parquet(os.path.join(tmp, f"new_{a}.parquet")) for a in apts], how="diagonal")
    fl = [c for c, dt in new.schema.items() if dt == pl.Float64 and c != "MVT_ID_mvt"]
    new = new.with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in fl])
    new.write_parquet(os.path.join(OUT, "new.parquet"))
    print("new", new.shape, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
