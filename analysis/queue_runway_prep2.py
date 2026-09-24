"""queue_runway: second-stage label-free features (residualised neighbour proxies, estimated-off-block queue).

All reference medians here are of quantities that are observable in ranking.parquet (NM taxi proxy
g = MVT - AOBT_3 for departures, arrival taxi-in for ARR rows), so they are fitted on training+ranking rows
together without touching any departure label.

Run: POLARS_MAX_THREADS=2 .venv/bin/python analysis/queue_runway_prep2.py   (after queue_runway_prep.py)
Out: data/interim/queue_runway/new2.parquet
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queue_runway_prep import OUT, _count_between, ep, load  # noqa: E402


def win_mean(anchor, times, vals, lo_off, hi_off, exclude_self=None):
    ok = np.isfinite(times) & np.isfinite(vals)
    o = np.argsort(times[ok])
    t, v = times[ok][o], vals[ok][o]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    hi = np.searchsorted(t, anchor + hi_off, side="right")
    lo = np.searchsorted(t, anchor + lo_off, side="left")
    s = cs[hi] - cs[lo]
    n = (hi - lo).astype(float)
    if exclude_self is not None:
        es = np.where(np.isfinite(exclude_self), exclude_self, 0.0)
        inw = np.isfinite(exclude_self)
        s = s - es
        n = n - inw
    with np.errstate(all="ignore"):
        return np.where(n > 0, s / n, np.nan), n


def main() -> None:
    frames = []
    apts = sorted(load().filter(pl.col("PHASE_mvt") == "DEP").select("ADEP_mvt").unique().collect()["ADEP_mvt"].drop_nulls())
    for apt in apts:
        sub = load(apt).collect()
        d = sub.filter(pl.col("PHASE_mvt") == "DEP")
        a = sub.filter(pl.col("PHASE_mvt") == "ARR")
        # --- departure NM-taxi proxy residual g - median_g(stand, runway) ---
        d = d.with_columns(
            ((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64)).alias("g")
        ).with_columns(pl.when(pl.col("g").is_between(0, 7200)).then(pl.col("g")).otherwise(None).alias("g"))
        d = d.with_columns(pl.col("g").median().over("STAND_mvt", "RUNWAY_mvt").alias("g_ref"),
                           pl.col("g").count().over("STAND_mvt", "RUNWAY_mvt").alias("g_n"))
        d = d.with_columns(pl.when(pl.col("g_n") >= 20).then(pl.col("g_ref"))
                           .otherwise(pl.col("g").median().over("RUNWAY_mvt")).alias("g_ref"))
        d = d.with_columns((pl.col("g") - pl.col("g_ref")).alias("rg"))
        t = ep(d["MVT_TIME_UTC_mvt"])
        rg = d["rg"].cast(pl.Float64).to_numpy()
        g_ref = d["g_ref"].cast(pl.Float64).to_numpy()
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        # airport-wide time windows, symmetric and one-sided, self excluded
        for nm, lo, hi in (("pm20", -1200, 1200), ("prev30", -1800, -1), ("next30", 1, 1800), ("pm60", -3600, 3600)):
            m, n = win_mean(t, t, rg, lo, hi, exclude_self=rg if lo < 0 < hi else None)
            cols[f"nbr_rg_{nm}"] = m
        cols["nbr_rg_asym"] = cols["nbr_rg_next30"] - cols["nbr_rg_prev30"]
        # same runway sequence +-5 (self excluded)
        rw = d["RUNWAY_mvt"].fill_null("NA").to_numpy()
        seq = np.full(d.height, np.nan)
        for r in np.unique(rw):
            idx = np.where(rw == r)[0]
            o = idx[np.argsort(t[idx], kind="stable")]
            v = rg[o]
            vv = np.nan_to_num(v)
            ok = np.isfinite(v).astype(float)
            c1 = np.concatenate([[0.0], np.cumsum(vv)])
            c2 = np.concatenate([[0.0], np.cumsum(ok)])
            i = np.arange(len(o))
            lo_ = np.clip(i - 5, 0, len(o))
            hi_ = np.clip(i + 6, 0, len(o))
            s = c1[hi_] - c1[lo_] - vv
            c = c2[hi_] - c2[lo_] - ok
            with np.errstate(all="ignore"):
                seq[o] = np.where(c > 0, s / c, np.nan)
        cols["nbr_seq_rg_pm5"] = seq
        # estimated off-block (label-free) b_hat = t - g_ref ; queue in system at b_hat (same runway & airport)
        bh = t - g_ref
        okq = np.isfinite(bh) & np.isfinite(t) & (bh < t)
        s_ = np.sort(bh[okq])
        e_ = np.sort(t[okq])
        insys = (np.searchsorted(s_, bh, side="right") - np.searchsorted(e_, bh, side="right")).astype(float)
        cols["qhat_in_system_at_bhat"] = np.where(okq, insys - 1, np.nan)
        cols["qhat_tko_between_bhat_t"] = np.where(okq, _count_between(np.sort(t[np.isfinite(t)]), bh, t), np.nan)
        qr = np.full(d.height, np.nan)
        qt = np.full(d.height, np.nan)
        for r in np.unique(rw):
            m = (rw == r) & okq
            if m.sum() < 3:
                continue
            s_r, e_r = np.sort(bh[m]), np.sort(t[m])
            qr[m] = (np.searchsorted(s_r, bh[m], side="right") - np.searchsorted(e_r, bh[m], side="right")) - 1
            qt[m] = _count_between(e_r, bh[m], t[m])
        cols["qhat_in_system_at_bhat_rwy"] = qr
        # take-offs on my runway between my estimated off-block and my take-off, relative to what the
        # runway could have served in that time at its minimum headway (~ queue I waited behind)
        cols["qhat_tko_between_rwy"] = qt
        # --- arrival taxi-in residual, symmetric window (ARR rows fully observed in ranking) ---
        a = a.with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ti")).with_columns(
            pl.when(pl.col("ti").is_between(0, 3600)).then(pl.col("ti")).otherwise(None).alias("ti"))
        a = a.with_columns(pl.col("ti").median().over("STAND_mvt", "RUNWAY_mvt").alias("ti_ref"),
                           pl.col("ti").count().over("STAND_mvt", "RUNWAY_mvt").alias("ti_n"))
        a = a.with_columns(pl.when(pl.col("ti_n") >= 20).then(pl.col("ti_ref"))
                           .otherwise(pl.col("ti").median().over("RUNWAY_mvt")).alias("ti_ref"))
        at = ep(a["MVT_TIME_UTC_mvt"])
        ar = (a["ti"] - a["ti_ref"]).cast(pl.Float64).to_numpy()
        for nm, lo, hi in (("pm30", -1800, 1800), ("prev30", -1800, 0), ("next30", 0, 1800)):
            m, n = win_mean(t, at, ar, lo, hi)
            cols[f"arr_ti_resid_{nm}"] = m
        # anchored at estimated off-block: arrivals landing in [bhat-15m, bhat+15m]
        m, _ = win_mean(bh, at, ar, -900, 900)
        cols["arr_ti_resid_at_bhat"] = m
        frames.append(pl.DataFrame(cols))
        print(apt, d.height, flush=True)
    new = pl.concat(frames, how="diagonal")
    fl = [c for c, dt in new.schema.items() if dt == pl.Float64 and c != "MVT_ID_mvt"]
    new = new.with_columns([pl.col(c).fill_nan(None).cast(pl.Float32) for c in fl])
    new.write_parquet(os.path.join(OUT, "new2.parquet"))
    print(new.shape)


if __name__ == "__main__":
    main()
