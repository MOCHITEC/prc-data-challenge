"""Feature table for the temporal_shift proxy model.

The proxy is NOT the production model (src/prc/train.py, mixture v12). It is a single LightGBM on y with the
row-local clock gaps + categorical context + a few backward windows that mimic what src/prc/features.py already
has, so that *relative* effects (training-period weighting, day-level context) can be measured cheaply.

Every feature is computable from ranking.parquet: departures contribute only MVT_TIME / NM clocks / SCHED /
RUNWAY / STAND; arrivals contribute their taxi-in and in-block time.

New (candidate) features are prefixed `day_` (whole local day of the airport, both directions in time — legal
because the full ranking day is given at once) and `cw_` (centred +-60 min window).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from temporal_shift_common import CACHE, load

FEAT = CACHE / "features.parquet"

BASE_NUM = ["m_aobt3", "m_sched", "m_eobt1", "m_lobt", "aobt3_m_sched", "nm", "hour_local", "dow_local",
            "bw_nbr_m_aobt3_60m", "bw_arr_taxi_in_60m", "bw_dep_n_30m", "fw_dep_n_30m"]
BASE_CAT = ["APT", "RUNWAY_mvt", "STAND_mvt", "airline", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt"]
DAY_FEATS = ["day_m_aobt3_med", "day_m_aobt3_mean", "day_arr_taxi_in_mean", "day_dep_n_rel", "day_nm_unmatched",
             "day_msched_gt3h", "day_arr_blk_eq_sched", "day_aobt3_eq_sched", "cw_nbr_m_aobt3_60m",
             "cw_arr_taxi_in_60m", "day_dev_m_aobt3", "day_dev_taxi_in"]


def _window_mean(anchor, times, values, lo_off, hi_off, exclude_self_times=None):
    """mean of values with time in [t+lo_off, t+hi_off) ; NaN ignored."""
    order = np.argsort(times, kind="stable")
    t_, v_ = times[order], values[order]
    ok = np.isfinite(v_) & np.isfinite(t_)
    t_, v_ = t_[ok], v_[ok]
    cs = np.concatenate([[0.0], np.cumsum(v_)])
    hi = np.searchsorted(t_, anchor + hi_off, side="left")
    lo = np.searchsorted(t_, anchor + lo_off, side="left")
    n = (hi - lo).astype(float)
    s = cs[hi] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan), n


def build() -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "set", "PHASE_mvt", "APT", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
            "date_local", "year", "month", "hour_local", "dow_local", "m_aobt3", "m_sched", "m_eobt1", "m_lobt",
            "aobt3_m_sched", "nm", "y", "clock_src", "RUNWAY_mvt", "STAND_mvt", "airline", "AIRCRAFT_TYPE_mvt",
            "MARKET_SEGMENT_flt"]
    d = load(cols).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"))
    out = []
    for apt in sorted(d["APT"].cast(pl.Utf8).unique().to_list()):
        a = d.filter(pl.col("APT") == apt)
        dep = a.filter(pl.col("PHASE_mvt") == "DEP").sort("t")
        arr = a.filter(pl.col("PHASE_mvt") == "ARR")
        t = dep["t"].to_numpy()
        ma = dep["m_aobt3"].to_numpy().astype(float)
        ma_c = np.where((ma > 0) & (ma < 7200), ma, np.nan)
        arr_t = arr["BLOCK_TIME_UTC_mvt"].dt.epoch("s").cast(pl.Float64).to_numpy()
        arr_y = arr["y"].to_numpy().astype(float)
        arr_y = np.where((arr_y >= 0) & (arr_y < 7200), arr_y, np.nan)
        # backward windows (existing-feature analogues). lag 60 s like src/prc/features.py
        bw_m, _ = _window_mean(t, t, ma_c, -3600 - 60, -60)
        bw_ti, _ = _window_mean(t, arr_t, arr_y, -3600, 0)
        ones = np.ones_like(t)
        _, bw_n = _window_mean(t, t, ones, -1800, 0)
        _, fw_n = _window_mean(t, t, ones, 1, 1801)
        # centred windows, excluding self: sum over window minus own contribution
        s_m, n_m = _window_mean(t, t, ma_c, -3600, 3601)
        own = np.isfinite(ma_c)
        with np.errstate(invalid="ignore", divide="ignore"):
            cw_m = np.where(n_m - own > 0, (s_m * n_m - np.nan_to_num(ma_c) * own) / (n_m - own), np.nan)
        cw_ti, _ = _window_mean(t, arr_t, arr_y, -3600, 3601)
        dep = dep.with_columns(
            pl.Series("bw_nbr_m_aobt3_60m", bw_m), pl.Series("bw_arr_taxi_in_60m", bw_ti),
            pl.Series("bw_dep_n_30m", bw_n), pl.Series("fw_dep_n_30m", fw_n),
            pl.Series("cw_nbr_m_aobt3_60m", cw_m), pl.Series("cw_arr_taxi_in_60m", cw_ti),
            pl.Series("_ma_c", ma_c).fill_nan(None),
        )
        # day-level aggregates (local day of this airport)
        arr_day = arr.with_columns(
            pl.when(pl.col("y").is_between(0, 7200)).then(pl.col("y")).alias("_ti"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 5).alias("_bs"),
        ).group_by("date_local").agg(pl.col("_ti").mean().alias("day_arr_taxi_in_mean"),
                                     pl.col("_bs").mean().alias("day_arr_blk_eq_sched"))
        dday = dep.group_by("date_local").agg(
            pl.col("_ma_c").median().alias("day_m_aobt3_med"), pl.col("_ma_c").mean().alias("day_m_aobt3_mean"),
            pl.len().alias("day_dep_n"), (1 - pl.col("nm").mean()).alias("day_nm_unmatched"),
            (pl.col("m_sched") > 3 * 3600).mean().alias("day_msched_gt3h"),
            (pl.col("aobt3_m_sched").abs() <= 5).mean().alias("day_aobt3_eq_sched"))
        dday = dday.join(arr_day, on="date_local", how="left")
        # relative day volume vs the airport's median day of the same weekday (computed on the observable counts
        # of all periods pooled — counts are not labels)
        dday = dday.with_columns(pl.col("date_local").dt.weekday().alias("_wd"))
        dday = dday.with_columns((pl.col("day_dep_n") / pl.col("day_dep_n").median().over("_wd")).alias("day_dep_n_rel"))
        dep = dep.join(dday.drop("_wd", "day_dep_n"), on="date_local", how="left").with_columns(
            (pl.col("day_m_aobt3_mean") - pl.col("cw_nbr_m_aobt3_60m")).alias("day_dev_m_aobt3"),
            (pl.col("day_arr_taxi_in_mean") - pl.col("cw_arr_taxi_in_60m")).alias("day_dev_taxi_in"),
        )
        out.append(dep.drop("_ma_c", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
    f = pl.concat(out, how="vertical_relaxed")
    return f.with_columns([pl.col(c).cast(pl.Float32) for c in BASE_NUM + DAY_FEATS if f.schema[c] != pl.Float32])


def features(refresh: bool = False) -> pl.DataFrame:
    if refresh or not FEAT.exists():
        build().write_parquet(FEAT)
    return pl.read_parquet(FEAT)


if __name__ == "__main__":
    f = features(refresh=True)
    print(f.shape, f.estimated_size("mb"))
    print(f.select(BASE_NUM + DAY_FEATS).describe())
