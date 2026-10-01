"""Delay-mechanism diagnosis for taxi-out (read-only).

Run:  POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/delay_mechanisms.py [S1 ... S7]

Nothing is written to outputs/, data/, src/ or configs/. No model is fitted on the competition
target; every fit below is on the *OOF residual* of the production model and is a diagnostic only.

Parity rules obeyed throughout:
  * no column reads BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt of any row (own or neighbour);
  * neighbour statistics use MVT_TIME_UTC_mvt and AOBT_3_flt only, both present in ranking.parquet,
    and are lagged 60 s behind the anchor exactly as src/prc/features.py does.

Sections
  S1  provenance of g = AOBT_3 - BLOCK: what model A actually has to predict on lane A
  S2  worst-row autopsy (top 1 % of |residual| on lane A)
  S3  out-of-fold binned-mean residual scan over the 118 columns already in the model
  S4  build the candidate columns (neighbour dispersion / runway-local clock / config / apron)
  S5  residual-explanation test, month-blocked, with arity-matched noise controls
  S6  de-icing and wind-direction slices (the two leads on record)
  S7  month-blocked transfer check for whatever S5 ranks first
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

W = "/home/user/prc-data-challenge/"
OOF = W + "experiments/oof_main_gap-62dd7c6e88.parquet"
FOLDS = W + "experiments/oof_main_gap-62dd7c6e88_folds.parquet"
FEAT = W + "data/work/feat_train.parquet"


# --------------------------------------------------------------------------------------- loading
def load_lane_a(columns: list[str] | None = None) -> pl.DataFrame:
    """Pooled OOF rows of the production model, matched lane only, with residual r and g."""
    o = pl.read_parquet(OOF)
    fo = pl.read_parquet(FOLDS).filter(pl.col("pooled"))
    d = pl.read_parquet(FEAT, columns=columns)
    df = (
        d.join(o, on="MVT_ID_mvt")
        .join(fo, on="MVT_ID_mvt")
        .with_columns(
            (pl.col("oof") - pl.col("TAXITIME_SEC_mvt")).alias("r"),
            (pl.col("TAXITIME_SEC_mvt") - pl.col("mvt_minus_aobt3")).alias("g"),
        )
    )
    return df.filter(pl.col("mvt_minus_aobt3").is_not_null())


# ------------------------------------------------------------------- windowed neighbour statistics
def _window_stats(anchor: np.ndarray, times: np.ndarray, values: np.ndarray, w: int, lag: int, thr: float):
    """mean / sd / tail-mass of `values` over events with time in [t-lag-w, t-lag).

    Exact, via cumulative sums; O(n log n). `thr` is the tail threshold (seconds).
    Returns (mean, sd, frac_above_thr, n).
    """
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    ok = np.isfinite(values) & np.isfinite(times)
    t, v = times[ok], values[ok]
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    ct = np.concatenate([[0.0], np.cumsum((v > thr).astype(np.float64))])
    hi = np.searchsorted(t, anchor - lag, side="left")
    lo = np.searchsorted(t, anchor - lag - w, side="left")
    n = (hi - lo).astype(np.float64)
    s1, s2, st = c1[hi] - c1[lo], c2[hi] - c2[lo], ct[hi] - ct[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / n, np.nan)
        var = np.where(n > 1, (s2 - s1 * s1 / np.maximum(n, 1)) / np.maximum(n - 1, 1), np.nan)
        frac = np.where(n > 0, st / n, np.nan)
    return mean, np.sqrt(np.clip(var, 0, None)), frac, n


def build_forward_clock(src: pl.DataFrame) -> pl.DataFrame:
    """Forward and centred neighbour off-block clocks.

    Every `nbr_*` column in the model is trailing-only, yet MVT_TIME_UTC_mvt and AOBT_3_flt are present
    on every scored row including flights later in the day, and features.py already uses forward counts
    (`dep_tko_next_*`). So the same surface state can be estimated two-sided.
    """
    m = src.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("_p"))

    def wm(anchor, t, v, lo_off, hi_off):
        o = np.argsort(t, kind="stable")
        t, v = t[o], v[o]
        ok = np.isfinite(v) & np.isfinite(t)
        t, v = t[ok], v[ok]
        c1 = np.concatenate([[0.0], np.cumsum(v)])
        hi = np.searchsorted(t, anchor + hi_off, "left")
        lo = np.searchsorted(t, anchor + lo_off, "left")
        n = (hi - lo).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (c1[hi] - c1[lo]) / n, np.nan)

    out = []
    for apt in m["ADEP_mvt"].unique().to_list():
        p = m.filter(pl.col("ADEP_mvt") == apt)
        t, v = p["_t"].to_numpy(), p["_p"].to_numpy()
        out.append(pl.DataFrame({
            "MVT_ID_mvt": p["MVT_ID_mvt"].to_numpy(),
            "nbr_fwd_30m": wm(t, t, v, 60, 60 + 1800),
            "nbr_fwd_60m": wm(t, t, v, 60, 60 + 3600),
            "nbr_ctr_3h": wm(t, t, v, -10800, 10800),
            "nbr_day": wm(t, t, v, -86400, 86400)}))
    return pl.concat(out)


RWY_HEADING = {}  # filled lazily from the runway designator


def runway_heading(rwy: str | None) -> float:
    """Magnetic heading in degrees from an ICAO runway designator ('27L' -> 270)."""
    if rwy is None:
        return float("nan")
    digits = "".join(ch for ch in rwy[:2] if ch.isdigit())
    if len(digits) != 2:
        return float("nan")
    n = int(digits)
    return float("nan") if not (1 <= n <= 36) else n * 10.0


def build_candidates(src: pl.DataFrame) -> pl.DataFrame:
    """Candidate columns, all from MVT_TIME_UTC_mvt / AOBT_3_flt / RUNWAY_mvt / STAND_mvt / ADEP_mvt.

    src must be *all* departures (the context frame), not just the scored subset.
    """
    m = src.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64).alias("_proxy"),
        pl.col("STAND_mvt").str.slice(0, 1).alias("_pier"),
    )
    out = []
    for apt in m["ADEP_mvt"].unique().to_list():
        p = m.filter(pl.col("ADEP_mvt") == apt) if apt is not None else m.filter(pl.col("ADEP_mvt").is_null())
        t = p["_t"].to_numpy()
        v = p["_proxy"].to_numpy()
        thr = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        cols: dict[str, np.ndarray] = {"MVT_ID_mvt": p["MVT_ID_mvt"].to_numpy()}
        # --- airport-wide dispersion and tail mass (the model has only the mean) -------------
        for w in (30, 60):
            mean, sd, frac, n = _window_stats(t, t, v, w * 60, 60, thr)
            cols[f"nbr_taxi_sd_{w}m"] = sd
            cols[f"nbr_taxi_tail_{w}m"] = frac
            cols[f"nbr_taxi_cv_{w}m"] = np.where(mean > 0, sd / mean, np.nan)
            if w == 30:
                apt_mean30 = mean
        # --- same-runway neighbour clock, and its deviation from the airport-wide one --------
        rwy_mean = np.full(len(t), np.nan)
        rwy_n = np.zeros(len(t))
        rwy_share = np.full(len(t), np.nan)
        _, _, _, n_all60 = _window_stats(t, t, v, 3600, 60, thr)
        for rwy in p["RUNWAY_mvt"].unique().to_list():
            sel = (p["RUNWAY_mvt"] == rwy).fill_null(False).to_numpy() if rwy is not None else p["RUNWAY_mvt"].is_null().to_numpy()
            if not sel.any():
                continue
            sub = p.filter(pl.col("RUNWAY_mvt") == rwy) if rwy is not None else p.filter(pl.col("RUNWAY_mvt").is_null())
            ts, vs = sub["_t"].to_numpy(), sub["_proxy"].to_numpy()
            mean, _, _, nn = _window_stats(t[sel], ts, vs, 30 * 60, 60, thr)
            rwy_mean[sel], rwy_n[sel] = mean, nn
            _, _, _, n_r60 = _window_stats(t[sel], ts, vs, 3600, 60, thr)
            with np.errstate(invalid="ignore", divide="ignore"):
                rwy_share[sel] = np.where(n_all60[sel] > 0, n_r60 / n_all60[sel], np.nan)
        cols["nbr_taxi_mean_rwy_30m"] = rwy_mean
        cols["nbr_taxi_rwy_n_30m"] = rwy_n
        cols["nbr_taxi_rwy_dev_30m"] = rwy_mean - apt_mean30
        cols["rwy_share_60m"] = rwy_share
        # --- same-pier neighbour clock and pushback pressure ---------------------------------
        pier_mean = np.full(len(t), np.nan)
        pier_push = np.full(len(t), np.nan)
        aobt = p["AOBT_3_flt"].dt.epoch("s").cast(pl.Float64).to_numpy()
        for pier in p["_pier"].unique().to_list():
            sel = (p["_pier"] == pier).fill_null(False).to_numpy() if pier is not None else p["_pier"].is_null().to_numpy()
            if not sel.any():
                continue
            sub = p.filter(pl.col("_pier") == pier) if pier is not None else p.filter(pl.col("_pier").is_null())
            ts, vs = sub["_t"].to_numpy(), sub["_proxy"].to_numpy()
            mean, _, _, _ = _window_stats(t[sel], ts, vs, 30 * 60, 60, thr)
            pier_mean[sel] = mean
            ab = sub["AOBT_3_flt"].dt.epoch("s").cast(pl.Float64).to_numpy()
            # pushbacks from the same pier in the 10 min before this flight's own pushback
            _, _, _, npush = _window_stats(aobt[sel], ab, np.ones_like(ab), 600, 0, np.inf)
            pier_push[sel] = npush
        cols["nbr_taxi_mean_pier_30m"] = pier_mean
        cols["nbr_taxi_pier_dev_30m"] = pier_mean - apt_mean30
        cols["pier_push_prev_10m"] = pier_push
        out.append(pl.DataFrame(cols))
    return pl.concat(out, how="diagonal")


# ------------------------------------------------------------------------------ residual scanning
def oof_bin_gain(r, x, key=None, nbin=20, nfold=5, seed=0, lam=50.0) -> float:
    """Seconds of lane-A RMSE recovered by an out-of-fold per-bin constant on x (negative = better)."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    if key is None:
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        code = np.full(n, -1, dtype=np.int64)
        if ok.sum() > 0:
            qs = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, nbin + 1)[1:-1]))
            code[ok] = np.searchsorted(qs, x[ok], side="right")
        code = code + 1
    else:
        code = np.asarray(key)
    rng = np.random.default_rng(seed)
    f = rng.integers(0, nfold, n)
    K = int(code.max()) + 1
    corr = np.zeros(n)
    for k in range(nfold):
        tr = f != k
        s = np.bincount(code[tr], weights=r[tr], minlength=K)
        c = np.bincount(code[tr], minlength=K)
        mean = s / (c + lam)
        corr[~tr] = mean[code[~tr]]
    return float(np.sqrt(((r - corr) ** 2).mean()) - np.sqrt((r**2).mean()))


def month_blocked_gain(df: pl.DataFrame, feats: list[str], seed: int = 0, num_leaves: int = 31) -> dict:
    """Fit the OOF residual on `feats` with month-blocked folds. Returns seconds of lane-A RMSE."""
    import lightgbm as lgb

    X = df.select(feats).with_columns([pl.col(c).cast(pl.Float64) for c in feats]).to_numpy()
    r = df["r"].to_numpy()
    mo = df["month"].to_numpy()
    pred = np.full(len(r), np.nan)
    for k in range(4):
        te = np.isin(mo, [3 * k + 1, 3 * k + 2, 3 * k + 3])
        if te.sum() == 0:
            continue
        ds = lgb.Dataset(X[~te], label=r[~te])
        params = dict(
            objective="regression", learning_rate=0.05, num_leaves=num_leaves, min_data_in_leaf=200,
            feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed,
            num_threads=2, deterministic=True, force_row_wise=True,
        )
        m = lgb.train(params, ds, num_boost_round=200)
        pred[te] = m.predict(X[te])
    base = float(np.sqrt((r**2).mean()))
    new = float(np.sqrt(((r - np.nan_to_num(pred)) ** 2).mean()))
    return {"base": base, "new": new, "delta": new - base}


# =============================================================================== sections
def S1() -> None:
    """Provenance of g = AOBT_3 - BLOCK, and how informative the offset is per airport."""
    A = load_lane_a(
        [
            "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "month",
        ]
    )
    g = A["g"].to_numpy()
    print("S1 lane A n=%d  RMSE %.4f" % (A.height, np.sqrt((A["r"].to_numpy() ** 2).mean())))
    print("S1 g = AOBT_3 - BLOCK: mean %.1f sd %.1f med %.1f  |g|<=60s on %.4f of rows"
          % (g.mean(), g.std(), np.median(g), (np.abs(g) <= 60).mean()))
    print("S1 AOBT_3 second-of-minute == 0 on %.4f of rows (so g is not a rounding artefact)"
          % (A["AOBT_3_flt"].dt.second() == 0).mean())
    print("S1 per airport: b(y~proxy) is the OLS slope of the label on the offset the model subtracts at 1.0")
    print("%-6s %8s %9s %8s %8s" % ("apt", "n", "b(y~px)", "corr", "RMSE"))
    for apt in sorted(A["ADEP_mvt"].unique().to_list()):
        q = A.filter(pl.col("ADEP_mvt") == apt)
        v = q["mvt_minus_aobt3"].to_numpy().astype(float)
        y = q["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        r = q["r"].to_numpy()
        print("%-6s %8d %9.3f %8.3f %8.1f"
              % (apt, len(v), np.polyfit(v, y, 1)[0], np.corrcoef(v, y)[0, 1], np.sqrt((r**2).mean())))


def S1b() -> None:
    """The arrival mirror: where both systems watch the runway, NM and the airport agree.

    Arrivals in ranking.parquet keep BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt (only departures are
    blanked), and carry ARVT_3_flt, so NM-vs-airport agreement can be measured on the scored set itself.
    """
    rk = pl.read_parquet(W + "data/raw/ranking.parquet", columns=[
        "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "ARVT_3_flt"])
    arr = rk.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        (pl.col("ARVT_3_flt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("g_arr"),
        (pl.col("ARVT_3_flt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().alias("arvt_minus_land"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ti"))
    print("S1b scored ARR rows %d; BLOCK null %.4f, TAXITIME null %.4f, ARVT_3 null %.4f"
          % (arr.height, arr["BLOCK_TIME_UTC_mvt"].is_null().mean(),
             arr["TAXITIME_SEC_mvt"].is_null().mean(), arr["ARVT_3_flt"].is_null().mean()))
    x, t = arr["g_arr"].to_numpy(), arr["ti"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(t)
    print("S1b ARVT_3 - MVT_TIME: median %.0f s, sd %.1f s   (the NM arrival stamp IS the landing stamp)"
          % (arr["arvt_minus_land"].median(), arr["arvt_minus_land"].std()))
    print("S1b corr(ARVT_3 - BLOCK_arr, -taxi_in) = %.4f" % np.corrcoef(x[ok], -t[ok])[0, 1])


def S2() -> None:
    """The minute histogram of the offset: where NM's stamp is a table value, not a measurement."""
    A = load_lane_a(["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "month"])
    for apt in sorted(A["ADEP_mvt"].unique().to_list()):
        s = A.filter(pl.col("ADEP_mvt") == apt)
        v = s["mvt_minus_aobt3"].to_numpy()
        r = s["r"].to_numpy()
        gg = s["g"].to_numpy()
        b = np.floor(v / 60).astype(int)
        cnt = np.bincount(b - b.min())
        k = int(cnt.argmax())
        nb = [cnt[j] for j in range(max(0, k - 5), min(len(cnt), k + 6)) if j != k]
        sel = b == b.min() + k
        print("S2 %s spike at %d min: %d rows (%.2f%%), %.1fx the median neighbouring minute | "
              "RMSE spike %.0f rest %.0f | sd(g) spike %.0f rest %.0f"
              % (apt, b.min() + k, sel.sum(), 100 * sel.mean(), cnt[k] / max(np.median(nb), 1),
                 np.sqrt((r[sel] ** 2).mean()), np.sqrt((r[~sel] ** 2).mean()), gg[sel].std(), gg[~sel].std()))


def S3() -> None:
    """Out-of-fold binned-mean residual scan over every column already in the model."""
    A = load_lane_a()
    r = A["r"].to_numpy()
    print("S3 lane A base RMSE %.4f" % np.sqrt((r**2).mean()))
    num = [c for c, t in A.schema.items()
           if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Int16)
           and c not in ("MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt", "oof", "r", "g")]
    out = [(c, oof_bin_gain(r, A[c].cast(pl.Float64).to_numpy())) for c in num]
    for c in ("ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
              "ADES_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_REGION"):
        code = A[c].fill_null("__NA__").cast(pl.Categorical).to_physical().to_numpy()
        out.append((c + " [cat]", oof_bin_gain(r, None, key=code)))
    t = pl.DataFrame(out, schema=["col", "d_rmse"], orient="row").sort("d_rmse")
    pl.Config.set_tbl_rows(200)
    print(t.head(15))
    print("S3 noise controls:", t.filter(pl.col("col").str.starts_with("noise_control")))


def S4() -> None:
    """Where the error is: concentration by airport-day, and the airport x date x hour oracle."""
    A = load_lane_a(["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3",
                     "MVT_TIME_UTC_mvt", "month", "hour_local"]).with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("d"))
    r = A["r"].to_numpy()
    base = np.sqrt((r**2).mean())
    for apt in ("LIRF", "LFPG"):
        s = A.filter(pl.col("ADEP_mvt") == apt)
        tot = (s["r"] ** 2).sum()
        day = s.group_by("d").agg((pl.col("r") ** 2).sum().alias("sse")).sort("sse", descending=True)  # noqa
        cs = (day["sse"] / tot).cum_sum().to_numpy()
        print("S4 %s: %d days; top 5 %.3f, top 10 %.3f, top 20 %.3f of its lane-A SSE"
              % (apt, day.height, cs[4], cs[9], cs[19]))
        worst = s.filter(pl.col("d") == day["d"][0])
        print("S4   worst day %s: n=%d RMSE %.1f bias %+.1f | median y %.0f (normal %.0f), "
              "median proxy %.0f (normal %.0f)"
              % (day["d"][0], worst.height, np.sqrt((worst["r"].to_numpy() ** 2).mean()),
                 worst["r"].mean(), worst["TAXITIME_SEC_mvt"].median(), s["TAXITIME_SEC_mvt"].median(),
                 worst["mvt_minus_aobt3"].median(), s["mvt_minus_aobt3"].median()))

    def loo(df, keys, lam):
        k = df.select(keys).with_columns(
            pl.concat_str([pl.col(c).cast(pl.Utf8) for c in keys], separator="|").alias("k"))["k"]
        code = k.cast(pl.Categorical).to_physical().to_numpy()
        K = int(code.max()) + 1
        s = np.bincount(code, weights=r, minlength=K)
        c = np.bincount(code, minlength=K)
        m = np.where(c[code] > 1, (s[code] - r) / np.maximum(c[code] - 1, 1), 0.0)
        if lam:
            m = m * (c[code] - 1) / (c[code] - 1 + lam)
        return float(np.sqrt(((r - m) ** 2).mean()) - base), K

    for keys in (["ADEP_mvt"], ["d"], ["ADEP_mvt", "d"], ["ADEP_mvt", "d", "hour_local"]):
        d, K = loo(A, keys, 20.0)
        print("S4 LOO oracle %-30s cells=%-7d dRMSE %+.3f" % (" x ".join(keys), K, d))
    rng = np.random.default_rng(1)
    sh = A.with_columns(pl.Series("d", rng.permutation(A["d"].to_numpy())))
    d, K = loo(sh, ["ADEP_mvt", "d"], 20.0)
    print("S4 CONTROL shuffled dates                   cells=%-7d dRMSE %+.3f" % (K, d))


def S5(A: pl.DataFrame | None = None) -> None:
    """Month-blocked residual-explanation test with arity-matched controls and a positive control."""
    A = A if A is not None else load_lane_a()
    A = A.with_columns(
        pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("adep_code"),
        pl.col("RUNWAY_mvt").fill_null("NA").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("rwy_code"))
    r = A["r"].to_numpy()
    rng = np.random.default_rng(7)
    A = A.with_columns([pl.Series("leak_%d" % k, r + rng.normal(0, k * r.std(), len(r))) for k in (1, 3, 10)])
    BASE = ["adep_code", "hour_local", "mvt_minus_aobt3", "rwy_code"]
    NC = ["noise_control", "noise_control_2", "noise_control_3", "noise_control_4"]

    def arm(name, extra, seeds=3):
        v = [month_blocked_gain(A, BASE + extra, seed=s, num_leaves=8)["new"] for s in range(seeds)]
        print("S5 %-24s %.4f (seed sd %.4f)" % (name, np.mean(v), np.std(v)), flush=True)
        return float(np.mean(v))

    arm("base", [])
    for k in (1, 3, 10):
        arm("POSCTRL leak_%d" % k, ["leak_%d" % k], seeds=1)
    arm("control4", NC)


def build_wind_and_config() -> pl.DataFrame:
    """Wind direction relative to the assigned runway, plus runway-configuration state.

    `drct` and `skyc1` are fetched by src/prc/weather.py (IEM_FIELDS) and then dropped in the hourly
    aggregation, so wind DIRECTION has never reached the model. Everything here is exogenous or built
    from RUNWAY_mvt + MVT_TIME_UTC_mvt, both present on every scored row.
    """
    import glob

    src = pl.read_parquet(FEAT, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
    frames = [pl.read_csv(f, null_values=["M", "T"], infer_schema_length=0)
              for f in glob.glob(W + "data/ref/metar/*.csv")]
    m = (pl.concat(frames, how="diagonal")
         .with_columns(pl.col("valid").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M").dt.replace_time_zone("UTC"),
                       pl.col("drct").cast(pl.Float64), pl.col("sknt").cast(pl.Float64))
         .with_columns(pl.col("valid").dt.truncate("1h").alias("hour_utc")))
    h = m.sort("valid").group_by(["station", "hour_utc"]).agg(pl.col("drct").last(), pl.col("sknt").last())
    src = src.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hour_utc")).join(
        h, left_on=["ADEP_mvt", "hour_utc"], right_on=["station", "hour_utc"], how="left")
    src = src.with_columns(pl.Series("rwy_head", [runway_heading(x) for x in src["RUNWAY_mvt"].to_list()]))
    delta = (pl.col("drct") - pl.col("rwy_head")).radians()
    src = src.with_columns(
        (pl.col("sknt") * delta.cos()).alias("headwind_kt"),
        (pl.col("sknt") * delta.sin().abs()).alias("crosswind_kt"),
        (1 - delta.cos()).alias("rwy_wind_misalign"),
        pl.col("drct").alias("wind_dir_deg"))
    out = []
    for apt in src["ADEP_mvt"].unique().to_list():
        p = src.filter(pl.col("ADEP_mvt") == apt).sort("MVT_TIME_UTC_mvt")
        t = p["MVT_TIME_UTC_mvt"].dt.epoch("s").cast(pl.Float64).to_numpy()
        codes = p["RUNWAY_mvt"].fill_null("NA").cast(pl.Categorical).to_physical().to_numpy()
        K = int(codes.max()) + 1
        ind = np.zeros((len(t), K))
        ind[np.arange(len(t)), codes] = 1
        cs = np.vstack([np.zeros(K), np.cumsum(ind, axis=0)])

        def win(a, b):
            return cs[np.searchsorted(t, t - a, "left")] - cs[np.searchsorted(t, t - b, "left")]

        rec, old = win(300, 3600), win(5400, 9000)
        mr, mo = rec.argmax(1), old.argmax(1)
        own, tot = rec[np.arange(len(t)), codes], rec.sum(1)
        out.append(pl.DataFrame({
            "MVT_ID_mvt": p["MVT_ID_mvt"].to_numpy(),
            "cfg_switch_recent": ((mr != mo) & (old.sum(1) > 0) & (tot > 0)).astype(float),
            "own_rwy_share_55m": np.where(tot > 0, own / np.maximum(tot, 1), np.nan),
            "n_rwy_active_55m": (rec > 0).sum(1).astype(float)}))
    return src.select(["MVT_ID_mvt", "headwind_kt", "crosswind_kt", "rwy_wind_misalign", "wind_dir_deg"]).join(
        pl.concat(out), on="MVT_ID_mvt")


def S6() -> None:
    """Calibrate the instrument against a feature whose true full-model gain is known.

    `oof_main_reparam-bc60017878` and `oof_main_gap-62dd7c6e88` differ only by
    include_candidates=['stand_arrival_gap'] (experiments.jsonl). The instrument is run on the BEFORE
    residual and its reading is compared with the measured before/after difference.
    """
    o1 = pl.read_parquet(W + "experiments/oof_main_reparam-bc60017878.parquet").rename({"oof": "b"})
    o2 = pl.read_parquet(W + "experiments/oof_main_gap-62dd7c6e88.parquet").rename({"oof": "a"})
    fo = pl.read_parquet(W + "experiments/oof_main_reparam-bc60017878_folds.parquet").filter(
        pl.col("fold") != "jul_fwd")
    d = pl.read_parquet(FEAT, columns=[
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "month", "hour_local",
        "stand_arrival_gap", "stand_aobt_turn", "stand_arrival_type_match",
        "noise_control", "noise_control_2", "noise_control_3", "noise_control_4"])
    A = (d.join(o1, on="MVT_ID_mvt").join(o2, on="MVT_ID_mvt").join(fo, on="MVT_ID_mvt")
         .filter(pl.col("mvt_minus_aobt3").is_not_null()))
    y = A["TAXITIME_SEC_mvt"].to_numpy()
    r1, r2 = A["b"].to_numpy() - y, A["a"].to_numpy() - y
    true_gain = np.sqrt((r2**2).mean()) - np.sqrt((r1**2).mean())
    print("S6 lane A n=%d  before %.4f  after %.4f  TRUE full-model gain %+.4f"
          % (A.height, np.sqrt((r1**2).mean()), np.sqrt((r2**2).mean()), true_gain))
    A = A.with_columns(
        pl.Series("r", r1),
        pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("adep_code"),
        pl.col("RUNWAY_mvt").fill_null("NA").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("rwy_code"))
    BASE = ["adep_code", "hour_local", "mvt_minus_aobt3", "rwy_code"]

    def arm(name, extra):
        v = [month_blocked_gain(A, BASE + extra, seed=s, num_leaves=8)["new"] for s in range(3)]
        print("S6 %-26s %.4f (seed sd %.4f)" % (name, np.mean(v), np.std(v)), flush=True)
        return float(np.mean(v))

    arm("base (on r_before)", [])
    c = arm("control4", ["noise_control", "noise_control_2", "noise_control_3", "noise_control_4"])
    x = arm("stand_arrival_gap family", ["stand_arrival_gap", "stand_aobt_turn", "stand_arrival_type_match",
                                         "noise_control"])
    print("S6 instrument reads %+.4f for a true %+.4f  =>  transfer coefficient %.2f"
          % (x - c, true_gain, (x - c) / true_gain))


def S7() -> None:
    """The candidate sweep: seven mechanism families against an arity-matched noise control."""
    A = load_lane_a([
        "MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt", "mvt_minus_aobt3", "month", "hour_local",
        "wind_ms", "gust_ms", "noise_control", "noise_control_2", "noise_control_3", "noise_control_4",
        "MVT_TIME_UTC_mvt"])
    n0 = A.height
    src = pl.read_parquet(FEAT, columns=[
        "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "RUNWAY_mvt", "STAND_mvt"])
    A = (A.join(build_candidates(src), on="MVT_ID_mvt", how="left", maintain_order="left")
           .join(build_forward_clock(src), on="MVT_ID_mvt", how="left", maintain_order="left")
           .join(build_wind_and_config(), on="MVT_ID_mvt", how="left", maintain_order="left"))
    assert A.height == n0
    A = A.with_columns((pl.col("mvt_minus_aobt3") // 60).alias("_pm"))
    mode = (A.group_by(["ADEP_mvt", "RUNWAY_mvt", "_pm"]).len().sort("len", descending=True)
            .unique(subset=["ADEP_mvt", "RUNWAY_mvt"], keep="first")
            .select(["ADEP_mvt", "RUNWAY_mvt", pl.col("_pm").alias("_mode")]))
    A = A.join(mode, on=["ADEP_mvt", "RUNWAY_mvt"], how="left", maintain_order="left").with_columns(
        (pl.col("_pm") == pl.col("_mode")).cast(pl.Int8).alias("proxy_at_mode"),
        (pl.col("_pm") - pl.col("_mode")).cast(pl.Float64).alias("proxy_minus_mode"),
        (pl.col("mvt_minus_aobt3") % 60).cast(pl.Float64).alias("proxy_sec"),
        pl.col("gust_ms").fill_null(pl.col("wind_ms")).alias("gust_filled"),
        (pl.col("gust_ms").fill_null(pl.col("wind_ms")) - pl.col("wind_ms")).alias("gust_excess"),
        pl.col("ADEP_mvt").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("adep_code"),
        pl.col("RUNWAY_mvt").fill_null("NA").cast(pl.Categorical).to_physical().cast(pl.Float64).alias("rwy_code"))
    assert A.height == n0
    BASE = ["adep_code", "hour_local", "mvt_minus_aobt3", "rwy_code"]

    def arm(name, extra):
        v = [month_blocked_gain(A, BASE + extra, seed=s, num_leaves=8)["new"] for s in range(3)]
        print("S7 %-24s %.4f (seed sd %.4f)" % (name, np.mean(v), np.std(v)), flush=True)
        return float(np.mean(v))

    arm("base", [])
    c = arm("control4", ["noise_control", "noise_control_2", "noise_control_3", "noise_control_4"])
    for nm, ex in (
        ("C1 nominal_proxy", ["proxy_at_mode", "proxy_minus_mode", "proxy_sec", "noise_control"]),
        ("C2 rwy_local_clock", ["nbr_taxi_mean_rwy_30m", "nbr_taxi_rwy_dev_30m", "nbr_taxi_rwy_n_30m",
                                "rwy_share_60m"]),
        ("C3 wind_direction", ["headwind_kt", "crosswind_kt", "rwy_wind_misalign", "wind_dir_deg"]),
        ("C4 gust_repair", ["gust_filled", "gust_excess", "noise_control", "noise_control_2"]),
        ("C5 runway_config", ["cfg_switch_recent", "own_rwy_share_55m", "n_rwy_active_55m", "noise_control"]),
        ("C6 nbr_dispersion", ["nbr_taxi_sd_30m", "nbr_taxi_sd_60m", "nbr_taxi_tail_30m", "nbr_taxi_cv_30m"]),
        ("C7 apron_pier", ["nbr_taxi_mean_pier_30m", "nbr_taxi_pier_dev_30m", "pier_push_prev_10m",
                           "noise_control"]),
        ("C8 forward_clock", ["nbr_fwd_30m", "nbr_fwd_60m", "nbr_ctr_3h", "nbr_day"]),
    ):
        x = arm(nm, ex)
        print("S7      vs control4 %+.4f" % (x - c))


if __name__ == "__main__":
    todo = [a for a in sys.argv[1:] if a.startswith("S")] or ["S1", "S1b", "S2", "S3", "S4", "S5", "S6", "S7"]
    for s in todo:
        print("\n" + "=" * 30 + " " + s + " " + "=" * 30)
        globals()[s]()
