"""Does METAR weather explain the OOF residual of the current pipeline? (local, read-only)

Residual r = y - p, p = b + c_C, from $SCRATCH/learn/cv_rows_months{7,12}_v2.parquet. Weather per (airport, hour of
MVT) from $SCRATCH/agents/wx/wx_hourly.parquet (built by local_wx_build.py). NB: b already carries the production
hourly weather columns (temp, dew-point spread, wind, gust, visibility, frozen flag, de-icing gate; this hour and
the hour before), so this measures what *more* weather adds.

  1. simple: per-airport mean residual on snow / de-icing hours vs other hours
  2. leave-one-month-out small LightGBM on clip(r, +-1800): arm W = all weather + airport + hour + lane,
     arm F = weather flags only (snow / de-icing / frost / fog / low vis) + airport + hour + lane,
     arm A = airport + hour + lane only (control). The last arm in ARMS is compared with A. gain = (SSE(r) - SSE(r - f)) / N on the held-out month,
     day-block bootstrap (1000) for the CI.
  3. 2026: fit W on all months, predict the ranking DEP rows, report rows touched and change size.

    OOF=7|12 ARMS=A,W|A,F POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_wx_resid.py
"""
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
OUT = os.path.join(S, "agents", "wx")
OOF = os.environ.get("OOF", "7")
CLIP = 1800.0
WX = ["temp_c", "dewp_c", "spread_c", "vis_km", "wind_kt", "gust_kt", "snow", "heavy_snow", "fz_precip", "fzfg", "rain",
      "graupel", "fog", "mist", "ts", "precip", "deice", "frost", "deice_wide", "snow_3h", "deice_3h", "deice_wide_3h",
      "precip_3h", "fog_3h", "snow_hrs_12h", "snow_hrs_24h", "deice_hrs_12h", "vis_min_3h", "temp_min_3h", "temp_min_24h"]
BASE = ["apt", "hod", "lane"]
# flags only (no continuous temperature / wind, which can act as a month / day identifier)
FLAGS = ["snow", "heavy_snow", "snow_3h", "snow_hrs_12h", "deice", "deice_3h", "deice_hrs_12h", "deice_wide", "frost", "fzfg",
         "fz_precip", "fog", "fog_3h", "lowvis"]
ARMS = os.environ.get("ARMS", "A,W").split(",")
COLS = {"A": BASE, "W": BASE + WX, "F": BASE + FLAGS}
PRM = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=2, verbose=-1, seed=7)
ROUNDS = 300
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def wx() -> pl.DataFrame:
    h = pl.read_parquet(os.path.join(OUT, "wx_hourly.parquet"))
    h = h.with_columns((pl.col("vis_km") < 1.5).fill_null(False).alias("lowvis"))
    return h.select(["ADEP_mvt", "hour_utc", "lowvis"] + WX).with_columns([pl.col(c).cast(pl.Float32) for c in WX + ["lowvis"]])


def design(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(pl.col("ADEP_mvt").replace_strict({a: i for i, a in enumerate(APTS)}).cast(pl.Int32).alias("apt"),
                          pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hod"),
                          pl.col("has").cast(pl.Int32).alias("lane"))


def load_oof() -> pl.DataFrame:
    cv = pl.read_parquet(os.path.join(S, "learn", f"cv_rows_months{OOF}_v2.parquet"))
    t = pl.concat([pl.scan_parquet(f).select("MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt")
                   for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))]).filter(pl.col("PHASE_mvt") == "DEP") \
        .drop("PHASE_mvt").collect()
    d = cv.join(t, on="MVT_ID_mvt", how="left")
    assert d["MVT_TIME_UTC_mvt"].null_count() == 0
    d = d.with_columns((pl.col("y") - pl.col("b") - pl.col("c_C")).alias("r"),
                       pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hour_utc"),
                       pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").alias("day"))
    d = d.join(wx(), on=["ADEP_mvt", "hour_utc"], how="left")
    return design(d)


def simple(d: pl.DataFrame) -> None:
    for flag in ["snow", "snow_3h", "deice", "deice_wide"]:
        g = d.with_columns(pl.col("r").clip(-CLIP, CLIP).alias("rc")).group_by("ADEP_mvt", pl.col(flag) > 0).agg(
            pl.len().alias("n"), pl.col("r").mean().alias("mean_r"), pl.col("rc").mean().alias("mean_rc"),
            (pl.col("r") ** 2).mean().sqrt().alias("rmse"), pl.col("day").n_unique().alias("days"),
            pl.col("y").mean().alias("mean_y"), (pl.col("b") + pl.col("c_C")).mean().alias("mean_p"))
        w = g.filter(pl.col(flag)).join(g.filter(~pl.col(flag)).select("ADEP_mvt", pl.col("mean_rc").alias("mean_rc_other"),
                                                                        pl.col("rmse").alias("rmse_other"),
                                                                        pl.col("mean_y").alias("mean_y_other")), on="ADEP_mvt")
        print(f"\n== simple: residual on {flag} hours vs others (OOF {OOF} months)")
        print(w.sort("ADEP_mvt").drop(flag))


def boot(gd: np.ndarray, n: int, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(gd), size=(reps, len(gd)))
    v = gd[idx].sum(1) / n
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def lomo(d: pl.DataFrame) -> dict:
    months = sorted(d["ym"].unique().to_list())
    rc = d["r"].clip(-CLIP, CLIP).to_numpy()
    r = d["r"].to_numpy()
    ym = d["ym"].to_numpy()
    days = d["day"].to_numpy()
    snow = (d["snow_3h"].fill_null(0).to_numpy() > 0) | (d["deice"].fill_null(0).to_numpy() > 0)
    res = {}
    preds = {}
    for arm in ARMS:
        cols = COLS[arm]
        X = d.select(cols).to_numpy().astype(np.float32)
        f = np.zeros(len(r))
        for m in months:
            tr, te = ym != m, ym == m
            ds = lgb.Dataset(X[tr], rc[tr], feature_name=cols, categorical_feature=["apt"], free_raw_data=True)
            bst = lgb.train(PRM, ds, ROUNDS)
            f[te] = bst.predict(X[te], num_threads=2)
            print(arm, m, "done", flush=True)
        preds[arm] = f
    out = []
    for arm in ARMS:
        f = preds[arm]
        g = r ** 2 - (r - f) ** 2
        for m in months + ["ALL"]:
            sel = (ym == m) if m != "ALL" else np.ones(len(r), bool)
            dd = pl.DataFrame({"day": days[sel], "g": g[sel]}).group_by("day").agg(pl.col("g").sum())["g"].to_numpy()
            lo, hi = boot(dd, sel.sum())
            gs = g[sel & snow].sum() / max(1, (sel & snow).sum())
            out.append(dict(arm=arm, ym=m, n=int(sel.sum()), gain=float(g[sel].sum() / sel.sum()), lo=lo, hi=hi,
                            n_wx=int((sel & snow).sum()), gain_per_wx_row=float(gs), mean_f_wx=float(f[sel & snow].mean()) if (sel & snow).any() else 0.0))
    # weather increment: W - A (W = the last arm)
    WARM = ARMS[-1]
    gW = r ** 2 - (r - preds[WARM]) ** 2
    gA = r ** 2 - (r - preds["A"]) ** 2
    for m in months + ["ALL"]:
        sel = (ym == m) if m != "ALL" else np.ones(len(r), bool)
        dd = pl.DataFrame({"day": days[sel], "g": (gW - gA)[sel]}).group_by("day").agg(pl.col("g").sum())["g"].to_numpy()
        lo, hi = boot(dd, sel.sum())
        out.append(dict(arm=f"{WARM}-A", ym=m, n=int(sel.sum()), gain=float((gW - gA)[sel].sum() / sel.sum()), lo=lo, hi=hi,
                        n_wx=int((sel & snow).sum()), gain_per_wx_row=float((gW - gA)[sel & snow].sum() / max(1, (sel & snow).sum())),
                        mean_f_wx=0.0))
    T = pl.DataFrame(out)
    pl.Config.set_tbl_rows(100); pl.Config.set_float_precision(2); pl.Config.set_tbl_width_chars(200)
    print(f"\n== LOMO gains (s^2 per row), OOF {OOF} months; n_wx = rows with snow_3h or deice")
    print(T)
    T.write_csv(os.path.join(OUT, f"lomo_oof{OOF}_{''.join(ARMS)}.csv"))
    # per-airport weather increment on snow/deice rows
    pa = pl.DataFrame({"apt": d["ADEP_mvt"], "ym": ym, "wx": snow, "dg": gW - gA, "fW": preds[WARM], "fA": preds["A"], "r": r}) \
        .filter(pl.col("wx")).group_by("apt").agg(pl.len(), pl.col("dg").sum().alias("sum_gain_WminusA"),
                                                    (pl.col("fW") - pl.col("fA")).mean().alias("mean_fW_minus_fA"),
                                                    pl.col("r").mean().alias("mean_r")).sort("apt")
    print("\n== snow/deice rows by airport: W-A gain (sum s^2), mean extra shift, mean residual")
    print(pa)
    return res


def rank2026(d: pl.DataFrame) -> None:
    rk = pl.read_parquet("data/raw/ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]) \
        .filter(pl.col("PHASE_mvt") == "DEP")
    fh = pl.read_parquet(os.path.join(S, "learn", "feat_2026.parquet"), columns=["MVT_ID_mvt", "icao"]) \
        .filter(pl.col("icao").is_not_null()).select("MVT_ID_mvt").unique().with_columns(pl.lit(True).alias("has"))
    rk = rk.join(fh, on="MVT_ID_mvt", how="left").with_columns(pl.col("has").fill_null(False),
                                                                pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hour_utc"),
                                                                pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"))
    rk = design(rk.join(wx(), on=["ADEP_mvt", "hour_utc"], how="left"))
    print("\n2026 lane share has:", rk["has"].mean())
    rc = d["r"].clip(-CLIP, CLIP).to_numpy()
    fs = {}
    for arm in ("A", ARMS[-1]):
        cols = COLS[arm]
        bst = lgb.train(PRM, lgb.Dataset(d.select(cols).to_numpy().astype(np.float32), rc, feature_name=cols,
                                         categorical_feature=["apt"]), ROUNDS)
        fs[arm] = bst.predict(rk.select(cols).to_numpy().astype(np.float32), num_threads=2)
        if arm != "A":
            imp = sorted(zip(cols, bst.feature_importance("gain")), key=lambda x: -x[1])[:12]
            print("W importance (gain):", [(c, round(v / 1e6, 1)) for c, v in imp])
    rk = rk.with_columns(pl.Series("fW", fs[ARMS[-1]]), pl.Series("fA", fs["A"]), (pl.Series(fs[ARMS[-1]] - fs["A"])).alias("dWA"),
                         ((pl.col("snow_3h") > 0) | (pl.col("deice") > 0)).alias("wx"))
    s = rk.group_by("ym", "ADEP_mvt", "wx").agg(pl.len(), pl.col("dWA").mean().alias("mean_dWA"),
                                               pl.col("dWA").abs().mean().alias("mean_abs_dWA"),
                                               (pl.col("dWA") ** 2).mean().sqrt().alias("rms_dWA")).sort("ym", "ADEP_mvt", "wx")
    print("\n== 2026 DEP rows: weather-induced change (W - A) by airport and snow/deice flag")
    print(s.filter(pl.col("ym") == "2026-01"))
    tot = rk.group_by("ym", "wx").agg(pl.len(), pl.col("dWA").mean(), pl.col("dWA").abs().mean().alias("abs"),
                                      (pl.col("dWA") ** 2).mean().sqrt().alias("rms"))
    print(tot.sort("ym", "wx"))
    rk.select("MVT_ID_mvt", "ADEP_mvt", "ym", "wx", "fW", "fA", "dWA").write_parquet(os.path.join(OUT, f"rank2026_oof{OOF}_{ARMS[-1]}.parquet"))


def main() -> None:
    d = load_oof()
    print("rows", d.height, "wx join null share", d["temp_c"].is_null().mean(), flush=True)
    simple(d)
    lomo(d)
    rank2026(d)


if __name__ == "__main__":
    main()
