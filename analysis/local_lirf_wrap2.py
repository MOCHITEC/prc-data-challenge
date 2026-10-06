"""LIRF long-delay unmatched departures: can 'was the off-block entered?' be learned on ordinary rows and transferred? (local)

Population: LIRF departures with AOBT_3 null. Class: copy |y - ms| <= 12 (BLOCK = SCHED), else 'entered'
(normal taxi, time-zone class, one-day wrap). Long-delay set L = ms >= 45 000 (2025: 27 rows), core set ms >= 50 400.
Ordinary set O = 1 h <= ms < 6 h (about 1 300 rows, about 50 % copy).

Signals (each measured on O / on arrivals first, then on L, then shown for the 2026 rows with ms >= 45 000):
  1  LightGBM 'copy vs entered' trained on unmatched rows with ms < 45 000 (fields present in feat_rank), three feature
     sets: full (as local_lirf_unm_fill.py), no-delay (delay-derived fields removed), small (behaviour fields only).
  2a share of schedule copies among LIRF arrivals that landed before SCHED (only those can be copies), same local day
     and +-6 h around the departure's estimated off-block (MVT - 1 100 s); 2b leave-one-out copy share among the
     other unmatched departures of the day (2025 only); 2c local hour of off-block, weekday, flight-number format.
  3  inbound arrival of the rotation: same STAND last in-block before the off-block; flight-number neighbour
     (same 3-letter prefix, number n or n - 1 / n + 1) landing in the 36 h before. Copy / entered-early / late.
  4  ADS-B per-flight features ($SCRATCH/learn/feat_*.parquet): presence, ground fixes, push detection.

    .venv/bin/python analysis/local_lirf_wrap2.py          # all signals, writes signals_2025/2026.parquet
    .venv/bin/python analysis/local_lirf_wrap2.py extra    # adjusted logits + permutation p-values (needs the first run)
Output tables: $SCRATCH/agents/wrap2/
"""
import glob
import os
import re

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(20)
S = os.environ["SCRATCH"]
O = S + "/agents/wrap2/"
os.makedirs(O, exist_ok=True)
L_MIN, CORE = 45000, 50400
FIVE = {"WMTT3508": 0.65, "ITYTY672": 0.65, "SVII3006": 0.65, "TAY9LA": 0.05, "NOSOS431": 0.15}
ROME = "Europe/Rome"
OB_OFF = 1100


def load(paths):
    return (pl.concat([pl.scan_parquet(p) for p in paths]).filter((pl.col("ADEP_mvt") == "LIRF") | (pl.col("ADES_mvt") == "LIRF"))
            .collect())


def deps(a):
    y = pl.col("TAXITIME_SEC_mvt")
    d = (a.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
         .with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds()))
    d = d.with_columns(cls=pl.when(y.is_null()).then(pl.lit("?")).when((y - pl.col("ms")).abs() <= 12).then(pl.lit("copy"))
                       .when(y < 3600).then(pl.lit("norm")).when((y - 86400).is_between(0, 4000)).then(pl.lit("wrap"))
                       .otherwise(pl.lit("oth")),
                       ob=pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=OB_OFF))
    d = d.with_columns(copy=pl.when(pl.col("cls") == "?").then(None).otherwise(pl.col("cls") == "copy"),
                       obL=pl.col("ob").dt.convert_time_zone(ROME),
                       pre=pl.col("FLIGHT_mvt").str.slice(0, 3),
                       num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)\D*$", 1).cast(pl.Int64, strict=False),
                       fmt=pl.col("FLIGHT_mvt").map_elements(lambda s: "dbl" if len(s) > 4 and (s[3:5] == s[1:3] or s[3:4] == s[2:3]) else "plain",
                                                             return_dtype=pl.String).fill_null("null"))
    return d.with_columns(obH=pl.col("obL").dt.hour(), wd=pl.col("obL").dt.weekday(), day=pl.col("obL").dt.date())


def arrs(a):
    r = (a.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF"))
         .with_columns(bs=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                       elig=pl.col("MVT_TIME_UTC_mvt") <= pl.col("SCHED_TIME_UTC_mvt"))
         .with_columns(acopy=pl.col("bs").abs() <= 12,
                       pre=pl.col("FLIGHT_mvt").str.slice(0, 3),
                       num=pl.col("FLIGHT_mvt").str.extract(r"(\d+)\D*$", 1).cast(pl.Int64, strict=False),
                       day=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(ROME).dt.date()))
    return r


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = np.isfinite(s) & np.isfinite(y)
    return float("nan") if len(set(y[ok])) < 2 else roc_auc_score(y[ok], s[ok])


def rate_table(df, by, tgt="copy"):
    return df.group_by(by).agg(n=pl.len(), copy_rate=pl.col(tgt).mean().round(3)).sort(by)


def long_table(df, score, label):
    """counts and AUC (score = P(entered)) of wrap vs copy on the long rows."""
    out = []
    for lo in (L_MIN, CORE):
        z = df.filter((pl.col("ms") >= lo) & pl.col("cls").is_in(["wrap", "copy"]))
        out.append(f"ms>={lo}: n={z.height} AUC(entered->wrap)={auc(z['cls'] == 'wrap', z[score]):.3f}")
    print(f"[{label}] " + " | ".join(out))


# ------------------------------------------------------------------------------------------------ 1 classifier
DELAY_PAT = ("sched", "eobt", "lobt", "iobt", "aobt3", "mvt_minus", "proxy", "ta_delay", "ta_gap", "occ_lb", "revision",
             "stand_aobt", "occ_aobt", "delay_neighbor", "nm_matched", "aobt3_null", "lobt_null", "eobt1_null")
SMALL = ["hour_local", "dow", "month", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "fl_fmt_code", "fl_len", "fl_nm_rate_other",
         "ta_copy", "ta_same_op", "ta_taxi", "dest_distance_km", "ADES_REGION", "dup_n", "stand_null", "runway_null"]
DROP = {"TAXITIME_SEC_mvt", "fill", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
        "FLIGHT_ID_mvt"}
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=2, seed=5)


def classifier(d25, d26):
    sel = pl.col("mvt_minus_aobt3").is_null() & (pl.col("ADEP_mvt") == "LIRF")
    T = pl.scan_parquet("data/work_flseq/feat_train.parquet").filter(sel).collect()
    R = pl.scan_parquet("data/work_flseq/feat_rank.parquet").filter(sel).collect()
    T = T.join(d25.select("MVT_ID_mvt", "cls", "copy", "ms"), on="MVT_ID_mvt")
    R = R.join(d26.select("MVT_ID_mvt", "cls", "ms", "FLIGHT_mvt"), on="MVT_ID_mvt", suffix="_d")
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric() and c in R.columns and T[c].null_count() < T.height
           and c not in ("ms", "copy")]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String and c in R.columns and c not in ("cls", "FLIGHT_mvt")]
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list()) | set(R[c].drop_nulls().to_list())))} for c in cat}
    full = num + cat
    nodelay = [c for c in full if not any(p in c.lower() for p in DELAY_PAT)]
    small = [c for c in SMALL if c in full]

    def X(df, fs):
        return df.select([pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32) if c in cat else pl.col(c).cast(pl.Float64)
                          for c in fs]).to_numpy().astype(np.float64)

    tr = (T["ms"] < L_MIN).to_numpy()
    y = (~T["copy"]).to_numpy().astype(int)  # 1 = entered
    m = T["month"].to_numpy()
    O1 = ((T["ms"] >= 3600) & (T["ms"] < 6 * 3600)).to_numpy()
    res = {}
    for name, fs in (("full", full), ("nodelay", nodelay), ("small", small)):
        cf = [c for c in fs if c in cat]
        Xt, Xr = X(T, fs), X(R, fs)
        # month-grouped CV on the ms < 45 000 rows (4 folds by month mod 4)
        oof = np.full(T.height, np.nan)
        for k in range(4):
            va = tr & (m % 4 == k)
            mdl = lgb.train(P, lgb.Dataset(Xt[tr & ~va], y[tr & ~va], feature_name=fs, categorical_feature=cf), 300)
            oof[va] = mdl.predict(Xt[va])
        mdl = lgb.train(P, lgb.Dataset(Xt[tr], y[tr], feature_name=fs, categorical_feature=cf), 300)
        pT, pR = mdl.predict(Xt), mdl.predict(Xr)
        T = T.with_columns(pl.Series("p_" + name, np.where(tr, oof, pT)))
        R = R.with_columns(pl.Series("p_" + name, pR))
        imp = sorted(zip(mdl.feature_importance("gain"), fs), reverse=True)[:8]
        print(f"\n[1 {name}] {len(fs)} feats; month-CV AUC entered on ms<45000: {auc(y[tr], oof[tr]):.3f}; on 1-6 h rows: "
              f"{auc(y[O1], oof[O1]):.3f}; top gain: " + ", ".join(f"{f}" for g, f in imp))
        long_table(T, "p_" + name, "1 " + name)
        res[name] = fs
    L = T.filter(pl.col("ms") >= L_MIN).sort("ms").select("FLIGHT_mvt", "ms", "cls", "p_full", "p_nodelay", "p_small")
    print(L)
    print("2026 rows ms >= 45000 (P(entered)):")
    print(R.filter(pl.col("ms") >= L_MIN).sort("ms").select("FLIGHT_mvt", "ms", "p_full", "p_nodelay", "p_small"))
    T.select("MVT_ID_mvt", "p_full", "p_nodelay", "p_small").write_parquet(O + "clf_2025.parquet")
    R.select("MVT_ID_mvt", "p_full", "p_nodelay", "p_small").write_parquet(O + "clf_2026.parquet")


# ------------------------------------------------------------------------------------------------ 2 day / time proxies
def arr_share(d, ar, half_h=6):
    """share of copies among eligible arrivals (landed <= SCHED) in +-half_h around the estimated off-block, and same local day."""
    e = ar.filter(pl.col("elig")).select("MVT_TIME_UTC_mvt", "acopy", "day").sort("MVT_TIME_UTC_mvt")
    t = e["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy()
    c = np.concatenate([[0], np.cumsum(e["acopy"].to_numpy().astype(int))])
    ob = d["ob"].dt.epoch("s").to_numpy()
    lo, hi = np.searchsorted(t, ob - half_h * 3600), np.searchsorted(t, ob + half_h * 3600)
    n = hi - lo
    w = np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan)
    day = e.group_by("day").agg(a_day_n=pl.len(), a_day=pl.col("acopy").mean())
    return d.with_columns(pl.Series("a_win", w), pl.Series("a_win_n", n)).join(day, on="day", how="left")


def day_signals(d25, d26, a25, a26):
    print("\n[2a] LIRF arrivals: copy rate (|BLOCK-SCHED|<=12) among arrivals landed <= SCHED")
    e = a25.filter(pl.col("elig"))
    print(f"2025 eligible arrivals {e.height}, copy rate {e['acopy'].mean():.3f}; 2026 eligible {a26.filter(pl.col('elig')).height}, "
          f"copy rate {a26.filter(pl.col('elig'))['acopy'].mean():.3f}")
    dd = e.group_by("day").agg(n=pl.len(), r=pl.col("acopy").mean())
    p0 = e["acopy"].mean()
    print(f"daily arrival copy rate: sd {dd['r'].std():.3f} vs binomial sd {np.sqrt(p0 * (1 - p0) / dd['n'].mean()):.3f} (mean n/day {dd['n'].mean():.0f})")
    print(rate_table(e.with_columns(h=pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(ROME).dt.hour() // 3 * 3), "h", "acopy"))
    d25 = arr_share(d25, a25)
    d26 = arr_share(d26, a26)
    Ox = d25.filter((pl.col("ms") >= 3600) & (pl.col("ms") < 6 * 3600))
    for c in ("a_win", "a_day"):
        q = Ox.with_columns(pl.col(c).qcut(4, allow_duplicates=True).alias("qq"))
        print(f"1-6 h rows: AUC(copy) of {c} {auc(Ox['copy'], Ox[c]):.3f}; within 1-2 h band {auc(Ox.filter(pl.col('ms') < 7200)['copy'], Ox.filter(pl.col('ms') < 7200)[c]):.3f}")
        print(rate_table(q, "qq"))
        long_table(d25.with_columns((-pl.col(c)).alias("s")), "s", "2a " + c + " (low share -> wrap)")

    print("\n[2b] leave-one-out copy share among the other unmatched departures of the same local day (2025)")
    lab = d25.filter(pl.col("cls") != "?").with_columns(band=(pl.col("ms") / 3600).cut([1, 2, 3, 4, 6, 12.5, 100]))
    # expected copy by delay band, residual, LOO day mean of residual and of raw copy
    lab = lab.with_columns(ex=pl.col("copy").cast(pl.Float64).mean().over("band"))
    lab = lab.with_columns(res=pl.col("copy").cast(pl.Float64) - pl.col("ex"))
    lab = lab.with_columns(dn=pl.len().over("day"), ds=pl.col("copy").cast(pl.Float64).sum().over("day"),
                           drs=pl.col("res").sum().over("day"))
    lab = lab.with_columns(loo=pl.when(pl.col("dn") > 1).then((pl.col("ds") - pl.col("copy").cast(pl.Float64)) / (pl.col("dn") - 1)),
                           loo_res=pl.when(pl.col("dn") > 1).then((pl.col("drs") - pl.col("res")) / (pl.col("dn") - 1)))
    Ox = lab.filter((pl.col("ms") >= 3600) & (pl.col("ms") < 6 * 3600))
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(200):
        sh = Ox.with_columns(pl.Series("cp", rng.permutation(Ox["copy"].to_numpy())))
        sh = sh.with_columns(ds=pl.col("cp").cast(pl.Float64).sum().over("day"), dn=pl.len().over("day"))
        sh = sh.filter(pl.col("dn") > 1).with_columns(l=(pl.col("ds") - pl.col("cp").cast(pl.Float64)) / (pl.col("dn") - 1))
        perm.append(auc(sh["cp"], sh["l"]))
    O2 = Ox.with_columns(ds=pl.col("copy").cast(pl.Float64).sum().over("day"), dn=pl.len().over("day")).filter(pl.col("dn") > 1)
    O2 = O2.with_columns(l=(pl.col("ds") - pl.col("copy").cast(pl.Float64)) / (pl.col("dn") - 1))
    print(f"1-6 h rows, LOO over 1-6 h rows of the day: AUC {auc(O2['copy'], O2['l']):.3f} (n={O2.height}); permutation AUC mean "
          f"{np.nanmean(perm):.3f}, 97.5% {np.nanpercentile(perm, 97.5):.3f}")
    print(f"1-6 h rows, LOO residual (delay-band adjusted) over all unmatched of the day: AUC {auc(Ox['copy'], Ox['loo_res']):.3f}")
    print(rate_table(Ox.with_columns(pl.col("loo_res").qcut(4, allow_duplicates=True).alias("qq")), "qq"))
    long_table(lab.with_columns((-pl.col("loo_res")).alias("s")), "s", "2b loo_res (low copy share -> wrap)")

    print("\n[2c] local hour of off-block / weekday / flight-number format, 1-6 h rows")
    Ox = d25.filter((pl.col("ms") >= 3600) & (pl.col("ms") < 6 * 3600))
    print(rate_table(Ox.with_columns(h3=pl.col("obH") // 3 * 3), "h3"))
    print(rate_table(Ox, "wd"))
    print(rate_table(Ox, "fmt"))
    print(rate_table(Ox.with_columns(band=(pl.col("ms") / 3600).cut([2, 3, 4])), ["band", "fmt"]))
    return d25, d26


# ------------------------------------------------------------------------------------------------ 3 pairing
def pairing(d, ar):
    """stand predecessor and flight-number neighbour inbound for each departure."""
    a = ar.filter(pl.col("STAND_mvt").is_not_null()).select(
        pl.col("STAND_mvt"), pl.col("BLOCK_TIME_UTC_mvt").alias("ain"), pl.col("acopy").alias("st_copy"),
        pl.col("elig").alias("st_elig"), pl.col("FLIGHT_mvt").alias("st_fl")).sort("ain")
    x = d.sort("ob").join_asof(a, left_on="ob", right_on="ain", by="STAND_mvt", strategy="backward", tolerance="36h")
    x = x.with_columns(st=pl.when(pl.col("ain").is_null()).then(pl.lit("none")).when(pl.col("st_copy")).then(pl.lit("copy"))
                       .when(pl.col("st_elig")).then(pl.lit("early_entered")).otherwise(pl.lit("late")))
    b = ar.filter(pl.col("num").is_not_null()).select("pre", "num", pl.col("MVT_TIME_UTC_mvt").alias("aland"), pl.col("acopy").alias("fn_copy"),
                                                     pl.col("elig").alias("fn_elig"), pl.col("FLIGHT_mvt").alias("fn_fl"))
    cand = []
    for dn in (0, -1, 1):
        cand.append(x.select("MVT_ID_mvt", "pre", "num", "ob").with_columns(pl.col("num") + dn).join(b, on=["pre", "num"])
                    .filter((pl.col("aland") < pl.col("ob")) & (pl.col("aland") > pl.col("ob") - pl.duration(hours=36)))
                    .with_columns(pri=pl.lit(abs(dn))))
    c = (pl.concat(cand).sort("MVT_ID_mvt", "pri", "aland", descending=[False, False, True]).unique("MVT_ID_mvt", keep="first")
         .select("MVT_ID_mvt", "fn_copy", "fn_elig", "fn_fl"))
    x = x.join(c, on="MVT_ID_mvt", how="left").with_columns(
        fn=pl.when(pl.col("fn_fl").is_null()).then(pl.lit("none")).when(pl.col("fn_copy")).then(pl.lit("copy"))
        .when(pl.col("fn_elig")).then(pl.lit("early_entered")).otherwise(pl.lit("late")))
    return x


def pair_signals(d25, d26, a25, a26):
    print("\n[3] inbound pairing")
    x25, x26 = pairing(d25, a25), pairing(d26, a26)
    Ox = x25.filter((pl.col("ms") >= 3600) & (pl.col("ms") < 6 * 3600))
    for k in ("st", "fn"):
        print(f"1-6 h rows by {k}:"); print(rate_table(Ox, k))
        Lx = x25.filter(pl.col("ms") >= L_MIN)
        print(f"long rows by {k}:"); print(Lx.group_by(k).agg(n=pl.len(), wrap=(pl.col("cls") == "wrap").sum(),
                                                               copy=(pl.col("cls") == "copy").sum()).sort(k))
    return x25, x26


# ------------------------------------------------------------------------------------------------ 4 ADS-B
def adsb(d25, d26):
    print("\n[4] ADS-B per-flight features")
    cols = ["MVT_ID_mvt", "join", "ground_fix", "seen", "n_ground", "push", "t_push", "t_move", "t_mvt", "stand_dur", "n_win_ground"]
    ids25 = d25["MVT_ID_mvt"].to_list()
    f25 = pl.concat([pl.scan_parquet(p).filter(pl.col("MVT_ID_mvt").is_in(ids25)).select(cols)
                     for p in sorted(glob.glob(S + "/learn/feat_2025-*.parquet"))]).collect()
    f26 = pl.scan_parquet(S + "/learn/feat_2026.parquet").filter(pl.col("MVT_ID_mvt").is_in(d26["MVT_ID_mvt"].to_list())).select(cols).collect()

    def prep(d, f):
        f = f.unique("MVT_ID_mvt")
        x = d.join(f, on="MVT_ID_mvt", how="left").with_columns(
            has=pl.col("join").is_not_null(), gfix=pl.col("ground_fix").fill_null(False),
            haspush=pl.col("t_push").is_not_null(), taxi_adsb=(pl.col("t_mvt") - pl.col("t_push")))
        return x
    x25, x26 = prep(d25, f25), prep(d26, f26)
    Ox = x25.filter((pl.col("ms") >= 3600) & (pl.col("ms") < 6 * 3600))
    for k in ("has", "gfix", "haspush"):
        print(rate_table(Ox, k))
    print(x25.filter(pl.col("ms") >= L_MIN).sort("ms").select("FLIGHT_mvt", "ms", "cls", "TAXITIME_SEC_mvt", "has", "gfix", "n_ground",
                                                              "haspush", "taxi_adsb", "stand_dur"))
    return x25, x26


def main():
    a25 = load(sorted(glob.glob("data/raw/training_*.parquet")))
    a26 = load(["data/raw/ranking.parquet"])
    d25, d26 = deps(a25), deps(a26)
    ar25, ar26 = arrs(a25), arrs(a26)
    print(f"2025 unmatched LIRF deps {d25.height}; long (>= {L_MIN}) {d25.filter(pl.col('ms') >= L_MIN).height}; 1-6 h "
          f"{d25.filter((pl.col('ms') >= 3600) & (pl.col('ms') < 21600)).height}")
    classifier(d25, d26)
    d25, d26 = day_signals(d25, d26, ar25, ar26)
    x25, x26 = pair_signals(d25, d26, ar25, ar26)
    b25, b26 = adsb(d25, d26)
    keep = ["FLIGHT_mvt", "ms", "cls", "obH", "wd", "fmt", "a_win", "a_win_n", "a_day", "st", "st_fl", "fn", "fn_fl"]
    j25 = x25.select(["MVT_ID_mvt"] + keep).join(b25.select("MVT_ID_mvt", "has", "gfix", "haspush", "taxi_adsb"), on="MVT_ID_mvt")
    j26 = x26.select(["MVT_ID_mvt"] + keep).join(b26.select("MVT_ID_mvt", "has", "gfix", "haspush", "taxi_adsb"), on="MVT_ID_mvt")
    c25, c26 = pl.read_parquet(O + "clf_2025.parquet"), pl.read_parquet(O + "clf_2026.parquet")
    j25, j26 = j25.join(c25, on="MVT_ID_mvt", how="left"), j26.join(c26, on="MVT_ID_mvt", how="left")
    print("\n=== 2025 long rows, all signals"); print(j25.filter(pl.col("ms") >= L_MIN).sort("ms").drop("MVT_ID_mvt"))
    print("\n=== 2026 rows ms >= 45000, all signals"); print(j26.filter(pl.col("ms") >= L_MIN).sort("ms").drop("MVT_ID_mvt", "cls"))
    j25.write_parquet(O + "signals_2025.parquet"); j26.write_parquet(O + "signals_2026.parquet")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# ------------------------------------------------------------------------------------------------ supplementary checks
def extra():
    """confounder-adjusted versions of the large-sample relationships, and permutation p-values on the long rows."""
    from sklearn.linear_model import LogisticRegression
    s = pl.read_parquet(O + "signals_2025.parquet").filter(pl.col("cls") != "?")
    s = s.with_columns(copy=(pl.col("cls") == "copy").cast(pl.Int32), night=((pl.col("obH") >= 21) | (pl.col("obH") < 6)).cast(pl.Int32),
                       dbl=(pl.col("fmt") != "plain").cast(pl.Int32), band=(pl.col("ms") / 3600).cut([2, 3, 4]).cast(pl.String))
    Ox = s.filter((pl.col("ms") >= 3600) & (pl.col("ms") < 21600))
    rng = np.random.default_rng(1)
    base = ["night", "dbl"]
    D = Ox.with_columns([(pl.col("band") == b).cast(pl.Int32).alias(f"b{i}") for i, b in enumerate(sorted(set(Ox["band"].to_list()))[1:])],
                        st_copy=(pl.col("st") == "copy").cast(pl.Int32), st_ee=(pl.col("st") == "early_entered").cast(pl.Int32),
                        fn_copy=(pl.col("fn") == "copy").cast(pl.Int32), fn_ee=(pl.col("fn") == "early_entered").cast(pl.Int32))
    bands = [c for c in D.columns if re.fullmatch(r"b\d", c)]
    for extra_cols in ([], ["a_win"], ["st_copy", "st_ee"], ["fn_copy", "fn_ee"]):
        cols = bands + base + extra_cols
        Z = D.drop_nulls(cols)
        X, y = Z.select(cols).to_numpy().astype(float), Z["copy"].to_numpy()
        coef = LogisticRegression(C=1e6, max_iter=2000).fit(X, y).coef_[0]
        bs = np.array([LogisticRegression(C=1e6, max_iter=2000).fit(X[i], y[i]).coef_[0]
                       for i in (rng.integers(0, len(y), len(y)) for _ in range(200))])
        print(f"\ncopy ~ band + {' + '.join(base + extra_cols)} (1-6 h rows, n={len(y)}): log-odds [bootstrap 95% CI]")
        for j, c in enumerate(cols):
            if c in bands:
                continue
            print(f"  {c:8s} {coef[j]:+.2f} [{np.percentile(bs[:, j], 2.5):+.2f}, {np.percentile(bs[:, j], 97.5):+.2f}]")
    rng = np.random.default_rng(1)
    for lo in (L_MIN, CORE):
        Lx = s.filter((pl.col("ms") >= lo) & pl.col("cls").is_in(["wrap", "copy"]))
        w = (Lx["cls"] == "wrap").to_numpy()
        print(f"\nlong rows ms>={lo}: n={len(w)} wrap={w.sum()}")
        for k in ("night", "dbl"):
            for v in (0, 1):
                z = Lx.filter(pl.col(k) == v)
                print(f"  {k}={v}: n={z.height} wrap={int((z['cls'] == 'wrap').sum())}")
        for c in ("p_full", "p_nodelay", "p_small"):
            x = Lx[c].to_numpy()
            a0 = auc(w, x)
            pv = np.mean([auc(rng.permutation(w), x) >= a0 for _ in range(2000)])
            print(f"  {c}: AUC {a0:.3f}, one-sided permutation p {pv:.3f}; mean P(entered) {x.mean():.3f} vs observed wrap share {w.mean():.3f}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "extra":
    extra()
