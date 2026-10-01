"""Transductive same-period congestion/disruption features (parity: only columns visible in ranking).

DEP rows: MVT_TIME, AOBT_3, SCHED (other departures' MVT-AOBT3 / MVT-SCHED, take-off counts, NM-unmatched).
ARR rows: all columns incl. labelled taxi-in (visible in ranking).
Output: tx_feats.parquet with one row per DEP MVT_ID (2025 training + 2026 ranking).
"""
import glob
import numpy as np, polars as pl

ROOT = "/home/user/prc-data-challenge/data/raw/"
OUT = "/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/longtaxi/tx_feats.parquet"
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
files = sorted(glob.glob(ROOT + "training_*.parquet")) + [ROOT + "ranking.parquet"]
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "AOBT_3_flt", "TAXITIME_SEC_mvt", "RUNWAY_mvt", "STAND_mvt"]
raw = pl.concat([pl.scan_parquet(f).select(cols).with_columns(pl.lit("rank" in f).alias("is_rank")) for f in files]).collect()
raw = raw.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"))
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").select(
    "MVT_ID_mvt", "ADEP_mvt", "t", "is_rank", "MVT_TIME_UTC_mvt",
    ((pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().cast(pl.Float64)).alias("mma"),
    ((pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64)).alias("mms"))
arr = raw.filter(pl.col("PHASE_mvt") == "ARR").select(
    pl.col("ADES_mvt").alias("apt"), "is_rank", "RUNWAY_mvt", "STAND_mvt",
    pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("tib"),
    pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ti"))
del raw
# arrival taxi-in residual vs 2025 airport x runway x stand median (fallback airport x runway, airport)
arr = arr.filter(pl.col("ti").is_between(0, 7200) & pl.col("tib").is_not_null())
a25 = arr.filter(~pl.col("is_rank"))
m3 = a25.group_by("apt", "RUNWAY_mvt", "STAND_mvt").agg(pl.col("ti").median().alias("m3"), pl.len().alias("n3"))
m2 = a25.group_by("apt", "RUNWAY_mvt").agg(pl.col("ti").median().alias("m2"))
m1 = a25.group_by("apt").agg(pl.col("ti").median().alias("m1"))
arr = (arr.join(m3, on=["apt", "RUNWAY_mvt", "STAND_mvt"], how="left").join(m2, on=["apt", "RUNWAY_mvt"], how="left")
       .join(m1, on="apt", how="left")
       .with_columns((pl.col("ti") - pl.when(pl.col("n3") >= 20).then("m3").otherwise(pl.coalesce("m2", "m1"))).alias("tir")))

dep = dep.with_columns(pl.col("mma").clip(-600, 5400).alias("mma_c"),
                       pl.col("mma").is_null().cast(pl.Float64).alias("unm"),
                       (pl.col("mms") > 10800).cast(pl.Float64).alias("late3h"))
# 2025 airport normals
d25 = dep.filter(~pl.col("is_rank"))
norm_mma = dict(d25.filter(pl.col("mma").is_between(0, 7200)).group_by("ADEP_mvt").agg(pl.col("mma").mean()).iter_rows())
# hourly departure count normal per airport x local hour (2025 median of hourly counts)
loc = []
for apt, tz in TZ.items():
    sub = dep.filter(pl.col("ADEP_mvt") == apt).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.convert_time_zone(tz).alias("lt"))
    loc.append(sub.with_columns(pl.col("lt").dt.date().alias("lday"), pl.col("lt").dt.hour().alias("lhour"),
                                pl.col("lt").dt.weekday().alias("ldow")).drop("lt"))
dep = pl.concat(loc)
hc = (dep.filter(~pl.col("is_rank") & pl.col("t").is_not_null()).group_by("ADEP_mvt", "lday", "lhour").agg(pl.len().alias("c"))
      .group_by("ADEP_mvt", "lhour").agg(pl.col("c").median().alias("hour_norm")))
dc = (dep.filter(~pl.col("is_rank")).group_by("ADEP_mvt", "lday", "ldow").agg(pl.len().alias("c"))
      .group_by("ADEP_mvt", "ldow").agg(pl.col("c").median().alias("day_norm")))


def wsum(ev_t, ev_v, q, lo, hi):
    """sum and count of ev_v (nan-safe) for events with time in [q+lo, q+hi]."""
    o = np.argsort(ev_t); et = ev_t[o]; ev = ev_v[o]
    ok = ~np.isnan(ev)
    cs = np.concatenate([[0], np.cumsum(np.where(ok, ev, 0))]); cn = np.concatenate([[0], np.cumsum(ok)])
    a = np.searchsorted(et, q + lo, "left"); b = np.searchsorted(et, q + hi, "right")
    return cs[b] - cs[a], cn[b] - cn[a]


out = []
for apt in TZ:
    d = dep.filter(pl.col("ADEP_mvt") == apt).filter(pl.col("t").is_not_null())
    a = arr.filter(pl.col("apt") == apt)
    t = d["t"].to_numpy(); nm = norm_mma[apt]
    cols_ = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
    for name, v in (("mma", d["mma_c"].to_numpy()), ("unm", d["unm"].to_numpy()), ("late3h", d["late3h"].to_numpy())):
        self_ok = ~np.isnan(v)
        for tag, lo, hi in (("c30", -1800, 1800), ("c60", -3600, 3600), ("c120", -7200, 7200), ("f60", 1, 3600), ("p60", -3600, -1)):
            if name != "mma" and tag not in ("c60", "c120"):
                continue
            s, n = wsum(t, v, t, lo, hi)
            if lo <= 0 <= hi:  # exclude self
                s = s - np.where(self_ok, v, 0); n = n - self_ok
            m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            if name == "mma":
                cols_[f"tx_mma_{tag}_ratio"] = m / nm
                cols_[f"tx_mma_{tag}_n"] = n.astype(float)
            else:
                cols_[f"tx_{name}_{tag}"] = m
    # take-off throughput centered 60 vs hour normal
    s, n = wsum(t, np.ones_like(t), t, -1800, 1800)
    cols_["tx_tko_c60"] = n - 1.0
    # arrival taxi-in residual windows (by in-block time)
    at = a["tib"].to_numpy(); ar = a["tir"].to_numpy(); araw = a["ti"].to_numpy()
    for tag, lo, hi in (("c60", -3600, 3600), ("c120", -7200, 7200), ("f60", 0, 3600), ("p60", -3600, 0)):
        s, n = wsum(at, ar, t, lo, hi)
        cols_[f"tx_tir_{tag}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        if tag == "c60":
            cols_["tx_tir_c60_n"] = n.astype(float)
            s2, n2 = wsum(at, (ar > 600).astype(float), t, lo, hi)
            cols_["tx_tir_c60_gt10"] = np.where(n2 > 0, s2 / np.maximum(n2, 1), np.nan)
    out.append(pl.DataFrame(cols_))
tx = pl.concat(out)
# day-level (local day) aggregates
arr_day = []
for apt, tz in TZ.items():
    s = arr.filter(pl.col("apt") == apt).with_columns(
        pl.from_epoch(pl.col("tib").cast(pl.Int64), "s").dt.replace_time_zone("UTC").dt.convert_time_zone(tz).dt.date().alias("lday"))
    arr_day.append(s.group_by("apt", "lday").agg(pl.col("tir").mean().alias("tx_day_tir"),
                                                 (pl.col("tir") > 600).mean().alias("tx_day_tir_gt10")))
arr_day = pl.concat(arr_day).rename({"apt": "ADEP_mvt"})
day = (dep.group_by("ADEP_mvt", "lday", "ldow").agg(
    (pl.col("mma").filter(pl.col("mma").is_between(0, 7200)).mean()).alias("day_mma"),
    pl.col("unm").mean().alias("tx_day_unm"), pl.col("late3h").mean().alias("tx_day_late3h"), pl.len().alias("day_n"))
    .join(dc, on=["ADEP_mvt", "ldow"], how="left").join(arr_day, on=["ADEP_mvt", "lday"], how="left")
    .with_columns((pl.col("day_mma") / pl.col("ADEP_mvt").replace_strict(norm_mma, return_dtype=pl.Float64)).alias("tx_day_mma_ratio"),
                  (pl.col("day_n") / pl.col("day_norm")).alias("tx_day_n_ratio")))
key = dep.select("MVT_ID_mvt", "ADEP_mvt", "lday", "lhour", "ldow", "is_rank").join(hc, on=["ADEP_mvt", "lhour"], how="left")
tx = (key.join(tx, on="MVT_ID_mvt", how="left")
      .join(day.select("ADEP_mvt", "lday", "tx_day_mma_ratio", "tx_day_unm", "tx_day_late3h", "tx_day_n_ratio", "tx_day_tir", "tx_day_tir_gt10"),
            on=["ADEP_mvt", "lday"], how="left")
      .with_columns((pl.col("tx_tko_c60") / pl.col("hour_norm")).alias("tx_tko_c60_rel"))
      .with_columns([(pl.col(c) - pl.col("tx_day_mma_ratio")).alias(c + "_vsday") for c in ("tx_mma_c60_ratio",)]))
tx = tx.with_columns([pl.col(c).cast(pl.Float32) for c in tx.columns if c.startswith("tx_")])
tx.write_parquet(OUT)
print(tx.shape)
print(tx.group_by("is_rank").agg([pl.col(c).null_count() / pl.len() for c in tx.columns if c.startswith("tx_")]).transpose(include_header=True))
