"""Weather (METAR) features per (airport, hour) + 2025 vs 2026 prevalence of snow / de-icing hours (local, read-only).

Source: the cached IEM ASOS/METAR download in data/ref/metar/<ICAO>_2025-01-01_2026-08-01.csv (Iowa Environmental
Mesonet, public domain; routine METARs at hh:50, 24/day, all 10 stations, 2025-01-01..2026-07-31).
A METAR at hh:50 is keyed to hour hh+1 (it describes the start of that hour); a departure is joined on the
UTC hour of MVT_TIME_UTC_mvt (BLOCK is hidden in the ranking set), and the 3 h / 12 h rolling columns cover the
off-block period.

Outputs ($SCRATCH/agents/wx/):
  wx_hourly.parquet   one row per (ADEP_mvt, hour_utc) on a full hourly grid
  dep_wx.parquet      every 2025 training DEP + 2026 ranking DEP row (MVT_ID_mvt, ADEP_mvt, ym, set) with wx flags
  prevalence.csv      share of hours and of departures under snow / de-icing per airport x month

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_wx_build.py
"""
import glob
import os
import re
from datetime import datetime, timezone

import polars as pl

OUT = os.path.join(os.environ["SCRATCH"], "agents", "wx")
os.makedirs(OUT, exist_ok=True)


def tokens(s: str) -> list[str]:
    if s is None or s == "M":
        return []
    return [t for t in s.split() if not t.startswith("VC")]   # vicinity phenomena are not at the airport


DESC = ("MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ")


def parse(t: str) -> tuple[str, set, set]:
    """METAR present-weather group -> (intensity, descriptors, 2-letter phenomena)."""
    inten = t[0] if t[:1] in "+-" else ""
    r = t[len(inten):]
    desc = set()
    while r[:2] in DESC:
        desc.add(r[:2]); r = r[2:]
    return inten, desc, {r[i:i + 2] for i in range(0, len(r), 2)}


def flags(s: str) -> dict:
    groups = [parse(t) for t in tokens(s)]
    ph = set().union(*[g[2] for g in groups]) if groups else set()
    has = lambda pred: any(pred(i, d, p) for i, d, p in groups)  # noqa: E731
    d = {
        "snow": bool(ph & {"SN", "SG"}),
        "heavy_snow": has(lambda i, d_, p: bool(p & {"SN"}) and i != "-" and not (d_ & {"BL", "DR"})),  # moderate/heavy
        "fz_precip": has(lambda i, d_, p: "FZ" in d_ and bool(p & {"RA", "DZ"})) or bool(ph & {"PL", "IC"}),
        "fzfg": has(lambda i, d_, p: "FZ" in d_ and "FG" in p),
        "rain": has(lambda i, d_, p: "FZ" not in d_ and bool(p & {"RA", "DZ"})),
        "graupel": bool(ph & {"GS", "GR"}),
        "fog": has(lambda i, d_, p: "FG" in p and not (d_ & {"MI", "PR", "BC"})),
        "mist": "BR" in ph,
        "ts": has(lambda i, d_, p: "TS" in d_),
    }
    d["precip"] = d["snow"] or d["fz_precip"] or d["rain"] or d["graupel"] or ("UP" in ph)
    return d


def load_metar() -> pl.DataFrame:
    fr = []
    for f in sorted(glob.glob("data/ref/metar/*_2025-01-01_2026-08-01.csv")):
        m = pl.read_csv(f, null_values=["M", ""], infer_schema_length=0)
        m = m.filter(pl.col("valid").is_not_null() & (pl.col("valid") != "valid"))
        fr.append(m)
    m = pl.concat(fr)
    num = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "vsby"]
    m = m.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    m = m.with_columns(pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("t"))
    fl = [flags(s) for s in m["wxcodes"].to_list()]
    F = pl.DataFrame(fl)
    m = pl.concat([m, F], how="horizontal_extend")
    m = m.with_columns(
        ((pl.col("tmpf") - 32) / 1.8).alias("temp_c"),
        ((pl.col("dwpf") - 32) / 1.8).alias("dewp_c"),
        (pl.col("vsby") * 1.609).alias("vis_km"),
        pl.col("sknt").alias("wind_kt"),
        (pl.col("t") + pl.duration(minutes=10)).dt.truncate("1h").alias("hour_utc"),
        pl.col("station").alias("ADEP_mvt"),
    )
    return m


def hourly(m: pl.DataFrame) -> pl.DataFrame:
    bools = ["snow", "heavy_snow", "fz_precip", "fzfg", "rain", "graupel", "fog", "mist", "ts", "precip"]
    h = m.group_by(["ADEP_mvt", "hour_utc"]).agg(
        pl.col("temp_c").mean(), pl.col("dewp_c").mean(), pl.col("vis_km").min(), pl.col("wind_kt").max(),
        pl.col("gust").max().alias("gust_kt"), *[pl.col(b).any() for b in bools])
    # full grid so rolling windows are in real hours
    grid = []
    for a in h["ADEP_mvt"].unique().sort().to_list():
        grid.append(pl.DataFrame({"hour_utc": pl.datetime_range(datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 8, 1, 1, tzinfo=timezone.utc),
                                                                "1h", eager=True, time_zone="UTC")}).with_columns(pl.lit(a).alias("ADEP_mvt")))
    h = pl.concat(grid).join(h, on=["ADEP_mvt", "hour_utc"], how="left").sort("ADEP_mvt", "hour_utc")
    # fill single missing hours from the previous observation (max 2 h)
    cont = ["temp_c", "dewp_c", "vis_km", "wind_kt"]
    h = h.with_columns([pl.col(c).forward_fill(limit=2).over("ADEP_mvt") for c in cont] +
                       [pl.col(b).fill_null(False) for b in bools])
    h = h.with_columns(
        (pl.col("temp_c") - pl.col("dewp_c")).alias("spread_c"),
        # de-icing proxy: any precipitation while T < 3 C, or freezing precipitation / freezing fog
        ((pl.col("precip") & (pl.col("temp_c") < 3)) | pl.col("fz_precip") | pl.col("fzfg")).alias("deice"),
        # frost: sub-zero and near-saturated (hoar frost on parked aircraft)
        ((pl.col("temp_c") <= 0) & ((pl.col("temp_c") - pl.col("dewp_c")) <= 2)).alias("frost"),
    )
    h = h.with_columns((pl.col("deice") | pl.col("frost")).alias("deice_wide"))
    roll = []
    for b in ["snow", "deice", "deice_wide", "precip", "fog"]:
        roll.append(pl.col(b).cast(pl.Int8).rolling_max(3, min_samples=1).over("ADEP_mvt").alias(f"{b}_3h"))
    roll += [
        pl.col("snow").cast(pl.Int16).rolling_sum(12, min_samples=1).over("ADEP_mvt").alias("snow_hrs_12h"),
        pl.col("snow").cast(pl.Int16).rolling_sum(24, min_samples=1).over("ADEP_mvt").alias("snow_hrs_24h"),
        pl.col("deice").cast(pl.Int16).rolling_sum(12, min_samples=1).over("ADEP_mvt").alias("deice_hrs_12h"),
        pl.col("vis_km").rolling_min(3, min_samples=1).over("ADEP_mvt").alias("vis_min_3h"),
        pl.col("temp_c").rolling_min(3, min_samples=1).over("ADEP_mvt").alias("temp_min_3h"),
        pl.col("temp_c").rolling_min(24, min_samples=1).over("ADEP_mvt").alias("temp_min_24h"),
    ]
    h = h.with_columns(roll)
    return h


def deps() -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]
    tr = pl.concat([pl.scan_parquet(f).select(cols) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))]) \
        .filter(pl.col("PHASE_mvt") == "DEP").with_columns(pl.lit("train").alias("set")).collect()
    rk = pl.read_parquet("data/raw/ranking.parquet", columns=cols).filter(pl.col("PHASE_mvt") == "DEP") \
        .with_columns(pl.lit("rank").alias("set"))
    d = pl.concat([tr, rk]).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hour_utc"),
                                         pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"))
    return d.drop("PHASE_mvt")


def main() -> None:
    m = load_metar()
    print("metar obs", m.height, m["t"].min(), m["t"].max(), flush=True)
    h = hourly(m)
    h.write_parquet(os.path.join(OUT, "wx_hourly.parquet"))
    print("hourly", h.height, "missing temp share", h["temp_c"].is_null().mean())
    d = deps().join(h.select("ADEP_mvt", "hour_utc", "snow", "snow_3h", "deice", "deice_3h", "deice_wide", "frost", "fzfg",
                             "temp_c", "vis_km"), on=["ADEP_mvt", "hour_utc"], how="left")
    d.write_parquet(os.path.join(OUT, "dep_wx.parquet"))
    hp = h.with_columns(pl.col("hour_utc").dt.strftime("%Y-%m").alias("ym")).group_by("ADEP_mvt", "ym").agg(
        pl.col("snow").mean().alias("h_snow"), pl.col("deice").mean().alias("h_deice"),
        pl.col("deice_wide").mean().alias("h_deice_wide"))
    dp = d.group_by("ADEP_mvt", "ym").agg(pl.len().alias("n_dep"), pl.col("snow").mean().alias("d_snow"),
                                          pl.col("snow_3h").cast(pl.Float64).mean().alias("d_snow_3h"),
                                          pl.col("deice").mean().alias("d_deice"), pl.col("deice_wide").mean().alias("d_deice_wide"),
                                          pl.col("snow").sum().alias("n_snow"), pl.col("deice").sum().alias("n_deice"))
    p = dp.join(hp, on=["ADEP_mvt", "ym"], how="left").sort("ADEP_mvt", "ym")
    p.write_csv(os.path.join(OUT, "prevalence.csv"))
    pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(20); pl.Config.set_float_precision(3)
    for mo in ["01", "02", "07", "11", "12"]:
        print(p.filter(pl.col("ym").str.ends_with(mo)).select("ADEP_mvt", "ym", "n_dep", "d_snow", "d_snow_3h", "d_deice", "d_deice_wide", "n_snow", "n_deice"))


if __name__ == "__main__":
    main()
