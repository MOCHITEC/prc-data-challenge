"""Stress test of C7 (LIRF 'one day + taxi' class): which observables separate wrap vs schedule copy (local).

Unmatched (AOBT_3 null) LIRF departures. Classes: copy |y-ms|<=12, norm y<3600, wrap y-86400 in [0,4000].
Prints the class table by delay band x midnight crossing (local and UTC, between SCHED and approx. off-block),
the per-row feature list for ms >= 45000, and the same features for the 2026 rows with ms >= 45000.

    .venv/bin/python analysis/local_lirf_wrap_stress.py
"""
import glob
import os
import re

import polars as pl

pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(20)
O = os.environ["SCRATCH"] + "/agents/lirf/"


def load(paths):
    return pl.concat([pl.scan_parquet(p) for p in paths]).filter((pl.col("ADEP_mvt") == "LIRF") | (pl.col("ADES_mvt") == "LIRF")).collect()


def feats(a):
    d = a.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("AOBT_3_flt").is_null())
    y = pl.col("TAXITIME_SEC_mvt")
    d = d.with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds())
    d = d.with_columns(cls=pl.when(y.is_null()).then(pl.lit("?")).when((y - pl.col("ms")).abs() <= 12).then(pl.lit("copy"))
                       .when(y < 3600).then(pl.lit("norm")).when((y - 86400).is_between(0, 4000)).then(pl.lit("wrap")).otherwise(pl.lit("oth")),
                       ob=pl.col("MVT_TIME_UTC_mvt") - pl.duration(seconds=1100))
    loc = lambda c: pl.col(c).dt.convert_time_zone("Europe/Rome")
    d = d.with_columns(crossU=(pl.col("ob").dt.date() - pl.col("SCHED_TIME_UTC_mvt").dt.date()).dt.total_days(),
                       crossL=(loc("ob").dt.date() - loc("SCHED_TIME_UTC_mvt").dt.date()).dt.total_days(),
                       schH=loc("SCHED_TIME_UTC_mvt").dt.hour() + loc("SCHED_TIME_UTC_mvt").dt.minute() / 60,
                       obH=loc("ob").dt.hour() + loc("ob").dt.minute() / 60,
                       wd=loc("SCHED_TIME_UTC_mvt").dt.weekday(), mon=pl.col("SCHED_TIME_UTC_mvt").dt.month(),
                       fmt=pl.col("FLIGHT_mvt").map_elements(lambda s: "null" if s is None else ("dbl" if re.match(r"^([A-Z]{3})\1?", s) and len(s) > 2 and (s[3:5] == s[1:3] or s[3:4] == s[2:3]) else "plain"), return_dtype=pl.String))
    # nearest-date rule: recorded block - sched = delta wrapped into [-12h, 12h)
    d = d.with_columns(dh=(pl.col("ob") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds() / 3600)
    return d


def main():
    tr = load(sorted(glob.glob("data/raw/training_*.parquet")))
    d = feats(tr)
    d.write_parquet(O + "lirf_unm_2025_feats.parquet")
    f = lambda g: g.agg(n=pl.len(), copy=(pl.col("cls") == "copy").sum(), norm=(pl.col("cls") == "norm").sum(),
                        wrap=(pl.col("cls") == "wrap").sum(), oth=(pl.col("cls") == "oth").sum())
    d2 = d.filter(pl.col("ms") > 3600).with_columns(band=(pl.col("ms") / 3600).cut([2, 3, 4, 5, 7, 10, 14, 15.5, 100]))
    print("class by band x crossL x crossU (2025 unmatched LIRF)")
    print(f(d2.group_by("band", "crossL", "crossU")).sort("band", "crossL", "crossU"))
    cols = ["FLIGHT_mvt", "fmt", "STAND_mvt", "RUNWAY_mvt", "mon", "wd", "schH", "obH", "dh", "crossL", "crossU", "ms", "TAXITIME_SEC_mvt", "cls"]
    print(d.filter(pl.col("ms") >= 45000).sort("ms").select(cols))
    print(f(d.filter(pl.col("ms") >= 45000).group_by("fmt")))
    print(f(d.filter(pl.col("ms") >= 45000).group_by("crossL", "crossU")))
    r = feats(load(["data/raw/ranking.parquet"])).filter(pl.col("ms") >= 45000).sort("ms")
    print("2026"); print(r.select(["MVT_ID_mvt"] + cols))


if __name__ == "__main__":
    main()
