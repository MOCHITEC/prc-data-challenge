"""Leave-one-month-out checks on the 7-month OOF (p = b + c_C) of label-artefact post-processes (local).

A  binned recalibration on d = MVT-SCHED - p (schedule-copy pull), per airport x lane: p' = p + shrunk mean residual of
   the (airport, lane, d-bin) cell estimated on the other 6 months.
B  LIRF unmatched 'local-time-entered off-block' class (y = UTC offset + taxi): per FLIGHT prefix rate from 2025 months
   other than the held-out one, p' = q (offset + T) + (1 - q) p on rows with MVT-SCHED < 1 h.
C  LIRF unmatched rows with 7 h <= approx off-block - SCHED < 13.5 h (2025: all schedule copies): p' = MVT - SCHED.
Prints gain in MSE units (SSE reduction / N_oof) per held-out month.

    .venv/bin/python analysis/local_copy_recal_lomo.py
"""
import glob
import os

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(300)
S = os.environ["SCRATCH"]
O = S + "/agents/lirf/"


def report(name, j, pnew):
    N = 1233234
    g = j.with_columns(pn=pnew).with_columns(g=(pl.col("p") - pl.col("y")) ** 2 - (pl.col("pn") - pl.col("y")) ** 2)
    t = g.group_by("ym").agg(gain=pl.col("g").sum() / N, touched=(pl.col("pn") != pl.col("p")).sum()).sort("ym")
    print(name, f"total gain {t['gain'].sum():+.2f} MSE, rows touched {t['touched'].sum()}")
    print(t.transpose(include_header=True))


def main():
    cv = pl.read_parquet(S + "/learn/cv_rows_months7_v2.parquet").with_columns(p=pl.col("b") + pl.col("c_C"))
    fl = pl.read_parquet(O + "dep_2025_copyflags.parquet")
    j = cv.join(fl.select("MVT_ID_mvt", "ms", "un", "copy", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt"), on="MVT_ID_mvt", how="left")
    assert j["ms"].null_count() == 0
    j = j.with_columns(d=pl.col("ms") - pl.col("p"))
    edges = [-600, -300, -180, -120, -60, -30, 0, 30, 60, 120, 180, 300, 600]
    j = j.with_columns(db=pl.col("d").cut(edges).cast(pl.String))
    # A
    for k in (200, 1000):
        out = []
        for m in j["ym"].unique().sort():
            tr = j.filter(pl.col("ym") != m).group_by("ADEP_mvt", "un", "db").agg(s=(pl.col("y") - pl.col("p")).sum(), n=pl.len())
            te = j.filter(pl.col("ym") == m).join(tr, on=["ADEP_mvt", "un", "db"], how="left")
            out.append(te.with_columns(adj=(pl.col("s") / (pl.col("n") + k)).fill_null(0)))
        a = pl.concat(out)
        report(f"A recal airport x lane x bin(ms-p), shrink k={k}", a, pl.col("p") + pl.col("adj"))
        print(a.filter(pl.col("ym") == "2025-07").group_by("ADEP_mvt").agg(mean_abs_adj=pl.col("adj").abs().mean()).sort("ADEP_mvt").transpose(include_header=True))
    # B
    raw = (pl.concat([pl.scan_parquet(f).select("MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "AOBT_3_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
                      for f in sorted(glob.glob("data/raw/training_*.parquet"))])
           .filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("PHASE_mvt") == "DEP") & pl.col("AOBT_3_flt").is_null()).collect())
    off = 3600 + pl.col("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.dst_offset().dt.total_seconds()
    raw = raw.with_columns(ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(), pre=pl.col("FLIGHT_mvt").str.slice(0, 3).fill_null("?"),
                           ym=pl.col("SCHED_TIME_UTC_mvt").dt.strftime("%Y-%m"), off=off)
    raw = raw.with_columns(tz=((pl.col("TAXITIME_SEC_mvt") - pl.col("off")).is_between(240, 2700)) & ((pl.col("TAXITIME_SEC_mvt") - pl.col("ms")).abs() > 12))
    small = raw.filter(pl.col("ms") < 3600)
    T = float((small.filter("tz")["TAXITIME_SEC_mvt"] - small.filter("tz")["off"]).median())
    print("B: tz class among unmatched LIRF ms<1h:", small["tz"].sum(), "/", small.height, "T", T)
    jb = j.join(raw.select("MVT_ID_mvt", "pre", "off"), on="MVT_ID_mvt", how="left")
    for a0 in (2, 5):
        out = []
        for m in j["ym"].unique().sort():
            st = small.filter(pl.col("ym") != m)
            base = st["tz"].mean()
            r = st.group_by("pre").agg(t=pl.col("tz").sum(), n=pl.len())
            te = jb.filter(pl.col("ym") == m).join(r, on="pre", how="left")
            q = ((pl.col("t").fill_null(0) + a0 * base) / (pl.col("n").fill_null(0) + a0))
            te = te.with_columns(pn=pl.when(pl.col("un") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") < 3600))
                                 .then(q * (pl.col("off") + T) + (1 - q) * pl.col("p")).otherwise(pl.col("p")))
            out.append(te)
        b = pl.concat(out)
        report(f"B tz-class by prefix, prior strength {a0}", b, pl.col("pn"))
    # C
    c = j.with_columns(dh=(pl.col("ms") - 1100) / 3600)
    report("C LIRF unmatched 7h<=dh<13.5h -> ms", c, pl.when((pl.col("ADEP_mvt") == "LIRF") & pl.col("un") & pl.col("dh").is_between(7, 13.5)).then(pl.col("ms")).otherwise(pl.col("p")))


if __name__ == "__main__":
    main()
