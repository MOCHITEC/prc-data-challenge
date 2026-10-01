"""Seconds-of-minute fingerprints and the 'true copy' excess (report §2-§3).

* seconds histograms of BLOCK / MVT per airport (minute grid + jitter vs. second resolution)
* BLOCK - clock offset histogram in [-30, 30] s
* per airport x clock: counts in the minute bins k = -2..2 (|d - 60k| <= 6), copy excess = n(k=0) - mean(n(k=+-1)),
  and the purity of the +-60 s label used by prc.clocks (share of +-60 s matches that are genuine copies).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from reporting_fingerprint_common import *  # noqa
import numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)

df = load_all().filter(pl.col("split") == "train")
dep = df.filter(pl.col("PHASE_mvt") == "DEP")
print("share of timestamps whose seconds are within [54, 6] (13/60 = 0.217 if uniform):")
grid = lambda c: ((sec(c) >= 54) | (sec(c) <= 6)).mean().round(3)  # noqa: E731
print(df.group_by("apt", "PHASE_mvt").agg(grid("BLOCK_TIME_UTC_mvt").alias("BLOCK_on_grid"), grid("MVT_TIME_UTC_mvt").alias("MVT_on_grid"),
      grid("AOBT_3_flt").alias("AOBT3_on_grid"), (sec("SCHED_TIME_UTC_mvt") == 0).mean().round(3).alias("SCHED_sec0")).sort("PHASE_mvt", "apt"))
rows = []
for name, ref in CAND.items():
    x = dep.with_columns((pl.col("BLOCK_TIME_UTC_mvt") - pl.col(ref)).dt.total_seconds().alias("dd"))
    for apt in sorted(dep["apt"].unique()):
        v = x.filter(pl.col("apt") == apt)["dd"].drop_nulls().to_numpy()
        n = {k: int(((v >= 60 * k - 6) & (v <= 60 * k + 6)).sum()) for k in (-2, -1, 0, 1, 2)}
        n60 = int((np.abs(v) <= 60).sum())
        ex = n[0] - (n[-1] + n[1]) / 2
        rows.append(dict(clock=name, apt=apt, n=len(v), k_m1=n[-1], k0=n[0], k_p1=n[1], n_pm60=n60, true_copies=round(ex),
                         true_copy_rate=round(ex / len(v), 4), purity_pm60=round(ex / max(n60, 1), 3), purity_pm6=round(ex / max(n[0], 1), 3)))
t = pl.DataFrame(rows)
print(t)
t.write_csv(CACHE / "copy_excess.csv")
print("totals by clock:", t.group_by("clock").agg(pl.col("true_copies").sum(), pl.col("n_pm60").sum()).with_columns((pl.col("true_copies") / pl.col("n_pm60")).round(3).alias("purity")))
