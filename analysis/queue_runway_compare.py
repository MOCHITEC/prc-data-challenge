"""queue_runway: paired comparison of two eval arms by subset and airport (jan+jul OOF).

Run: POLARS_MAX_THREADS=1 .venv/bin/python analysis/queue_runway_compare.py base all
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prc.cv import paired_bootstrap_rmse_diff  # noqa: E402

C = os.path.join(ROOT, "data", "interim", "queue_runway")
Y = "TAXITIME_SEC_mvt"


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    A = pl.read_parquet(os.path.join(C, f"eval_{a}.parquet")).rename({"pred": "pa"})
    B = pl.read_parquet(os.path.join(C, f"eval_{b}.parquet")).select("MVT_ID_mvt", pl.col("pred").alias("pb"))
    j = A.join(B, on="MVT_ID_mvt")
    clean = pl.col(Y).is_between(30, 7200)
    subs = {"genuine clean": (pl.col("clock_src") == "genuine") & clean,
            "phys clean": pl.col("clock_src").is_in(["genuine", "aobt3"]) & clean,
            "NM-unmatched clean": (pl.col("nm_matched") == 0) & clean}
    print(f"| subset | fold/apt | n | rmse {a} | rmse {b} | diff | CI95 |")
    print("|---|---|---|---|---|---|---|")
    for s, pr in subs.items():
        groups = [("jan", pl.col("fold") == "jan"), ("jul", pl.col("fold") == "jul"), ("jan+jul", pl.lit(True))]
        if s == "genuine clean":
            groups += [(ap, pl.col("ADEP_mvt") == ap) for ap in sorted(j["ADEP_mvt"].unique().to_list())]
        for g, gp in groups:
            z = j.filter(pr & gp)
            y = z[Y].to_numpy().astype(float)
            r = paired_bootstrap_rmse_diff(y, z["pa"].to_numpy(), z["pb"].to_numpy(), n_boot=300)
            ra = float(np.sqrt(np.mean((y - z["pa"].to_numpy()) ** 2)))
            rb = float(np.sqrt(np.mean((y - z["pb"].to_numpy()) ** 2)))
            print(f"| {s} | {g} | {len(y):,} | {ra:.2f} | {rb:.2f} | {r['diff_rmse']:+.2f} | "
                  f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] |")


if __name__ == "__main__":
    main()
