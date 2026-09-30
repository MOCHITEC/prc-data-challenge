"""queue_runway: markdown tables of the ablation (all arms vs base) for docs/analysis/queue_runway.md.

Run: POLARS_MAX_THREADS=1 .venv/bin/python analysis/queue_runway_tables.py
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
ARMS = ["base_seed1", "seq", "nbr", "sched_arr", "qhat", "cfg", "all", "all_nocfgcat", "seq_sched", "seq_nbr_sched"]


def main() -> None:
    b = pl.read_parquet(os.path.join(C, "eval_base.parquet")).rename({"pred": "p0"})
    arms = [a for a in ARMS if os.path.exists(os.path.join(C, f"eval_{a}.parquet"))]
    j = b
    for a in arms:
        j = j.join(pl.read_parquet(os.path.join(C, f"eval_{a}.parquet")).select("MVT_ID_mvt", pl.col("pred").alias(a)),
                   on="MVT_ID_mvt")
    clean = pl.col(Y).is_between(30, 7200)
    subsets = {
        "genuine clean": (pl.col("clock_src") == "genuine") & clean,
        "phys clean": pl.col("clock_src").is_in(["genuine", "aobt3"]) & clean,
        "NM未マッチ clean": (pl.col("nm_matched") == 0) & clean,
    }
    for sname, pr in subsets.items():
        print(f"\n#### {sname}\n")
        print("| arm | jan Δ | jul Δ | jan+jul RMSE | jan+jul Δ | CI95 |")
        print("|---|---|---|---|---|---|")
        z = j.filter(pr)
        y = z[Y].to_numpy().astype(float)
        print(f"| base | — | — | {np.sqrt(np.mean((y - z['p0'].to_numpy()) ** 2)):.2f} | — | — |")
        for a in arms:
            ds = []
            for f in ("jan", "jul"):
                zz = z.filter(pl.col("fold") == f)
                yy = zz[Y].to_numpy().astype(float)
                ds.append(np.sqrt(np.mean((yy - zz[a].to_numpy()) ** 2)) - np.sqrt(np.mean((yy - zz["p0"].to_numpy()) ** 2)))
            r = paired_bootstrap_rmse_diff(y, z["p0"].to_numpy(), z[a].to_numpy(), n_boot=300)
            print(f"| {a} | {ds[0]:+.2f} | {ds[1]:+.2f} | {np.sqrt(np.mean((y - z[a].to_numpy()) ** 2)):.2f} | "
                  f"{r['diff_rmse']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] |")
    # airport table: genuine clean, delta vs base per arm
    z = j.filter(subsets["genuine clean"])
    print("\n#### 空港別 Δ（genuine clean、jan+jul）\n")
    print("| 空港 | n | base RMSE | " + " | ".join(arms) + " |")
    print("|---|---|---|" + "---|" * len(arms))
    for ap in sorted(z["ADEP_mvt"].unique().to_list()):
        zz = z.filter(pl.col("ADEP_mvt") == ap)
        yy = zz[Y].to_numpy().astype(float)
        r0 = np.sqrt(np.mean((yy - zz["p0"].to_numpy()) ** 2))
        cells = [f"{np.sqrt(np.mean((yy - zz[a].to_numpy()) ** 2)) - r0:+.2f}" for a in arms]
        print(f"| {ap} | {len(yy):,} | {r0:.2f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
