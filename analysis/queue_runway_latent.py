"""queue_runway: the latent 'congestion state' left in the base model's residual, and how much of it is
recoverable from ranking-observable quantities.

1. Oracle ceiling: add to each prediction the leave-one-out mean OOF residual of the *other* departures in the
   same airport x runway x clock-hour (needs their labels -> not available in ranking; ceiling only).
2. Observable version: predict that LOO hour residual from hour-level (LOO) means of observable features with a
   small ridge per airport, fitted on jan and applied to jul and vice versa (cross-fold), then add it.

Run: POLARS_MAX_THREADS=1 .venv/bin/python analysis/queue_runway_latent.py [arm]   (default arm=base)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queue_runway_explore import md  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "data", "interim", "queue_runway")
Y = "TAXITIME_SEC_mvt"
OBS = ["mvt_minus_aobt3", "nbr_rg_pm20", "arr_ti_resid_pm30", "sched_backlog_60m", "hw_mean_next5", "burst_len",
       "q_in_system_at_tko", "dep_tko_prev_60m", "arr_land_prev_60m", "mvt_minus_sched"]


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else "base"
    o = pl.read_parquet(os.path.join(C, f"eval_{arm}.parquet"))
    b = pl.read_parquet(os.path.join(C, "base.parquet"), columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt",
                                                                  "mvt_minus_aobt3", "q_in_system_at_tko",
                                                                  "dep_tko_prev_60m", "arr_land_prev_60m",
                                                                  "mvt_minus_sched"])
    n = pl.read_parquet(os.path.join(C, "new.parquet"), columns=["MVT_ID_mvt", "sched_backlog_60m", "hw_mean_next5",
                                                                 "burst_len"])
    n2 = pl.read_parquet(os.path.join(C, "new2.parquet"), columns=["MVT_ID_mvt", "nbr_rg_pm20", "arr_ti_resid_pm30"])
    o = o.join(b, on="MVT_ID_mvt").join(n, on="MVT_ID_mvt").join(n2, on="MVT_ID_mvt")
    o = o.filter(pl.col(Y).is_between(30, 7200))
    o = o.with_columns((pl.col(Y) - pl.col("pred")).alias("e"),
                       pl.col("MVT_TIME_UTC_mvt").dt.truncate("1h").alias("hr"),
                       pl.col("clock_src").is_in(["genuine", "aobt3"]).alias("phys"))
    # the latent state is measured on phys rows only (clock-copy rows have their own error process)
    grp = ["ADEP_mvt", "RUNWAY_mvt", "hr"]
    o = o.with_columns(pl.when(pl.col("phys")).then(pl.col("e")).alias("e_phys"))
    o = o.with_columns(pl.col("e_phys").sum().over(grp).alias("s"), pl.col("e_phys").count().over(grp).alias("c"))
    den = pl.col("c") - pl.col("e_phys").is_not_null().cast(pl.Int64)
    o = o.with_columns(pl.when(den > 0).then((pl.col("s") - pl.col("e_phys").fill_null(0)) / den).alias("loo"))
    obs_cl = [f for f in OBS]
    o = o.with_columns([pl.col(f).cast(pl.Float64).clip(-1e4, 1e4).alias(f) for f in obs_cl])
    o = o.with_columns([pl.col(f).mean().over(grp).alias(f"h_{f}") for f in obs_cl])
    rows = []
    ev = o.filter(pl.col("clock_src") == "genuine")
    y = ev[Y].to_numpy().astype(float)
    p0 = ev["pred"].to_numpy()
    loo = ev["loo"].to_numpy()
    for shrink in (0.5, 1.0):
        p = p0 + shrink * np.nan_to_num(loo)
        rows.append({"variant": f"oracle LOO runway-hour residual x{shrink}", "rmse": float(np.sqrt(np.mean((y - p) ** 2)))})
    rows.insert(0, {"variant": f"{arm} (no correction)", "rmse": float(np.sqrt(np.mean((y - p0) ** 2)))})
    # observable version: cross-fold ridge per airport on hour-level means
    hcols = [f"h_{f}" for f in obs_cl]
    corr = np.zeros(o.height)
    for fold_fit, fold_app in (("jan", "jul"), ("jul", "jan")):
        for a in o["ADEP_mvt"].unique().to_list():
            fit = o.filter((pl.col("fold") == fold_fit) & (pl.col("ADEP_mvt") == a) & pl.col("loo").is_not_null())
            X = np.nan_to_num(fit.select(hcols).to_numpy())
            mu, sd = X.mean(0), X.std(0) + 1e-9
            Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
            lam = 10.0 * len(X) ** 0.5
            beta = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ fit["loo"].to_numpy())
            m = ((o["fold"] == fold_app) & (o["ADEP_mvt"] == a)).to_numpy()
            Xa = np.nan_to_num(o.filter(pl.Series(m)).select(hcols).to_numpy())
            corr[m] = np.column_stack([np.ones(len(Xa)), (Xa - mu) / sd]) @ beta
    o = o.with_columns(pl.Series("corr_obs", corr))
    ev = o.filter(pl.col("clock_src") == "genuine")
    p = ev["pred"].to_numpy() + ev["corr_obs"].to_numpy()
    rows.append({"variant": "observable hour-level ridge correction (cross-fold)", "rmse": float(np.sqrt(np.mean((y - p) ** 2)))})
    print(f"## genuine clean val rows (jan+jul), n={len(y):,}")
    print(md(pl.DataFrame(rows)))
    # how much of the LOO residual does the observable model explain?
    z = o.filter(pl.col("loo").is_not_null() & pl.col("phys"))
    r2 = 1 - ((z["loo"] - z["corr_obs"]) ** 2).mean() / ((z["loo"] - z["loo"].mean()) ** 2).mean()
    print(f"\nR2 of observable hour-level model on LOO runway-hour residual (cross-fold): {r2:.4f}")
    print(f"corr(e, loo) on phys rows: {z.select(pl.corr('e', 'loo')).item():.4f}")


if __name__ == "__main__":
    main()
