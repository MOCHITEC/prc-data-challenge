"""queue_runway: how much of the base model's out-of-fold residual do the new features explain?

Uses eval_base.parquet (jan+jul OOF of the base LightGBM, queue_runway_eval.py). For each new feature, the
share of residual variance explained by its 20-quantile bin means (eta^2), within airport, on the validation
rows. Positive eta^2 on an OOF residual is an upper-bound-ish signal that the feature adds information the base
model did not have (bins are fitted on the same rows, so ~20/n of eta^2 is noise: < 0.0002 here).

Run: POLARS_MAX_THREADS=1 .venv/bin/python analysis/queue_runway_resid.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from queue_runway_eval import GROUPS, NEW_CATS  # noqa: E402
from queue_runway_explore import eta2, md  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "data", "interim", "queue_runway")
Y = "TAXITIME_SEC_mvt"


def main() -> None:
    o = pl.read_parquet(os.path.join(C, "eval_base.parquet"))
    o = o.with_columns((pl.col(Y) - pl.col("pred")).alias("res"))
    new = pl.read_parquet(os.path.join(C, "new.parquet")).join(
        pl.read_parquet(os.path.join(C, "new2.parquet")), on="MVT_ID_mvt")
    o = o.join(new, on="MVT_ID_mvt", how="left")
    subsets = {
        "genuine": o.filter((pl.col("clock_src") == "genuine") & pl.col(Y).is_between(30, 7200)),
        "nm_unmatched": o.filter((pl.col("nm_matched") == 0) & (pl.col("clock_src") == "genuine")
                                 & pl.col(Y).is_between(30, 7200)),
    }
    apts = sorted(o["ADEP_mvt"].unique().to_list())
    rows = []
    for g, fs in GROUPS.items():
        for f in fs:
            row = {"group": g, "feature": f}
            for sname, z in subsets.items():
                if f in NEW_CATS:
                    x = z[f].cast(pl.Categorical).to_physical().cast(pl.Float64).to_numpy()
                else:
                    x = z[f].cast(pl.Float64).to_numpy()
                r = z["res"].to_numpy().astype(float)
                e, w = [], []
                for a in apts:
                    m = (z["ADEP_mvt"] == a).to_numpy()
                    if m.sum() < 200:
                        continue
                    e.append(eta2(x[m], r[m] - r[m].mean()))
                    w.append(m.sum())
                row[f"eta2_{sname}"] = float(np.average(e, weights=w))
            rows.append(row)
    t = pl.DataFrame(rows).sort("eta2_genuine", descending=True)
    print(md(t))
    # per airport for the top features
    top = t["feature"].head(8).to_list()
    z = subsets["genuine"]
    pa = []
    for a in apts:
        zz = z.filter(pl.col("ADEP_mvt") == a)
        r = zz["res"].to_numpy().astype(float)
        row = {"apt": a, "n": zz.height, "rmse_base": float(np.sqrt(np.mean(r**2)))}
        for f in top:
            if f in NEW_CATS:
                continue
            row[f] = eta2(zz[f].cast(pl.Float64).to_numpy(), r - r.mean())
        pa.append(row)
    print("\n## per airport eta2 on base OOF residual (genuine clean)")
    print(md(pl.DataFrame(pa)))
    # noise floor: eta2 of a random feature
    rnd = np.random.default_rng(0).random(z.height)
    print("\nnoise floor eta2(random) =", eta2(rnd, z["res"].to_numpy().astype(float)))


if __name__ == "__main__":
    main()
