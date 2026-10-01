"""Plan §3 criterion 3, second half: base -> opdi_stress on the rows whose day the stress arm blanked.

    .venv/bin/python docs/opdi_2026-09-30/evaluate_stress.py BASE_ID STRESS_ID

The stress arm empties the two OPDI columns on the target side for the CRC32-picked third of the days
(`day_null_mask`, share 0.3333). Passes when the K=0 95% CI on those rows does not exclude zero on the worse side."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

from prc.cv import paired_bootstrap_rmse_diff
from prc.train import day_null_mask

LABEL, WORK, EXP = "TAXITIME_SEC_mvt", Path("data/work_flseq"), Path("experiments")


def main(base: str, stress: str) -> None:
    f = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, "MVT_TIME_UTC_mvt", "ADEP_mvt"])
    d = f.join(pl.read_parquet(EXP / f"oof_{base}.parquet").rename({"oof": "a"}), on="MVT_ID_mvt")
    d = d.join(pl.read_parquet(EXP / f"oof_{stress}.parquet").rename({"oof": "b"}), on="MVT_ID_mvt")
    d = d.filter(pl.col("a").is_not_null() & pl.col("b").is_not_null() & pl.col(LABEL).is_finite())
    blank = day_null_mask(d, 0.3333)
    y, a, b = d[LABEL].to_numpy().astype(float), d["a"].to_numpy(), d["b"].to_numpy()
    for name, sel in (("blanked days", blank), ("other days", ~blank)):
        r = paired_bootstrap_rmse_diff(y[sel], a[sel], b[sel], n_boot=500, seed=0)
        lo, hi = r["ci95"]
        print(
            f"{name:>13}: n={int(sel.sum()):>7,}  base {np.sqrt(np.mean((y[sel] - a[sel]) ** 2)):8.2f}  "
            f"stress {np.sqrt(np.mean((y[sel] - b[sel]) ** 2)):8.2f}  diff {r['diff_rmse']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        )
    r = paired_bootstrap_rmse_diff(y[blank], a[blank], b[blank], n_boot=500, seed=0)
    print("criterion 3b:", "FAIL (worse on blanked days)" if r["ci95"][0] > 0 else "PASS")


if __name__ == "__main__":
    main(*sys.argv[1:3])
