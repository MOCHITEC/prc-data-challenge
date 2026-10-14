"""Plan §4 criterion 3, second half: base -> ox_stress on the rows whose day the stress arm blanked.

    .venv/bin/python docs/opdi_events_2026-10-05/evaluate_stress.py BASE_ID STRESS_ID

The stress arm empties the five OPDI stand-exit columns on the target side for the CRC32-picked third of the days
(`day_null_mask`, share 0.3333, no salt - the same days as `opdi_exit_target_null_day_share`). Criterion 3b passes
when the K=0 95% CI on those rows does not exclude zero on the worse side (same procedure as
docs/opdi_2026-09-30/evaluate_stress.py). The lines below the verdict are reporting only: the matched lane alone
(post-hoc, the lane production would change) and the rows that lost their OPDI exit."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

from prc.cv import paired_bootstrap_rmse_diff
from prc.train import day_null_mask

LABEL, PROXY, WORK, EXP = "TAXITIME_SEC_mvt", "mvt_minus_aobt3", Path("data/work_flseq"), Path("experiments")


def main(base: str, stress: str) -> None:
    cols = ["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt", "ADEP_mvt"]
    f = pl.read_parquet(WORK / "feat_train.parquet", columns=cols)
    d = f.join(pl.read_parquet(EXP / f"oof_{base}.parquet").rename({"oof": "a"}), on="MVT_ID_mvt")
    d = d.join(pl.read_parquet(EXP / f"oof_{stress}.parquet").rename({"oof": "b"}), on="MVT_ID_mvt")
    d = d.filter(pl.col("a").is_not_null() & pl.col("b").is_not_null() & pl.col(LABEL).is_finite())
    ex = pl.read_parquet("data/work/opdi_exitpark.parquet").select("MVT_ID_mvt", pl.lit(True).alias("has"))
    d = d.join(ex, on="MVT_ID_mvt", how="left").with_columns(pl.col("has").fill_null(False))
    blank = day_null_mask(d, 0.3333)
    y, a, b = d[LABEL].to_numpy().astype(float), d["a"].to_numpy(), d["b"].to_numpy()
    mt, has, apt = d[PROXY].is_not_null().to_numpy(), d["has"].to_numpy(), d["ADEP_mvt"].to_numpy()

    def line(name: str, sel: np.ndarray) -> dict:
        r = paired_bootstrap_rmse_diff(y[sel], a[sel], b[sel], n_boot=500, seed=0)
        lo, hi = r["ci95"]
        print(
            f"{name:>42}: n={int(sel.sum()):>7,}  base {np.sqrt(np.mean((y[sel] - a[sel]) ** 2)):8.2f}  "
            f"stress {np.sqrt(np.mean((y[sel] - b[sel]) ** 2)):8.2f}  diff {r['diff_rmse']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        )
        return r

    r = line("blanked days", blank)
    line("other days", ~blank)
    print("criterion 3b:", "FAIL (worse on blanked days)" if r["ci95"][0] > 0 else "PASS")
    print("\nreporting only (post-hoc):")
    line("blanked days, matched", blank & mt)
    line("blanked days, unmatched", blank & ~mt)
    line("blanked days, matched, lost their exit", blank & mt & has)
    line("blanked days, matched, never had one", blank & mt & ~has)
    line("blanked days, LSZH matched", blank & mt & (apt == "LSZH"))
    line("blanked days, LSZH matched, lost their exit", blank & mt & has & (apt == "LSZH"))
    line("other days, matched", ~blank & mt)
    line("other days, matched, with exit", ~blank & mt & has)


if __name__ == "__main__":
    main(*sys.argv[1:3])
