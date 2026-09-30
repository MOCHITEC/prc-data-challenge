"""Plan §2 criterion 3: how often each arm predicts more than 12 h on the ranking set's unmatched rows.

    .venv/bin/python docs/modelb_capacity_2026-09-30/ranking_tail.py outputs/mb_ctl.parquet outputs/mb_b255.parquet ...

Each file is a full-data refit's ranking prediction (`prc.pipeline --skip-cv`). The yardstick is the share of the
training set's unmatched labels above 12 h; an arm passes when its share is at most 1.5 times that."""

import sys
from pathlib import Path

import polars as pl

LABEL, PROXY, H12 = "TAXITIME_SEC_mvt", "mvt_minus_aobt3", 43_200
WORK = Path("data/work_flseq")


def main(files: list[str]) -> None:
    tr = pl.read_parquet(WORK / "feat_train.parquet", columns=[LABEL, PROXY]).filter(
        pl.col(PROXY).is_null() & pl.col(LABEL).is_finite()
    )
    share_y = float((tr[LABEL] > H12).mean())
    print(f"training unmatched rows {tr.height:,}: label > 12 h share {share_y:.4%} (limit {1.5 * share_y:.4%})")
    rk = pl.read_parquet(WORK / "feat_rank.parquet", columns=["MVT_ID_mvt", PROXY, "ADEP_mvt"])
    unm = rk.filter(pl.col(PROXY).is_null()).select("MVT_ID_mvt", "ADEP_mvt")
    for f in files:
        p = pl.read_parquet(f)
        col = next(c for c in p.columns if c != "MVT_ID_mvt" and p[c].dtype.is_numeric())
        d = unm.join(p.select("MVT_ID_mvt", pl.col(col).alias("pred")), on="MVT_ID_mvt", how="inner")
        share = float((d["pred"] > H12).mean())
        by = d.group_by("ADEP_mvt").agg((pl.col("pred") > H12).sum().alias("n_gt12h")).filter(pl.col("n_gt12h") > 0)
        print(
            f"{f}: {d.height:,} unmatched ranking rows, pred > 12 h {share:.4%} "
            f"({'PASS' if share <= 1.5 * share_y else 'FAIL'}); by airport {dict(by.iter_rows())}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
