"""Build the LB probe for mix255 (results.md §5): v20 with only its unmatched rows replaced. Writes, never uploads.

    .venv/bin/python docs/modelb_capacity_2026-09-30/build_probe.py V20_PARQUET OUT_PARQUET

v20's unmatched predictions equal the mb_ctl refit's exactly (checked on all 5,290 rows), so on those rows the probe is
0.5 * (v20 + mb_b255 refit) - the mix255 of the CV - and every matched row is v20's, byte for byte."""

import sys
from pathlib import Path

import polars as pl

from prc.predict import write_submission

PROXY = "mvt_minus_aobt3"


def main(v20_path: str, out_path: str) -> None:
    v20 = pl.read_parquet(v20_path)
    ctl = pl.read_parquet("outputs/mb_ctl.parquet").rename({"TAXITIME_SEC_mvt": "ctl"})
    b255 = pl.read_parquet("outputs/mb_b255.parquet").rename({"TAXITIME_SEC_mvt": "b255"})
    unm = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=["MVT_ID_mvt", PROXY]).filter(
        pl.col(PROXY).is_null()
    )
    d = v20.join(ctl, on="MVT_ID_mvt", how="left").join(b255, on="MVT_ID_mvt", how="left")
    d = d.with_columns(pl.col("MVT_ID_mvt").is_in(unm["MVT_ID_mvt"].implode()).alias("unm"))
    u = d.filter("unm")
    if u.height != unm.height or (u["ctl"] != u["TAXITIME_SEC_mvt"]).any():
        raise AssertionError("v20's unmatched rows are not the mb_ctl refit's; the probe would not be mix255")
    pred = d.select(
        "MVT_ID_mvt",
        pl.when("unm")
        .then((pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) + pl.col("b255").cast(pl.Float64)) / 2)
        .otherwise(pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        .alias("pred"),
    )
    template = next(iter(sorted(Path("data/raw").rglob("submitting.parquet"))))
    ref = pl.read_parquet("data/work_flseq/feat_train.parquet", columns=["TAXITIME_SEC_mvt"])["TAXITIME_SEC_mvt"]
    write_submission(pred, template, Path(out_path), reference_median=float(ref.median()))
    out = pl.read_parquet(out_path).join(v20.rename({"TAXITIME_SEC_mvt": "v20"}), on="MVT_ID_mvt")
    changed = out.filter(pl.col("TAXITIME_SEC_mvt") != pl.col("v20"))
    print(
        f"wrote {out_path}: {out.height:,} rows, {changed.height:,} differ from v20 (all unmatched: "
        f"{changed['MVT_ID_mvt'].is_in(unm['MVT_ID_mvt'].implode()).all()})"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
