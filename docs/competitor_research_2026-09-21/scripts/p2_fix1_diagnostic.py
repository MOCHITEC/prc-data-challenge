"""P2 Step 1 - quantify the encoding self-inclusion defect on MODEL A.

The only quantification the repo owns is model B (22,470 training rows, 1.2 % of the data):
`docs/investigation_2026-09-16/results/holdout-encoding-self-inclusion.md`. Its §5 says model A was
never run. Model A is 98.8 % of the rows, so this script fills that gap and nothing else.

Read-only. No model is fitted, nothing under outputs/ is touched, no production file is imported
other than `prc.train`'s own builders - `fit_encodings` / `attach_encodings` / `_inner_holdout_mask`
are reused rather than reimplemented, so a diagnostic cannot drift from the code it diagnoses.

Sections match `docs/competitor_research_2026-09-21/results/P2-fix1-earlystop-encoder.md`:

  §1  self-inclusion dilution      (plan §4 Step 1 item 1)
  §2  column displacement, LOMO    (plan §4 Step 1 item 2)
  §3  the order-statistic worst case (plan §4 Step 1 item 3, the quantity §1.5 says min_count misses)
  §4  the C1 baseline: train vs 2026 ranking distributions (plan §4 Step 1 item 4)

Populations, stated once because the whole document depends on them:
  * `fold`  - the production forward fold: train months 202501-202510, valid 202511-202512.
  * `A_tr`  - model A's training rows = labelled fold-training rows with `mvt_minus_aobt3` non-null.
  * `A_es`  - the inner early-stopping holdout inside `A_tr`, carved by `_inner_holdout_mask` with
              `inner_holdout_months = 1`. This is the population Fix 1 repairs.
  * `A_fit` - `A_tr` minus `A_es`, the rows the stage-1 booster is fitted on. Fix 1 does not touch it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from prc.train import (  # noqa: E402
    ENCODING_COLS,
    LABEL,
    PROXY,
    TrainConfig,
    _inner_holdout_mask,
    attach_encodings,
    fit_encodings,
)

TRAIN = ROOT / "data" / "work" / "feat_train.parquet"
RANK = ROOT / "data" / "work" / "feat_rank.parquet"

# The keys of the three tables that actually reach the model. `op` is fitted and hidden
# (`use_offset_encoding` defaults False), which is plan §4 Step 0's scope correction.
TABLE_KEYS = {
    "ref": ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "arh": ["ADEP_mvt", "RUNWAY_mvt", "hour_local"],
    "apt": ["ADEP_mvt"],
}
# Columns the script needs. Everything else in the 286 MB feature frame is left on disk.
NEEDED = sorted(
    {
        LABEL,
        PROXY,
        "MVT_ID_mvt",
        "MVT_TIME_UTC_mvt",
        "ADEP_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "hour_local",
        "AIRCRAFT_OPERATOR_flt",
    }
)


def _say(x) -> None:
    print(x, flush=True)


def _ym(df: pl.DataFrame) -> pl.Series:
    return (df["MVT_TIME_UTC_mvt"].dt.year() * 100 + df["MVT_TIME_UTC_mvt"].dt.month()).alias("ym")


def _describe(s: pl.Series) -> dict:
    """median / IQR / p01 / p99 / null rate of one column, the four §4 asks for."""
    n = s.len()
    nulls = s.null_count()
    v = s.drop_nulls().cast(pl.Float64)
    if v.len() == 0:
        return {"n": n, "null_rate": nulls / max(1, n), "median": None, "iqr": None, "p01": None, "p99": None}
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    return {
        "n": n,
        "null_rate": nulls / max(1, n),
        "median": float(v.median()),
        "iqr": q3 - q1,
        "p01": float(v.quantile(0.01)),
        "p99": float(v.quantile(0.99)),
    }


# ---------------------------------------------------------------- §1


def s1_dilution(a_tr: pl.DataFrame, ym: pl.Series, es_mask: np.ndarray) -> dict:
    """Group size, own-month share and own-row share, per table, on A_tr and on A_es.

    Mirrors §3.1 of the model-B study so the two are directly comparable. The group is defined on the
    labelled rows the builder actually keeps: `reference_table` drops labels outside (0, 7200), the
    other two keep every labelled row, so `ref` is counted on its own filtered population."""
    out: dict = {}
    base = a_tr.with_columns(ym)
    for name, keys in TABLE_KEYS.items():
        pop = base
        if name == "ref":
            pop = base.filter((pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200))
        sizes = pop.group_by(keys).agg(pl.len().alias("_n"))
        own = pop.group_by([*keys, "ym"]).agg(pl.len().alias("_n_own"))
        j = (
            base.select([*keys, "ym"])
            .with_columns(pl.Series("_es", es_mask))
            .join(sizes, on=keys, how="left")
            .join(own, on=[*keys, "ym"], how="left")
        )
        rec = {}
        for label, frame in (("A_tr", j), ("A_es", j.filter(pl.col("_es")))):
            have = frame.filter(pl.col("_n").is_not_null())
            share = (have["_n_own"] / have["_n"]).cast(pl.Float64)
            rec[label] = {
                "rows": frame.height,
                "rows_with_a_group": have.height,
                "median_group_n": float(have["_n"].median()) if have.height else None,
                "median_own_month_share": float(share.median()) if have.height else None,
                "mean_own_month_share": float(share.mean()) if have.height else None,
                "max_own_month_share": float(share.max()) if have.height else None,
                "share_gt_20pct": float((share > 0.20).mean()) if have.height else None,
                "median_own_row_share": float((1.0 / have["_n"]).median()) if have.height else None,
                "max_own_row_share": float((1.0 / have["_n"]).max()) if have.height else None,
            }
        out[name] = rec
    return out


# ---------------------------------------------------------------- §2


def _attach(frame: pl.DataFrame, enc: dict) -> pl.DataFrame:
    return attach_encodings(frame, enc).select(ENCODING_COLS)


def _displacement(a: pl.DataFrame, b: pl.DataFrame) -> dict:
    """|a - b| per encoding column over the same rows, in the model-B study's shape."""
    rec = {}
    for c in ENCODING_COLS:
        x, y = a[c].cast(pl.Float64).to_numpy(), b[c].cast(pl.Float64).to_numpy()
        both = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[both] - y[both])
        moved = d > 0
        rec[c] = {
            "rows_changed": int(moved.sum() + int(((np.isfinite(x) != np.isfinite(y))).sum())),
            "median_abs_change": float(np.median(d[moved])) if moved.any() else 0.0,
            "p90_abs_change": float(np.quantile(d[moved], 0.90)) if moved.any() else 0.0,
            "nulls_before": int((~np.isfinite(x)).sum()),
            "nulls_after": int((~np.isfinite(y)).sum()),
        }
    return rec


def s2_displacement(train_f: pl.DataFrame, a_tr: pl.DataFrame, ym: pl.Series, es_mask: np.ndarray, cfg) -> dict:
    """Two reads, because Fix 1 and Fix 2 touch different populations.

    (a) FIX 1's scope: the A_es rows under the production map vs under a map fitted on the fold's
        training months excluding the holdout month. Nothing else in the pipeline changes.
    (b) FIX 2's scope: every A_tr row under the production map vs under a map fitted without that
        row's OWN month (leave-one-month-out, one encoder per training month).
    The encoder is always fitted on the full fold-training frame (both lanes), exactly as
    `predict_fold` does - model A's rows are not the only rows in the table."""
    full = fit_encodings(train_f, cfg)
    tr_ym = _ym(train_f)
    prod = _attach(a_tr, full)

    hold_months = sorted(set(ym.filter(pl.Series("m", es_mask)).unique().to_list()))
    enc_excl = fit_encodings(train_f.filter(~tr_ym.is_in(hold_months)), cfg)
    es_rows = a_tr.filter(pl.Series("m", es_mask))
    fix1 = _displacement(prod.filter(pl.Series("m", es_mask)), _attach(es_rows, enc_excl))

    months = sorted(set(ym.unique().to_list()))
    parts_prod, parts_lomo = [], []
    for m in months:
        sel = (ym == m).to_numpy()
        enc_m = fit_encodings(train_f.filter(tr_ym != m), cfg)
        parts_prod.append(prod.filter(pl.Series("m", sel)))
        parts_lomo.append(_attach(a_tr.filter(pl.Series("m", sel)), enc_m))
        _say(f"    lomo month {m}: {int(sel.sum()):,} rows")
    fix2 = _displacement(pl.concat(parts_prod), pl.concat(parts_lomo))

    sizes = {
        "table_groups_full": {k: full[k].height for k in ("ref", "arh", "apt")},
        "table_groups_excl_holdout_month": {k: enc_excl[k].height for k in ("ref", "arh", "apt")},
        "holdout_months": hold_months,
        "es_rows": es_rows.height,
        "train_rows": a_tr.height,
    }
    return {"sizes": sizes, "fix1_es_rows": fix1, "fix2_all_training_rows_lomo": fix2}


# ---------------------------------------------------------------- §3


def s3_order_statistic(a_tr: pl.DataFrame, es_mask: np.ndarray, min_count: int) -> dict:
    """How often is a row's own label EXACTLY the statistic its own encoding column carries?

    Correction to the plan, measured not argued: plan §1.5 says "with polars' default linear
    interpolation" and polars 1.44.2's default is `nearest`. `pl.col(y).quantile(0.10)` returns the
    order statistic at rank round(0.10 * (n - 1)) verbatim - not a blend of two. So `ref_p10` on a
    qualifying group IS one member's label, at weight 1.0, and `ref_p20` likewise; `median()` for even n
    is the mean of the two central labels, weight 1/2 each. `train.py:88`'s "leak of 1/n" is the bound
    for a mean and is false for all three.

    Counted the direct way - the row's own label equal to its own group statistic - rather than by
    reconstructing an index rule, so ties are handled by the same arithmetic the builder uses."""
    keys = TABLE_KEYS["ref"]
    pop = a_tr.with_columns(pl.Series("_es", es_mask)).filter(
        pl.col(LABEL).is_not_null() & (pl.col(LABEL) > 0) & (pl.col(LABEL) < 7200)
    )
    stats = (
        pop.group_by(keys)
        .agg(
            [
                pl.col(LABEL).quantile(0.10).alias("ref_p10"),
                pl.col(LABEL).quantile(0.20).alias("ref_p20"),
                pl.col(LABEL).median().alias("ref_med"),
                pl.len().alias("_n"),
            ]
        )
        .filter(pl.col("_n") >= min_count)
    )
    j = pop.join(stats, on=keys, how="inner")
    out = {"n_qualifying_groups": stats.height, "median_group_n": float(stats["_n"].median())}
    for label, frame in (("A_tr", j), ("A_es", j.filter(pl.col("_es")))):
        n = frame.height
        rec = {"rows_in_a_qualifying_ref_group": n, "groups_at_exactly_min_count": int(
            stats.filter(pl.col("_n") == min_count).height
        )}
        for c in ("ref_p10", "ref_p20", "ref_med"):
            hit = int((frame[LABEL] == frame[c]).sum()) if n else 0
            rec[f"rows_whose_label_equals_{c}"] = hit
            rec[f"share_{c}"] = hit / max(1, n)
        out[label] = rec
    return out


# ---------------------------------------------------------------- §4


def s4_c1_baseline(a_tr: pl.DataFrame, rank: pl.DataFrame, train_f: pl.DataFrame, cfg) -> dict:
    """Per encoding column, the train-side and serve-side distributions AS THEY ARE TODAY.

    Serve side is built the way `predict.predict_ranking` builds it: encoder fitted on all 12 labelled
    months, attached to the 2026 ranking frame. Train side is the fold encoder on A_tr. The two are
    therefore not identical maps today either - that is the baseline any later change is measured
    against, not a defect Fix 1 introduces."""
    tr_enc = _attach(a_tr, fit_encodings(train_f, cfg))
    full12 = pl.read_parquet(TRAIN, columns=NEEDED).filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    rk_enc = _attach(rank, fit_encodings(full12, cfg))
    return {
        c: {"train": _describe(tr_enc[c]), "rank2026": _describe(rk_enc[c])}
        for c in ENCODING_COLS
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/competitor_research_2026-09-21/results/p2-fix1-step1.json")
    ap.add_argument("--sections", default="1234")
    a = ap.parse_args(argv)

    cfg = TrainConfig(reference_min_count=30, inner_holdout_months=1)
    feats = pl.read_parquet(TRAIN, columns=NEEDED)
    ym_all = _ym(feats)
    valid = sorted(x for x in set(ym_all.to_list()) if x % 100 >= 11)
    train_months = sorted(x for x in set(ym_all.to_list()) if x not in valid)
    _say(f"forward fold: train {train_months} -> valid {valid}")

    train_f = feats.filter(ym_all.is_in(train_months)).filter(
        pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite()
    )
    a_tr = train_f.filter(pl.col(PROXY).is_not_null())
    ym = _ym(a_tr)
    es_mask = _inner_holdout_mask(a_tr, cfg.inner_holdout_months)
    _say(
        f"train_f {train_f.height:,} rows; A_tr {a_tr.height:,}; A_es {int(es_mask.sum()):,} "
        f"({sorted(set(ym.filter(pl.Series('m', es_mask)).unique().to_list()))})"
    )

    res: dict = {
        "fold": {"train_months": train_months, "valid_months": valid},
        "rows": {
            "train_f": train_f.height,
            "A_tr": a_tr.height,
            "A_es": int(es_mask.sum()),
            "A_fit": int((~es_mask).sum()),
        },
    }
    if "1" in a.sections:
        _say("§1 dilution")
        res["s1_dilution"] = s1_dilution(a_tr, ym, es_mask)
    if "2" in a.sections:
        _say("§2 displacement")
        res["s2_displacement"] = s2_displacement(train_f, a_tr, ym, es_mask, cfg)
    if "3" in a.sections:
        _say("§3 order statistic")
        res["s3_order_statistic"] = s3_order_statistic(a_tr, es_mask, cfg.reference_min_count)
    if "4" in a.sections:
        _say("§4 C1 baseline")
        rank = pl.read_parquet(RANK, columns=[c for c in NEEDED if c != LABEL])
        res["s4_c1_baseline"] = s4_c1_baseline(a_tr, rank, train_f, cfg)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=str))
    _say(json.dumps(res, indent=2, default=str))
    _say(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
