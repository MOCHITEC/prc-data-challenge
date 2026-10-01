"""Do the new fold-time columns look the same on the 2026 ranking rows as on the CV validation rows?

Read-only. Builds each family the way production would (training side = all of 2025, target = ranking.parquet)
and the way the forward CV fold does (training = its ten months, target = Nov-Dec 2025), then prints each
column's null rate and quartiles on the two target sides. A column that is dense in CV and empty in 2026 - or
whose scale moves - sends the model down branches it never validated (docs/README; the OPDI incident in
docs/03 §8 is the precedent).

    .venv/bin/python docs/article_fe_2026-09-24/parity_check.py
"""

from __future__ import annotations

import polars as pl

from prc.cv import month_block_folds, year_month_key
from prc.train import (
    LABEL,
    TrainConfig,
    aggdiff_features,
    attach_encodings,
    cyclic_features,
    fit_encodings,
    frequency_features,
)

COLS = [
    "freq_stand",
    "freq_stand_rwy",
    "freq_operator",
    "freq_actype",
    "freq_ades",
    "freq_rwy_hour",
    "agg_proxy_minus_ref_med",
    "agg_proxy_minus_ref_p10",
    "agg_proxy_minus_arh_med",
    "agg_proxy_cell_dev",
    "agg_sched_minus_ref_med",
    "agg_ref_iqr",
    "cyc_sin",
    "cyc_cos",
]


def build(tr: pl.DataFrame, tg: pl.DataFrame, cfg: TrainConfig) -> pl.DataFrame:
    tr = tr.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    enc = fit_encodings(tr, cfg)
    tr, tg = attach_encodings(tr, enc), attach_encodings(tg, enc)
    tr, tg = frequency_features(tr, tg)
    tr, tg = aggdiff_features(tr, tg, cfg.reference_min_count)
    return cyclic_features(tg)


def describe(df: pl.DataFrame, c: str) -> str:
    s = df[c].cast(pl.Float64)
    nn = s.drop_nulls().drop_nans()
    null = 1 - nn.len() / max(1, s.len())
    if nn.len() == 0:
        return f"{null:6.1%} null |  —"
    q = [nn.quantile(p) for p in (0.1, 0.5, 0.9)]
    return f"{null:6.1%} null | p10 {q[0]:10.4g}  p50 {q[1]:10.4g}  p90 {q[2]:10.4g}"


def main() -> int:
    cfg = TrainConfig(reference_min_count=30)
    feats = pl.read_parquet("data/work/feat_train.parquet")
    rank = pl.read_parquet("data/work/feat_rank.parquet")
    months = year_month_key(feats)
    fold = next(f for f in month_block_folds(months.unique().to_list()) if f.name == "forward")
    cv_tg = build(
        feats.filter(months.is_in(list(fold.train_months))), feats.filter(months.is_in(list(fold.valid_months))), cfg
    )
    prod_tg = build(feats, rank, cfg)
    print(f"CV forward target rows {cv_tg.height:,} | production ranking rows {prod_tg.height:,}\n")
    print(f"{'column':26s} | {'CV forward (Nov-Dec 2025)':52s} | production (Jan+Jul 2026)")
    for c in COLS:
        print(f"{c:26s} | {describe(cv_tg, c):52s} | {describe(prod_tg, c)}")
    # the one place 2026 is known to differ: stands that appear in July 2026 and not in 2025
    new = prod_tg.filter(pl.col("ref_n").is_null() & pl.col("STAND_mvt").is_not_null())
    print(f"\nranking rows whose stand x runway cell has no 2025 reference: {new.height:,} ({new.height / prod_tg.height:.2%})")
    print(f"  of which freq_stand non-null: {new['freq_stand'].is_not_null().mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
