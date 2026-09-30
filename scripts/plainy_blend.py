"""Matched-lane blend of the anchored and the plain-y model A, on H2's lanes.

    .venv/bin/python scripts/plainy_blend.py oof  --plain <exp_id> [--scratch DIR]
    .venv/bin/python scripts/plainy_blend.py detail --a hybrid_H2 --b hybrid_H2_plainy [--scratch DIR]
    .venv/bin/python scripts/plainy_blend.py full [--out outputs/hybrid_H2_plainy.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/plainy-blend.md.

`oof` builds three hybrid OOF files in a scratch experiment directory (H2, the fixed-0.5 blend and the
leave-one-fold-out optimal-weight blend) plus the two single members, and prints the fitted weights. Those files
go through `python -m prc.compare --work-dir data/work_flseq --exp-dir DIR` unchanged; `detail` prints the lane /
month / airport breakdowns for a pair. `full` writes the full-fit blended hybrid in the template's order and schema and validates it.

Lane = `mvt_minus_aobt3` non-null, as in prc.train.predict_fold and osm-on-flseq.md §8."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from prc import compare
from prc.compare import group_diff, per_group_diff
from prc.cv import year_month_key
from prc.train import LABEL, PROXY

ANCHORED = "main_flseq_osm_nolirf-926741d779"
UNMATCHED = "main_flseq-8615e6a194"
WORK = Path("data/work_flseq")
EXP = Path("experiments")
W_FIXED = 0.5


def _oof(exp_id: str) -> pl.DataFrame:
    return pl.read_parquet(EXP / f"oof_{exp_id}.parquet")


def build_oof(plain: str, scratch: Path) -> dict:
    cols = ["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt", "ADEP_mvt"]
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=cols)
    for name, col in ((ANCHORED, "a"), (plain, "p"), (UNMATCHED, "u")):
        d = d.join(_oof(name).rename({"oof": col}), on="MVT_ID_mvt", how="left", maintain_order="left")
    folds = pl.read_parquet(EXP / f"oof_{ANCHORED}_folds.parquet").filter(pl.col("pooled")).drop("pooled")
    d = d.join(folds, on="MVT_ID_mvt", how="left", maintain_order="left")
    matched = pl.col(PROXY).is_not_null()
    scored = d.filter(pl.col(LABEL).is_not_null() & pl.col("a").is_not_null())
    # The three members must cover the same scored rows, and the pooled fold must be known for every one.
    for c in ("p", "u", "fold"):
        n = scored[c].null_count()
        if n:
            raise AssertionError(f"{n} scored rows without {c}")
    # Leave-one-fold-out weight on the anchored member, least squares on matched rows of the other folds.
    m = scored.filter(matched)
    y, a, p = (m[c].to_numpy().astype(float) for c in (LABEL, "a", "p"))
    fold = m["fold"].to_numpy()
    w_by_fold = {}
    for f in sorted(set(fold)):
        o = fold != f
        dd = a[o] - p[o]
        w_by_fold[f] = float(((y[o] - p[o]) * dd).sum() / (dd * dd).sum())
    dd = a - p
    w_all = float(((y - p) * dd).sum() / (dd * dd).sum())
    w_row = pl.col("fold").replace_strict(w_by_fold, default=None, return_dtype=pl.Float64)
    d = d.with_columns(
        pl.when(matched).then(pl.col("a")).otherwise(pl.col("u")).alias("h2"),
        pl.when(matched).then(W_FIXED * pl.col("a") + (1 - W_FIXED) * pl.col("p")).otherwise(pl.col("u")).alias("b05"),
        pl.when(matched).then(w_row * pl.col("a") + (1 - w_row) * pl.col("p")).otherwise(pl.col("u")).alias("bopt"),
    )
    scratch.mkdir(parents=True, exist_ok=True)
    for c, name in (("h2", "hybrid_H2"), ("b05", "hybrid_H2_plainy"), ("bopt", "hybrid_H2_plainy_lofo")):
        d.select("MVT_ID_mvt", pl.col(c).alias("oof")).write_parquet(scratch / f"oof_{name}.parquet")
    # The two single members too, so `plain alone vs anchored alone` can go through prc.compare unchanged.
    for c, name in (("a", "member_anchored"), ("p", "member_plain")):
        d.select("MVT_ID_mvt", pl.col(c).alias("oof")).write_parquet(scratch / f"oof_{name}.parquet")
    return {"w_by_fold": w_by_fold, "w_all": w_all}


def detail(scratch: Path, a: str, b: str, n_boot: int = 500) -> None:
    d = compare.load_pair(a, b, WORK, scratch)
    y, pa_, pb = d[LABEL].to_numpy().astype(float), d["p_a"].to_numpy(), d["p_b"].to_numpy()
    matched = d[PROXY].is_not_null().to_numpy()
    apt = d["ADEP_mvt"].to_numpy()
    month = year_month_key(d).to_numpy()

    def line(label, r):
        print(
            f"  {label:<22} n={r['n']:>7,}  A={r['rmse_a']:8.2f}  B={r['rmse_b']:8.2f}  "
            f"diff={r['diff']:+7.2f} [{r['ci95'][0]:+6.2f}, {r['ci95'][1]:+6.2f}]  P(B)={r['b_better_prob']:.3f}"
        )

    print(f"\n== detail {a} -> {b}")
    print("lanes:")
    line("all", group_diff(y, pa_, pb, None, n_boot))
    line("matched", group_diff(y, pa_, pb, matched, n_boot))
    line("unmatched", group_diff(y, pa_, pb, ~matched, n_boot))
    line("all except LIRF", group_diff(y, pa_, pb, apt != "LIRF", n_boot))
    print("per calendar month (all rows | matched):")
    for mo in np.unique(month):
        line(f"{mo} all", group_diff(y, pa_, pb, month == mo, n_boot))
        line(f"{mo} matched", group_diff(y, pa_, pb, (month == mo) & matched, n_boot))
    print("per airport, all rows (ordered by share of squared disagreement):")
    for r in per_group_diff(y, pa_, pb, apt, None, n_boot):
        line(f"{r['group']} ({r['gap_share']:.1%})", r)


def build_full(out: Path) -> None:
    tmpl_path = Path("data/raw/submitting.parquet")
    template = pq.read_table(tmpl_path)
    ids = pl.from_arrow(template.select(["MVT_ID_mvt"]))
    lane = pl.read_parquet(WORK / "feat_rank.parquet", columns=["MVT_ID_mvt", PROXY])
    d = ids.join(lane, on="MVT_ID_mvt", how="left", maintain_order="left")
    for f, c in (
        ("outputs/main_flseq_osm_nolirf.parquet", "a"),
        ("outputs/main_flseq_osm_nolirf_plainy.parquet", "p"),
        ("outputs/main_flseq.parquet", "u"),
    ):
        s = pl.read_parquet(f)
        if not s["MVT_ID_mvt"].equals(ids["MVT_ID_mvt"]):
            raise AssertionError(f"{f}: ids or order differ from the template")
        d = d.with_columns(s["TAXITIME_SEC_mvt"].cast(pl.Float64).alias(c))
    if d.height != template.num_rows or d[PROXY].len() != d.height:
        raise AssertionError("lane join changed the row count")
    m = d[PROXY].is_not_null().to_numpy()
    a, p, u = (d[c].to_numpy() for c in ("a", "p", "u"))
    vals = np.where(m, np.rint(W_FIXED * a + (1 - W_FIXED) * p), u)
    tgt_type = template.schema.field("TAXITIME_SEC_mvt").type
    sub = pa.table(
        {"MVT_ID_mvt": template["MVT_ID_mvt"], "TAXITIME_SEC_mvt": pa.array(vals, type=pa.float64()).cast(tgt_type)},
        schema=template.schema,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out)
    back = pq.read_table(out)
    got = pl.from_arrow(back)
    h2 = pl.read_parquet("outputs/hybrid_H2.parquet")
    v = got["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    print(f"wrote {out}: rows {back.num_rows:,} (matched {int(m.sum()):,}, unmatched {int((~m).sum()):,})")
    print(
        f"  schema == template: {back.schema.equals(template.schema)}  ({back.schema.to_string(show_schema_metadata=False)!r})"
    )
    print(
        f"  ids and order == template: {back['MVT_ID_mvt'].combine_chunks().equals(template['MVT_ID_mvt'].combine_chunks())}"
    )
    print(
        f"  nulls {got['TAXITIME_SEC_mvt'].null_count()}  negatives {int((v < 0).sum())}  min {v.min():.0f}  max {v.max():.0f}"
    )
    diff = v - h2["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    print(
        f"  vs hybrid_H2: differs on {int((diff != 0).sum()):,} rows (matched only: {int((diff[~m] != 0).sum()) == 0}), "
        f"max |d| {np.abs(diff).max():.0f} s, RMS {np.sqrt((diff**2).mean()):.2f} s"
    )
    dp = a - p
    print(
        f"  members on matched rows: anchored - plain RMS {np.sqrt((dp[m] ** 2).mean()):.2f} s, max |d| {np.abs(dp[m]).max():.0f} s"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["oof", "detail", "full"])
    ap.add_argument("--a", default="hybrid_H2")
    ap.add_argument("--b", default="hybrid_H2_plainy")
    ap.add_argument("--plain", default=None, help="experiment id of the plain-y run")
    ap.add_argument("--scratch", type=Path, required=False, default=None, help="scratch experiment directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_H2_plainy.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        build_full(a.out)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    if not a.plain:
        raise SystemExit("--plain is required for oof")
    w = build_oof(a.plain, a.scratch)
    print(f"LOFO weight on the anchored member: {w['w_by_fold']}; fitted on all pooled folds: {w['w_all']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
