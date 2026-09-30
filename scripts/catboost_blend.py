"""Matched-lane equal-weight blend of three model-A members on hybrid_H2_plainy's lanes.

    .venv/bin/python scripts/catboost_blend.py oof  --cb <exp_id> --scratch DIR
    .venv/bin/python scripts/catboost_blend.py detail --a hybrid_H2_plainy --b hybrid_plainy_cb --scratch DIR
    .venv/bin/python scripts/catboost_blend.py full [--out outputs/hybrid_plainy_cb.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/catboost-blend.md.

Members on the matched lane (`mvt_minus_aobt3` non-null): the anchored LightGBM (main_flseq_osm_nolirf), the
plain-y LightGBM (main_flseq_osm_nolirf_plainy) and the anchored CatBoost (main_flseq_osm_nolirf_cb). The unmatched
lane is main_flseq's model B, as in hybrid_H2 and hybrid_H2_plainy.

`oof` writes, in a scratch experiment directory: hybrid_H2_plainy (the baseline, rebuilt from the stored OOF),
hybrid_plainy_cb (the primary: the equal-weight mean of the three), hybrid_plainy_cb_lofo (secondary:
leave-one-fold-out least-squares weights, constrained to sum to one) and two single-member hybrids for context.
Those go through `python -m prc.compare --work-dir data/work_flseq --exp-dir DIR` unchanged."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plainy_blend import ANCHORED, EXP, UNMATCHED, WORK, _oof, detail  # noqa: E402

from prc.train import LABEL, PROXY  # noqa: E402

PLAIN = "main_flseq_osm_nolirf_plainy-b971f4b81b"
MEMBERS = ("a", "p", "c")  # anchored LightGBM, plain LightGBM, anchored CatBoost


def sum_to_one_weights(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Least-squares weights w (sum w = 1, sign free) for y ~ X @ w. The last column is the reference:
    y - X[:, -1] is regressed on X[:, :-1] - X[:, -1:], and its weight is one minus the others."""
    ref = X[:, -1]
    D = X[:, :-1] - ref[:, None]
    w, *_ = np.linalg.lstsq(D, y - ref, rcond=None)
    return np.append(w, 1.0 - w.sum())


def build_oof(cb: str, scratch: Path) -> dict:
    cols = ["MVT_ID_mvt", LABEL, PROXY]
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=cols)
    for name, col in ((ANCHORED, "a"), (PLAIN, "p"), (cb, "c"), (UNMATCHED, "u")):
        d = d.join(_oof(name).rename({"oof": col}), on="MVT_ID_mvt", how="left", maintain_order="left")
    folds = pl.read_parquet(EXP / f"oof_{ANCHORED}_folds.parquet").filter(pl.col("pooled")).drop("pooled")
    d = d.join(folds, on="MVT_ID_mvt", how="left", maintain_order="left")
    matched = pl.col(PROXY).is_not_null()
    scored = d.filter(pl.col(LABEL).is_not_null() & pl.col("a").is_not_null())
    for c in ("p", "c", "u", "fold"):
        n = scored[c].null_count()
        if n:
            raise AssertionError(f"{n} scored rows without {c}")
    m = scored.filter(matched)
    y = m[LABEL].to_numpy().astype(float)
    X = np.column_stack([m[c].to_numpy().astype(float) for c in MEMBERS])
    fold = m["fold"].to_numpy()
    w_by_fold = {}
    for f in sorted(set(fold)):
        o = fold != f
        w_by_fold[f] = sum_to_one_weights(y[o], X[o]).tolist()
    w_all = sum_to_one_weights(y, X).tolist()

    def w_col(j: int):
        return pl.col("fold").replace_strict({f: w[j] for f, w in w_by_fold.items()}, default=None, return_dtype=pl.Float64)

    lofo = sum(w_col(j) * pl.col(c) for j, c in enumerate(MEMBERS))
    d = d.with_columns(
        pl.when(matched).then(0.5 * pl.col("a") + 0.5 * pl.col("p")).otherwise(pl.col("u")).alias("base"),
        pl.when(matched).then((pl.col("a") + pl.col("p") + pl.col("c")) / 3.0).otherwise(pl.col("u")).alias("mean3"),
        pl.when(matched).then(lofo).otherwise(pl.col("u")).alias("lofo"),
        pl.when(matched).then(pl.col("a")).otherwise(pl.col("u")).alias("h_a"),
        pl.when(matched).then(pl.col("c")).otherwise(pl.col("u")).alias("h_c"),
    )
    scratch.mkdir(parents=True, exist_ok=True)
    for c, name in (
        ("base", "hybrid_H2_plainy"),
        ("mean3", "hybrid_plainy_cb"),
        ("lofo", "hybrid_plainy_cb_lofo"),
        ("h_a", "member_anchored"),
        ("h_c", "member_cb"),
    ):
        d.select("MVT_ID_mvt", pl.col(c).alias("oof")).write_parquet(scratch / f"oof_{name}.parquet")
    # Member RMSE on the matched scored rows, and their pairwise correlation of errors.
    e = X - y[:, None]
    rmse = {c: float(np.sqrt((e[:, j] ** 2).mean())) for j, c in enumerate(MEMBERS)}
    corr = np.corrcoef(e.T).round(4).tolist()
    return {"w_by_fold": w_by_fold, "w_all": w_all, "rmse_matched": rmse, "err_corr": corr, "n_matched": len(y)}


def build_full(out: Path) -> None:
    tmpl_path = Path("data/raw/submitting.parquet")
    template = pq.read_table(tmpl_path)
    ids = pl.from_arrow(template.select(["MVT_ID_mvt"]))
    lane = pl.read_parquet(WORK / "feat_rank.parquet", columns=["MVT_ID_mvt", PROXY])
    d = ids.join(lane, on="MVT_ID_mvt", how="left", maintain_order="left")
    for f, c in (
        ("outputs/main_flseq_osm_nolirf.parquet", "a"),
        ("outputs/main_flseq_osm_nolirf_plainy.parquet", "p"),
        ("outputs/main_flseq_osm_nolirf_cb.parquet", "c"),
        ("outputs/main_flseq.parquet", "u"),
    ):
        s = pl.read_parquet(f)
        if not s["MVT_ID_mvt"].equals(ids["MVT_ID_mvt"]):
            raise AssertionError(f"{f}: ids or order differ from the template")
        d = d.with_columns(s["TAXITIME_SEC_mvt"].cast(pl.Float64).alias(c))
    if d.height != template.num_rows or d[PROXY].len() != d.height:
        raise AssertionError("lane join changed the row count")
    m = d[PROXY].is_not_null().to_numpy()
    a, p, c, u = (d[k].to_numpy() for k in ("a", "p", "c", "u"))
    vals = np.where(m, np.rint((a + p + c) / 3.0), u)
    tgt_type = template.schema.field("TAXITIME_SEC_mvt").type
    sub = pa.table(
        {"MVT_ID_mvt": template["MVT_ID_mvt"], "TAXITIME_SEC_mvt": pa.array(vals, type=pa.float64()).cast(tgt_type)},
        schema=template.schema,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out)
    back = pq.read_table(out)
    got = pl.from_arrow(back)
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
    base = pl.read_parquet("outputs/hybrid_H2_plainy.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    diff = v - base
    print(
        f"  vs hybrid_H2_plainy: differs on {int((diff != 0).sum()):,} rows (unmatched untouched: "
        f"{int((diff[~m] != 0).sum()) == 0}), max |d| {np.abs(diff).max():.0f} s, RMS {np.sqrt((diff**2).mean()):.2f} s"
    )
    for x, y_, lab in ((c, a, "cb - anchored"), (c, p, "cb - plain"), (a, p, "anchored - plain")):
        dd = (x - y_)[m]
        print(f"  members on matched rows: {lab} RMS {np.sqrt((dd**2).mean()):.2f} s, max |d| {np.abs(dd).max():.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["oof", "detail", "full"])
    ap.add_argument("--a", default="hybrid_H2_plainy")
    ap.add_argument("--b", default="hybrid_plainy_cb")
    ap.add_argument("--cb", default=None, help="experiment id of the CatBoost run")
    ap.add_argument("--scratch", type=Path, default=None, help="scratch experiment directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_plainy_cb.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        build_full(a.out)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    if not a.cb:
        raise SystemExit("--cb is required for oof")
    r = build_oof(a.cb, a.scratch)
    print(f"matched rows {r['n_matched']:,}; member RMSE (a, p, c): {r['rmse_matched']}")
    print(f"error correlation (a, p, c): {r['err_corr']}")
    print(f"LOFO weights (a, p, c) by held-out fold: {r['w_by_fold']}")
    print(f"weights fitted on all pooled folds: {r['w_all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
