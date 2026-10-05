"""LightGBM linear_tree in both LGB members of hybrid_blend_l2 (CatBoost+L2 unchanged).

    .venv/bin/python scripts/blend_lt.py oof    --scratch DIR --a-lt <exp_id> --p-lt <exp_id>
    .venv/bin/python scripts/blend_lt.py detail --scratch DIR [--a hybrid_blend_l2_P --b hybrid_blend_l2_lt]
    .venv/bin/python scripts/blend_lt.py full   [--out outputs/hybrid_blend_l2_lt.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/linear-tree.md.

Matched lane (`mvt_minus_aobt3` non-null), equal-weight mean of three members; unmatched lane main_flseq's model B
(unchanged):
- base hybrid_blend_l2_P:   anchored LGB + L2, plain-y LGB + L2, anchored CatBoost + L2  (outputs/hybrid_blend_l2.parquet)
- hybrid_blend_l2_lt:       anchored LGB + L2 + LT, plain-y LGB + L2 + LT, anchored CatBoost + L2"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend_l2 import ANCHORED_L2, mean3
from blend_l2 import FULL_FILES as L2_FILES
from plainy_blend import UNMATCHED, WORK, _oof, detail

from prc.train import LABEL, PROXY

PLAIN_L2 = "main_flseq_osm_nolirf_plainy_l2-c74a27239b"
CB_L2 = "main_flseq_osm_nolirf_cb_l2-4aa341b206"

HYBRIDS = {
    "hybrid_blend_l2_P": ("a2", "p2", "c2"),
    "hybrid_blend_l2_lt": ("a3", "p3", "c2"),
}
FULL_FILES = {
    "a2": L2_FILES["a2"],
    "p2": L2_FILES["p2"],
    "c2": L2_FILES["c2"],
    "a3": "outputs/main_flseq_osm_nolirf_l2_lt.parquet",
    "p3": "outputs/main_flseq_osm_nolirf_plainy_l2_lt.parquet",
    "u": L2_FILES["u"],
}
BASE_FILE = "outputs/hybrid_blend_l2.parquet"


def build_oof(scratch: Path, a_lt: str, p_lt: str) -> dict:
    members = {"a2": ANCHORED_L2, "p2": PLAIN_L2, "c2": CB_L2, "a3": a_lt, "p3": p_lt, "u": UNMATCHED}
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY])
    for col, name in members.items():
        d = d.join(_oof(name).rename({"oof": col}), on="MVT_ID_mvt", how="left", maintain_order="left")
    matched = pl.col(PROXY).is_not_null()
    mask = d[LABEL].is_not_null() & d["a2"].is_not_null()
    scored = d.filter(mask)
    for c, name in members.items():
        n = scored[c].null_count()
        if n:
            raise AssertionError(f"{n} scored rows without {c} ({name})")
    out = {"n_scored": scored.height, "rmse": {}}
    scratch.mkdir(parents=True, exist_ok=True)
    for name, cols in HYBRIDS.items():
        h = d.select("MVT_ID_mvt", pl.when(matched).then(mean3(cols)).otherwise(pl.col("u")).alias("oof"))
        h.write_parquet(scratch / f"oof_{name}.parquet")
        s = scored.select(LABEL).with_columns(h.filter(mask)["oof"])
        out["rmse"][name] = float(np.sqrt(((s["oof"] - s[LABEL]) ** 2).mean()))
    # Information only: the plain-y member swapped alone (matched lane p3 instead of p2, same unmatched lane).
    for name, col in (("member_plainy_l2", "p2"), ("member_plainy_l2_lt", "p3")):
        d.select("MVT_ID_mvt", pl.when(matched).then(pl.col(col)).otherwise(pl.col("u")).alias("oof")).write_parquet(
            scratch / f"oof_{name}.parquet"
        )
    m = scored.filter(matched)
    y = m[LABEL].to_numpy().astype(float)
    out["member_rmse_matched"] = {c: float(np.sqrt(((m[c].to_numpy() - y) ** 2).mean())) for c in members if c != "u"}
    return out


def build_full(out: Path) -> None:
    cols = HYBRIDS["hybrid_blend_l2_lt"]
    template = pq.read_table("data/raw/submitting.parquet")
    ids = pl.from_arrow(template.select(["MVT_ID_mvt"]))
    lane = pl.read_parquet(WORK / "feat_rank.parquet", columns=["MVT_ID_mvt", PROXY])
    d = ids.join(lane, on="MVT_ID_mvt", how="left", maintain_order="left")
    if d.height != template.num_rows:
        raise AssertionError("lane join changed the row count")
    for c in (*cols, "u"):
        s = pl.read_parquet(FULL_FILES[c])
        if not s["MVT_ID_mvt"].equals(ids["MVT_ID_mvt"]):
            raise AssertionError(f"{FULL_FILES[c]}: ids or order differ from the template")
        d = d.with_columns(s["TAXITIME_SEC_mvt"].cast(pl.Float64).alias(c))
    m = d[PROXY].is_not_null().to_numpy()
    x0, x1, x2, u = (d[c].to_numpy() for c in (*cols, "u"))
    vals = np.where(m, np.rint((x0 + x1 + x2) / 3.0), u)
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
    print(f"wrote {out} (members {', '.join(FULL_FILES[c] for c in cols)}; unmatched {FULL_FILES['u']})")
    print(
        f"  rows {back.num_rows:,} == template {template.num_rows:,}: {back.num_rows == template.num_rows}"
        f"  (matched {int(m.sum()):,}, unmatched {int((~m).sum()):,})"
    )
    print(
        f"  schema == template: {back.schema.equals(template.schema)}  ({back.schema.to_string(show_schema_metadata=False)!r})"
    )
    print(
        f"  ids and order == template: {back['MVT_ID_mvt'].combine_chunks().equals(template['MVT_ID_mvt'].combine_chunks())}"
    )
    print(
        f"  nulls {got['TAXITIME_SEC_mvt'].null_count()}  negatives {int((v < 0).sum())}  min {v.min():.0f}  max {v.max():.0f}"
    )
    base = pl.read_parquet(BASE_FILE)["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    diff = v - base
    print(
        f"  vs {BASE_FILE}: differs on {int((diff != 0).sum()):,} rows (unmatched untouched: "
        f"{int((diff[~m] != 0).sum()) == 0}), max |d| {np.abs(diff).max():.0f} s, RMS {np.sqrt((diff**2).mean()):.2f} s"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["oof", "detail", "full"])
    ap.add_argument("--a", default="hybrid_blend_l2_P")
    ap.add_argument("--b", default="hybrid_blend_l2_lt")
    ap.add_argument("--a-lt", default=None, help="experiment id of the anchored LT run")
    ap.add_argument("--p-lt", default=None, help="experiment id of the plain-y LT run")
    ap.add_argument("--scratch", type=Path, default=None, help="scratch experiment directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_blend_l2_lt.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        build_full(a.out)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    if not (a.a_lt and a.p_lt):
        raise SystemExit("--a-lt and --p-lt are required for oof")
    r = build_oof(a.scratch, a.a_lt, a.p_lt)
    print(f"scored rows {r['n_scored']:,}; hybrid RMSE: {r['rmse']}")
    print(f"member RMSE on matched scored rows: {r['member_rmse_matched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
