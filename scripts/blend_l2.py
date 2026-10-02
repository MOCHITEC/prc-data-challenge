"""Lever L2 (forward NM windows) in the three matched-lane members of hybrid_plainy_cb.

    .venv/bin/python scripts/blend_l2.py oof  --scratch DIR [--plain-l2 <exp_id> --cb-l2 <exp_id>]
    .venv/bin/python scripts/blend_l2.py detail --a hybrid_plainy_cb --b hybrid_blend_l2_P --scratch DIR
    .venv/bin/python scripts/blend_l2.py full --which P|F [--out outputs/hybrid_blend_l2.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/blend-l2.md.

Matched lane (`mvt_minus_aobt3` non-null), equal-weight mean of three members; unmatched lane main_flseq's model B
(unchanged from hybrid_plainy_cb):
- base hybrid_plainy_cb: anchored LGB, plain-y LGB, anchored CatBoost (no L2 anywhere)
- P (primary):  anchored LGB + L2, plain-y LGB + L2, anchored CatBoost + L2
- F (fallback): anchored LGB + L2, plain-y LGB,      anchored CatBoost

`oof` writes oof_<name>.parquet files in a scratch experiment directory for `prc.compare --exp-dir DIR`. P is
written only when both new experiment ids are given."""

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
CB = "main_flseq_osm_nolirf_cb-6297d62f5f"
ANCHORED_L2 = "main_flseq_osm_nolirf_l2-ef204b44ff"

# Matched-lane members of each hybrid, by column name in build_oof / build_full.
HYBRIDS = {
    "hybrid_plainy_cb": ("a", "p", "c"),
    "hybrid_blend_l2_F": ("a2", "p", "c"),
    "hybrid_blend_l2_P": ("a2", "p2", "c2"),
}
FULL_FILES = {
    "a": "outputs/main_flseq_osm_nolirf.parquet",
    "p": "outputs/main_flseq_osm_nolirf_plainy.parquet",
    "c": "outputs/main_flseq_osm_nolirf_cb.parquet",
    "a2": "outputs/main_flseq_osm_nolirf_l2.parquet",
    "p2": "outputs/main_flseq_osm_nolirf_plainy_l2.parquet",
    "c2": "outputs/main_flseq_osm_nolirf_cb_l2.parquet",
    "u": "outputs/main_flseq.parquet",
}


def mean3(cols: tuple[str, str, str]) -> pl.Expr:
    return (pl.col(cols[0]) + pl.col(cols[1]) + pl.col(cols[2])) / 3.0


def build_oof(scratch: Path, plain_l2: str | None, cb_l2: str | None) -> dict:
    members = {"a": ANCHORED, "p": PLAIN, "c": CB, "a2": ANCHORED_L2, "u": UNMATCHED}
    if plain_l2 and cb_l2:
        members |= {"p2": plain_l2, "c2": cb_l2}
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY])
    for col, name in members.items():
        d = d.join(_oof(name).rename({"oof": col}), on="MVT_ID_mvt", how="left", maintain_order="left")
    matched = pl.col(PROXY).is_not_null()
    scored = d.filter(pl.col(LABEL).is_not_null() & pl.col("a").is_not_null())
    for c in members:
        n = scored[c].null_count()
        if n:
            raise AssertionError(f"{n} scored rows without {c} ({members[c]})")
    # The untouched unmatched lane: every member's own OOF is irrelevant there, the hybrid takes u.
    out = {"n_scored": scored.height, "rmse": {}}
    scratch.mkdir(parents=True, exist_ok=True)
    for name, cols in HYBRIDS.items():
        if not set(cols) <= set(members):
            continue
        h = d.select("MVT_ID_mvt", pl.when(matched).then(mean3(cols)).otherwise(pl.col("u")).alias("oof"))
        h.write_parquet(scratch / f"oof_{name}.parquet")
        s = scored.select(LABEL).with_columns(h.filter(scored_mask(d))["oof"])
        out["rmse"][name] = float(np.sqrt(((s["oof"] - s[LABEL]) ** 2).mean()))
    m = scored.filter(matched)
    y = m[LABEL].to_numpy().astype(float)
    out["member_rmse_matched"] = {
        c: float(np.sqrt(((m[c].to_numpy() - y) ** 2).mean())) for c in members if c != "u"
    }
    return out


def scored_mask(d: pl.DataFrame) -> pl.Series:
    return d[LABEL].is_not_null() & d["a"].is_not_null()


def build_full(which: str, out: Path) -> None:
    cols = HYBRIDS[{"P": "hybrid_blend_l2_P", "F": "hybrid_blend_l2_F", "base": "hybrid_plainy_cb"}[which]]
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
    print(f"wrote {out} ({which}: members {', '.join(FULL_FILES[c] for c in cols)})")
    print(f"  rows {back.num_rows:,} == template {template.num_rows:,}: {back.num_rows == template.num_rows}"
          f"  (matched {int(m.sum()):,}, unmatched {int((~m).sum()):,})")
    print(f"  schema == template: {back.schema.equals(template.schema)}  ({back.schema.to_string(show_schema_metadata=False)!r})")
    print(f"  ids and order == template: {back['MVT_ID_mvt'].combine_chunks().equals(template['MVT_ID_mvt'].combine_chunks())}")
    print(f"  nulls {got['TAXITIME_SEC_mvt'].null_count()}  negatives {int((v < 0).sum())}  min {v.min():.0f}  max {v.max():.0f}")
    base = pl.read_parquet("outputs/hybrid_plainy_cb.parquet")["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    diff = v - base
    print(
        f"  vs hybrid_plainy_cb: differs on {int((diff != 0).sum()):,} rows (unmatched untouched: "
        f"{int((diff[~m] != 0).sum()) == 0}), max |d| {np.abs(diff).max():.0f} s, RMS {np.sqrt((diff**2).mean()):.2f} s"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["oof", "detail", "full"])
    ap.add_argument("--a", default="hybrid_plainy_cb")
    ap.add_argument("--b", default="hybrid_blend_l2_P")
    ap.add_argument("--plain-l2", default=None, help="experiment id of the plain-y + L2 run")
    ap.add_argument("--cb-l2", default=None, help="experiment id of the CatBoost + L2 run")
    ap.add_argument("--which", choices=["P", "F", "base"], default=None)
    ap.add_argument("--scratch", type=Path, default=None, help="scratch experiment directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_blend_l2.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        if a.which is None:
            raise SystemExit("--which is required for full")
        build_full(a.which, a.out)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    r = build_oof(a.scratch, a.plain_l2, a.cb_l2)
    print(f"scored rows {r['n_scored']:,}; hybrid RMSE: {r['rmse']}")
    print(f"member RMSE on matched scored rows: {r['member_rmse_matched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
