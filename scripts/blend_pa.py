"""Per-airport anchored LGB+L2 member (PA) in the matched lane of hybrid_blend_l2.

    .venv/bin/python scripts/blend_pa.py oof    --scratch DIR --pa <exp_id>
    .venv/bin/python scripts/blend_pa.py detail --scratch DIR [--a hybrid_blend_l2_P --b hybrid_blend_l2_pa]
    .venv/bin/python scripts/blend_pa.py full   [--out outputs/hybrid_blend_l2_pa.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/per-airport-member.md.

Matched lane (`mvt_minus_aobt3` non-null); unmatched lane main_flseq's model B (unchanged) in every hybrid:
- base hybrid_blend_l2_P:   mean(a2, p2, c2)                      (outputs/hybrid_blend_l2.parquet)
- primary hybrid_blend_l2_pa: mean(0.5*a2 + 0.5*pa, p2, c2)
- secondary hybrid_blend_l2_pa4: mean(a2, pa, p2, c2)             (report only)
a2 = anchored LGB+L2 pooled, pa = the same recipe fitted per airport, p2 = plain-y LGB+L2, c2 = CatBoost+L2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend_l2 import ANCHORED_L2
from blend_l2 import FULL_FILES as L2_FILES
from plainy_blend import UNMATCHED, WORK, _oof, detail

from prc.train import LABEL, PROXY

PLAIN_L2 = "main_flseq_osm_nolirf_plainy_l2-c74a27239b"
CB_L2 = "main_flseq_osm_nolirf_cb_l2-4aa341b206"

# Matched-lane formula of each hybrid: {member column: weight}.
HYBRIDS = {
    "hybrid_blend_l2_P": {"a2": 1 / 3, "p2": 1 / 3, "c2": 1 / 3},
    "hybrid_blend_l2_pa": {"a2": 1 / 6, "pa": 1 / 6, "p2": 1 / 3, "c2": 1 / 3},
    "hybrid_blend_l2_pa4": {"a2": 1 / 4, "pa": 1 / 4, "p2": 1 / 4, "c2": 1 / 4},
}
FULL_FILES = {
    "a2": L2_FILES["a2"],
    "p2": L2_FILES["p2"],
    "c2": L2_FILES["c2"],
    "pa": "outputs/main_flseq_osm_nolirf_l2_pa.parquet",
    "u": L2_FILES["u"],
}
BASE_FILE = "outputs/hybrid_blend_l2.parquet"


def matched_value(cols: dict[str, float], get) -> object:
    """The matched-lane value of a hybrid, written in the pre-registered form (not via the weight table), so the
    primary is literally mean(0.5*a2 + 0.5*pa, p2, c2)."""
    if set(cols) == {"a2", "p2", "c2"}:
        return (get("a2") + get("p2") + get("c2")) / 3.0
    if set(cols) == {"a2", "pa", "p2", "c2"} and cols["p2"] == 1 / 3:
        return (0.5 * get("a2") + 0.5 * get("pa") + get("p2") + get("c2")) / 3.0
    if set(cols) == {"a2", "pa", "p2", "c2"}:
        return (get("a2") + get("pa") + get("p2") + get("c2")) / 4.0
    raise ValueError(cols)


def build_oof(scratch: Path, pa_id: str) -> dict:
    members = {"a2": ANCHORED_L2, "pa": pa_id, "p2": PLAIN_L2, "c2": CB_L2, "u": UNMATCHED}
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
        v = matched_value(cols, pl.col)
        h = d.select("MVT_ID_mvt", pl.when(matched).then(v).otherwise(pl.col("u")).alias("oof"))
        h.write_parquet(scratch / f"oof_{name}.parquet")
        s = scored.select(LABEL).with_columns(h.filter(mask)["oof"])
        out["rmse"][name] = float(np.sqrt(((s["oof"] - s[LABEL]) ** 2).mean()))
    # Secondary: each LGB member alone in the matched lane, with the same unmatched lane.
    for name, col in (("member_a2", "a2"), ("member_pa", "pa")):
        d.select("MVT_ID_mvt", pl.when(matched).then(pl.col(col)).otherwise(pl.col("u")).alias("oof")).write_parquet(
            scratch / f"oof_{name}.parquet"
        )
    m = scored.filter(matched)
    y = m[LABEL].to_numpy().astype(float)
    out["member_rmse_matched"] = {c: float(np.sqrt(((m[c].to_numpy() - y) ** 2).mean())) for c in members if c != "u"}
    dd = m["a2"].to_numpy() - m["pa"].to_numpy()
    out["a2_minus_pa_rms_matched"] = float(np.sqrt((dd**2).mean()))
    return out


def build_full(out: Path, which: str = "hybrid_blend_l2_pa") -> None:
    cols = HYBRIDS[which]
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
    vals = np.where(m, np.rint(matched_value(cols, lambda c: d[c].to_numpy())), d["u"].to_numpy())
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
    print(f"wrote {out} ({which}: members {', '.join(FULL_FILES[c] for c in cols)}; unmatched {FULL_FILES['u']})")
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
    ap.add_argument("--b", default="hybrid_blend_l2_pa")
    ap.add_argument("--pa", default=None, help="experiment id of the per-airport run")
    ap.add_argument("--which", default="hybrid_blend_l2_pa", choices=sorted(HYBRIDS))
    ap.add_argument("--scratch", type=Path, default=None, help="scratch experiment directory")
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_blend_l2_pa.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        build_full(a.out, a.which)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    if not a.pa:
        raise SystemExit("--pa is required for oof")
    r = build_oof(a.scratch, a.pa)
    print(f"scored rows {r['n_scored']:,}; hybrid RMSE: {r['rmse']}")
    print(f"member RMSE on matched scored rows: {r['member_rmse_matched']}")
    print(f"a2 - pa RMS on matched scored rows: {r['a2_minus_pa_rms_matched']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
