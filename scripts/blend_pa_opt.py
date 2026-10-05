"""hybrid_blend_l2_pa with the retuned model-A parameters in M1 (pooled + per-airport) and M2 (plain-y).

    .venv/bin/python scripts/blend_pa_opt.py oof    --scratch DIR --a2o <id> --pao <id> --p2o <id>
    .venv/bin/python scripts/blend_pa_opt.py detail --scratch DIR [--a hybrid_blend_l2_pa --b hybrid_blend_l2_pa_opt]
    .venv/bin/python scripts/blend_pa_opt.py full   [--out outputs/hybrid_blend_l2_pa_opt.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/optuna-retune.md §0.5.

Matched lane (`mvt_minus_aobt3` non-null); unmatched lane main_flseq's model B in both:
- base      hybrid_blend_l2_pa:     ((0.5*a2  + 0.5*pa ) + p2  + c2) / 3   (outputs/hybrid_blend_l2_pa.parquet)
- candidate hybrid_blend_l2_pa_opt: ((0.5*a2o + 0.5*pao) + p2o + c2) / 3
c2 (CatBoost+L2) is not retuned and is the same member in both."""

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
from blend_pa import CB_L2, PLAIN_L2
from blend_pa import FULL_FILES as PA_FILES
from plainy_blend import UNMATCHED, WORK, _oof, detail

from prc.train import LABEL, PROXY

PA_L2 = "main_flseq_osm_nolirf_l2_pa-85f815edf4"
HYBRIDS = {
    "hybrid_blend_l2_pa": ("a2", "pa", "p2", "c2"),
    "hybrid_blend_l2_pa_opt": ("a2o", "pao", "p2o", "c2"),
}
FULL_FILES = {
    **PA_FILES,
    "a2o": "outputs/main_flseq_osm_nolirf_l2_opt.parquet",
    "pao": "outputs/main_flseq_osm_nolirf_l2_pa_opt.parquet",
    "p2o": "outputs/main_flseq_osm_nolirf_plainy_l2_opt.parquet",
}
BASE_FILE = "outputs/hybrid_blend_l2_pa.parquet"


def matched_value(members: tuple[str, str, str, str], get):
    a, p_a, p, c = members
    return (0.5 * get(a) + 0.5 * get(p_a) + get(p) + get(c)) / 3.0


def build_oof(scratch: Path, a2o: str, pao: str, p2o: str) -> dict:
    members = {
        "a2": ANCHORED_L2,
        "pa": PA_L2,
        "p2": PLAIN_L2,
        "c2": CB_L2,
        "a2o": a2o,
        "pao": pao,
        "p2o": p2o,
        "u": UNMATCHED,
    }
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
        h = d.select(
            "MVT_ID_mvt", pl.when(matched).then(matched_value(cols, pl.col)).otherwise(pl.col("u")).alias("oof")
        )
        h.write_parquet(scratch / f"oof_{name}.parquet")
        s = scored.select(LABEL).with_columns(h.filter(mask)["oof"])
        out["rmse"][name] = float(np.sqrt(((s["oof"] - s[LABEL]) ** 2).mean()))
    m = scored.filter(matched)
    y = m[LABEL].to_numpy().astype(float)
    out["member_rmse_matched"] = {c: float(np.sqrt(((m[c].to_numpy() - y) ** 2).mean())) for c in members if c != "u"}
    return out


def build_full(out: Path) -> None:
    cols = HYBRIDS["hybrid_blend_l2_pa_opt"]
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
    print(f"wrote {out}: members {', '.join(FULL_FILES[c] for c in cols)}; unmatched {FULL_FILES['u']}")
    print(
        f"  rows {back.num_rows:,} == template {template.num_rows:,}: {back.num_rows == template.num_rows}"
        f"  (matched {int(m.sum()):,}, unmatched {int((~m).sum()):,})"
    )
    print(
        f"  schema == template: {back.schema.equals(template.schema)}"
        f"  ({back.schema.to_string(show_schema_metadata=False)!r})"
    )
    same_ids = back["MVT_ID_mvt"].combine_chunks().equals(template["MVT_ID_mvt"].combine_chunks())
    print(f"  ids and order == template: {same_ids}")
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
    ap.add_argument("--a", default="hybrid_blend_l2_pa")
    ap.add_argument("--b", default="hybrid_blend_l2_pa_opt")
    ap.add_argument("--a2o")
    ap.add_argument("--pao")
    ap.add_argument("--p2o")
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_blend_l2_pa_opt.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        build_full(a.out)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    if not (a.a2o and a.pao and a.p2o):
        raise SystemExit("--a2o, --pao and --p2o are required for oof")
    r = build_oof(a.scratch, a.a2o, a.pao, a.p2o)
    print(f"scored rows {r['n_scored']:,}; hybrid RMSE: {r['rmse']}")
    print(f"member RMSE on matched scored rows: {r['member_rmse_matched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
