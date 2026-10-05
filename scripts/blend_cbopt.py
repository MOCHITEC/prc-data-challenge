"""hybrid_blend_l2_pa_opt_mlp with the retuned CatBoost member in place of M3.

    .venv/bin/python scripts/blend_cbopt.py oof    --scratch DIR --c2o <exp id of main_flseq_osm_nolirf_cb_l2_cbopt>
    .venv/bin/python scripts/blend_cbopt.py detail --scratch DIR
    .venv/bin/python scripts/blend_cbopt.py full   [--out outputs/hybrid_blend_l2_pa_opt_mlp_cbopt.parquet]

Pre-registration and results: docs/competitor_research_2026-09-21/results/catboost-retune.md.

Matched lane (`mvt_minus_aobt3` non-null); unmatched lane main_flseq's model B in both hybrids:
- base      hybrid_blend_l2_pa_opt_mlp:       (M1 + M2 + M3  + M4) / 4   (outputs/hybrid_blend_l2_pa_opt_mlp.parquet)
- candidate hybrid_blend_l2_pa_opt_mlp_cbopt: (M1 + M2 + M3' + M4) / 4
with M1 = 0.5*a2o + 0.5*pao, M2 = p2o, M3 = c2 (main_flseq_osm_nolirf_cb_l2), M3' = c2o (the retuned CatBoost),
M4 = m4 (the MLP). Only M3 changes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend_mlp import A2O, C2, P2O, PAO  # noqa: E402
from blend_mlp import FULL_FILES as MLP_FILES  # noqa: E402
from plainy_blend import EXP, UNMATCHED, WORK, _oof, detail  # noqa: E402

from prc.train import LABEL, PROXY  # noqa: E402

M4 = "main_flseq_osm_nolirf_l2_mlp-2b9a3b9348"
BASE = "hybrid_blend_l2_pa_opt_mlp"
CAND = "hybrid_blend_l2_pa_opt_mlp_cbopt"
FULL_FILES = {**MLP_FILES, "c2o": "outputs/main_flseq_osm_nolirf_cb_l2_cbopt.parquet"}
BASE_FILE = "outputs/hybrid_blend_l2_pa_opt_mlp.parquet"


def base_value(get):
    return ((0.5 * get("a2o") + 0.5 * get("pao")) + get("p2o") + get("c2") + get("m4")) / 4.0


def candidate_value(get):
    return ((0.5 * get("a2o") + 0.5 * get("pao")) + get("p2o") + get("c2o") + get("m4")) / 4.0


def build_oof(scratch: Path, c2o: str) -> dict:
    ids = {"a2o": A2O, "pao": PAO, "p2o": P2O, "c2": C2, "c2o": c2o, "m4": M4, "u": UNMATCHED}
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY])
    for col, name in ids.items():
        d = d.join(_oof(name).rename({"oof": col}), on="MVT_ID_mvt", how="left", maintain_order="left")
    folds = pl.read_parquet(EXP / f"oof_{A2O}_folds.parquet").filter(pl.col("pooled")).drop("pooled")
    d = d.join(folds, on="MVT_ID_mvt", how="left", maintain_order="left")
    matched = pl.col(PROXY).is_not_null()
    mask = d[LABEL].is_not_null() & d["a2o"].is_not_null()
    scored = d.filter(mask)
    for c in (*ids, "fold"):
        n = scored[c].null_count()
        if n:
            raise AssertionError(f"{n} scored rows without {c} ({ids.get(c, c)})")
    hyb = {BASE: base_value(pl.col), CAND: candidate_value(pl.col), "member_c2": pl.col("c2"), "member_c2o": pl.col("c2o")}
    scratch.mkdir(parents=True, exist_ok=True)
    out = {"n_scored": scored.height, "rmse": {}}
    for name, expr in hyb.items():
        h = d.select("MVT_ID_mvt", pl.when(matched).then(expr).otherwise(pl.col("u")).alias("oof"))
        h.write_parquet(scratch / f"oof_{name}.parquet")
        s = h.filter(mask)["oof"].to_numpy() - scored[LABEL].to_numpy()
        out["rmse"][name] = float(np.sqrt((s**2).mean()))
    m = scored.filter(matched)
    y = m[LABEL].to_numpy().astype(float)
    m = m.with_columns((0.5 * pl.col("a2o") + 0.5 * pl.col("pao")).alias("m1"))
    cols = ("m1", "p2o", "c2", "c2o", "m4")
    e = np.column_stack([m[c].to_numpy().astype(float) for c in cols]) - y[:, None]
    fold = m["fold"].to_numpy()
    out["n_matched"] = len(y)
    out["member_rmse_matched"] = {c: float(np.sqrt((e[:, j] ** 2).mean())) for j, c in enumerate(cols)}
    out["member_rmse_matched_by_fold"] = {
        f: {c: float(np.sqrt((e[fold == f, j] ** 2).mean())) for j, c in enumerate(cols)} for f in sorted(set(fold))
    }
    out["err_corr_cols"] = list(cols)
    out["err_corr"] = np.corrcoef(e.T).round(4).tolist()
    return out


def build_full(out: Path) -> None:
    template = pq.read_table("data/raw/submitting.parquet")
    ids = pl.from_arrow(template.select(["MVT_ID_mvt"]))
    lane = pl.read_parquet(WORK / "feat_rank.parquet", columns=["MVT_ID_mvt", PROXY])
    d = ids.join(lane, on="MVT_ID_mvt", how="left", maintain_order="left")
    if d.height != template.num_rows:
        raise AssertionError("lane join changed the row count")
    for c, f in FULL_FILES.items():
        s = pl.read_parquet(f)
        if not s["MVT_ID_mvt"].equals(ids["MVT_ID_mvt"]):
            raise AssertionError(f"{f}: ids or order differ from the template")
        d = d.with_columns(s["TAXITIME_SEC_mvt"].cast(pl.Float64).alias(c))
    m = d[PROXY].is_not_null().to_numpy()
    get = lambda c: d[c].to_numpy()  # noqa: E731
    base_check = np.where(m, np.rint(base_value(get)), d["u"].to_numpy())
    base = pl.read_parquet(BASE_FILE)["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    print(f"  base formula reproduces {BASE_FILE}: {np.array_equal(base_check, base)}")
    vals = np.where(m, np.rint(candidate_value(get)), d["u"].to_numpy())
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
    print(f"wrote {out}: members {', '.join(FULL_FILES[c] for c in ('a2o', 'pao', 'p2o', 'c2o', 'm4'))}; unmatched {FULL_FILES['u']}")
    print(
        f"  rows {back.num_rows:,} == template {template.num_rows:,}: {back.num_rows == template.num_rows}"
        f"  (matched {int(m.sum()):,}, unmatched {int((~m).sum()):,})"
    )
    print(f"  schema == template: {back.schema.equals(template.schema)}  ({back.schema.to_string(show_schema_metadata=False)!r})")
    same_ids = back["MVT_ID_mvt"].combine_chunks().equals(template["MVT_ID_mvt"].combine_chunks())
    print(f"  ids and order == template: {same_ids}")
    print(
        f"  nulls {got['TAXITIME_SEC_mvt'].null_count()}  negatives {int((v < 0).sum())}  min {v.min():.0f}  max {v.max():.0f}"
    )
    diff = v - base
    print(
        f"  vs {BASE_FILE}: differs on {int((diff != 0).sum()):,} rows (unmatched untouched: "
        f"{int((diff[~m] != 0).sum()) == 0}), max |d| {np.abs(diff).max():.0f} s, RMS {np.sqrt((diff**2).mean()):.2f} s"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["oof", "detail", "full"])
    ap.add_argument("--a", default=BASE)
    ap.add_argument("--b", default=CAND)
    ap.add_argument("--c2o")
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/hybrid_blend_l2_pa_opt_mlp_cbopt.parquet"))
    a = ap.parse_args()
    if a.what == "full":
        build_full(a.out)
        return 0
    if a.scratch is None:
        raise SystemExit("--scratch is required for oof and detail (keep hybrid OOF files out of experiments/)")
    if a.what == "detail":
        detail(a.scratch, a.a, a.b)
        return 0
    if not a.c2o:
        raise SystemExit("--c2o is required for oof")
    r = build_oof(a.scratch, a.c2o)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
