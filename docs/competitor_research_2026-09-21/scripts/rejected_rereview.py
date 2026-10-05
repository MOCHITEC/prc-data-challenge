"""OOF-only re-checks of rejected levers on the hybrid_blend_l2_pa_opt_mlp base (rejected-rereview.md §1).

    POLARS_MAX_THREADS=1 .venv/bin/python docs/competitor_research_2026-09-21/scripts/rejected_rereview.py align   --scratch DIR
    POLARS_MAX_THREADS=1 .venv/bin/python docs/competitor_research_2026-09-21/scripts/rejected_rereview.py build   --scratch DIR

`align` runs the row-alignment checks of §1.1. `build` rebuilds the base (and LOFO) with scripts/blend_mlp.py's own
code, checks them against the stored files, and writes the candidate hybrid OOFs to DIR. The comparisons are then
`python -m prc.compare hybrid_blend_l2_pa_opt_mlp <B> --work-dir data/work_flseq --exp-dir DIR`.
Nothing is written outside DIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from blend_mlp import A2O, C2, P2O, PAO, build_oof  # noqa: E402
from plainy_blend import EXP, UNMATCHED, WORK  # noqa: E402

from prc.train import LABEL, PROXY  # noqa: E402

M4 = "main_flseq_osm_nolirf_l2_mlp-2b9a3b9348"
L2B = "main_flseq_l2b-e40f03b86d"
OSM = "main_flseq_osm-bb204cd922"
LT = "main_flseq_osm_nolirf_l2_lt-a9fb149ab5"
STORED = ROOT / "logs/mlp/scratch"
ARMS = {"a2o": A2O, "pao": PAO, "p2o": P2O, "c2": C2, "m4": M4, "u": UNMATCHED, "u_l2b": L2B, "u_osm": OSM, "lt": LT}


def frame() -> pl.DataFrame:
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY])
    for col, name in ARMS.items():
        o = pl.read_parquet(EXP / f"oof_{name}.parquet", columns=["MVT_ID_mvt", "oof"]).rename({"oof": col})
        d = d.join(o, on="MVT_ID_mvt", how="left", maintain_order="left")
    return d


def align() -> None:
    base_folds = pl.read_parquet(EXP / f"oof_{A2O}_folds.parquet").filter(pl.col("pooled")).drop("pooled")
    labels = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY])
    print(f"feat_train rows {labels.height:,}, unique ids {labels['MVT_ID_mvt'].n_unique():,}")
    a = pl.read_parquet(EXP / f"oof_{A2O}.parquet")
    scored = labels.join(a, on="MVT_ID_mvt", how="inner").filter(pl.col(LABEL).is_not_null() & pl.col("oof").is_not_null())
    ids = set(scored["MVT_ID_mvt"].to_list())
    print(f"base scored set (label & a2o non-null): {len(ids):,}")
    for col, name in ARMS.items():
        o = pl.read_parquet(EXP / f"oof_{name}.parquet")
        dup = o.height - o["MVT_ID_mvt"].n_unique()
        j = labels.join(o, on="MVT_ID_mvt", how="inner").filter(pl.col(LABEL).is_not_null() & pl.col("oof").is_not_null())
        s = set(j["MVT_ID_mvt"].to_list())
        f = pl.read_parquet(EXP / f"oof_{name}_folds.parquet").filter(pl.col("pooled")).drop("pooled")
        fj = base_folds.join(f, on="MVT_ID_mvt", how="left", suffix="_arm")
        fj = fj.join(scored.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
        fold_mismatch = int((fj["fold"] != fj["fold_arm"]).fill_null(True).sum())
        print(
            f"  {col:<6} {name:<46} rows {o.height:>9,} dup {dup}  scored {len(s):>7,}  "
            f"same set {s == ids}  missing {len(ids - s)}  extra {len(s - ids)}  fold mismatches {fold_mismatch}"
        )


def build(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    r = build_oof(scratch, M4)  # writes base, mlp, lofo, members with blend_mlp's own code
    print(f"blend_mlp.build_oof rmse: {r['rmse']}")
    for name in ("hybrid_blend_l2_pa_opt_mlp", "hybrid_blend_l2_pa_opt_mlp_lofo"):
        new = pl.read_parquet(scratch / f"oof_{name}.parquet")
        old = pl.read_parquet(STORED / f"oof_{name}.parquet")
        j = new.join(old, on="MVT_ID_mvt", how="inner", suffix="_s").drop_nulls()
        print(f"  {name}: rebuilt vs stored rows {j.height:,}, max |diff| {float((j['oof'] - j['oof_s']).abs().max()):.3g}")
    d = frame()
    matched = pl.col(PROXY).is_not_null()
    mask = d[LABEL].is_not_null() & d["a2o"].is_not_null()
    m1 = 0.5 * pl.col("a2o") + 0.5 * pl.col("pao")
    base_m = (m1 + pl.col("p2o") + pl.col("c2") + pl.col("m4")) / 4.0
    hyb = {
        "rr_base": (base_m, pl.col("u")),
        "rr_R2_pa5": ((pl.col("a2o") + pl.col("pao") + pl.col("p2o") + pl.col("c2") + pl.col("m4")) / 5.0, pl.col("u")),
        "rr_R3_l2b_mean": (base_m, 0.5 * pl.col("u") + 0.5 * pl.col("u_l2b")),
        "rr_R4_osm_mean": (base_m, 0.5 * pl.col("u") + 0.5 * pl.col("u_osm")),
        "rr_R5_lt5": ((m1 + pl.col("p2o") + pl.col("c2") + pl.col("m4") + pl.col("lt")) / 5.0, pl.col("u")),
        "rr_R3s_l2b_only": (base_m, pl.col("u_l2b")),
        "rr_R4s_osm_only": (base_m, pl.col("u_osm")),
        "rr_R34_u3": (base_m, (pl.col("u") + pl.col("u_l2b") + pl.col("u_osm")) / 3.0),
        "rr_R25_pa_lt6": (
            (pl.col("a2o") + pl.col("pao") + pl.col("p2o") + pl.col("c2") + pl.col("m4") + pl.col("lt")) / 6.0,
            pl.col("u"),
        ),
    }
    y = d.filter(mask)[LABEL].to_numpy().astype(float)
    stored = pl.read_parquet(STORED / "oof_hybrid_blend_l2_pa_opt_mlp.parquet")
    for name, (mexpr, uexpr) in hyb.items():
        h = d.select("MVT_ID_mvt", pl.when(matched).then(mexpr).otherwise(uexpr).alias("oof"))
        nn = h.filter(mask)["oof"].null_count()
        if nn:
            raise AssertionError(f"{name}: {nn} scored rows null")
        h.write_parquet(scratch / f"oof_{name}.parquet")
        e = h.filter(mask)["oof"].to_numpy() - y
        print(f"  {name:<20} rmse {np.sqrt((e**2).mean()):.4f}")
        if name == "rr_base":
            j = h.join(stored, on="MVT_ID_mvt", how="inner", suffix="_s").drop_nulls()
            print(f"    rr_base vs stored base: rows {j.height:,}, max |diff| {float((j['oof'] - j['oof_s']).abs().max()):.3g}")
    # error correlation of lt with M1..M4 on matched rows
    mm = d.filter(mask & matched).with_columns(m1.alias("m1"))
    ym = mm[LABEL].to_numpy().astype(float)
    E = np.column_stack([mm[c].to_numpy().astype(float) - ym for c in ("m1", "p2o", "c2", "m4", "lt")])
    print("matched-lane error RMSE m1 p2o c2 m4 lt:", np.sqrt((E**2).mean(0)).round(3).tolist())
    print("error corr (rows/cols m1 p2o c2 m4 lt):")
    print(np.corrcoef(E.T).round(4))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["align", "build"])
    ap.add_argument("--scratch", type=Path, required=False)
    a = ap.parse_args()
    if a.what == "align":
        align()
    else:
        if a.scratch is None:
            raise SystemExit("--scratch required")
        build(a.scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
