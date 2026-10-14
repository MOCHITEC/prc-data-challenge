"""plan_standmap.md section 2-3: the appear residual (4 months, base a2) with the own-stand table from OSM only (osm),
OSM else the 2026 ADS-B map (fill), map else OSM (mapfirst), and the control with the map positions permuted (rand).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_standmap.py

Same rows, folds, inner early-stopping split and parameters as cv_months.py. Writes cv_standmap_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months as CM  # noqa: E402  (sets up CV with --base a2)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

AP, CV, L, _paired = CM.AP, CM.CV, CM.L, CM._paired
ARMS = ("osm", "fill", "mapfirst", "rand")


def load(variant: str) -> pl.DataFrame:
    D = pl.concat([CV.load_labelled().select(CM.KEEP), CM.load_novdec()])
    F = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"feat_{k}.parquet")) for k in ("2025", "2025nd")],
                  how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"))
    T = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"appear_{variant}_{k}.parquet")) for k in ("2025", "2025nd")])
    return D.drop([c for c in AP.COLS if c in D.columns]).join(T, on="MVT_ID_mvt", how="left").sort("MVT_ID_mvt")


def differs(a: pl.DataFrame, b: pl.DataFrame) -> np.ndarray:
    """Rows whose own-stand-based appear columns differ between two variants (null-aware, 0.5 m / 0.5 s tolerance)."""
    out = np.zeros(a.height, bool)
    for c in ("own_ok", "d_dep1_own", "d_stand_own", "n_in_own", "t_leave_own_lo", "t_appear_adj"):
        x, y = a[c].cast(pl.Float64).to_numpy(), b[c].cast(pl.Float64).to_numpy()
        nx, ny = np.isnan(x), np.isnan(y)
        out |= (nx != ny) | (~nx & ~ny & (np.abs(x - y) > 0.5))
    return out


def main() -> None:
    frames = {v: load(v) for v in ARMS}
    ids = frames["osm"]["MVT_ID_mvt"]
    assert all((f["MVT_ID_mvt"] == ids).all() for f in frames.values())
    D = frames["osm"]
    days = sorted(D["day"].unique().to_list())
    fold_of = CM.folds(days)
    extra = [pl.col("day").replace_strict(fold_of).alias("cvfold"),
             pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx")]
    frames = {v: f.with_columns(extra) for v, f in frames.items()}
    masks = {v: differs(frames["osm"], frames[v]) for v in ARMS if v != "osm"}
    print({v: int(m.sum()) for v, m in masks.items()}, "rows differ from osm", flush=True)
    preds, res = {}, {"final_rounds": {}, "features": {}}
    for v in ARMS:
        print(f"arm {v}", flush=True)
        c, its, feats = CM.run_arm(frames[v], True, fold_of, days)
        preds[v] = frames[v]["b"].to_numpy() + c
        res["final_rounds"][v] = int(np.median(its))
        res["features"][v] = feats
    y, di, nd = D["y"].to_numpy(), frames["osm"]["day_idx"].to_numpy(), len(days)
    apt = D["ADEP_mvt"].to_numpy()
    out, verdict = {}, {}
    for v in ("fill", "mapfirst"):
        m = masks[v]
        r = {}
        for ref in ("osm", "rand"):
            for label, sel in (("map rows", m), ("all rows", np.ones(len(y), bool))):
                dd, lo, hi = _paired(y[sel], preds[ref][sel], preds[v][sel], di[sel], nd)
                r[f"vs {ref} {label}"] = {"n": int(sel.sum()), "d": dd, "lo": lo, "hi": hi}
                print(f"  {v:9s} vs {ref:4s} {label:9s} n={int(sel.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        for a in ("EDDF", "EDDM", "LEBL", "LSZH", "EGLL", "EHAM", "LEMD", "LIRF"):
            sel = m & (apt == a)
            if sel.sum() >= 200:
                dd, lo, hi = _paired(y[sel], preds["osm"][sel], preds[v][sel], di[sel], nd)
                print(f"  {v:9s} vs osm  {a} map rows n={int(sel.sum()):>6,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        ok = (r["vs osm map rows"]["hi"] < 0 and r["vs rand map rows"]["hi"] < 0 and not r["vs osm all rows"]["lo"] > 0)
        verdict[v] = ok
        out[v] = r
    passed = [v for v in ("fill", "mapfirst") if verdict[v]]
    choice = min(passed, key=lambda v: out[v]["vs osm all rows"]["d"]) if passed else None
    print(f"\nDECISION (plan_standmap.md section 3): fill {verdict['fill']}, mapfirst {verdict['mapfirst']} -> "
          f"{'ADOPT ' + choice if choice else 'REJECT'}")
    res.update(results=out, verdict=verdict, choice=choice)
    json.dump(res, open(os.path.join(L.OUT, "cv_standmap_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
