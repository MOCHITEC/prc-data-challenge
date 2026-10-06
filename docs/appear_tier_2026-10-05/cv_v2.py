"""plan_standmap.md section 4: the appear tier (fill stand table) on v38's V2 residual settings.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/cv_v2.py

Three arms, all with adsb_learn_cv.PARAMS + V2 (num_leaves 255, min_data_in_leaf 300, feature_fraction 0.7), on the
4-month a2 rows and folds of cv_months.py: plain_v2, appear_fill_v2, shuffle_fill_v2. Writes cv_months_v2fill_a2.json
in cv_months_a2.json's shape (final_rounds / features under plain4 and appear4) for apply_delta.py --params v2."""

import json
import os
import sys

sys.argv = [sys.argv[0], "--base", "a2"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_months as CM  # noqa: E402
import cv_standmap as CS  # noqa: E402

import numpy as np  # noqa: E402

V2 = dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7)


def main() -> None:
    CM.CV.PARAMS = {**CM.CV.PARAMS, **V2}
    print("params:", CM.CV.PARAMS, flush=True)
    D = CS.load("fill")
    days = sorted(D["day"].unique().to_list())
    fold_of = CM.folds(days)
    import polars as pl
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    arms, res = {}, {"final_rounds": {}, "features": {}, "params": CM.CV.PARAMS}
    for name, frame, appear in (("plain4", D, False), ("shuffle4", CM.shuffled(D), True), ("appear4", D, True)):
        print(f"arm {name} (V2)", flush=True)
        c, its, feats = CM.run_arm(frame, appear, fold_of, days)
        arms[name] = D["b"].to_numpy() + c
        res["final_rounds"][name] = int(np.median(its))
        res["features"][name] = feats
    y, di, nd = D["y"].to_numpy(), D["day_idx"].to_numpy(), len(days)
    mo, apt = D["day"].str.slice(0, 7).to_numpy(), D["ADEP_mvt"].to_numpy()
    out = {}
    for a, b_ in (("plain4", "appear4"), ("shuffle4", "appear4"), ("plain4", "shuffle4")):
        key = f"{b_} vs {a}"
        out[key] = {}
        for label, m in [("all rows", np.ones(len(y), bool))] + [(x, mo == x) for x in CM.MONTHS] + \
                        [(x, apt == x) for x in sorted(set(apt))]:
            dd, lo, hi = CM._paired(y[m], arms[a][m], arms[b_][m], di[m], nd)
            out[key][label] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:22s} {label:>9s} n={int(m.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    ok = out["appear4 vs plain4"]["all rows"]["hi"] < 0 and out["appear4 vs shuffle4"]["all rows"]["hi"] < 0
    verdict = "ADOPT" if ok else "REJECT"
    print(f"\nDECISION (plan_standmap.md section 4): -> {verdict}")
    res.update(results=out, verdict=verdict)
    json.dump(res, open(os.path.join(CM.L.OUT, "cv_months_v2fill_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
