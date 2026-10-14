"""plan.md section 6: OPDI stand-exit columns on the current ADS-B residual (4 months, a2, V2 settings, appear fill).

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/opdi_events_2026-10-05/cv_opdi4.py

Arms cur / opdi / opdishuf on the rows and folds of docs/appear_tier_2026-10-05/cv_months.py; OPDI columns from
scripts/adsb_opdi.py, kept at LSZH and EDDF. Writes cv_opdi4_a2.json."""

import json
import os
import sys

sys.argv = [sys.argv[0], "--base", "a2"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "appear_tier_2026-10-05"))
import cv_field as CF  # noqa: E402  (cv_months / cv_standmap set up with --base a2)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import adsb_opdi as O  # noqa: E402

CM, CS, CV, L, AP = CF.CM, CF.CS, CF.CV, CF.L, CF.AP
O.APTS = ("LSZH", "EDDF")


def arm_frame(D: pl.DataFrame, mode: str) -> pl.DataFrame:
    O.MODE = mode
    try:
        return O.add(D, CV.ROOT)
    finally:
        O.MODE = ""


def run_arm(D: pl.DataFrame, with_opdi: bool, fold_of: dict, days: list):
    orig = CF.FD.design
    if with_opdi:
        CF.FD.design = lambda X, feats, base: O.design(X, feats, base)
    try:
        return CF.run_arm(D, with_opdi, fold_of, days)
    finally:
        CF.FD.design = orig


def main() -> None:
    CV.PARAMS = {**CV.PARAMS, **CF.V2}
    D = CS.load("fill")
    days = sorted(D["day"].unique().to_list())
    fold_of = CM.folds(days)
    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    frames = {"cur": D, "opdi": arm_frame(D, "opdi"), "opdishuf": arm_frame(D, "opdishuf")}
    assert all((f["MVT_ID_mvt"] == D["MVT_ID_mvt"]).all() for f in frames.values())
    print(f"rows {D.height:,}; OPDI rows {int(frames['opdi']['opdi_exit_est_s'].is_not_null().sum()):,} "
          f"({dict(frames['opdi'].filter(pl.col('opdi_exit_est_s').is_not_null()).group_by('ADEP_mvt').len().iter_rows())})",
          flush=True)
    preds, res = {}, {"final_rounds": {}, "features": {}, "params": CV.PARAMS, "apts": O.APTS}
    for name in ("cur", "opdishuf", "opdi"):
        print(f"arm {name}", flush=True)
        c, its, feats = run_arm(frames[name], name != "cur", fold_of, days)
        preds[name] = D["b"].to_numpy() + c
        res["final_rounds"][name] = int(np.median(its))
        res["features"][name] = feats
    y, di, nd = D["y"].to_numpy(), D["day_idx"].to_numpy(), len(days)
    apt = D["ADEP_mvt"].to_numpy()
    has_o = frames["opdi"]["opdi_exit_est_s"].is_not_null().to_numpy()
    out = {}
    for a, b_ in (("cur", "opdi"), ("opdishuf", "opdi"), ("cur", "opdishuf")):
        key = f"{b_} vs {a}"
        out[key] = {}
        for label, m in (("all rows", np.ones(len(y), bool)), ("with OPDI exit", has_o), ("LSZH", apt == "LSZH"),
                         ("EDDF", apt == "EDDF")):
            dd, lo, hi = CM._paired(y[m], preds[a][m], preds[b_][m], di[m], nd)
            out[key][label] = {"n": int(m.sum()), "d_rmse": dd, "lo": lo, "hi": hi}
            print(f"  {key:18s} {label:>15s} n={int(m.sum()):>7,}  {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    ok = out["opdi vs cur"]["all rows"]["hi"] < 0 and out["opdi vs opdishuf"]["all rows"]["hi"] < 0
    verdict = "ADOPT" if ok else "REJECT"
    print(f"\nDECISION (docs/opdi_events_2026-10-05/plan.md section 6): -> {verdict}")
    res.update(results=out, verdict=verdict)
    json.dump(res, open(os.path.join(L.OUT, "cv_opdi4_a2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
