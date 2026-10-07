"""adsbaudit step 3b: leave-MONTH-out vs leave-days-out for the learned ADS-B residual and its apron arm (local).

Every residual CV so far holds out random DAYS inside the training months; the 2026 application is a different year.
Month-level structure in y - b (the OOF base never saw that month) is shared by the training and test days of a month,
and the coverage features (fix counts, gaps) act as month signatures, so leave-days-out may credit the residual, and
especially a fine location code such as the apron, with month-specific corrections that cannot transfer.
Harness = rescat (7 months, V2 params, same rows and features). For each held-out month H, arms A0 (production
features) and A3 (+ cat_apron) are fitted on the ADS-B rows of the other 6 months with the rescat final rounds
(A0 460, A3 441) and predict H. Compared, on the same rows, with the rescat leave-days-out c (agents/rescat/cv_rows).
Gain = mean over all labelled rows of H of (y-b)^2 - (y-b-c)^2, c applied on ground-fix rows except LTFM/LIRF.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsbaudit_lmo.py 2025-01,2025-07,2025-11
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

HOLD = sys.argv[1].split(",")
sys.path.insert(0, "analysis")
import local_adsb_rescat as R  # noqa: E402

L, CV = R.L, R.CV
OUT = os.path.join(L.SCR, "agents", "adsbaudit")
P = {**R.PRM, "num_threads": 2}
ROUNDS = {"A0": 460, "A3": 441}


def main():
    D, days, fold_of, inner = R.load()
    C, V = R.code_tables()
    D = R.add_codes(D, C, V)
    X, feats, cats = L.design(D, "b")
    allf = feats + ["cat_apron"]
    Xn = X.select(allf).to_numpy().astype(np.float32)
    del X
    has = D["has"].to_numpy()
    gnd = D["ground_fix"].fill_null(False).to_numpy() & ~D["ADEP_mvt"].is_in(["LTFM", "LIRF"]).to_numpy()
    y, b = D["y"].to_numpy(), D["b"].to_numpy()
    r = np.clip(y - b, -CV.CLIP_Y, CV.CLIP_Y).astype(np.float32)
    ym, apt = D["ym"].to_numpy(), D["ADEP_mvt"].to_numpy()
    di, nd = D["day_idx"].to_numpy(), len(days)
    ldo = D.select("MVT_ID_mvt").join(pl.read_parquet(os.path.join(R.RES, "cv_rows.parquet")).select("MVT_ID_mvt", "c_A0", "c_A3"),
                                      on="MVT_ID_mvt", how="left")
    c_ldo = {a: ldo[f"c_{a}"].fill_null(0.0).to_numpy() for a in ("A0", "A3")}
    res = {}
    e0 = y - b
    for H in HOLD:
        dst = os.path.join(OUT, f"lmo_{H}.npz")
        te = ym == H
        if not os.path.exists(dst):
            tr = has & ~te
            out = {}
            for a, fn in (("A0", feats), ("A3", allf)):
                idx = [allf.index(f) for f in fn]
                ds = lgb.Dataset(Xn[np.ix_(tr, idx)], r[tr], feature_name=fn, categorical_feature=[c for c in cats + ["cat_apron"] if c in fn],
                                 free_raw_data=True)
                m = lgb.train(P, ds, num_boost_round=ROUNDS[a])
                out[a] = np.clip(m.predict(Xn[np.ix_(te, idx)]), -CV.CLIP_C, CV.CLIP_C)
                del ds, m
                print(f"{H} {a} fitted on {int(tr.sum())} rows", flush=True)
            np.savez(dst, **out)
        z = np.load(dst)
        g = {}
        for lab, cc in (("LMO A0", z["A0"]), ("LMO A3", z["A3"]), ("LDO A0", c_ldo["A0"][te]), ("LDO A3", c_ldo["A3"][te])):
            cc = np.where(gnd[te], cc, 0.0)
            g[lab] = e0[te] ** 2 - (e0[te] - cc) ** 2
        rec = {}
        for k, v in g.items():
            rec[k] = round(float(v.mean()), 1)
        for lab, d in (("LMO A3-A0", g["LMO A3"] - g["LMO A0"]), ("LDO A3-A0", g["LDO A3"] - g["LDO A0"]),
                       ("A0 LDO-LMO", g["LDO A0"] - g["LMO A0"])):
            gg, lo, hi = CV.boot(d, di[te], nd)
            rec[lab] = (round(gg, 1), round(lo, 1), round(hi, 1))
        dl = np.where(gnd[te], z["A3"] - z["A0"], 0.0)
        dd = np.where(gnd[te], c_ldo["A3"][te] - c_ldo["A0"][te], 0.0)
        rec["sum d^2/N LMO, LDO"] = (round(float((dl ** 2).mean()), 1), round(float((dd ** 2).mean()), 1))
        rec["corr(d_LMO, d_LDO)"] = round(float(np.corrcoef(dl[gnd[te]], dd[gnd[te]])[0, 1]), 3)
        per = {}
        for a_ in sorted(set(apt[te])):
            m_ = apt[te] == a_
            per[a_] = dict(n=int(m_.sum()), lmo=round(float((g["LMO A3"] - g["LMO A0"])[m_].mean()), 1),
                           ldo=round(float((g["LDO A3"] - g["LDO A0"])[m_].mean()), 1),
                           a0_lmo=round(float(g["LMO A0"][m_].mean()), 1), a0_ldo=round(float(g["LDO A0"][m_].mean()), 1))
        rec["per_airport"] = per
        res[H] = rec
        print(H, json.dumps(rec, indent=0), flush=True)
    json.dump(res, open(os.path.join(OUT, "lmo_" + "_".join(HOLD) + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
