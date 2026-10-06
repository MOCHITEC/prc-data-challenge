"""Member disagreement and stacking of the base members, validated leave-one-month-out (local, read-only).

Part A  Jan+Jul 2025 (the only months with all three matched-lane members): members
        a = main_flseq_osm_nolirf (anchored LGB), p = ..._plainy, c = ..._cb (CatBoost), f = main_flseq (unmatched-lane
        model, also available on matched rows). Base b = mean(a,p,c) on the matched lane, f on the unmatched lane
        (== production base in cv_rows_months7_v2). ADS-B residual c_C from cv_rows_months7_v2 (trained on y-b).
  A1 disagreement vs error (spread deciles, sign regression, per-member MSE on high-spread rows by segment)
  A2 linear stacking (weights summing to 1, ridge toward the plain mean; per airport and per airport x lane x
     delay band; optional per-group intercept), fit on one month, tested on the other
  A3 small LightGBM stacker on y-b (member deviations + airport, lane, hour, MVT-AOBT_3, MVT-SCHED), LOMO
Part B  all 12 months of 2025 with the two members that exist everywhere (a, f): base b2 = a (matched) / f (unmatched);
        linear and LGB stacking, 12-fold leave-one-month-out.
Gain = (SSE_base - SSE_new) / rows in that month (s^2; ~480 s^2 ~ 1 s on the board).

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_member_stack.py
"""
from __future__ import annotations

import glob
import os

import lightgbm as lgb
import numpy as np
import polars as pl

SCR = os.environ["SCRATCH"]
OUT = os.path.join(SCR, "agents", "ens")
os.makedirs(OUT, exist_ok=True)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_tbl_cols(20)


def oof(stem: str, tag: str) -> pl.DataFrame:
    s = f"{stem}_{tag}" if tag else stem
    f = [g for g in glob.glob(f"experiments/oof_{s}-*_folds.parquet") if "_jj-" not in g or tag == "jj"]
    f = [g for g in f if os.path.basename(g).startswith(f"oof_{s}-")]
    assert len(f) == 1, (s, f)
    fo = pl.read_parquet(f[0]).filter("pooled").select("MVT_ID_mvt")
    return pl.read_parquet(f[0].replace("_folds", "")).join(fo, on="MVT_ID_mvt", how="semi")


def feats() -> pl.DataFrame:
    return pl.read_parquet("data/work_flseq/feat_train.parquet",
                           columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3",
                                    "mvt_minus_sched", "aobt3_minus_sched", "hour_local"]).with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.String).alias("day"),
        pl.col("mvt_minus_aobt3").is_not_null().alias("matched"),
        pl.when(pl.col("aobt3_minus_sched").is_null()).then(pl.lit("na"))
        .when(pl.col("aobt3_minus_sched") < 300).then(pl.lit("d<5m"))
        .when(pl.col("aobt3_minus_sched") < 1200).then(pl.lit("d5-20m")).otherwise(pl.lit("d>20m")).alias("dband"))


def gain_table(D: pl.DataFrame, cols: dict[str, np.ndarray], base: np.ndarray, extra: np.ndarray | None = None) -> pl.DataFrame:
    """per-month gain (SSE reduction / rows of the month) of each candidate vs base; extra is added to both."""
    y, ym = D["y"].to_numpy(), D["ym"].to_numpy()
    add = 0.0 if extra is None else extra
    rows = []
    for m in sorted(set(ym)) + ["ALL"]:
        s = np.ones(len(y), bool) if m == "ALL" else ym == m
        e0 = (y - base - add)[s]
        r = {"month": m, "n": int(s.sum()), "rmse_base": round(float(np.sqrt((e0 ** 2).mean())), 2)}
        for k, v in cols.items():
            r[k] = round(float((e0 ** 2 - (y - v - add)[s] ** 2).mean()), 1)
        rows.append(r)
    return pl.DataFrame(rows)


def fit_w(dev: np.ndarray, e: np.ndarray, lam: float, icpt: bool) -> np.ndarray:
    """e ~ dev @ beta (+ intercept), ridge lam*n on beta (shrink to the plain mean); dev columns are m_k - mean."""
    X = np.column_stack([dev, np.ones(len(e))]) if icpt else dev
    P = np.eye(X.shape[1]) * lam * len(e)
    if icpt:
        P[-1, -1] = 1e-6 * len(e)
    return np.linalg.solve(X.T @ X + P, X.T @ e)


def lin_stack(D: pl.DataFrame, M: np.ndarray, base: np.ndarray, grp: np.ndarray, lam: float, icpt: bool,
              clip: float = 1800.0) -> np.ndarray:
    """leave-one-month-out per-group linear stack; deviation basis = members minus base on rows where members apply."""
    y, ym = D["y"].to_numpy(), D["ym"].to_numpy()
    dev = M - base[:, None]
    e = np.clip(y - base, -clip, clip)
    out = base.copy()
    for m in sorted(set(ym)):
        te, tr = ym == m, ym != m
        for g in np.unique(grp):
            gt, gr = te & (grp == g), tr & (grp == g)
            if gt.sum() == 0 or gr.sum() < 500:
                continue
            beta = fit_w(dev[gr], e[gr], lam, icpt)
            X = np.column_stack([dev[gt], np.ones(gt.sum())]) if icpt else dev[gt]
            out[gt] = base[gt] + X @ beta
    return out


LGB_P = dict(objective="regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=2000, lambda_l2=50.0,
             feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, max_bin=127, num_threads=2, verbose=-1, seed=7)


def lgb_stack(D: pl.DataFrame, X: np.ndarray, names: list[str], cats: list[str], base: np.ndarray,
              clip_y: float = 1500.0, clip_c: float = 400.0) -> tuple[np.ndarray, list[int]]:
    y, ym, day = D["y"].to_numpy(), D["ym"].to_numpy(), D["day"].to_numpy()
    r = np.clip(y - base, -clip_y, clip_y)
    out, its = base.copy(), []
    rng = np.random.default_rng(11)
    days = np.unique(day)
    vday = set(days[rng.random(len(days)) < 0.2])
    isv = np.isin(day, list(vday))
    for m in sorted(set(ym)):
        te, tr = ym == m, ym != m
        dtr = lgb.Dataset(X[tr & ~isv], r[tr & ~isv], feature_name=names, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(X[tr & isv], r[tr & isv], reference=dtr)
        b = lgb.train(LGB_P, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        its.append(int(b.best_iteration))
        out[te] = base[te] + np.clip(b.predict(X[te], num_iteration=b.best_iteration), -clip_c, clip_c)
    return out, its


def design(D: pl.DataFrame, M: np.ndarray, mnames: list[str], base: np.ndarray, members: bool):
    apt = D["ADEP_mvt"].cast(pl.Categorical).to_physical().to_numpy().astype(float)
    ctx = [apt, D["matched"].to_numpy().astype(float), D["hour_local"].to_numpy().astype(float),
           D["mvt_minus_aobt3"].fill_null(np.nan).to_numpy().astype(float),
           D["mvt_minus_sched"].to_numpy().astype(float), base]
    names = ["apt", "matched", "hour", "mvt_m_aobt3", "mvt_m_sched", "base"]
    if members:
        dev = M - base[:, None]
        ctx += [dev[:, i] for i in range(dev.shape[1])] + [M.std(axis=1)]
        names += [f"d_{n}" for n in mnames] + ["spread"]
    return np.column_stack(ctx), names, ["apt"]


def part_a() -> None:
    print("=" * 30, "PART A: Jan+Jul 2025, members a,p,c (+f)", "=" * 30)
    D = feats()
    for k, s in (("a", "main_flseq_osm_nolirf"), ("p", "main_flseq_osm_nolirf_plainy"),
                 ("c", "main_flseq_osm_nolirf_cb"), ("f", "main_flseq")):
        D = D.join(oof(s, "").rename({"oof": k}), on="MVT_ID_mvt", how="inner" if k == "a" else "left")
    R = pl.read_parquet(os.path.join(SCR, "learn", "cv_rows_months7_v2.parquet")).select("MVT_ID_mvt", "b", "has", "c_C")
    D = D.join(R, on="MVT_ID_mvt", how="left").sort("MVT_ID_mvt")
    D = D.with_columns(pl.when("matched").then((pl.col("a") + pl.col("p") + pl.col("c")) / 3).otherwise("f").alias("b0"))
    assert D["b"].null_count() == 0 and float((D["b"] - D["b0"]).abs().max()) < 1e-6
    print("rows", D.height, D.group_by("ym", "matched").len().sort("ym", "matched").rows())
    y, b, cC = D["y"].to_numpy(), D["b"].to_numpy(), D["c_C"].to_numpy()
    mt = D["matched"].to_numpy()
    M3 = D.select("a", "p", "c").to_numpy()
    M4 = D.select("a", "p", "c", "f").to_numpy()
    e = y - b

    # ---- A1 disagreement vs error (matched lane) ----
    Dm = D.filter("matched").with_columns(
        pl.concat_list("a", "p", "c").list.std().alias("sd3"),
        (pl.max_horizontal("a", "p", "c") - pl.min_horizontal("a", "p", "c")).alias("rng3"),
        (pl.col("f") - pl.col("b")).alias("f_minus_b"),
        (pl.col("y") - pl.col("b")).alias("e"), (pl.col("y") - pl.col("b") - pl.col("c_C")).alias("ec"))
    Dm = Dm.with_columns(pl.col("sd3").qcut(10, labels=[f"q{i}" for i in range(10)]).alias("sdq"))
    print("\nA1 matched lane: error vs member spread (sd of a,p,c) deciles")
    print(Dm.group_by("sdq").agg(pl.len(), pl.col("sd3").mean().round(1).alias("sd3"), pl.col("rng3").mean().round(1),
                                 (pl.col("e") ** 2).mean().sqrt().round(1).alias("rmse_b"),
                                 (pl.col("ec") ** 2).mean().sqrt().round(1).alias("rmse_b+c"),
                                 pl.col("e").mean().round(1).alias("bias"), pl.col("e").abs().median().round(1).alias("med|e|"),
                                 *[((pl.col("y") - pl.col(k)) ** 2).mean().sqrt().round(1).alias(f"rmse_{k}") for k in "apcf"],
                                 ).sort("sdq"))
    for m in ("2025-01", "2025-07"):
        x = Dm.filter(pl.col("ym") == m)
        print(m, "spearman(|e|, sd3) =", round(x.select(pl.corr(pl.col("e").abs(), "sd3", method="spearman")).item(), 3),
              "| spearman(|e+c|, sd3) =", round(x.select(pl.corr(pl.col("ec").abs(), "sd3", method="spearman")).item(), 3))
    # sign: regress e on member deviations (beta>0 for member k means truth lies towards k)
    print("\nA1 sign: OLS e = sum_k beta_k (m_k - b) [+f-b], per month and spread tercile (beta_k>0: truth leans to k)")
    rows = []
    Dm = Dm.with_columns(pl.col("sd3").qcut([0.5, 0.9], labels=["lo50", "mid40", "top10"]).alias("sdt"))
    for m in ("2025-01", "2025-07"):
        for t in ("lo50", "mid40", "top10", "all"):
            x = Dm.filter((pl.col("ym") == m) & ((pl.col("sdt") == t) | (t == "all")))
            dv = x.select(*[pl.col(k) - pl.col("b") for k in "apcf"]).to_numpy()
            # a+p+c deviations sum to 0 -> drop c (beta_c implied = 0 reference); report a,p relative to c and f
            X = dv[:, [0, 1, 3]]
            beta = np.linalg.lstsq(X, np.clip(x["e"].to_numpy(), -1800, 1800), rcond=None)[0]
            rows.append({"month": m, "spread": t, "n": x.height, "b_a(vs c)": round(beta[0], 3), "b_p(vs c)": round(beta[1], 3),
                         "b_f": round(beta[2], 3),
                         **{f"win_{k}": round(float(((x["y"] - x[k]).abs() < (x["y"] - x["b"]).abs()).mean()), 3) for k in "apcf"}})
    print(pl.DataFrame(rows))
    print("\nA1 top-10% spread rows: per-member MSE minus MSE(b) (s^2; negative = member better than mean), by segment")
    T = Dm.filter(pl.col("sdt") == "top10")
    for seg in ("ADEP_mvt", "dband", "has", "ym"):
        print(T.group_by(seg).agg(pl.len(), (pl.col("e") ** 2).mean().sqrt().round(1).alias("rmse_b"),
                                  *[(((pl.col("y") - pl.col(k)) ** 2) - pl.col("e") ** 2).mean().round(0).alias(f"dMSE_{k}") for k in "apcf"],
                                  *[(((pl.col("y") - pl.col(k)) ** 2) - pl.col("e") ** 2).filter(pl.col("ym") == "2025-01").mean().round(0).alias(f"jan_{k}") for k in "apcf"],
                                  *[(((pl.col("y") - pl.col(k)) ** 2) - pl.col("e") ** 2).filter(pl.col("ym") == "2025-07").mean().round(0).alias(f"jul_{k}") for k in "apcf"],
                                  ).sort(seg))

    # ---- A2 linear stacking ----
    apt = D["ADEP_mvt"].to_numpy()
    seg = (D["ADEP_mvt"] + "|" + D["matched"].cast(pl.String) + "|" + D["dband"]).to_numpy()
    lane = np.where(mt, "M", "U")
    cand = {}
    # matched lane only stacks a,p,c (,f); unmatched lane base is f, members a,p,c available too
    for lam in (1e-6, 0.05):
        for nm, M in (("3m", M3), ("4m", M4)):
            for gname, g in (("glob", lane), ("apt", np.char.add(apt.astype(str), lane)), ("aptseg", seg)):
                for icpt in (False, True):
                    cand[f"{nm}_{gname}_l{lam}{'_i' if icpt else ''}"] = lin_stack(D, M, b, g, lam, icpt)
    print("\nA2 linear stack, leave-one-month-out (fit Jan->test Jul and v.v.); gain s^2 per month row (base = production mean)")
    G = gain_table(D, cand, b)
    print(G.transpose(include_header=True, column_names=G["month"].to_list()).slice(3))
    print("\nA2 same, after adding the ADS-B residual c_C (trained on y-b of the plain mean) to both")
    Gc = gain_table(D, cand, b, cC)
    print(Gc.transpose(include_header=True, column_names=Gc["month"].to_list()).slice(3))
    # print fitted global weights on all data for interpretation
    for nm, M in (("3m", M3), ("4m", M4)):
        s = mt
        beta = fit_w(M[s] - b[s, None], np.clip(e[s], -1800, 1800), 0.0, False)
        print(f"global matched-lane deviation coefs ({nm}, a,p,c[,f]; weight_k = 1/3 + beta_k - mean(beta_abc)...):", np.round(beta, 3))
        for m in ("2025-01", "2025-07"):
            s2 = mt & (D["ym"].to_numpy() == m)
            print("   ", m, np.round(fit_w(M[s2] - b[s2, None], np.clip(e[s2], -1800, 1800), 1e-6, False), 3))

    # ---- A3 LightGBM stacker ----
    lg = {}
    for members in (False, True):
        X, names, cats = design(D, M4, list("apcf"), b, members)
        p_, its = lgb_stack(D, X, names, cats, b)
        lg["lgb_" + ("members+ctx" if members else "ctx_only")] = p_
        print("A3 lgb", "members" if members else "ctx-only", "best iters", its)
    print("\nA3 LGB stacker on y-b, LOMO; gain s^2 per month row")
    G = gain_table(D, lg, b)
    print(G)
    print("A3 after adding c_C to both")
    print(gain_table(D, lg, b, cC))
    pl.DataFrame({"MVT_ID_mvt": D["MVT_ID_mvt"], "ym": D["ym"], "y": y, "b": b, "c_C": cC,
                  **{k: v for k, v in {**cand, **lg}.items() if k in ("4m_apt_l0.05", "3m_apt_l0.05", "lgb_members+ctx", "lgb_ctx_only")}}
                 ).write_parquet(os.path.join(OUT, "stack_partA_rows.parquet"))


def part_b() -> None:
    print("\n" + "=" * 30, "PART B: 12 months 2025, members a (anchored LGB) and f (main_flseq)", "=" * 30)
    D = feats()
    parts = []
    for tag in ("jj", "fmj", "aso", "fw"):
        x = oof("main_flseq_osm_nolirf", tag).rename({"oof": "a"}).join(oof("main_flseq", tag).rename({"oof": "f"}),
                                                                          on="MVT_ID_mvt", how="inner")
        parts.append(x)
    D = D.join(pl.concat(parts), on="MVT_ID_mvt", how="inner").sort("MVT_ID_mvt")
    print("rows", D.height, D.group_by("ym").len().sort("ym").rows())
    D = D.with_columns(pl.when("matched").then("a").otherwise("f").alias("b2"))
    y, b2 = D["y"].to_numpy(), D["b2"].to_numpy()
    mt = D["matched"].to_numpy()
    M = D.select("a", "f").to_numpy()
    # deviation basis relative to b2: on matched rows d_f = f - a is the only informative column
    apt = D["ADEP_mvt"].to_numpy()
    lane = np.where(mt, "M", "U")
    seg = (D["ADEP_mvt"] + "|" + D["matched"].cast(pl.String) + "|" + D["dband"]).to_numpy()
    cand = {}
    dev_only = M[:, [1]] * mt[:, None] + M[:, [0]] * (~mt)[:, None]  # the "other" member
    for gname, g in (("glob", lane), ("apt", np.char.add(apt.astype(str), lane)), ("aptseg", seg)):
        for icpt in (False, True):
            cand[f"af_{gname}{'_i' if icpt else ''}"] = lin_stack(D, dev_only, b2, g, 1e-6, icpt)
    print("B linear stack (b2 + w*(other - b2)), 12-fold LOMO; gain s^2 per month row")
    print(gain_table(D, cand, b2))
    s = mt
    print("global matched-lane weight on f:", np.round(fit_w((M[s, 1] - b2[s])[:, None], np.clip((y - b2)[s], -1800, 1800), 1e-6, False), 3))
    lg = {}
    for members in (False, True):
        X, names, cats = design(D, M, ["a", "f"], b2, members)
        p_, its = lgb_stack(D, X, names, cats, b2)
        lg["lgb_" + ("members+ctx" if members else "ctx_only")] = p_
        print("B lgb", "members" if members else "ctx-only", "best iters", its, flush=True)
    print("B LGB stacker on y-b2, 12-fold LOMO; gain s^2 per month row")
    print(gain_table(D, lg, b2))
    # per-airport weight stability for the af_apt stack: weight fitted per month separately
    rows = []
    for a in sorted(set(apt)):
        r = {"apt": a}
        for m in sorted(set(D["ym"].to_numpy())):
            ss = mt & (apt == a) & (D["ym"].to_numpy() == m)
            r[m[-2:]] = round(float(fit_w((M[ss, 1] - b2[ss])[:, None], np.clip((y - b2)[ss], -1800, 1800), 1e-6, False)[0]), 2)
        rows.append(r)
    print("B per-airport per-month fitted weight on f (matched lane)")
    print(pl.DataFrame(rows))


def part_c() -> None:
    """1-parameter blend b' = b + w (f - b) on the matched lane, on the 7-month stack rows (b, c_C of cv_rows_months7_v2;
    b = mean(a,p,c) in Jan/Jul, = a in Aug-Dec). w global or per airport (shrunk to global with k rows), LOMO 7 folds.
    Gains split by ADS-B status: on has=False rows c=0, so the after-c gain there is free of the c/b interaction."""
    print("\n" + "=" * 30, "PART C: 7-month blend toward main_flseq on the matched lane, before/after c_C", "=" * 30)
    R = pl.read_parquet(os.path.join(SCR, "learn", "cv_rows_months7_v2.parquet"))
    f = pl.concat([oof("main_flseq", t) for t in ("jj", "aso", "fw")]).rename({"oof": "f"})
    D = (R.join(f, on="MVT_ID_mvt", how="left")
         .join(feats().select("MVT_ID_mvt", "matched", "day"), on="MVT_ID_mvt", how="left").sort("MVT_ID_mvt"))
    assert D["f"].null_count() == 0
    y, b, c, fv = (D[k].to_numpy() for k in ("y", "b", "c_C", "f"))
    mt, has, ym, apt = D["matched"].to_numpy(), D["has"].to_numpy(), D["ym"].to_numpy(), D["ADEP_mvt"].to_numpy()
    d = np.where(mt, fv - b, 0.0)
    e = np.clip(y - b, -1800, 1800)
    ec = np.clip(y - b - c, -1800, 1800)
    for target_name, tgt in (("fit on y-b", e), ("fit on y-b-c", ec)):
        out = {"glob": np.zeros(len(y)), "apt_k20000": np.zeros(len(y))}
        ws = []
        for m in sorted(set(ym)):
            te, tr = ym == m, (ym != m) & mt
            wg = float((d[tr] * tgt[tr]).sum() / (d[tr] ** 2).sum())
            out["glob"][te] = wg * d[te]
            ws.append(round(wg, 3))
            for a in set(apt):
                ta, ra = te & (apt == a), tr & (apt == a)
                sxx = (d[ra] ** 2).sum()
                k = 20000 * (d[tr] ** 2).mean()
                wa = ((d[ra] * tgt[ra]).sum() + k * wg) / (sxx + k)
                out["apt_k20000"][ta] = wa * d[ta]
        print(f"\n[{target_name}] LOMO global w per held-out month:", ws)
        rows = []
        for mm in sorted(set(ym)) + ["ALL"]:
            s = np.ones(len(y), bool) if mm == "ALL" else ym == mm
            n = s.sum()
            r = {"month": mm, "n": int(n)}
            for kname, delta in out.items():
                g0 = (((y - b) ** 2 - (y - b - delta) ** 2) * s).sum() / n
                g1 = (((y - b - c) ** 2 - (y - b - c - delta) ** 2) * s).sum() / n
                g1n = (((y - b) ** 2 - (y - b - delta) ** 2) * s * ~has).sum() / n
                r[f"{kname}:noC"] = round(float(g0), 1)
                r[f"{kname}:withC"] = round(float(g1), 1)
                r[f"{kname}:withC_hasF_part"] = round(float(g1n), 1)
            rows.append(r)
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["a", "b", "c"]
    for p_ in parts:
        {"a": part_a, "b": part_b, "c": part_c}[p_]()
