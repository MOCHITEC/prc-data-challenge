"""rescat: airline + stand-apron categoricals in the learned ADS-B residual (local, read-only analysis).

Pre-registration: $SCRATCH/agents/rescat/prereg.md (written before any fit). Reuses the 7-month harness of
`analysis/local_adsb_learn_months7.py` (PARAMS=v2): same rows, day folds, inner early-stopping split, clip, LTFM c = 0,
day-block bootstrap. Arms: A0 production features, A1 + cat_airline, A2 + cat_airline + cat_apron.

    POLARS_MAX_THREADS=3 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
        .venv/bin/python analysis/local_adsb_rescat.py codes   # code tables + drift
    ... local_adsb_rescat.py cv       # 5 folds x 3 arms, resumable (per-fold npz in $SCRATCH/agents/rescat/folds)
    ... local_adsb_rescat.py eval     # gains, CIs, concentration, decision
    ... local_adsb_rescat.py apply A1|A2 <in.parquet>   # 2026 delta vs production 7:v2 (only if adopted)

Airline rule: ICAO prefix of CALLSIGN_flt (`^[A-Z]{3}[0-9]`), else of FLIGHT_mvt (ICAO `^[A-Z]{3}[0-9]`, garbled
`^[A-Z]{3}[A-Z]{1,2}[0-9]` -> first 3), else IATA `^[A-Z0-9]{2}[0-9]{1,4}[A-Z]?$` mapped to ICAO via a learned
callsign co-occurrence table (else "I:<iata>"), else "none". Apron: ADEP|<leading letters (<= 2)> or
ADEP|<first digit>d<n digits>. Codes with < 50 rows in the 2025 7-month ADS-B rows -> "other".
Writes only under $SCRATCH/agents/rescat/.
"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

ARGV = list(sys.argv)
sys.path.insert(0, "analysis")
import local_adsb_learn_months as M  # noqa: E402  (overwrites sys.argv)

L, CV = M.L, M.CV
OUT = L.OUT
RES = os.path.join(L.SCR, "agents", "rescat")
FOLDS = os.path.join(RES, "folds")
os.makedirs(FOLDS, exist_ok=True)
PRM = {**CV.PARAMS, **dict(num_leaves=255, min_data_in_leaf=300, feature_fraction=0.7, num_threads=3)}
RARE = 50
MONTHS7 = [1, 7, 8, 9, 10, 11, 12]
ARMS = {"A0": [], "A1": ["cat_airline"], "A2": ["cat_airline", "cat_apron"],
        "A3": ["cat_apron"]}  # A3: follow-up arm requested after the first report (apron only)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_cols(20)


# ----------------------------------------------------------------------------------------------- raw codes
def raw_dep(year: str) -> pl.DataFrame:
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"]
    files = sorted(glob.glob("data/raw/training_2025-*.parquet")) if year == "2025" else ["data/raw/ranking.parquet"]
    R = pl.concat([pl.read_parquet(f, columns=cols) for f in files])
    return R.filter(pl.col("PHASE_mvt") == "DEP").drop("PHASE_mvt").with_columns(pl.lit(year).alias("yr"))


def iata_map(R: pl.DataFrame) -> dict:
    cs = pl.col("CALLSIGN_flt").str.to_uppercase()
    f = pl.col("FLIGHT_mvt").str.to_uppercase()
    T = (R.filter(f.str.contains(r"^[A-Z0-9]{2}[0-9]{1,4}[A-Z]?$") & f.str.slice(0, 2).str.contains(r"[A-Z]")
                  & cs.str.contains(r"^[A-Z]{3}[0-9]"))
         .group_by(f.str.slice(0, 2).alias("iata"), cs.str.slice(0, 3).alias("icao")).len())
    T = T.with_columns(pl.col("len").sum().over("iata").alias("tot")).sort("iata", "len", "icao", descending=[False, True, False])
    T = T.group_by("iata", maintain_order=True).first().filter((pl.col("tot") >= 20) & (pl.col("len") / pl.col("tot") >= 0.6))
    return dict(T.select("iata", "icao").iter_rows())


def raw_codes(R: pl.DataFrame, imap: dict) -> pl.DataFrame:
    cs = pl.col("CALLSIGN_flt").str.to_uppercase().str.strip_chars()
    f = pl.col("FLIGHT_mvt").str.to_uppercase().str.strip_chars()
    iata = f.str.slice(0, 2)
    airline = (pl.when(cs.str.contains(r"^[A-Z]{3}[0-9]")).then(cs.str.slice(0, 3))
               .when(f.str.contains(r"^[A-Z]{3}[0-9]")).then(f.str.slice(0, 3))
               .when(f.str.contains(r"^[A-Z]{3}[A-Z]{1,2}[0-9]")).then(f.str.slice(0, 3))
               .when(f.str.contains(r"^[A-Z0-9]{2}[0-9]{1,4}[A-Z]?$") & iata.str.contains(r"[A-Z]"))
               .then(iata.replace_strict(imap, default=None).fill_null(pl.lit("I:") + iata))
               .otherwise(pl.lit("none")).fill_null("none"))
    s = pl.col("STAND_mvt").str.to_uppercase().str.strip_chars()
    lead_alpha = s.str.extract(r"^([A-Z]+)", 1).str.slice(0, 2)
    lead_dig = s.str.extract(r"^([0-9]+)", 1)
    code = (pl.when(s.is_null() | (s == "")).then(pl.lit("null"))
            .when(lead_alpha.is_not_null()).then(lead_alpha)
            .when(lead_dig.is_not_null()).then(lead_dig.str.slice(0, 1) + pl.lit("d") + lead_dig.str.len_chars().cast(pl.String))
            .otherwise(pl.lit("misc")))
    return R.select(pl.col("MVT_ID_mvt").cast(pl.Float64), "ADEP_mvt", "yr", airline.alias("airline"),
                    (pl.col("ADEP_mvt") + pl.lit("|") + code).alias("apron"))


def code_tables():
    """Returns (codes frame for 2025+2026 departures, kept-vocabulary dicts)."""
    p = os.path.join(RES, "codes.parquet")
    if os.path.exists(p):
        C = pl.read_parquet(p)
        V = json.load(open(os.path.join(RES, "vocab.json")))
        return C, V
    R = pl.concat([raw_dep("2025"), raw_dep("2026")])
    imap = iata_map(R)
    C = raw_codes(R, imap)
    del R
    assert C["MVT_ID_mvt"].n_unique() == C.height
    # training-row counts: 2025 7-month ADS-B rows
    keep_ids = train_adsb_ids()
    tr = C.filter(pl.col("MVT_ID_mvt").is_in(keep_ids))
    V = {"iata_map": imap}
    for col in ("airline", "apron"):
        cnt = dict(tr.group_by(col).len().iter_rows())
        union = sorted(set(C[col].unique().to_list()))
        kept = [v for v in union if cnt.get(v, 0) >= RARE]
        V[col] = kept
        V[col + "_n_union"] = len(union)
    C.write_parquet(p)
    json.dump(V, open(os.path.join(RES, "vocab.json"), "w"), indent=0)
    return C, V


def train_adsb_ids() -> pl.Series:
    F = pl.concat([pl.read_parquet(os.path.join(OUT, f), columns=["MVT_ID_mvt", "icao"])
                   for f in ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet")])
    return F.filter(pl.col("icao").is_not_null())["MVT_ID_mvt"].cast(pl.Float64)


def add_codes(D: pl.DataFrame, C: pl.DataFrame, V: dict) -> pl.DataFrame:
    D = D.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64).alias("_id")).join(
        C.select(pl.col("MVT_ID_mvt").alias("_id"), "airline", "apron"), on="_id", how="left").drop("_id")
    ex = []
    for col in ("airline", "apron"):
        kept = V[col]
        ex.append(pl.col(col).replace_strict(kept, list(range(len(kept))), default=len(kept)).cast(pl.Int32)
                  .alias(f"cat_{col}"))  # "other" (incl. null) = len(kept)
    return D.with_columns(ex)


# ----------------------------------------------------------------------------------------------- data (months7)
def load():
    keep = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_minus_aobt3", "b", "y", "day"]
    D = pl.concat([CV.load_labelled().select(keep), M.load_novdec().select(keep), M.load_novdec("aso", "aso").select(keep)])
    F = pl.concat([pl.read_parquet(os.path.join(OUT, f)) for f in ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet")],
                  how="diagonal_relaxed").drop("ADEP_mvt", "day")
    D = D.join(F, on="MVT_ID_mvt", how="left").with_columns(pl.col("icao").is_not_null().alias("has"),
                                                           pl.col("day").str.slice(0, 7).alias("ym"))
    days = sorted(D["day"].unique().to_list())
    months = sorted({d[:7] for d in days})
    rng = np.random.default_rng(CV.SEED)
    fold_of = {}
    for m in months:
        ds = sorted(d for d in days if d.startswith(m))
        for i, j in enumerate(rng.permutation(len(ds))):
            fold_of[ds[j]] = i % CV.K

    def inner(tr_days, seed):
        r = np.random.default_rng(seed)
        val = set()
        for m in months:
            ds = sorted(d for d in tr_days if d.startswith(m))
            val |= {ds[j] for i, j in enumerate(r.permutation(len(ds))) if i % 5 == 0}
        return val

    D = D.with_columns(pl.col("day").replace_strict(fold_of).alias("cvfold"),
                       pl.col("day").replace_strict({d: i for i, d in enumerate(days)}).alias("day_idx"))
    return D, days, fold_of, inner


def cmd_codes():
    C, V = code_tables()
    print("IATA->ICAO map size", len(V["iata_map"]), "| kept airline", len(V["airline"]), "of", V["airline_n_union"],
          "| kept apron", len(V["apron"]), "of", V["apron_n_union"])
    tr_ids = train_adsb_ids()
    F26 = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"), columns=["MVT_ID_mvt", "ADEP_mvt", "icao", "ground_fix"])
    T = add_codes(C.filter(pl.col("MVT_ID_mvt").is_in(tr_ids)).select("MVT_ID_mvt", "ADEP_mvt"), C, V).with_columns(pl.lit(0).alias("is26"))
    tr_mon = pl.concat([pl.read_parquet(os.path.join(OUT, "feat_2025.parquet"), columns=["MVT_ID_mvt", "icao"])]).filter(
        pl.col("icao").is_not_null())["MVT_ID_mvt"].cast(pl.Float64)
    S = add_codes(F26.filter(pl.col("icao").is_not_null()).select(pl.col("MVT_ID_mvt").cast(pl.Float64), "ADEP_mvt"), C, V
                  ).with_columns(pl.lit(1).alias("is26"))
    print("2026 ADS-B rows", S.height, "| null airline", S["airline"].null_count())
    cntA = dict(T.group_by("airline").len().iter_rows())
    cntS = dict(T.group_by("apron").len().iter_rows())
    out = {}
    for col, cnt in (("airline", cntA), ("apron", cntS)):
        v = S[col].to_list()
        unseen = np.mean([cnt.get(x, 0) == 0 for x in v])
        rare = np.mean([0 < cnt.get(x, 0) < RARE for x in v])
        other_tr = float((T[f"cat_{col}"] == len(V[col])).mean())
        out[col] = dict(unseen26=float(unseen), rare26=float(rare), other_share_2025=other_tr)
        print(f"{col}: 2026 ADS-B rows unseen in 2025-7m {unseen:.4f}, rare(<{RARE}) {rare:.4f} | 2025 rows mapped to other {other_tr:.4f}")
    print("airline 'none'/I: share 2025", float(T["airline"].is_in(["none"]).mean()), float(T["airline"].str.starts_with("I:").mean()),
          "2026", float(S["airline"].is_in(["none"]).mean()), float(S["airline"].str.starts_with("I:").mean()))
    for a in sorted(S["ADEP_mvt"].unique()):
        s = S.filter(pl.col("ADEP_mvt") == a)
        print(f"  {a}: n26 {s.height:6d} airline->other {float((s['cat_airline'] == len(V['airline'])).mean()):.4f} "
              f"apron->other {float((s['cat_apron'] == len(V['apron'])).mean()):.4f} | top aprons 2026 "
              f"{s.group_by('apron').len().sort('len', descending=True).head(6).rows()}")
    # adversarial AUC
    from sklearn.metrics import roc_auc_score
    T = T.with_columns(pl.col("MVT_ID_mvt").is_in(tr_mon).alias("janjul"))
    T = T.with_columns(pl.col("ADEP_mvt").replace_strict(list(L.s1.AIRPORTS), list(range(10)), default=-1).alias("cat_adep"))
    S = S.with_columns(pl.col("ADEP_mvt").replace_strict(list(L.s1.AIRPORTS), list(range(10)), default=-1).alias("cat_adep"),
                       pl.lit(True).alias("janjul"))
    for lab, A_ in (("2025 7m vs 2026", T), ("2025 Jan+Jul vs 2026", T.filter(pl.col("janjul")))):
        Z = pl.concat([A_.select("cat_adep", "cat_airline", "cat_apron", "is26"), S.select("cat_adep", "cat_airline", "cat_apron", "is26")])
        yv = Z["is26"].to_numpy()
        fold = np.random.default_rng(0).integers(0, 5, Z.height)
        res = {}
        for cols in (["cat_adep"], ["cat_airline"], ["cat_apron"], ["cat_airline", "cat_apron"], ["cat_adep", "cat_airline", "cat_apron"]):
            X = Z.select(cols).to_numpy()
            p = np.zeros(Z.height)
            for k in range(5):
                m = lgb.train(dict(objective="binary", num_leaves=63, learning_rate=0.1, num_threads=3, verbose=-1,
                                   max_cat_threshold=256, min_data_per_group=50, cat_smooth=10),
                              lgb.Dataset(X[fold != k], yv[fold != k], categorical_feature=list(range(len(cols)))), 200)
                p[fold == k] = m.predict(X[fold == k])
            res["+".join(c[4:] for c in cols)] = round(float(roc_auc_score(yv, p)), 4)
        out["auc " + lab] = res
        print(f"adversarial AUC {lab}: {res}", flush=True)
    json.dump(out, open(os.path.join(RES, "drift.json"), "w"), indent=1)


def matrix():
    D, days, fold_of, inner = load()
    C, V = code_tables()
    D = add_codes(D, C, V)
    print("rows", D.height, "ADS-B", int(D["has"].sum()), "| null airline among ADS-B", int(D.filter("has")["airline"].null_count()), flush=True)
    X, feats, cats = L.design(D, "b")
    allf = feats + ["cat_airline", "cat_apron"]
    Xn = X.select(allf).to_numpy().astype(np.float64)
    del X
    return D, days, fold_of, inner, Xn, feats, cats, allf


def cmd_cv():
    D, days, fold_of, inner, Xn, feats, cats, allf = matrix()
    has = D["has"].to_numpy()
    r = np.clip(D["y"].to_numpy() - D["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    cvf, day_arr = D["cvfold"].to_numpy(), D["day"].to_numpy()
    for k in range(CV.K):
        val_days = inner([d for d in days if fold_of[d] != k], CV.SEED + 1 + k)
        is_te = has & (cvf == k)
        is_tr = has & (cvf != k)
        is_val = is_tr & np.isin(day_arr, list(val_days))
        is_fit = is_tr & ~is_val
        for arm, extra in ARMS.items():
            dst = os.path.join(FOLDS, f"f{k}_{arm}.npz")
            if os.path.exists(dst):
                continue
            L.mem_gate(9000.0)
            fn = feats + extra
            idx = [allf.index(f) for f in fn]
            ct = cats + extra
            dtr = lgb.Dataset(Xn[np.ix_(is_fit, idx)], r[is_fit], feature_name=fn, categorical_feature=ct, free_raw_data=True)
            dva = lgb.Dataset(Xn[np.ix_(is_val, idx)], r[is_val], reference=dtr)
            m = lgb.train(PRM, dtr, num_boost_round=5000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            c = np.clip(m.predict(Xn[np.ix_(is_te, idx)], num_iteration=m.best_iteration), -CV.CLIP_C, CV.CLIP_C)
            imp = m.feature_importance("gain", iteration=m.best_iteration)
            np.savez(dst, ids=D["MVT_ID_mvt"].to_numpy()[is_te], c=c, best=int(m.best_iteration), imp=imp, feats=np.array(fn))
            print(f"fold {k} arm {arm}: fit {is_fit.sum()} es {is_val.sum()} test {is_te.sum()} best_iter {m.best_iteration} "
                  f"| new-col gain share {[round(float(imp[fn.index(e)] / imp.sum()), 4) for e in extra]}", flush=True)
            del dtr, dva, m


def cmd_eval():
    D, days, fold_of, inner = load()
    C, V = code_tables()
    D = add_codes(D, C, V)
    ids = D["MVT_ID_mvt"].to_numpy()
    pos = {v: i for i, v in enumerate(ids)}
    c = {a: np.zeros(D.height) for a in ARMS}
    best, imps = {a: [] for a in ARMS}, {a: None for a in ARMS}
    for a in ARMS:
        for k in range(CV.K):
            z = np.load(os.path.join(FOLDS, f"f{k}_{a}.npz"))
            ii = np.array([pos[v] for v in z["ids"]])
            c[a][ii] = z["c"]
            best[a].append(int(z["best"]))
            imps[a] = z["imp"] if imps[a] is None else imps[a] + z["imp"]
            fn = list(z["feats"])
        imps[a] = {f: float(v / imps[a].sum()) for f, v in zip(fn, imps[a])}
    apt, ym, has = D["ADEP_mvt"].to_numpy(), D["ym"].to_numpy(), D["has"].to_numpy()
    for a in ARMS:
        c[a][apt == "LTFM"] = 0.0
    yv, bv = D["y"].to_numpy(), D["b"].to_numpy()
    e0 = yv - bv
    sq = {a: (e0 - c[a]) ** 2 for a in ARMS}
    di, nd = D["day_idx"].to_numpy(), len(days)
    sels = (("all", np.ones(D.height, bool)), ("Jan", ym == "2025-01"), ("Jul", ym == "2025-07"),
            ("Aug-Oct", np.isin(ym, ["2025-08", "2025-09", "2025-10"])), ("Nov-Dec", np.isin(ym, ["2025-11", "2025-12"])),
            ("Jan+Jul", np.isin(ym, ["2025-01", "2025-07"])))
    out = {"best_iters": best, "final_rounds": {a: int(np.median(v)) for a, v in best.items()},
           "new_col_gain_share": {a: {f: imps[a][f] for f in ARMS[a]} for a in ARMS}, "arms": {}}
    print("median best iter", out["final_rounds"], "| new-col gain share", out["new_col_gain_share"])
    print("A0 gain over base (all rows):", round(float((e0 ** 2 - sq["A0"]).mean()), 1))
    for a, ref in (("A1", "A0"), ("A2", "A0"), ("A2", "A1"), ("A3", "A0"), ("A2", "A3")):
        d = sq[ref] - sq[a]
        key = f"{a}-{ref}"
        out["arms"][key] = {}
        print(f"\n=== {key}: d = (y-b-c_{ref})^2 - (y-b-c_{a})^2, s^2 (>0 = {a} better); RMSE delta on ADS-B rows")
        for lab, sel in sels:
            g, lo, hi = CV.boot(d[sel], di[sel], nd)
            sh = sel & has
            dr = float(np.sqrt(sq[a][sh].mean()) - np.sqrt(sq[ref][sh].mean()))
            dr_all = float(np.sqrt(sq[a][sel].mean()) - np.sqrt(sq[ref][sel].mean()))
            out["arms"][key][lab] = dict(n=int(sel.sum()), gain=g, lo=lo, hi=hi, d_rmse_adsb=dr, d_rmse_all=dr_all)
            print(f"  {lab:8s} n={sel.sum():8d} gain {g:7.1f} [{lo:7.1f}, {hi:7.1f}] | dRMSE all {dr_all:+.3f} s, ADS-B rows {dr:+.3f} s")
        o = np.argsort(-np.abs(d))
        tot = d.sum()
        share = float(1 - d[o[50:]].sum() / tot) if tot != 0 else float("nan")
        g_ex = float(d[o[50:]].sum() / D.height)
        out["arms"][key]["top50_share"] = share
        out["arms"][key]["gain_excl_top50"] = g_ex
        print(f"  concentration: top-50 |d| rows carry share {share:.3f} of the gain; gain without them {g_ex:.1f}")
        rows = []
        for ap in sorted(set(apt)):
            for lab, sel0 in (("all", np.ones(D.height, bool)), ("Jan", ym == "2025-01"), ("Jul", ym == "2025-07")):
                sel = sel0 & (apt == ap)
                g, lo, hi = CV.boot(d[sel], di[sel], nd)
                rows.append({"airport": ap, "m": lab, "n": int(sel.sum()), "gain": round(g, 1), "lo": round(lo, 1), "hi": round(hi, 1)})
        P = pl.DataFrame(rows).pivot(on="m", index="airport", values=["gain", "lo", "hi"])
        print(P)
        out["arms"][key]["per_airport"] = rows
    passed = {}
    for a in ("A1", "A2", "A3"):
        r = out["arms"][f"{a}-A0"]
        passed[a] = bool(r["all"]["lo"] > 0 and r["Jan"]["gain"] > 0 and r["Jul"]["gain"] > 0 and r["top50_share"] <= 0.5)
    if passed["A1"] and passed["A2"]:
        r21 = out["arms"]["A2-A1"]["all"]
        adopt = "A2" if (out["arms"]["A2-A0"]["all"]["gain"] > out["arms"]["A1-A0"]["all"]["gain"] and r21["lo"] > 0) else "A1"
    else:
        adopt = next((a for a in ("A1", "A2") if passed[a]), None)
    out["passed"], out["adopt"] = passed, adopt
    print(f"\nDECISION (pre-registered): pass {passed} -> adopt {adopt}")
    json.dump(out, open(os.path.join(RES, "eval.json"), "w"), indent=1)
    pl.DataFrame({"MVT_ID_mvt": ids, "ym": ym, "ADEP_mvt": apt, "y": yv, "b": bv, "has": has,
                  **{f"c_{a}": c[a] for a in ARMS}}).write_parquet(os.path.join(RES, "cv_rows.parquet"))


def cmd_apply(arm: str, src: str):
    """2026 delta of the adopted arm on ground-fix rows, as an increment on top of the production residual:
    new = file + c_arm(b) - c_7v2(b), with c_arm = 7 months + new cats and c_7v2 = the same recipe without them (the
    pair the CV compared). b is recovered from the file by the fixed point b = file - c_prod(b) with the residual the
    production file actually carries on that airport's ground rows (v40: 12:v2, but 7:v2 at LFPG; LIRF/LTFM never
    swapped). Rows at LTFM and at $EXCLUDE (default LIRF,LFPG) are written to rank_delta_all.parquet only."""
    ev = json.load(open(os.path.join(RES, "eval.json")))
    sys.path.insert(0, "analysis")
    import local_adsb_apply_swap as W
    D12 = W.data()  # Jan+Jul, Nov+Dec, Aug-Oct, Feb-Jun (fmj) labelled rows joined to ADS-B features
    ym = D12["day"].str.slice(0, 7).to_numpy()
    C, V = code_tables()
    D12 = add_codes(D12, C, V)
    X, feats, cats = L.design(D12, "b")
    r = np.clip(D12["y"].to_numpy() - D12["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    sel7 = np.isin(ym, ["2025-%02d" % m for m in MONTHS7])
    models = {}
    for tag, fn, ct, sel, n in (("7v2", feats, cats, sel7, W.rounds("7:v2")),
                                ("12v2", feats, cats, np.ones(len(ym), bool), W.rounds("12:v2")),
                                (arm, feats + ARMS[arm], cats + ARMS[arm], sel7, ev["final_rounds"][arm])):
        Xa = X.select(fn).to_numpy().astype(np.float64)[sel]
        models[tag] = (lgb.train(PRM, lgb.Dataset(Xa, r[sel], feature_name=fn, categorical_feature=ct), num_boost_round=n), fn)
        del Xa
        print(f"{tag}: {int(sel.sum())} rows, {n} rounds", flush=True)
    del X, D12
    sub = pl.read_parquet(src)
    rk = pl.scan_parquet("data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"))
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter((pl.col("ADEP_mvt") != "LTFM") & pl.col("ground_fix").fill_null(False)).with_columns(pl.col("file").cast(pl.Float64)))
    G = add_codes(G, C, V)
    apt = G["ADEP_mvt"].to_numpy()

    def corr(tag, b):
        m, fn = models[tag]
        Xg, _, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        return np.clip(m.predict(Xg.select(fn).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    f = G["file"].to_numpy()
    b = f.copy()
    for _ in range(3):
        b = f - np.where(apt == "LFPG", corr("7v2", b), corr("12v2", b))
    c7, cn = corr("7v2", b), corr(arm, b)
    new_v = np.round(f + cn - c7)
    new_v = np.where(new_v < 0, f, new_v)
    excl = [a for a in os.environ.get("EXCLUDE", "LIRF,LFPG").split(",") if a]
    R = G.select("MVT_ID_mvt", "ADEP_mvt", "file", "airline", "apron").with_columns(
        pl.Series("b", b), pl.Series("c_7v2", c7), pl.Series(f"c_{arm}", cn), pl.Series("new", new_v),
        (pl.Series("new", new_v) - pl.col("file")).alias("delta"), (~pl.col("ADEP_mvt").is_in(excl)).alias("applied"))
    R.write_parquet(os.path.join(RES, "rank_delta_all.parquet"))
    A = R.filter("applied").select("MVT_ID_mvt", "ADEP_mvt", "file", "new", "delta")
    A.write_parquet(os.path.join(RES, "rank_delta.parquet"))
    d = A["delta"].to_numpy()
    print(f"input {src}; excluded LTFM + {excl}; {A.height} ground rows, {int((d != 0).sum())} changed; "
          f"sum delta^2 / N(ranking={sub.height}) {float((d ** 2).sum()) / sub.height:.1f}; mean delta {d.mean():+.2f}, "
          f"mean |delta| {np.abs(d).mean():.1f}, p99 |delta| {np.quantile(np.abs(d), 0.99):.0f}, max |delta| {np.abs(d).max():.0f}")
    print(R.group_by("ADEP_mvt").agg(pl.len(), pl.col("applied").first(), pl.col("delta").mean().alias("mean"),
                                     pl.col("delta").abs().mean().alias("mean_abs"),
                                     (pl.col("delta") ** 2).mean().sqrt().alias("rms"),
                                     ((pl.col("delta") ** 2).sum() / sub.height).alias("sq_per_N")).sort("ADEP_mvt"))


def cmd_sub():
    """Post-hoc breakdowns (not part of the decision): pooled gains excluding LIRF (production never swaps LIRF) and
    LIRF+LFPG; LIRF gain by airline; A2 top-50 concentration excluding LIRF."""
    D, days, *_ = load()
    C, V = code_tables()
    R = pl.read_parquet(os.path.join(RES, "cv_rows.parquet"))
    D = add_codes(D.select("MVT_ID_mvt", "day_idx"), C, V).join(R, on="MVT_ID_mvt")
    di, nd, apt, ym = D["day_idx"].to_numpy(), len(days), D["ADEP_mvt"].to_numpy(), D["ym"].to_numpy()
    e0 = (D["y"] - D["b"]).to_numpy()
    sq = {a: (e0 - D[f"c_{a}"].to_numpy()) ** 2 for a in ARMS}
    out = {}
    for key, a, ref in (("A1-A0", "A1", "A0"), ("A2-A0", "A2", "A0"), ("A2-A1", "A2", "A1"), ("A3-A0", "A3", "A0"),
                        ("A2-A3", "A2", "A3")):
        d = sq[ref] - sq[a]
        for exl in (["LIRF"], ["LIRF", "LFPG"]):
            keep = ~np.isin(apt, exl)
            dd = np.where(keep, d, 0.0)  # excluded rows count as zero change, all rows in the denominator
            for lab, sel in (("all", np.ones(len(d), bool)), ("Jan", ym == "2025-01"), ("Jul", ym == "2025-07"),
                             ("Nov-Dec", np.isin(ym, ["2025-11", "2025-12"]))):
                g, lo, hi = CV.boot(dd[sel], di[sel], nd)
                out[f"{key} ex {'+'.join(exl)} {lab}"] = (round(g, 1), round(lo, 1), round(hi, 1))
            o = np.argsort(-np.abs(dd))
            out[f"{key} ex {'+'.join(exl)} top50_share"] = round(float(1 - dd[o[50:]].sum() / dd.sum()), 3)
    for k, v in out.items():
        print(f"{k:40s} {v}")
    d = sq["A0"] - sq["A1"]
    m = apt == "LIRF"
    T = pl.DataFrame({"airline": D["airline"].to_numpy()[m], "d": d[m]}).group_by("airline").agg(
        pl.len(), pl.col("d").sum().alias("dsum")).with_columns((pl.col("dsum") / d[m].sum()).alias("share")).sort("dsum", descending=True)
    print("LIRF A1-A0 gain by airline (top 8):\n", T.head(8))
    json.dump(out, open(os.path.join(RES, "sub.json"), "w"), indent=1)


def cmd_apply12(src: str, with_lfpg: str):
    """Follow-up: production-matched swap with the apron arm (A3). Ground-fix rows, LIRF and LTFM excluded:
    12:v2 -> 12:v2 + cat_apron (all 12 months, rounds = the 12:v2 rule, W.rounds('12:v2')) at every airport except
    LFPG; at LFPG (only if with_lfpg == 'yes') 7:v2 -> 7:v2 + cat_apron (7 months, rounds = median CV best iteration
    of A3). b = file - c_old(b) by fixed point, new = file + c_new(b) - c_old(b), exactly as local_adsb_apply_swap.py.
    Writes rank_delta_12m.parquet (MVT_ID_mvt, delta, + ADEP_mvt, file, new)."""
    ev = json.load(open(os.path.join(RES, "eval.json")))
    sys.path.insert(0, "analysis")
    import local_adsb_apply_swap as W
    D12 = W.data()
    ym = D12["day"].str.slice(0, 7).to_numpy()
    C, V = code_tables()
    D12 = add_codes(D12, C, V)
    X, feats, cats = L.design(D12, "b")
    r = np.clip(D12["y"].to_numpy() - D12["b"].to_numpy(), -CV.CLIP_Y, CV.CLIP_Y)
    allf = feats + ["cat_apron"]
    Xn = X.select(allf).to_numpy().astype(np.float64)
    del X, D12
    sel7 = np.isin(ym, ["2025-%02d" % m for m in MONTHS7])
    P = {**CV.PARAMS, **W.V2}  # production params (as apply_swap)
    spec = [("old12", feats, None, W.rounds("12:v2")), ("new12", allf, None, W.rounds("12:v2"))]
    if with_lfpg == "yes":
        spec += [("old7", feats, sel7, W.rounds("7:v2")), ("new7", allf, sel7, ev["final_rounds"]["A3"])]
    models = {}
    for tag, fn, sel, n in spec:
        idx = [allf.index(f) for f in fn]
        Xa = Xn[:, idx] if sel is None else Xn[np.ix_(sel, idx)]
        ra = r if sel is None else r[sel]
        ds = lgb.Dataset(Xa, ra, feature_name=fn, categorical_feature=[c for c in cats + ["cat_apron"] if c in fn], free_raw_data=True)
        models[tag] = (lgb.train(P, ds, num_boost_round=n), fn)
        del Xa, ds
        print(f"{tag}: {len(ra)} rows, {n} rounds, mem {L.mem_mb():.0f} MB", flush=True)
    del Xn
    sub = pl.read_parquet(src)
    rk = pl.scan_parquet("data/raw/ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt").collect()
    F26 = pl.read_parquet(os.path.join(OUT, "feat_2026.parquet"))
    excl = ["LTFM", "LIRF"] + ([] if with_lfpg == "yes" else ["LFPG"])
    G = (F26.drop("ADEP_mvt").join(sub.rename({"TAXITIME_SEC_mvt": "file"}).join(rk, on="MVT_ID_mvt"), on="MVT_ID_mvt")
         .filter(~pl.col("ADEP_mvt").is_in(excl) & pl.col("ground_fix").fill_null(False)).with_columns(pl.col("file").cast(pl.Float64)))
    G = add_codes(G, C, V)
    lf = G["ADEP_mvt"].to_numpy() == "LFPG"

    def corr(tag, b):
        m, fn = models[tag]
        Xg, _, _ = L.design(G.with_columns(pl.Series("b", b)), "b")
        return np.clip(m.predict(Xg.select(fn).to_numpy().astype(np.float64)), -CV.CLIP_C, CV.CLIP_C)

    def c_old(b):
        return np.where(lf, corr("old7", b), corr("old12", b)) if lf.any() else corr("old12", b)

    def c_new(b):
        return np.where(lf, corr("new7", b), corr("new12", b)) if lf.any() else corr("new12", b)

    f = G["file"].to_numpy()
    b = f.copy()
    for _ in range(3):
        b = f - c_old(b)
    new_v = np.round(f + c_new(b) - c_old(b))
    new_v = np.where(new_v < 0, f, new_v)
    R = G.select("MVT_ID_mvt", "ADEP_mvt", "file").with_columns(pl.Series("new", new_v), (pl.Series("new", new_v) - pl.col("file")).alias("delta"))
    R.select("MVT_ID_mvt", "delta", "ADEP_mvt", "file", "new").write_parquet(os.path.join(RES, "rank_delta_12m.parquet"))
    d = R["delta"].to_numpy()
    summ = dict(input=src, excluded=excl, ground_rows=R.height, changed=int((d != 0).sum()),
                sum_d2_over_N=float((d ** 2).sum()) / sub.height, N=sub.height, mean=float(d.mean()),
                mean_abs=float(np.abs(d).mean()), p99_abs=float(np.quantile(np.abs(d), 0.99)), max_abs=float(np.abs(d).max()),
                rounds={t: n for t, _, _, n in spec})
    json.dump(summ, open(os.path.join(RES, "rank_delta_12m_summary.json"), "w"), indent=1)
    print(summ)
    print(R.group_by("ADEP_mvt").agg(pl.len(), (pl.col("delta") != 0).sum().alias("changed"), pl.col("delta").mean().alias("mean"),
                                     pl.col("delta").abs().quantile(0.99).alias("p99_abs"),
                                     ((pl.col("delta") ** 2).sum() / sub.height).alias("sq_per_N")).sort("ADEP_mvt"))


if __name__ == "__main__":
    cmd = ARGV[1]
    if cmd == "codes":
        cmd_codes()
    elif cmd == "cv":
        cmd_cv()
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "sub":
        cmd_sub()
    elif cmd == "apply12":
        cmd_apply12(ARGV[2], ARGV[3])
    elif cmd == "apply":
        cmd_apply(ARGV[2], ARGV[3])
