"""Matched-lane big errors: leave-one-month-out check of a LOBT/SCHED copy blend outside LIRF (local, read-only).

Regime: the label BLOCK equals LOBT (cL: |y - (MVT-LOBT)| <= 12 s) or SCHED (cS: |y - (MVT-SCHED)| <= 12 s, not cL).
Per airport a 3-class LightGBM (none / cL / cS) on the pipeline features (feat_train) is fit on the 11 months other
than the held-out OOF month and scored on that month; the OOF prediction p = b + c_C is then moved toward the copy
values:  p' = p + a * (qL * (s_L - p) + qS * (s_S - p)).  a is fitted per airport leave-one-month-out on the 7 OOF
months (and A=0.4, the LIRF value, is reported as a fixed alternative). Gains are SSE reductions / 1,233,234.

  lomo  : writes $SCRATCH/agents/matched/copyq_<APT>.parquet (MVT_ID_mvt, month, qL, qS, sL, sS) and prints gains
  apply : refit on all 12 months (rounds = median best_iter), score feat_rank, write per-airport q for 2026

    APTS=LEBL,LEMD .venv/bin/python analysis/local_bigerr_copy_lomo.py lomo
    APTS=LEBL .venv/bin/python analysis/local_bigerr_copy_lomo.py apply
    SFIRST=1 APTS=LIRF ... lomo / lirf / apply   (SCHED-copy class takes ties; LIRF split vs the production overlay)
    .venv/bin/python analysis/local_bigerr_copy_lomo.py lirf   (after lomo for LIRF)
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

S = os.environ["SCRATCH"]
O = os.path.join(S, "agents", "matched")
MONTHS = [1, 7, 8, 9, 10, 11, 12]
SFIRST = os.environ.get("SFIRST") == "1"  # class order: SCHED copy wins ties (needed for the LIRF split)
TAG = "_sf" if SFIRST else ""
DROP = {"TAXITIME_SEC_mvt", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt",
        "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "ARVT_3_flt", "FLIGHT_ID_mvt", "cls"}
P = dict(objective="multiclass", num_class=3, learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
         feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=1)


def frames(apt):
    T = (pl.scan_parquet("data/work_flseq/feat_train.parquet")
         .filter((pl.col("ADEP_mvt") == apt) & pl.col("mvt_minus_aobt3").is_not_null()).collect())
    yy = pl.col("TAXITIME_SEC_mvt")
    if SFIRST:  # class 1 = SCHED copy (the production LIRF target), class 2 = LOBT copy that is not a SCHED copy
        T = T.with_columns(pl.when((yy - pl.col("mvt_minus_sched")).abs() <= 12).then(2)
                           .when((yy - pl.col("mvt_minus_lobt")).abs() <= 12).then(1).otherwise(0).alias("cls"))
    else:
        T = T.with_columns(pl.when((yy - pl.col("mvt_minus_lobt")).abs() <= 12).then(1)
                           .when((yy - pl.col("mvt_minus_sched")).abs() <= 12).then(2).otherwise(0).alias("cls"))
    R = pl.read_parquet("data/work_flseq/feat_rank.parquet").filter(
        (pl.col("ADEP_mvt") == apt) & pl.col("mvt_minus_aobt3").is_not_null())
    num = [c for c, t in T.schema.items() if c not in DROP and t.is_numeric() and c in R.columns]
    cat = [c for c, t in T.schema.items() if c not in DROP and t == pl.String and c in R.columns]
    codes = {c: {v: i for i, v in enumerate(sorted(set(T[c].drop_nulls().to_list()) | set(R[c].drop_nulls().to_list())))}
             for c in cat}

    def X(df):
        return df.select(num + [pl.col(c).replace_strict(codes[c], default=None, return_dtype=pl.Int32).alias(c) for c in cat]
                         ).to_numpy().astype(np.float32)
    return T, R, num + cat, cat, X


def fit(Xt, y, fn, cat, rng, rounds=None):
    if rounds:
        return lgb.train(P, lgb.Dataset(Xt, y, feature_name=fn, categorical_feature=cat), rounds), rounds
    va = rng.random(len(y)) < 0.1
    m = lgb.train(P, lgb.Dataset(Xt[~va], y[~va], feature_name=fn, categorical_feature=cat), 600,
                  valid_sets=[lgb.Dataset(Xt[va], y[va], feature_name=fn, categorical_feature=cat)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    return m, m.best_iteration


def lomo(apt):
    T, R, fn, cat, X = frames(apt)
    Xt = X(T)
    y = T["cls"].to_numpy()
    mo = T["month"].to_numpy()
    rng = np.random.default_rng(7)
    out, its = [], []
    for m in MONTHS:
        tr, te = mo != m, mo == m
        mdl, it = fit(Xt[tr], y[tr], fn, cat, rng)
        its.append(it)
        q = mdl.predict(Xt[te], num_iteration=it)
        out.append(pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"].filter(te), "month": mo[te], "qL": q[:, 1], "qS": q[:, 2],
                                 "sL": T["mvt_minus_lobt"].filter(te), "sS": T["mvt_minus_sched"].filter(te), "cls": y[te]}))
        print(apt, m, "iter", it, flush=True)
    Q = pl.concat(out)
    Q.write_parquet(os.path.join(O, f"copyq_{apt}{TAG}.parquet"))
    json.dump({"iters": its}, open(os.path.join(O, f"copyq_{apt}{TAG}.json"), "w"))


def gains(apt):
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months7_v2.parquet")).with_columns((pl.col("b") + pl.col("c_C")).alias("p"))
    N = cv.height
    Q = pl.read_parquet(os.path.join(O, f"copyq_{apt}{TAG}.parquet"))
    L = cv.join(Q, on="MVT_ID_mvt", how="inner")
    p, y, qL, qS, sL, sS, mo = [L[c].to_numpy() for c in ("p", "y", "qL", "qS", "sL", "sS", "month")]
    d = qL * (sL - p) + qS * (sS - p)
    e = p - y

    def best_a(sel):  # least squares a for  y ~ p + a d
        return float(np.sum(d[sel] * (y[sel] - p[sel])) / max(np.sum(d[sel] ** 2), 1e-9))
    res = {"apt": apt, "n": int(L.height)}
    tot, totA, per = 0.0, 0.0, []
    for m in MONTHS:
        te = mo == m
        a = max(0.0, min(1.0, best_a(~te)))
        g = float(np.sum(e[te] ** 2 - (p[te] + a * d[te] - y[te]) ** 2)) / N
        gA = float(np.sum(e[te] ** 2 - (p[te] + 0.4 * d[te] - y[te]) ** 2)) / N
        tot += g
        totA += gA
        per.append((m, round(a, 2), round(g, 1), round(gA, 1)))
    # row concentration: share of the gain from top-20 rows (fixed A=0.4)
    gi = e ** 2 - (p + 0.4 * d - y) ** 2
    top = np.sort(gi)[::-1]
    res.update(gain_lomo=round(tot, 1), gain_A04=round(totA, 1), a_all=round(best_a(np.ones_like(te)), 3), per_month=per,
               top20_share_of_pos=round(float(top[:20].sum() / max(gi[gi > 0].sum(), 1e-9)), 3), n_pos=int((gi > 0).sum()),
               n_neg=int((gi < 0).sum()), oracle=round(float(np.sum(np.where(L["cls"].to_numpy() == 1, e ** 2 - (sL - y) ** 2,
                                                                              np.where(L["cls"].to_numpy() == 2, e ** 2 - (sS - y) ** 2, 0)))) / N, 1))
    print(json.dumps(res), flush=True)
    return res


def apply(apt):
    T, R, fn, cat, X = frames(apt)
    its = json.load(open(os.path.join(O, f"copyq_{apt}{TAG}.json")))["iters"]
    it = int(np.median(its) * 1.1)
    mdl, _ = fit(X(T), T["cls"].to_numpy(), fn, cat, None, rounds=it)
    q = mdl.predict(X(R))
    pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "qL": q[:, 1], "qS": q[:, 2], "sL": R["mvt_minus_lobt"],
                  "sS": R["mvt_minus_sched"]}).write_parquet(os.path.join(O, f"copyq26_{apt}{TAG}.parquet"))
    print(apt, "2026 rows", R.height, "mean qL", q[:, 1].mean(), "mean qS", q[:, 2].mean(),
          "2025 rates", (T["cls"] == 1).mean(), (T["cls"] == 2).mean(), flush=True)


def lirf_split():
    """LIRF: SCHED-only (≈ production overlay) vs LOBT-only vs both, fixed A=0.4 and LOMO a, per month."""
    cv = pl.read_parquet(os.path.join(S, "learn", "cv_rows_months7_v2.parquet")).with_columns((pl.col("b") + pl.col("c_C")).alias("p"))
    N = cv.height
    L = cv.join(pl.read_parquet(os.path.join(O, f"copyq_LIRF{TAG}.parquet")), on="MVT_ID_mvt", how="inner")
    p, y, qL, qS, sL, sS, mo = [L[c].to_numpy() for c in ("p", "y", "qL", "qS", "sL", "sS", "month")]
    base = p + 0.4 * qS * (sS - p)  # production-like
    dL = qL * (sL - base)
    for lab, pp in (("S only A.4", base), ("S+L A.4", base + 0.4 * dL)):
        print(lab, [(m, round(float(np.sum((p[mo == m] - y[mo == m]) ** 2 - (pp[mo == m] - y[mo == m]) ** 2)) / N, 1)) for m in MONTHS],
              round(float(np.sum((p - y) ** 2 - (pp - y) ** 2)) / N, 1))
    tot = []
    for m in MONTHS:
        tr, te = mo != m, mo == m
        a = float(np.clip(np.sum(dL[tr] * (y[tr] - base[tr])) / np.sum(dL[tr] ** 2), 0, 1))
        tot.append((m, round(a, 2), round(float(np.sum((base[te] - y[te]) ** 2 - (base[te] + a * dL[te] - y[te]) ** 2)) / N, 1)))
    print("L on top of S, LOMO a:", tot, round(sum(t[2] for t in tot), 1))


if __name__ == "__main__":
    apts = os.environ.get("APTS", "LEBL").split(",")
    for a in apts:
        if sys.argv[1] == "lomo":
            if not os.path.exists(os.path.join(O, f"copyq_{a}{TAG}.parquet")):
                lomo(a)
            gains(a)
        elif sys.argv[1] == "lirf":
            lirf_split()
            break
        elif sys.argv[1] == "gains":
            gains(a)
        else:
            apply(a)
