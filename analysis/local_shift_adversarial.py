"""Shift worker, task 3: adversarial validation 2025 Jan+Jul vs 2026 Jan+Jul, and the residual bias it implies.

A. LightGBM classifier (label 1 = 2026 ranking row, 0 = 2025 Jan/Jul training row) on the base model's
   feature frame (feat_train / feat_rank), calendar / ID / noise columns dropped; random 50/50 split.
   Global (numeric + categoricals), global numeric-only, and per airport (numeric only).
B. For the top shifters: OOF residual e = y - (b + c_C) binned by 2025 deciles per airport; implied 2026 bias of
   feature f = sum_bins (share_2026 - share_2025JanJul) * mean_e_bin (bins with < 200 OOF rows shrunk).
C. Multivariate: LGB residual model on the top shifters + ADEP, leave-one-month-out over the seven OOF months
   (gain in MSE units with per-month sign), then refit on all seven and predict the 2026 rows: mean predicted
   residual per airport = implied 2026 bias, vs the same on 2025 Jan+Jul (LOMO predictions).

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_shift_adversarial.py
Outputs: $SCRATCH/agents/shift/adv_*.txt / .parquet
"""
import os

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

SCR = os.environ["SCRATCH"]
OUT = os.path.join(SCR, "agents", "shift")
os.makedirs(OUT, exist_ok=True)
CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
        "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_mvt", "ADES_REGION"]
DROP = {"MVT_ID_mvt", "FLIGHT_ID_mvt", "TAXITIME_SEC_mvt", "month", "dow", "noise_control", "noise_control_2",
        "noise_control_3", "noise_control_4"}
PRM = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
           verbose=-1, num_threads=2, seed=0, max_cat_to_onehot=4, cat_smooth=20)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def numeric_cols(schema) -> list[str]:
    ok = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32, pl.UInt8, pl.Int16)
    return [c for c, dt in schema.items() if dt in ok and c not in DROP]


def to_matrix(df: pl.DataFrame, num: list[str], cats: list[str]) -> np.ndarray:
    X = df.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy()
    if cats:
        C = df.select([pl.col(c).cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Float32) for c in cats]).to_numpy()
        C[np.isnan(C)] = -1
        X = np.hstack([X, C])
    return np.ascontiguousarray(X, dtype=np.float32)


def adv(df: pl.DataFrame, lab: np.ndarray, num: list[str], cats: list[str], top: int = 12):
    X = to_matrix(df, num, cats)
    cols = num + cats
    rng = np.random.default_rng(0)
    ho = rng.random(len(lab)) < 0.5
    ci = list(range(len(num), len(cols)))
    dtr = lgb.Dataset(X[~ho], lab[~ho], feature_name=cols, categorical_feature=ci, free_raw_data=True)
    bst = lgb.train(PRM, dtr, num_boost_round=300)
    auc = roc_auc_score(lab[ho], bst.predict(X[ho]))
    g = bst.feature_importance("gain")
    o = np.argsort(-g)[:top]
    return auc, [(cols[i], g[i] / g.sum()) for i in o]


def main() -> None:
    sch = pl.read_parquet_schema("data/work_flseq/feat_rank.parquet")
    num = numeric_cols(sch)
    need = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt"] + num + CATS
    tr = (pl.scan_parquet("data/work_flseq/feat_train.parquet").select(need)
          .filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])).collect())
    rk = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=need)
    D = pl.concat([tr, rk], how="vertical_relaxed").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon"))
    lab = np.r_[np.zeros(tr.height), np.ones(rk.height)]
    del tr, rk
    log(f"2025 Jan+Jul rows {int((lab == 0).sum()):,}  2026 rows {int(lab.sum()):,}  numeric features {len(num)}")

    # A. global
    for name, cats in (("numeric+cats", CATS), ("numeric only", [])):
        auc, imp = adv(D, lab, num, cats)
        log(f"\n[global {name}] AUC {auc:.4f}")
        for c, s in imp:
            log(f"   {c:30s} {s:.3f}")
    # per month pairing (Jan vs Jan, Jul vs Jul), numeric only
    for mo in (1, 7):
        m = (D["mon"] == mo).to_numpy()
        auc, imp = adv(D.filter(pl.Series(m)), lab[m], num, [], top=6)
        log(f"\n[month {mo} numeric only] AUC {auc:.4f}   " + ", ".join(f"{c} {s:.2f}" for c, s in imp))
    # per airport, numeric + cats except ADEP
    per_apt = {}
    for a in sorted(D["ADEP_mvt"].unique().to_list()):
        m = (D["ADEP_mvt"] == a).to_numpy()
        auc, imp = adv(D.filter(pl.Series(m)), lab[m], num, [c for c in CATS if c != "ADEP_mvt"], top=8)
        per_apt[a] = imp
        log(f"\n[{a}] AUC {auc:.4f}   " + ", ".join(f"{c} {s:.2f}" for c, s in imp))

    # B. univariate implied bias for the numeric top shifters
    shifters = []
    for a, imp in per_apt.items():
        for c, _ in imp[:6]:
            if c in num and c not in shifters:
                shifters.append(c)
    log(f"\nnumeric top shifters: {shifters}")
    oof = pl.read_parquet(os.path.join(SCR, "learn", "cv_rows_months7_v2.parquet")).with_columns(
        (pl.col("y") - pl.col("b") - pl.col("c_C")).alias("e"))
    F = pl.scan_parquet("data/work_flseq/feat_train.parquet").select(["MVT_ID_mvt"] + shifters).collect()
    O = oof.join(F, on="MVT_ID_mvt", how="left")
    del F
    D = D.with_columns(pl.Series("is26", lab.astype(bool)))
    rows = []
    for c in shifters:
        for a in sorted(O["ADEP_mvt"].unique().to_list()):
            oa = O.filter(pl.col("ADEP_mvt") == a)
            x = oa[c].cast(pl.Float64).to_numpy()
            ok = ~np.isnan(x)
            if ok.sum() < 1000:
                continue
            qs = np.unique(np.quantile(x[ok], np.linspace(0, 1, 11)[1:-1]))
            bo = np.where(ok, np.searchsorted(qs, x, side="right"), -1)
            e = oa["e"].to_numpy()
            nb = len(qs) + 2
            ssum = np.bincount(bo + 1, weights=e, minlength=nb)
            cnt = np.bincount(bo + 1, minlength=nb)
            me = ssum / np.maximum(cnt + 200, 1)  # shrunk toward 0 with 200 pseudo-rows
            me = me - (ssum.sum() / cnt.sum())  # centred: only the dependence on f, not the global bias
            da = D.filter(pl.col("ADEP_mvt") == a)
            sh = {}
            for side, flag in (("25", False), ("26", True)):
                xs = da.filter(pl.col("is26") == flag)[c].cast(pl.Float64).to_numpy()
                bs = np.where(np.isnan(xs), -1, np.searchsorted(qs, xs, side="right"))
                sh[side] = np.bincount(bs + 1, minlength=nb) / len(xs)
            x25 = da.filter(~pl.col("is26"))[c].cast(pl.Float64)
            x26 = da.filter(pl.col("is26"))[c].cast(pl.Float64)
            rows.append(dict(feat=c, apt=a, mean25=x25.mean(), mean26=x26.mean(), null25=x25.is_null().mean(),
                             null26=x26.is_null().mean(), resid_range=float(me[cnt > 500].max() - me[cnt > 500].min()),
                             bias=float(((sh["26"] - sh["25"]) * me).sum())))
    B = pl.DataFrame(rows)
    B.write_parquet(os.path.join(OUT, "adv_univariate_bias.parquet"))
    log("\nunivariate implied 2026 bias (s) = sum_bins (share26 - share25) * centred mean OOF residual; |bias| >= 2 s:")
    with pl.Config(tbl_rows=80, tbl_width_chars=200, float_precision=2):
        log(B.filter(pl.col("bias").abs() >= 2).sort("bias"))
    log("per-airport sum over shifters of univariate bias (not additive, a rough upper envelope):")
    log(B.group_by("apt").agg(pl.col("bias").sum().alias("sum_bias"), pl.col("bias").abs().max().alias("max_abs")).sort("apt"))

    # C. multivariate residual model on the shifters + ADEP, LOMO over the OOF months
    feats = shifters
    code = {a: i for i, a in enumerate(sorted(set(O["ADEP_mvt"].to_list())))}
    Xo = np.hstack([O.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(),
                    O["ADEP_mvt"].replace_strict(code, return_dtype=pl.Float32).to_numpy()[:, None]]).astype(np.float32)
    eo = O["e"].to_numpy()
    yms = O["ym"].to_numpy()
    rp = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.8,
              lambda_l2=10.0, verbose=-1, num_threads=2, seed=0)
    ci = [len(feats)]
    pred = np.zeros(len(eo))
    N = len(eo)
    gains = []
    for m in sorted(set(yms)):
        te = yms == m
        b = lgb.train(rp, lgb.Dataset(Xo[~te], eo[~te], categorical_feature=ci), num_boost_round=150)
        pred[te] = b.predict(Xo[te])
        gains.append((m, float((eo[te] ** 2).sum() - ((eo[te] - pred[te]) ** 2).sum()) / N))
    log(f"\nresidual model on shifters, LOMO gain (MSE units, + = better) total {sum(g for _, g in gains):+.1f}: "
        + "  ".join(f"{m[2:]}:{g:+.1f}" for m, g in gains))
    Xo2 = Xo
    bfull = lgb.train(rp, lgb.Dataset(Xo2, eo, categorical_feature=ci), num_boost_round=150)
    R = D.filter(pl.col("is26"))
    Xr = np.hstack([R.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(),
                    R["ADEP_mvt"].replace_strict(code, return_dtype=pl.Float32).to_numpy()[:, None]]).astype(np.float32)
    pr = bfull.predict(Xr)
    po = bfull.predict(Xo2)
    res = (pl.DataFrame({"apt": R["ADEP_mvt"], "p": pr}).group_by("apt").agg(pl.col("p").mean().alias("pred26"), pl.len().alias("n26"))
           .join(pl.DataFrame({"apt": O["ADEP_mvt"], "ym": yms, "p": po, "pl": pred, "e": eo})
                 .filter(pl.col("ym").is_in(["2025-01", "2025-07"])).group_by("apt")
                 .agg(pl.col("p").mean().alias("fit25JJ"), pl.col("pl").mean().alias("lomo25JJ"), pl.col("e").mean().alias("e25JJ")),
                 on="apt").with_columns((pl.col("pred26") - pl.col("fit25JJ")).alias("implied_shift")).sort("apt"))
    log("\nresidual model: mean predicted residual on 2026 vs fitted on 2025 Jan+Jul (s); implied_shift = pred26 - fit25JJ")
    with pl.Config(tbl_rows=20, float_precision=2):
        log(res)
    tot = float((pr - po[np.isin(yms, ["2025-01", "2025-07"])].mean()).mean())
    log(f"overall mean predicted residual 2026 {pr.mean():+.2f}  vs 2025 Jan+Jul fitted {po[np.isin(yms, ['2025-01', '2025-07'])].mean():+.2f}")
    log(f"implied MSE effect of removing this mean shift on 2026 ~ shift^2 = {tot ** 2:.1f}")
    with open(os.path.join(OUT, "adv_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
