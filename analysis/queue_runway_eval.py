"""queue_runway: month-block ablation of the new runway/queue features with a small LightGBM.

Folds follow prc.cv 'fast' (jan, jul held out; the unscored 'forward' fold is skipped, docs/06 §9-1).
Training rows: labelled departures whose block time is a real off-block (clock_src in genuine/aobt3) and
30 <= y <= 7200 - i.e. the taxi-time head of the v5+ mixture. The clock-copy classes are routed elsewhere.
Label statistics (stand x runway reference, airport x runway x hour median) are fitted on the fold's training
months only. Early stopping uses an inner month (Dec for jan, Jun for jul) that is dropped from training.

Run: POLARS_MAX_THREADS=2 .venv/bin/python analysis/queue_runway_eval.py [arm ...]
Out: data/interim/queue_runway/eval_<arm>.parquet (OOF predictions), printed tables
"""

from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prc.cv import paired_bootstrap_rmse_diff, rmse  # noqa: E402
from prc.features import numeric_feature_columns  # noqa: E402

C = os.path.join(ROOT, "data", "interim", "queue_runway")
Y = "TAXITIME_SEC_mvt"
CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt"]
FOLDS = {"jan": (202501, 202512), "jul": (202507, 202506)}  # (valid month, inner early-stop month)

GROUPS = {
    "seq": ["hw_prev_pairq", "hw_excess", "lead_wtc", "burst_pos", "burst_len", "burst_elapsed", "burst_remaining",
            "burst_pos_frac", "hw_mean_prev5", "hw_mean_next5", "hw_min_prev3", "arr_same_rwy_since_prevdep",
            "arr_same_rwy_prev_s", "arr_same_rwy_next_s", "arr_same_rwy_pm10m", "rwy_share_pm30"],
    "nbr": ["nbr_seq_mvt_aobt_prev5", "nbr_seq_mvt_aobt_next5", "nbr_seq_mvt_aobt_pm5", "nbr_seq_mvt_sched_pm5",
            "overtaken_by_aobt", "overtook_aobt", "rank_shift_aobt", "nbr_rg_pm20", "nbr_rg_prev30", "nbr_rg_next30",
            "nbr_rg_pm60", "nbr_rg_asym", "nbr_seq_rg_pm5"],
    "sched_arr": ["sched_backlog_60m", "sched_demand_next30m", "sched_demand_prev30m", "arr_taxiing_at_tko",
                  "arr_taxiing_at_aobt", "arr_taxi_in_landed_20m", "arr_ti_resid_pm30", "arr_ti_resid_prev30",
                  "arr_ti_resid_next30", "arr_ti_resid_at_bhat"],
    "qhat": ["qhat_in_system_at_bhat", "qhat_tko_between_bhat_t", "qhat_in_system_at_bhat_rwy", "qhat_tko_between_rwy"],
    "cfg": ["cfg_n_dep_rwy", "cfg_n_arr_rwy", "cfg_since_change_bins", "cfg_until_change_bins", "cfg_config",
            "cfg_arr_top"],
}
ARMS = {
    "base": [],
    "seq": ["seq"],
    "nbr": ["nbr"],
    "sched_arr": ["sched_arr"],
    "qhat": ["qhat"],
    "cfg": ["cfg"],
    "all": list(GROUPS),
    "all_nocfgcat": list(GROUPS),  # all, minus the two high-cardinality configuration categoricals
    "base_seed1": [],  # training-noise reference: base with another bagging/feature-fraction seed
}
DROP = {"all_nocfgcat": {"cfg_config", "cfg_arr_top"}}
SEED = {"base_seed1": 1}
NEW_CATS = {"cfg_config", "cfg_arr_top"}


SAMPLE_FRAC = float(os.environ.get("QR_FRAC", "0.5"))  # share of phys-clean rows of non-valid months kept


def load() -> tuple[pl.DataFrame, np.ndarray, list[str], pl.DataFrame]:
    """Memory-lean: labels/keys for all training rows (for encodings), and a float32 feature matrix only for
    the rows any fold needs (all rows of the valid months, a SAMPLE_FRAC sample of the phys-clean rest)."""
    sch = pl.read_parquet_schema(os.path.join(C, "base.parquet"))
    num = [c for c in numeric_feature_columns(pl.DataFrame(schema=sch))
           if c not in ("is_rank", "ym") and not c.endswith("_slog")]
    keys = ["MVT_ID_mvt", Y, "ym", "clock_src", "hour_local", "nm_matched"] + CATS
    lab = pl.scan_parquet(os.path.join(C, "base.parquet")).filter(pl.col("is_rank") == 0).select(keys).collect()
    phys = lab["clock_src"].is_in(["genuine", "aobt3"]) & lab[Y].is_between(30, 7200)
    rng = np.random.default_rng(0)
    valid_m = [v for v, _ in FOLDS.values()]
    keep = lab["ym"].is_in(valid_m).to_numpy() | (phys.to_numpy() & (rng.random(lab.height) < SAMPLE_FRAC))
    ids = lab.filter(pl.Series(keep)).select("MVT_ID_mvt")
    new_cols = [f for g in GROUPS.values() for f in g]
    feats_all = list(dict.fromkeys(num + CATS + new_cols))
    cats = CATS + sorted(NEW_CATS)
    parts = []
    base = ids.join(pl.scan_parquet(os.path.join(C, "base.parquet")).select(
        ["MVT_ID_mvt"] + [c for c in num + CATS if c != "MVT_ID_mvt"]).collect(), on="MVT_ID_mvt", how="left",
        maintain_order="left")
    base = base.join(pl.read_parquet(os.path.join(C, "new.parquet")), on="MVT_ID_mvt", how="left", maintain_order="left")
    base = base.join(pl.read_parquet(os.path.join(C, "new2.parquet")), on="MVT_ID_mvt", how="left", maintain_order="left")
    X = np.empty((base.height, len(feats_all)), dtype=np.float32)
    for i, f in enumerate(feats_all):
        if f in cats:
            vals = base[f].drop_nulls().unique().sort().to_list()
            mp = {v: k for k, v in enumerate(vals)}
            X[:, i] = base[f].replace_strict(mp, default=None).cast(pl.Float32).fill_null(np.nan).to_numpy()
        else:
            X[:, i] = base[f].cast(pl.Float32).fill_null(np.nan).to_numpy()
    meta = base.select("MVT_ID_mvt").join(lab, on="MVT_ID_mvt", how="left", maintain_order="left")
    del base, parts
    return meta, X, feats_all, lab


def encodings(tr: pl.DataFrame) -> list[tuple[list[str], pl.DataFrame]]:
    lab = tr.filter(pl.col(Y).is_between(30, 7200))
    ref = lab.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt").agg(
        pl.col(Y).quantile(0.1).alias("ref_p10"), pl.col(Y).median().alias("ref_med"), pl.len().alias("ref_n")
    ).filter(pl.col("ref_n") >= 30)
    arh = lab.group_by("ADEP_mvt", "RUNWAY_mvt", "hour_local").agg(pl.col(Y).median().alias("arh_med"))
    return [(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], ref), (["ADEP_mvt", "RUNWAY_mvt", "hour_local"], arh)]


ENC_COLS = ["ref_p10", "ref_med", "ref_n", "arh_med"]


def enc_matrix(meta: pl.DataFrame, enc) -> np.ndarray:
    df = meta.select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local")
    for keys, t in enc:
        df = df.join(t, on=keys, how="left", maintain_order="left")
    return np.column_stack([df[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in ENC_COLS])


def run_arm(meta: pl.DataFrame, X: np.ndarray, feats_all: list[str], lab: pl.DataFrame, arm: str) -> pl.DataFrame:
    extra = [f for g in ARMS[arm] for f in GROUPS[g] if f not in DROP.get(arm, set())]
    base_feats = [f for f in feats_all if not any(f in g for g in GROUPS.values())]
    feats = list(dict.fromkeys(base_feats + extra))
    col_idx = [feats_all.index(f) for f in feats]
    cats = [f for f in feats if f in CATS or f in NEW_CATS]
    ym = meta["ym"].to_numpy()
    phys = (meta["clock_src"].is_in(["genuine", "aobt3"]) & meta[Y].is_between(30, 7200)).to_numpy()
    yv = meta[Y].to_numpy().astype(np.float64)
    outs = []
    for fold, (vm, im) in FOLDS.items():
        t0 = time.time()
        enc = encodings(lab.filter((pl.col("ym") != vm) & (pl.col("ym") != im)))
        E = enc_matrix(meta, enc)
        valid = [v for v, _ in FOLDS.values()]
        tr_m = phys & (ym != vm) & (ym != im) & ~np.isin(ym, valid)
        es_m = phys & (ym == im)
        va_m = ym == vm
        names = feats + ENC_COLS
        cat_idx = [names.index(c) for c in cats]

        def mat(m):
            return np.hstack([X[np.ix_(m, col_idx)], E[m]])

        params = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      num_threads=2, seed=SEED.get(arm, 0), max_cat_to_onehot=8, cat_smooth=20, max_bin=127)
        dtr = lgb.Dataset(mat(tr_m), yv[tr_m], categorical_feature=cat_idx, feature_name=names, free_raw_data=True)
        des = lgb.Dataset(mat(es_m), yv[es_m], reference=dtr, categorical_feature=cat_idx, feature_name=names)
        m = lgb.train(params, dtr, 3000, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(mat(va_m), num_iteration=m.best_iteration)
        imp = dict(zip(names, m.feature_importance("gain").tolist()))
        with open(os.path.join(C, f"imp_{arm}_{fold}.json"), "w") as fh:
            json.dump(imp, fh)
        outs.append(meta.filter(pl.Series(va_m)).select("MVT_ID_mvt", "ADEP_mvt", Y, "clock_src", "nm_matched")
                    .with_columns(pl.Series("pred", p), pl.lit(fold).alias("fold")))
        print(f"  {arm}/{fold}: ntr={tr_m.sum()} iters={m.best_iteration} {time.time()-t0:.0f}s", flush=True)
        del dtr, des, m
    o = pl.concat(outs)
    o.write_parquet(os.path.join(C, f"eval_{arm}.parquet"))
    return o


def report(arms: list[str]) -> None:
    b = pl.read_parquet(os.path.join(C, "eval_base.parquet"))
    subsets = {
        "genuine clean": (pl.col("clock_src") == "genuine") & pl.col(Y).is_between(30, 7200),
        "phys clean": pl.col("clock_src").is_in(["genuine", "aobt3"]) & pl.col(Y).is_between(30, 7200),
        "NM-unmatched genuine clean": (pl.col("nm_matched") == 0) & (pl.col("clock_src") == "genuine")
        & pl.col(Y).is_between(30, 7200),
    }
    rows = []
    for arm in arms:
        o = pl.read_parquet(os.path.join(C, f"eval_{arm}.parquet"))
        j = b.select("MVT_ID_mvt", pl.col("pred").alias("p0")).join(o, on="MVT_ID_mvt")
        for sname, pred in subsets.items():
            for fold in ("jan", "jul", "both"):
                z = j.filter(pred) if fold == "both" else j.filter(pred & (pl.col("fold") == fold))
                y, p0, p1 = z[Y].to_numpy().astype(float), z["p0"].to_numpy(), z["pred"].to_numpy()
                bs = paired_bootstrap_rmse_diff(y, p0, p1, n_boot=300)
                rows.append({"arm": arm, "subset": sname, "fold": fold, "n": len(y), "rmse": rmse(y, p1),
                             "d_vs_base": bs["diff_rmse"], "ci_lo": bs["ci95"][0], "ci_hi": bs["ci95"][1]})
    r = pl.DataFrame(rows)
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(r)
    r.write_csv(os.path.join(C, "eval_summary.csv"))


def main() -> None:
    arms = sys.argv[1:] or list(ARMS)
    meta, X, feats_all, lab = load()
    print("loaded", X.shape, f"{X.nbytes/1e9:.2f} GB", flush=True)
    for arm in arms:
        if arm == "report":
            continue
        if os.path.exists(os.path.join(C, f"eval_{arm}.parquet")) and os.environ.get("QR_FORCE") != "1":
            print("skip", arm)
            continue
        run_arm(meta, X, feats_all, lab, arm)
    done = [a for a in ARMS if os.path.exists(os.path.join(C, f"eval_{a}.parquet"))]
    report(done)


if __name__ == "__main__":
    main()
