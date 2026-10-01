"""Month-blocked A/B: existing-style LightGBM (src/prc/features.py features + stand x runway reference) vs the same
model plus the new stand features. Folds 'jan' and 'jul' (train on the other 10 months of 2025, early stopping on
one inner month), exactly the months that are scored.

Model per fold (the v2 'two-clock' structure of src/prc/train.py):
  * NM-matched rows: target y - (MVT - AOBT_3)
  * NM-unmatched rows: target y
Usage:  python stand_03_model.py [variant ...]      variants: base, occ, nbr, geo, grp, occ_grp, all (see VARIANTS)
        python stand_03_model.py report <variant ...>
Env: RUN_TAG (suffix of the OOF files), TRAIN_FRAC (share of NM-matched fitting rows), LGB_LR, LGB_THREADS,
     LOAD_EXTRA=0 (skip neighbour/successor/geo files to save memory).
Writes SCRATCH/oof_<variant><RUN_TAG>.parquet and prints RMSE + paired bootstrap vs 'base'.
Peak RSS ~3.8 GB with all feature files loaded.
"""

from __future__ import annotations

import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from stand_common import SCRATCH

NUM_THREADS = int(os.environ.get("LGB_THREADS", "2"))
LABEL = "TAXITIME_SEC_mvt"
PROXY = "mvt_minus_aobt3"
CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
        "AIRCRAFT_OPERATOR_flt"]
OCC = ["g_arr_prev1", "g_arr_prev2", "g_arr_next1", "g_dep_prev", "g_dep_next", "n_arr_since_prev_dep",
       "g_next_dep_aobt", "aobt_minus_arr_prev1", "aobt_minus_arr_prev2", "occ_lb", "occ_lb_minus_aobt",
       "occ_lb_minus_sched"]
GEO_FILE = f"{SCRATCH}/stand_geo_feats.parquet"
EXTRA_FILES = [f"{SCRATCH}/stand_nbr_feats.parquet", f"{SCRATCH}/stand_succ_feats.parquet"]
if os.environ.get("LOAD_EXTRA", "1") == "0":  # memory: variants that need only OCC / grp columns
    EXTRA_FILES = []
NBR = ["nb_dep_aobt_5m", "nb_arr_inb_5m", "same_num_dep_5m", "nb_dep_tko_5m"]
SUCC = ["succ_taxiin", "succ_taxiin_excess", "succ_land_before_tko_s"]
GEO = ["geo_x_m", "geo_y_m", "geo_dist_thr_m"]
VARIANTS: dict[str, list[str]] = {
    "base": [],
    "occ": OCC,
    "occ_min": ["g_arr_prev1", "occ_lb_minus_aobt", "n_arr_since_prev_dep"],
    "nbr": NBR,
    "geo": GEO,
    "all": OCC + NBR + SUCC + GEO,
    "grp": ["grp_p10", "grp_med", "grp_n", "stand_num"],
    "occ_grp": OCC + ["grp_p10", "grp_med", "grp_n", "stand_num"],
}
PARAMS = dict(objective="regression", learning_rate=float(os.environ.get("LGB_LR", "0.06")), num_leaves=127, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              num_threads=NUM_THREADS, deterministic=True, seed=42, max_bin=127)


def load() -> pl.DataFrame:
    frames = [pl.read_parquet(f) for f in sorted(glob.glob(f"{SCRATCH}/base_*.parquet"))]
    df = pl.concat(frames, how="diagonal_relaxed")
    occ = pl.read_parquet(f"{SCRATCH}/stand_feats.parquet").drop(["n_arr_during_taxi", "lb_true", "turn_s"])
    df = df.join(occ, on="MVT_ID_mvt", how="left")
    df = df.with_columns(
        pl.when(pl.col("g_arr_prev1") < 1800).then(pl.col("g_arr_prev1")).alias("occ_lb"),
    ).with_columns(
        (pl.col("occ_lb") - pl.col(PROXY)).alias("occ_lb_minus_aobt"),
        (pl.col("occ_lb") - pl.col("mvt_minus_sched")).alias("occ_lb_minus_sched"),
    )
    for fpath in ([GEO_FILE] if EXTRA_FILES else []) + EXTRA_FILES:
        if os.path.exists(fpath):
            df = df.join(pl.read_parquet(fpath).drop("geo_matched", strict=False), on="MVT_ID_mvt", how="left")
    df = df.with_columns(
        (pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("_month"),
        pl.col("ADEP_mvt").alias("APT_s"),
        (pl.col("ADEP_mvt") + "|" + pl.col("STAND_mvt").fill_null("NA")).alias("STAND_mvt"),
        # stand group = airport | letter prefix | number // 10  (docs/analysis/stand.md section 1 and 5)
        (pl.col("ADEP_mvt") + "|" + pl.col("STAND_mvt").fill_null("").str.extract(r"^([A-Za-z]*)", 1).fill_null("")
         + "|" + (pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False) // 10).cast(pl.String)
         .fill_null("x")).cast(pl.Categorical).to_physical().cast(pl.Int32).alias("STAND_GRP"),
        pl.col("STAND_mvt").str.extract(r"(\d+)", 1).cast(pl.Float32, strict=False).alias("stand_num"),
    )
    for c in CATS:
        df = df.with_columns(pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32).alias(c))
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c, t in df.schema.items() if t == pl.Float64 and c not in (LABEL, "MVT_ID_mvt")])
    # the Datetime columns are only needed for the month; the NM clocks are already encoded as deltas
    df = df.drop([c for c, t in df.schema.items() if isinstance(t, pl.Datetime)])
    return df


def ref_table(tr: pl.DataFrame) -> pl.DataFrame:
    return (
        tr.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .agg(
            pl.col(LABEL).quantile(0.10).alias("ref_p10"),
            pl.col(LABEL).quantile(0.20).alias("ref_p20"),
            pl.col(LABEL).median().alias("ref_med"),
            pl.len().alias("ref_n"),
        )
        .filter(pl.col("ref_n") >= 30)
    )


TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.6"))
TAG = os.environ.get("RUN_TAG", "")  # e.g. '_fast' for the cheaper ablation config


def grp_table(tr: pl.DataFrame) -> pl.DataFrame:
    return (
        tr.group_by(["ADEP_mvt", "STAND_GRP", "RUNWAY_mvt"])
        .agg(pl.col(LABEL).quantile(0.10).alias("grp_p10"), pl.col(LABEL).median().alias("grp_med"),
             pl.len().alias("grp_n"))
        .filter(pl.col("grp_n") >= 30)
    )


def fit_predict(tr: pl.DataFrame, va: pl.DataFrame, cols: list[str], target_fn) -> tuple[np.ndarray, int]:
    inner_m = tr["_inner"][0]
    cat_idx = [i for i, c in enumerate(cols) if c in CATS]
    fit = tr.filter(pl.col("_month") != inner_m)  # already subsampled in run_variant
    hold = tr.filter(pl.col("_month") == inner_m)
    dtr = lgb.Dataset(fit.select(cols).to_numpy(), target_fn(fit), categorical_feature=cat_idx,
                      free_raw_data=True, params={"max_bin": 127})
    dtr.construct()
    del fit
    dva = lgb.Dataset(hold.select(cols).to_numpy(), target_fn(hold), categorical_feature=cat_idx, reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    p = m.predict(va.select(cols).to_numpy(), num_iteration=m.best_iteration)
    imp = dict(zip(cols, m.feature_importance("gain")))
    return p, m.best_iteration, imp


def run_variant(df: pl.DataFrame, name: str) -> pl.DataFrame:
    extra = VARIANTS[name]
    base_cols = [
        c for c in df.columns
        if c not in (LABEL, "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "APT_s", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                     "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "month", "_month", "_inner")
        and c not in OCC + NBR + SUCC + GEO + ["STAND_GRP", "stand_num"] and df.schema[c] != pl.Datetime
        and df.schema[c].is_numeric()
    ]
    cols_all = base_cols + ["ref_p10", "ref_p20", "ref_med", "ref_n"] + extra
    needed_extra = [c for c in extra if c in df.columns]
    outs = []
    for fold, inner in (("jan", 12), ("jul", 6)):
        m = 1 if fold == "jan" else 7
        needed = list(dict.fromkeys([c for c in cols_all if c in df.columns] + [
            "MVT_ID_mvt", "APT_s", LABEL, PROXY, "_month", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "STAND_GRP"]))
        tr = df.filter(pl.col("_month") != m).select(needed)
        va = df.filter(pl.col("_month") == m).select(needed)
        ref = ref_table(tr.filter(pl.col("_month") != inner))
        grp = grp_table(tr.filter(pl.col("_month") != inner))
        # subsample the NM-matched fitting rows (memory/time budget); keep the inner month and unmatched rows whole
        fit_m = (pl.col("_month") != inner) & pl.col(PROXY).is_not_null()
        if TRAIN_FRAC < 1:
            tr = pl.concat([tr.filter(~fit_m), tr.filter(fit_m).sample(fraction=TRAIN_FRAC, seed=7)])
        tr = tr.with_columns(pl.lit(inner).alias("_inner"))
        tr = tr.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
        va = va.join(ref, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
        if any(c.startswith("grp_") for c in extra):
            tr = tr.join(grp, on=["ADEP_mvt", "STAND_GRP", "RUNWAY_mvt"], how="left")
            va = va.join(grp, on=["ADEP_mvt", "STAND_GRP", "RUNWAY_mvt"], how="left")
        pred = np.zeros(va.height)
        for matched in (True, False):
            sel_tr = tr.filter(pl.col(PROXY).is_not_null() == matched)
            vmask = (va[PROXY].is_not_null() == matched).to_numpy()
            sel_va = va.filter(pl.col(PROXY).is_not_null() == matched)
            if matched:
                cols = cols_all
                p, it, imp = fit_predict(sel_tr, sel_va, cols, lambda f: (f[LABEL] - f[PROXY]).to_numpy().astype(float))
                pred[vmask] = p + sel_va[PROXY].to_numpy()
            else:
                cols = [c for c in cols_all if not c.startswith(("mvt_minus_aobt3", "aobt3", "lobt_minus",
                        "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt", "occ_lb_minus_aobt", "aobt_minus"))]
                p, it, imp2 = fit_predict(sel_tr, sel_va, cols, lambda f: f[LABEL].to_numpy().astype(float))
                pred[vmask] = p
            print(f"  {name} {fold} matched={matched} rounds={it}", flush=True)
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:40]
        tot = sum(imp.values())
        print("  top gain:", [(k, round(v / tot, 4)) for k, v in top[:12]])
        print("  new-feature gain:", [(k, round(imp[k] / tot, 4)) for k in extra if k in imp])
        outs.append(va.select(["MVT_ID_mvt", "APT_s", LABEL, PROXY]).with_columns(
            pl.Series("pred", np.clip(pred, 0, None)), pl.lit(fold).alias("fold")))
    oof = pl.concat(outs)
    oof.write_parquet(f"{SCRATCH}/oof_{name}{TAG}.parquet")
    return oof


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def paired(y, pa, pb, n=500, seed=0):
    rng = np.random.default_rng(seed)
    sa, sb = (y - pa) ** 2, (y - pb) ** 2
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        d[i] = np.sqrt(sb[idx].mean()) - np.sqrt(sa[idx].mean())
    return float(np.sqrt(sb.mean()) - np.sqrt(sa.mean())), np.percentile(d, [2.5, 97.5]), float((d < 0).mean())


def report(names: list[str]) -> None:
    base = pl.read_parquet(f"{SCRATCH}/oof_base{TAG}.parquet")
    for n in names:
        o = pl.read_parquet(f"{SCRATCH}/oof_{n}{TAG}.parquet")
        j = base.join(o.select(["MVT_ID_mvt", pl.col("pred").alias("pb")]), on="MVT_ID_mvt")
        y, pa, pb = j[LABEL].to_numpy().astype(float), j["pred"].to_numpy(), j["pb"].to_numpy()
        clean = (y >= 30) & (y <= 7200)
        d, ci, pr = paired(y, pa, pb)
        dc, cic, prc = paired(y[clean], pa[clean], pb[clean])
        print(f"{n}: rmse base={rmse(y, pa):.2f} new={rmse(y, pb):.2f} diff={d:+.2f} CI{ci.round(2)} P={pr:.3f} | "
              f"clean base={rmse(y[clean], pa[clean]):.2f} new={rmse(y[clean], pb[clean]):.2f} diff={dc:+.2f} "
              f"CI{cic.round(2)} P={prc:.3f}")
        rows = []
        for (apt,), g in j.group_by(["APT_s"]):
            yy, a, b = g[LABEL].to_numpy().astype(float), g["pred"].to_numpy(), g["pb"].to_numpy()
            c = (yy >= 30) & (yy <= 7200)
            rows.append((apt, len(yy), rmse(yy, a), rmse(yy, b), rmse(yy[c], a[c]), rmse(yy[c], b[c])))
        for f in ("jan", "jul"):
            g = j.filter(pl.col("fold") == f)
            yy, a, b = g[LABEL].to_numpy().astype(float), g["pred"].to_numpy(), g["pb"].to_numpy()
            print(f"   fold {f}: {rmse(yy, a):.2f} -> {rmse(yy, b):.2f}")
        print(pl.DataFrame(rows, schema=["apt", "n", "base", "new", "base_clean", "new_clean"], orient="row")
              .sort("apt"))


if __name__ == "__main__":
    names = sys.argv[1:] or ["base", "occ"]
    if names[0] == "report":
        report(names[1:])
        sys.exit()
    df = load()
    print("rows", df.height, "cols", df.width, flush=True)
    for n in names:
        if os.path.exists(f"{SCRATCH}/oof_{n}{TAG}.parquet") and n == "base":
            continue
        run_variant(df, n)
    report([n for n in names if n != "base"])
