"""operator_aircraft_route 06c: month-block A/B of the new identity / flight-TE / stand / route features.

Model = a light replica of the repo's v5/v7 clock mixture (docs/06 §8):
  * classifier over {genuine, aobt3, sched, eobt1, lobt+iobt} (LightGBM multiclass; rollover rows -> genuine),
  * taxi head fitted on genuine+aobt3 rows, target y - b with b = MVT-AOBT_3 (or the airport median when absent),
  * prediction = p_gen * taxi_head + sum_k p_k * (MVT - clock_k)   (posterior mean; missing clocks -> taxi head).
Base features = prc.features.build_departure_features (row-local + context; 06a) + the repo's reference encodings
(prc.train.fit_encodings, fit on the training side of each fold).
Folds = jan and jul (fit on the other 11 months of 2025), judged on the pooled jan+jul rows (docs/06 §9-1).
Fixed round counts, identical for all arms, num_threads=2. Memory-lean: one float32 matrix, per-fold row copies.

usage: python operator_aircraft_route_06c_ablation.py [arm ...]   (default: all arms)
Appends one JSON line per (arm, fold) to data/interim/oar_ablation.jsonl, OOF to data/interim/oar_oof_<arm>.parquet.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from operator_aircraft_route_common import INTERIM, load_dep  # noqa: E402
from prc.train import TrainConfig, attach_encodings, fit_encodings  # noqa: E402

CLASSES = ["genuine", "aobt3", "sched", "eobt1", "lobt"]
CAND = {"aobt3": "mvt_minus_aobt3", "sched": "mvt_minus_sched", "eobt1": "mvt_minus_eobt1", "lobt": "mvt_minus_lobt"}
BASE_CATS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
             "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "ADES_mvt", "ADES_REGION"]
IDENT_CATS = ["airline", "flight_fmt"]
REF_COLS = ["ref_p10", "ref_p10_fb", "ref_missing", "arh_med", "apt_med"]
TE_COLS = ["te_fl_r", "te_fl_n", "te_fl_sched", "te_fl_aobt3", "te_fl_eobt1", "te_al_r", "te_al_sched", "te_al_aobt3"]
TURN_COLS = ["ta_gap", "ta_aobt3_minus_inblock", "ta_sched_slack", "arr_inblock_delay", "ta_linked", "ta_same_airline",
             "ta_clean", "arr_taxi_in", "stand_lb", "sched_minus_lb", "aobt3_minus_lb"]
ROUTE_COLS = ["dist_km", "wide", "heli"]
ARMS = {
    "base": ([], []),
    "ident": (IDENT_CATS, []),
    "te": ([], TE_COLS),
    "turn": ([], TURN_COLS),
    "route": ([], ROUTE_COLS),
    "all": (IDENT_CATS, TE_COLS + TURN_COLS + ROUTE_COLS),
}
# window counts at 5/10 min duplicate the 15/30/60 ones; dropped to keep the matrix within the memory budget
DROP_BASE = ("dep_tko_prev_5m", "dep_tko_prev_10m", "arr_land_prev_5m", "arr_land_prev_10m", "arr_inblock_prev_5m",
             "arr_inblock_prev_10m", "dep_tko_next_60m", "arr_land_next_60m", "month")
CLF_PARAMS = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.12, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=2, seed=42, deterministic=True, force_row_wise=True)
REG_PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2, seed=42,
                  deterministic=True, force_row_wise=True)
DS_PARAMS = dict(max_bin=127, verbose=-1)
CLF_ROUNDS, REG_ROUNDS = 150, 500
TRAIN_FRAC = 0.5  # shared CPU: every arm trains on the same 50% row subsample of the training months
SMOKE = os.environ.get("OAR_SMOKE") == "1"
if SMOKE:
    CLF_ROUNDS, REG_ROUNDS = 10, 20
CLIP = 3600.0


class Data:
    """X: float32 [N, F] with named columns; K: small polars frame (keys, labels, candidates) in the same row order."""

    def __init__(self) -> None:
        bpath = os.path.join(INTERIM, "oar_basefeats_train.parquet")
        schema = pl.scan_parquet(bpath).collect_schema()
        lab = load_dep(["MVT_ID_mvt", "clock_src", "ym", "FLIGHT_mvt"]).filter(pl.col("clock_src").is_not_null())
        new = pl.read_parquet(os.path.join(INTERIM, "oar_newfeats.parquet")).drop("split")
        new = new.join(lab, on="MVT_ID_mvt", how="inner")
        del lab
        skip = set(BASE_CATS + IDENT_CATS + TURN_COLS + ROUTE_COLS + list(DROP_BASE)) | {
            "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}
        self.base_num = [c for c, t in schema.items() if c not in skip and t.is_numeric()]
        # global category levels (label-free, shared by every fold like the repo's pl.Enum)
        levels = {c: pl.scan_parquet(bpath).select(pl.col(c).drop_nulls().unique()).collect()[c].sort() for c in BASE_CATS}
        levels.update({c: new[c].drop_nulls().unique().sort() for c in IDENT_CATS})
        self.cols = self.base_num + [f"{c}__code" for c in BASE_CATS + IDENT_CATS] + TURN_COLS + ROUTE_COLS + REF_COLS + TE_COLS
        self.ci = {c: i for i, c in enumerate(self.cols)}
        n_all = new.height
        self.X = np.full((n_all, len(self.cols)), np.nan, dtype=np.float32)
        ks = []
        pos = 0
        for ym in sorted(new["ym"].unique().to_list()):
            nm = new.filter(pl.col("ym") == ym)
            if SMOKE:
                nm = nm.sample(fraction=0.05, seed=0)
            b = pl.scan_parquet(bpath).filter(pl.col("MVT_ID_mvt").is_in(nm["MVT_ID_mvt"].implode())).collect()
            df = b.join(nm, on="MVT_ID_mvt", how="inner")
            del b
            df = df.with_columns(
                (pl.col("mvt_minus_sched") - pl.col("stand_lb")).alias("sched_minus_lb"),
                (pl.col("mvt_minus_aobt3") - pl.col("stand_lb")).alias("aobt3_minus_lb"),
                pl.col("clock_src").replace({"iobt": "lobt", "rollover_24h": "genuine"}).alias("cls"),
                pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            )
            codes = []
            for c in BASE_CATS + IDENT_CATS:
                mp = pl.DataFrame({c: levels[c], f"{c}__code": np.arange(len(levels[c]), dtype=np.float32)})
                codes.append(df.select(c).join(mp, on=c, how="left", maintain_order="left")[f"{c}__code"])
            df = df.with_columns(codes)
            h = df.height
            num_cols = self.base_num + [f"{c}__code" for c in BASE_CATS + IDENT_CATS] + TURN_COLS + ROUTE_COLS
            self.X[pos:pos + h, : len(num_cols)] = df.select([pl.col(c).cast(pl.Float32) for c in num_cols]).to_numpy()
            ks.append(df.select("MVT_ID_mvt", "ym", "y", "TAXITIME_SEC_mvt", "cls", "clock_src", "ADEP_mvt", "STAND_mvt",
                                "RUNWAY_mvt", "hour_local", "FLIGHT_mvt", "airline", *CAND.values()))
            pos += h
            del df
            gc.collect()
        self.X = self.X[:pos]
        self.K = pl.concat(ks)
        print("X", self.X.shape, f"{self.X.nbytes / 1e9:.2f} GB", flush=True)

    def put(self, rows: np.ndarray, frame: pl.DataFrame, cols: list[str]) -> None:
        for c in cols:
            self.X[rows, self.ci[c]] = frame[c].cast(pl.Float32).fill_null(np.nan).to_numpy()


class Rows(lgb.Sequence):
    """Row/column view of D.X streamed to LightGBM in batches (avoids a full training-matrix copy)."""

    def __init__(self, X: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> None:
        self.X, self.rows, self.cols, self.batch_size = X, rows, cols, 65536

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.X[self.rows[idx], self.cols].astype(np.float64)
        return self.X[np.ix_(self.rows[idx], self.cols)].astype(np.float64)

    def __len__(self) -> int:
        return len(self.rows)


def add_te(fit: pl.DataFrame, tgt: pl.DataFrame) -> pl.DataFrame:
    """Label-derived encodings fit on `fit` only, joined onto `tgt` (order preserved)."""
    ref = fit.group_by("ADEP_mvt", "STAND_mvt").agg(pl.col("y").clip(0, CLIP).median().alias("_ref"), pl.len().alias("_rn"))
    apt = fit.group_by("ADEP_mvt").agg(pl.col("y").clip(0, CLIP).median().alias("_apt"))
    f = fit.join(ref, on=["ADEP_mvt", "STAND_mvt"], how="left").join(apt, on="ADEP_mvt", how="left").with_columns(
        (pl.col("y").clip(0, CLIP) - pl.when(pl.col("_rn") >= 10).then(pl.col("_ref")).otherwise(pl.col("_apt"))).alias("_r"))
    rates = ("sched", "aobt3", "eobt1")
    ap = f.group_by("ADEP_mvt").agg([(pl.col("cls") == c).mean().alias(f"_ap_{c}") for c in rates])
    al = f.group_by("ADEP_mvt", "airline").agg(pl.len().alias("_aln"), pl.col("_r").sum().alias("_alr"),
                                               *[(pl.col("cls") == c).sum().alias(f"_al_{c}") for c in rates])
    fl = f.group_by("ADEP_mvt", "FLIGHT_mvt").agg(pl.len().alias("te_fl_n"), pl.col("_r").sum().alias("_flr"),
                                                  *[(pl.col("cls") == c).sum().alias(f"_fl_{c}") for c in rates])
    t = tgt.select("ADEP_mvt", "airline", "FLIGHT_mvt")
    t = t.join(ap, on="ADEP_mvt", how="left", maintain_order="left")
    t = t.join(al, on=["ADEP_mvt", "airline"], how="left", maintain_order="left", nulls_equal=True)
    t = t.join(fl, on=["ADEP_mvt", "FLIGHT_mvt"], how="left", maintain_order="left")
    A, K = 20.0, 10.0
    t = t.with_columns(
        [(pl.col("_alr").fill_null(0) / (pl.col("_aln").fill_null(0) + 30.0)).alias("te_al_r")]
        + [((pl.col(f"_al_{c}").fill_null(0) + A * pl.col(f"_ap_{c}")) / (pl.col("_aln").fill_null(0) + A)).alias(f"te_al_{c}")
           for c in rates]
    )
    t = t.with_columns(
        [(pl.col("_flr").fill_null(0) / (pl.col("te_fl_n").fill_null(0) + K)).alias("te_fl_r"),
         pl.col("te_fl_n").fill_null(0)]
        + [((pl.col(f"_fl_{c}").fill_null(0) + K * pl.col(f"te_al_{c}")) / (pl.col("te_fl_n").fill_null(0) + K)).alias(f"te_fl_{c}")
           for c in rates]
    )
    assert t.height == tgt.height
    return t.select(TE_COLS)


def run_fold(D: Data, month: int, arm: str) -> tuple[dict, pl.DataFrame]:
    extra_cats, extra_num = ARMS[arm]
    K = D.K
    is_te = (K["ym"] % 100 == month).to_numpy()
    idx_tr, idx_te = np.where(~is_te)[0], np.where(is_te)[0]
    Ktr, Kte = K.filter(~pl.Series(is_te)), K.filter(pl.Series(is_te))
    # fixed subsample (same rows for every arm) used to fit the two heads; encodings still use all training rows
    sub_sel = np.random.default_rng(month).random(len(idx_tr)) < TRAIN_FRAC
    # repo reference encodings (fit on the training side)
    enc = fit_encodings(Ktr, TrainConfig(reference_min_count=30))
    D.put(idx_tr, attach_encodings(Ktr, enc), REF_COLS)
    D.put(idx_te, attach_encodings(Kte, enc), REF_COLS)
    if any(c in TE_COLS for c in extra_num):
        D.put(idx_te, add_te(Ktr, Kte), TE_COLS)
        for m in Ktr["ym"].unique().to_list():
            sel = (Ktr["ym"] == m).to_numpy()
            D.put(idx_tr[sel], add_te(Ktr.filter(~pl.Series(sel)), Ktr.filter(pl.Series(sel))), TE_COLS)
    cats = BASE_CATS + extra_cats
    feats = D.base_num + REF_COLS + extra_num + [f"{c}__code" for c in cats]
    cols = np.array([D.ci[c] for c in feats])
    cat_idx = [feats.index(f"{c}__code") for c in cats]
    cls_map = {c: i for i, c in enumerate(CLASSES)}
    ytr_cls = Ktr["cls"].replace_strict(cls_map, return_dtype=pl.Int32).to_numpy()[sub_sel]
    ds = lgb.Dataset(Rows(D.X, idx_tr[sub_sel], cols), label=ytr_cls, categorical_feature=cat_idx, free_raw_data=True,
                     params=DS_PARAMS)
    ds.construct()
    gc.collect()
    t0 = time.time()
    clf = lgb.train(CLF_PARAMS, ds, CLF_ROUNDS)
    t_clf = time.time() - t0
    b_tr = Ktr["mvt_minus_aobt3"].to_numpy().astype(float)[sub_sel]
    apt_tr = D.X[idx_tr[sub_sel], D.ci["apt_med"]].astype(float)
    b_tr = np.where(np.isfinite(b_tr), b_tr, apt_tr)
    gmask = (Ktr["cls"].is_in(["genuine", "aobt3"]).to_numpy() & (Ktr["clock_src"] != "rollover_24h").to_numpy())[sub_sel]
    gidx = np.where(gmask)[0]
    sub = ds.subset(gidx)
    sub.construct()  # a label set before construct() is silently ignored by subset datasets (lightgbm 4.7)
    sub.set_label((Ktr["y"].to_numpy()[sub_sel] - b_tr)[gidx])
    t0 = time.time()
    reg = lgb.train(REG_PARAMS, sub, REG_ROUNDS)
    t_reg = time.time() - t0
    del ds, sub
    gc.collect()
    Xte = D.X[np.ix_(idx_te, cols)]
    P = clf.predict(Xte)
    b_te = Kte["mvt_minus_aobt3"].to_numpy().astype(float)
    b_te = np.where(np.isfinite(b_te), b_te, D.X[idx_te, D.ci["apt_med"]].astype(float))
    g = b_te + reg.predict(Xte)
    pred = P[:, 0] * g
    for k, name in enumerate(CLASSES[1:], start=1):
        c = Kte[CAND[name]].to_numpy().astype(float)
        pred += np.where(np.isfinite(c), P[:, k] * np.nan_to_num(c), P[:, k] * g)
    y = Kte["y"].to_numpy()
    yc = Kte["cls"].replace_strict(cls_map, return_dtype=pl.Int32).to_numpy()
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(yc)), yc], 1e-9, 1))))
    newf = extra_num + [f"{c}__code" for c in extra_cats]
    imp = dict(zip(feats, reg.feature_importance("gain").tolist()))
    impc = dict(zip(feats, clf.feature_importance("gain").tolist()))
    res = dict(arm=arm, fold=month, n=len(y), rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
               rmse_clean=float(np.sqrt(np.mean(((y - pred)[y < 7200]) ** 2))), clf_logloss=ll,
               t_clf=round(t_clf), t_reg=round(t_reg), n_feats=len(feats),
               imp_reg_new={k: round(imp[k] / sum(imp.values()) * 100, 3) for k in newf},
               imp_clf_new={k: round(impc[k] / sum(impc.values()) * 100, 3) for k in newf})
    oof = pl.DataFrame({"MVT_ID_mvt": Kte["MVT_ID_mvt"], "y": y, "pred": pred, "g": g, "p_gen": P[:, 0],
                        "p_sched": P[:, 2], "cls": Kte["cls"]})
    # reset fold-specific columns
    D.X[:, [D.ci[c] for c in REF_COLS + TE_COLS]] = np.nan
    return res, oof


def main() -> None:
    arms = sys.argv[1:] or list(ARMS)
    D = Data()
    log = os.path.join(INTERIM, "oar_ablation_smoke.jsonl" if SMOKE else "oar_ablation.jsonl")
    for arm in arms:
        oofs = []
        for month in (1, 7):
            res, oof = run_fold(D, month, arm)
            print(json.dumps(res), flush=True)
            with open(log, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            oofs.append(oof)
            gc.collect()
        if not SMOKE:
            pl.concat(oofs).write_parquet(os.path.join(INTERIM, f"oar_oof_{arm}.parquet"))


if __name__ == "__main__":
    main()
