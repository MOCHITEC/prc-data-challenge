"""Shift worker, task 3b: frame-pooled features as a train/test skew (local, read-only).

The adversarial classifier (local_shift_adversarial.py) separates 2025 Jan+Jul from 2026 at AUC ~0.998, led by
features whose value depends on the FRAME they are computed over: fl_nm_rate_other (NM match rate of the other
departures with the same flight number), seq_hw_prev_pairq (P5 headway per airport x runway x WTC pair) and the
seq_* columns built on it. feat_train pools them over 12 months of 2025, feat_rank over Jan+Jul 2026.

This script:
 1. Adversarial AUC restricted to the columns v40's base model actually sees (main_flseq_osm_nolirf).
 2. Recomputes EXTRA_COLS for the 2025 Jan+Jul departures with the frame = Jan+Jul 2025 only (the 2026 pooling).
 3. Fits a proxy of the base model (lane A: g = y - proxy, lane B: y) on the other ten months with the original
    12-month-pooled features, predicts Jan+Jul 2025 with (a) the original and (b) the 2-month-pooled features.
    (b) - (a) estimates what the pooling artefact costs on 2026; a union-pooled 2026 frame would approximate (a).

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_shift_pool.py
Outputs: $SCRATCH/agents/shift/pool_*.parquet / pool_report.txt
"""
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from prc.features import CATEGORICALS, mask_departures  # noqa: E402
from prc.features_extra import EXTRA_COLS, extra_candidate_features  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import unmatched_columns, visible_numeric_columns  # noqa: E402

SCR = os.environ["SCRATCH"]
OUT = os.path.join(SCR, "agents", "shift")
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def catcode(frames: list[pl.DataFrame], cats: list[str]) -> list[np.ndarray]:
    lv = {c: pl.concat([f.select(pl.col(c).cast(pl.Utf8)) for f in frames])[c].drop_nulls().unique().sort() for c in cats}
    outs = []
    for f in frames:
        outs.append(np.column_stack([f[c].cast(pl.Utf8).cast(pl.Enum(lv[c])).to_physical().cast(pl.Float32)
                                     .fill_null(np.nan).to_numpy() for c in cats]).astype(np.float32))
    return outs


def mat(f: pl.DataFrame, num: list[str], C: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.hstack([f.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy(), C]),
                                dtype=np.float32)


def main() -> None:
    spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf.yaml"))
    spec.pop("name", None)
    for k in ("use_weather", "wx_codes", "wind_dir", "gust_repair", "coverage_gate", "coverage_allow", "note"):
        spec.pop(k, None)
    cfg = train_config_from_spec(spec)
    head = pl.read_parquet("data/work_flseq/feat_rank.parquet", n_rows=10)
    num = [c for c in visible_numeric_columns(head, cfg) if not c.startswith("noise_control")]
    numB = unmatched_columns(num)
    cats = [c for c in CATEGORICALS if c in head.columns]
    log(f"visible numeric {len(num)} (lane B {len(numB)}), categoricals {len(cats)}")
    pooled = [c for c in EXTRA_COLS if c in num]
    log(f"visible EXTRA cols recomputed: {pooled}")

    need = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"] + num + cats
    need = list(dict.fromkeys(need))
    f32 = [pl.col(c).cast(pl.Float32) for c in num if c != "TAXITIME_SEC_mvt"]
    tr = pl.scan_parquet("data/work_flseq/feat_train.parquet").select(need).with_columns(
        *f32, pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect()
    rk = pl.read_parquet("data/work_flseq/feat_rank.parquet", columns=need).with_columns(*f32)

    # 1. adversarial on visible columns
    jj = tr.filter(pl.col("mon").is_in([1, 7]))
    lab = np.r_[np.zeros(jj.height), np.ones(rk.height)]
    Cj, Cr = catcode([jj, rk], cats)
    X = np.vstack([mat(jj, num, Cj), mat(rk, num, Cr)])
    names = num + cats
    ho = np.random.default_rng(0).random(len(lab)) < 0.5
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               verbose=-1, num_threads=2, seed=0, cat_smooth=20)
    ci = list(range(len(num), len(names)))
    for tag, cols in (("visible numeric+cats", list(range(len(names)))),
                      ("visible minus pooled", [i for i, n in enumerate(names) if n not in pooled])):
        b = lgb.train(prm, lgb.Dataset(X[~ho][:, cols], lab[~ho], feature_name=[names[i] for i in cols],
                                       categorical_feature=[k for k, i in enumerate(cols) if i in ci]), 300)
        auc = roc_auc_score(lab[ho], b.predict(X[ho][:, cols]))
        g = b.feature_importance("gain")
        o = np.argsort(-g)[:10]
        log(f"[adv {tag}] AUC {auc:.4f}  " + ", ".join(f"{names[cols[i]]} {g[i] / g.sum():.2f}" for i in o))
    del X, Cj, Cr

    # 2. recompute pooled EXTRA cols over the Jan+Jul 2025 frame
    raw = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/raw/training_2025-0[17]-01_*.parquet"))],
                    how="vertical_relaxed")
    dep_ids = jj["MVT_ID_mvt"]
    ex = extra_candidate_features(mask_departures(raw), dep_ids)
    del raw
    chk = jj.select("MVT_ID_mvt", *pooled).join(ex.select("MVT_ID_mvt", *pooled), on="MVT_ID_mvt", suffix="_2m")
    for c in pooled:
        a, b2 = chk[c].cast(pl.Float64), chk[c + "_2m"].cast(pl.Float64)
        same = ((a - b2).abs() < 1e-4) | (a.is_null() & b2.is_null()) | (a.is_nan() & b2.is_nan())
        log(f"  {c:28s} identical {same.fill_null(False).mean():.3f}  mean 12m {a.mean():.3f}  2m {b2.mean():.3f}")
    jj2 = (jj.drop(pooled).join(ex.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float32) for c in pooled]), on="MVT_ID_mvt",
                                how="left").select(jj.columns))

    # 3. proxy models
    fit = tr.filter(~pl.col("mon").is_in([1, 7]))
    del tr
    res = []
    for lane in ("A", "B"):
        cols = num if lane == "A" else numB
        sel = pl.col("mvt_minus_aobt3").is_not_null() if lane == "A" else pl.col("mvt_minus_aobt3").is_null()
        f = fit.filter(sel)
        if lane == "A":
            f = f.filter((pl.col("MVT_ID_mvt").hash(7) % 100) < 45)
        v1, v2 = jj.filter(sel), jj2.filter(sel)
        Cf, C1, C2 = catcode([f, v1, v2], cats)
        Xf = mat(f, cols, Cf)
        off_f = f["mvt_minus_aobt3"].to_numpy() if lane == "A" else 0.0
        yf = f["TAXITIME_SEC_mvt"].to_numpy() - off_f
        p = dict(objective="regression", learning_rate=0.08 if lane == "A" else 0.05,
                 num_leaves=255 if lane == "A" else 127, min_data_in_leaf=100 if lane == "A" else 50,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                 num_threads=2, seed=0, max_bin=255)
        cif = list(range(len(cols), len(cols) + len(cats)))
        bst = lgb.train(p, lgb.Dataset(Xf, yf, categorical_feature=cif), 700 if lane == "A" else 1000)
        del Xf
        off_v = v1["mvt_minus_aobt3"].to_numpy() if lane == "A" else 0.0
        p1 = bst.predict(mat(v1, cols, C1)) + off_v
        p2 = bst.predict(mat(v2, cols, C2)) + off_v
        log(f"lane {lane}: fit rows {f.height:,}, eval rows {v1.height:,}")
        res.append(pl.DataFrame({"MVT_ID_mvt": v1["MVT_ID_mvt"], "ADEP_mvt": v1["ADEP_mvt"], "mon": v1["mon"],
                                 "lane": lane, "y": v1["TAXITIME_SEC_mvt"].cast(pl.Float64), "p12": p1, "p2": p2}))
    R = pl.concat(res)
    R.write_parquet(os.path.join(OUT, "pool_preds.parquet"))
    N = R.height
    summ = (R.group_by("lane", "ADEP_mvt").agg(
        pl.len().alias("n"),
        ((pl.col("y") - pl.col("p12")) ** 2).mean().sqrt().alias("rmse12"),
        ((pl.col("y") - pl.col("p2")) ** 2).mean().sqrt().alias("rmse2"),
        (pl.col("p2") - pl.col("p12")).mean().alias("mean_shift"),
        (pl.col("p2") - pl.col("p12")).abs().mean().alias("mean_abs_shift"),
        (((pl.col("y") - pl.col("p2")) ** 2 - (pl.col("y") - pl.col("p12")) ** 2).sum() / N).alias("dMSE_N"))
        .sort("lane", "ADEP_mvt"))
    with pl.Config(tbl_rows=30, float_precision=2, tbl_width_chars=200):
        log(summ)
    for m in (1, 7):
        s = R.filter(pl.col("mon") == m)
        d = float((((s["y"] - s["p2"]) ** 2) - ((s["y"] - s["p12"]) ** 2)).sum() / s.height)
        log(f"month {m}: rmse12 {float(((s['y'] - s['p12']) ** 2).mean()) ** .5:.2f}  rmse2 "
            f"{float(((s['y'] - s['p2']) ** 2).mean()) ** .5:.2f}  dMSE {d:+.1f}  mean shift {float((s['p2'] - s['p12']).mean()):+.2f}")
    d = float((((R["y"] - R["p2"]) ** 2) - ((R["y"] - R["p12"]) ** 2)).mean())
    log(f"ALL: dMSE (2-month pooling - 12-month pooling) {d:+.1f} MSE units  (~{d / 480:+.2f} s on the board scale)")
    with open(os.path.join(OUT, "pool_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
