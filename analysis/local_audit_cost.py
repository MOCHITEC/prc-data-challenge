"""Audit worker, step 3: price the train/test inconsistencies found by local_audit_frame.py / local_audit_dist.py.

Proxy of the production base model (same harness as local_shift_pool_ablate.py: lane A g = y - proxy on a 45 %
subsample, lane B y; LightGBM, 2 seeds), fitted on the ten months other than Jan+Jul 2025, now WITH the nine
label-derived encodings (ref_*/arh_*/apt_med) that production adds at fold time. Jan+Jul 2025 is predicted under
several arms; dMSE is (arm - base) summed over rows / N, i.e. MSE units on the Jan+Jul scale (~480 = 1 s board).

Model S = production encodings: tables fitted on all fit rows (each training row's own label and same-month
neighbours inside its cell). Arms predicted with model S:
  base   12-month-frame features (what training saw)
  edge   every column that differs between the 12- and 2-month frame EXCEPT the known pooled ones
         (fl_nm_rate_other, seq_hw_prev_pairq, seq_hw_excess, seq_burst_*): frame-boundary effects only
  all2m  the full 2-month rebuild (= how 2026 is actually computed)
Model X = same, but training rows' encodings cross-fitted by calendar month (each month encoded from the other
nine); the target rows use the full ten-month table exactly as for S. X_base - S_base prices the self-inclusion
(leak) in the encodings: negative = cross-fitting would help on unseen months.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_audit_cost.py [seed ...]
Outputs: $SCRATCH/agents/audit/cost_preds_s{seed}.parquet, cost_report.txt
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "analysis")
from local_audit_frame import visible  # noqa: E402
from local_shift_pool import catcode, mat  # noqa: E402
from prc.features import CATEGORICALS  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import ENCODING_COLS, attach_encodings, fit_encodings, unmatched_columns  # noqa: E402

OUT = os.path.join(os.environ["SCRATCH"], "agents", "audit")
LABEL = "TAXITIME_SEC_mvt"
POOLED = {"fl_nm_rate_other", "seq_hw_prev_pairq", "seq_hw_excess", "seq_burst_pos", "seq_burst_len",
          "seq_burst_elapsed", "seq_burst_remaining", "seq_burst_pos_frac"}
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def cfg_():
    spec = yaml.safe_load(open("configs/main_flseq_osm_nolirf.yaml"))
    for k in ("name", "use_weather"):
        spec.pop(k, None)
    return train_config_from_spec(spec)


def strip_enc(f):
    return f.drop([c for c in ENCODING_COLS + ["op_offset_med", "op_offset_n"] if c in f.columns])


def run_seed(seed: int):
    path = os.path.join(OUT, f"cost_preds_s{seed}.parquet")
    if os.path.exists(path):
        return pl.read_parquet(path)
    cfg = cfg_()
    num = visible()
    cats = list(CATEGORICALS)
    ekeys = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local", "AIRCRAFT_OPERATOR_flt",
             "mvt_minus_aobt3", LABEL]
    lab = (pl.scan_parquet("data/work_flseq/feat_train.parquet").select(ekeys + ["MVT_TIME_UTC_mvt"])
           .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
           .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect())
    labfit = lab.filter(~pl.col("mon").is_in([1, 7]))
    enc = fit_encodings(labfit, cfg)
    enc_m = {m: fit_encodings(labfit.filter(pl.col("mon") != m), cfg) for m in sorted(labfit["mon"].unique())}
    del lab, labfit
    need = list(dict.fromkeys(["MVT_ID_mvt", "MVT_TIME_UTC_mvt", LABEL, "hour_local", *num, *cats]))
    # lane A fit rows are subsampled to 45 % BEFORE loading (encodings above used every labelled fit row)
    keep = ((pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])) | pl.col("mvt_minus_aobt3").is_null()
            | ((pl.col("MVT_ID_mvt").hash(11 + seed) % 100) < 45))
    tr = (pl.scan_parquet("data/work_flseq/feat_train.parquet").select(need)
          .filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite() & keep)
          .with_columns(*[pl.col(c).cast(pl.Float32) for c in num if c != "hour_local"],
                        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect())
    jj = tr.filter(pl.col("mon").is_in([1, 7]))
    fit = tr.filter(~pl.col("mon").is_in([1, 7]))
    del tr
    new = pl.read_parquet(os.path.join(OUT, "feat_jj2m.parquet"))
    diff = [l.split()[0] for l in open(os.path.join(OUT, "frame_diff.txt")) if "differs on" in l]
    edge_cols = [c for c in diff if c not in POOLED and c in num]
    log(f"seed {seed}: edge columns {len(edge_cols)}: {edge_cols}")
    variants = {"base": jj}
    variants["edge"] = jj.drop(edge_cols).join(new.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float32) for c in edge_cols]),
                                               on="MVT_ID_mvt", how="left").select(jj.columns)
    d2 = [c for c in diff if c in num]
    variants["all2m"] = jj.drop(d2).join(new.select("MVT_ID_mvt", *[pl.col(c).cast(pl.Float32) for c in d2]),
                                         on="MVT_ID_mvt", how="left").select(jj.columns)
    del new, jj
    variants = {k: attach_encodings(v, enc) for k, v in variants.items()}
    fitS = attach_encodings(fit, enc)
    fitX = pl.concat([attach_encodings(fit.filter(pl.col("mon") == m), e) for m, e in enc_m.items()],
                     how="vertical_relaxed").select(fitS.columns)
    del fit
    numE = num + ENCODING_COLS
    res = []
    for lane in ("A", "B"):
        cols = numE if lane == "A" else unmatched_columns(numE)
        sel = pl.col("mvt_minus_aobt3").is_not_null() if lane == "A" else pl.col("mvt_minus_aobt3").is_null()
        vs = {k: v.filter(sel) for k, v in variants.items()}
        v0 = vs["base"]
        off_v = v0["mvt_minus_aobt3"].to_numpy() if lane == "A" else 0.0
        d = {"MVT_ID_mvt": v0["MVT_ID_mvt"], "ADEP_mvt": v0["ADEP_mvt"], "mon": v0["mon"],
             "y": v0[LABEL].cast(pl.Float64)}
        for mname, fr in (("S", fitS), ("X", fitX)):
            f = fr.filter(sel)
            keys = list(vs)
            Cs = catcode([f] + [vs[k] for k in keys], cats)
            Xf = mat(f, cols, Cs[0])
            off_f = f["mvt_minus_aobt3"].to_numpy() if lane == "A" else 0.0
            p = dict(objective="regression", learning_rate=0.08 if lane == "A" else 0.05,
                     num_leaves=255 if lane == "A" else 127, min_data_in_leaf=100 if lane == "A" else 50,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                     num_threads=2, seed=seed, max_bin=255)
            cif = list(range(len(cols), len(cols) + len(cats)))
            bst = lgb.train(p, lgb.Dataset(Xf, f[LABEL].to_numpy() - off_f, categorical_feature=cif),
                            700 if lane == "A" else 1000)
            del Xf
            for i, k in enumerate(keys):
                if mname == "X" and k != "base":
                    continue
                d[f"p_{mname}_{k}"] = bst.predict(mat(vs[k], cols, Cs[i + 1])) + off_v
            log(f"seed {seed} lane {lane} model {mname} done ({f.height:,} fit rows)")
        res.append(pl.DataFrame(d).with_columns(pl.lit(lane).alias("lane"), pl.lit(seed).alias("seed")))
    R = pl.concat(res)
    R.write_parquet(path)
    return R


def report(R: pl.DataFrame):
    arms = {"edge (S)": ("p_S_edge", "p_S_base"), "all2m (S)": ("p_S_all2m", "p_S_base"),
            "X-S self-incl": ("p_X_base", "p_S_base")}
    for seed in sorted(R["seed"].unique().to_list()):
        S = R.filter(pl.col("seed") == seed)
        N = S.height
        log(f"seed {seed}: base MSE {float(((S['y'] - S['p_S_base']) ** 2).mean()):.0f} (S), "
            f"{float(((S['y'] - S['p_X_base']) ** 2).mean()):.0f} (X)")
        for a, (pa, pb) in arms.items():
            S2 = S.with_columns(((pl.col("y") - pl.col(pa)) ** 2 - (pl.col("y") - pl.col(pb)) ** 2).alias("dd"),
                                (pl.col(pa) - pl.col(pb)).alias("sh"))
            parts = [f"{l}{m}:{S2.filter((pl.col('lane') == l) & (pl.col('mon') == m))['dd'].sum() / N:+.1f}"
                     for l in ("A", "B") for m in (1, 7)]
            apt = S2.group_by("ADEP_mvt").agg((pl.col("dd").sum() / N).alias("g")).sort("ADEP_mvt")
            log(f"  {a:14s} dMSE {S2['dd'].sum() / N:+8.1f} | " + " ".join(parts)
                + f" | rows moved {int((S2['sh'].abs() > 1e-6).sum()):,} | airports>0 {int((apt['g'] > 0).sum())}/10"
                + " | " + " ".join(f"{r[0]}:{r[1]:+.0f}" for r in apt.rows()))


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
    R = pl.concat([run_seed(s) for s in seeds])
    report(R)
    with open(os.path.join(OUT, "cost_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
