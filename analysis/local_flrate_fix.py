"""Train/test fix for fl_nm_rate_other: re-predict the 2026 ranking frame with the column recomputed over
2025 + 2026 departures together (training computes it over 12 months, ranking over Jan+Jul only; see
docs/shift_2026-10-06). One fit per base config, predicting the original and the fixed ranking rows together.

    .venv/bin/python analysis/local_flrate_fix.py build
    .venv/bin/python analysis/local_flrate_fix.py predict <config.yaml> <lane A|B> [n_seeds]
Outputs in $SCRATCH/flfix/: flrate_rank_fixed.parquet, pred_<name>_<lane>.parquet (MVT_ID_mvt, pred_orig, pred_fix).
"""
import glob
import os
import sys
import time

import polars as pl
import yaml

sys.path.insert(0, "src")
OUT = os.path.join(os.environ["SCRATCH"], "flfix")
os.makedirs(OUT, exist_ok=True)
WORK = "data/work_flseq"


def build():
    from prc.features_extra import flight_features
    cols = ["MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "FLIGHT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt"]
    tr = pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/raw/training_*.parquet"))])
    rk = pl.read_parquet("data/raw/ranking.parquet", columns=cols)
    both = pl.concat([tr, rk]).filter(pl.col("PHASE_mvt") == "DEP")
    fl = flight_features(both).select("MVT_ID_mvt", "fl_nm_rate_other")
    ids = rk.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt")
    fix = ids.join(fl, on="MVT_ID_mvt", how="left")
    old = pl.read_parquet(f"{WORK}/feat_rank.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "fl_nm_rate_other", "mvt_minus_aobt3"])
    j = old.join(fix.rename({"fl_nm_rate_other": "fix"}), on="MVT_ID_mvt")
    print("rows", j.height, "changed", int(((j["fl_nm_rate_other"] - j["fix"]).abs() > 1e-6).sum()),
          "share ==1 old/new", round(float((j["fl_nm_rate_other"] == 1).mean()), 3), round(float((j["fix"] == 1).mean()), 3))
    fix.write_parquet(f"{OUT}/flrate_rank_fixed.parquet")


def predict(cfg_path, lane, n_seeds):
    from prc.pipeline import drop_hidden_candidates, train_config_from_spec
    from prc.train import predict_fold
    spec = yaml.safe_load(open(cfg_path))
    if n_seeds:
        spec["n_seeds"] = int(n_seeds)
    cfg = train_config_from_spec(spec)
    t0 = time.time()
    f_tr = drop_hidden_candidates(pl.read_parquet(f"{WORK}/feat_train.parquet"), cfg)
    f_rk = drop_hidden_candidates(pl.read_parquet(f"{WORK}/feat_rank.parquet"), cfg)
    sel = pl.col("mvt_minus_aobt3").is_not_null() if lane == "A" else pl.col("mvt_minus_aobt3").is_null()
    f_rk = f_rk.filter(sel)
    # FIX=union (default): all pooled fl_/dup_/seq_ columns from the audit's union build; FIX=fl: fl_nm_rate_other only
    if os.environ.get("FIX", "union") == "union":
        fix = pl.read_parquet(os.path.join(os.environ["SCRATCH"], "agents/audit/feat_rank_extra_union.parquet"))
    else:
        fix = pl.read_parquet(f"{OUT}/flrate_rank_fixed.parquet")
    fcols = [c for c in fix.columns if c != "MVT_ID_mvt" and c in f_rk.columns]
    f_fx = (f_rk.drop(fcols).join(fix.select("MVT_ID_mvt", *fcols), on="MVT_ID_mvt", how="left", maintain_order="left")
            .with_columns([pl.col(c).cast(f_rk[c].dtype) for c in fcols] + [(-pl.col("MVT_ID_mvt")).alias("MVT_ID_mvt")]))
    print("swapped columns:", len(fcols), flush=True)
    tg = pl.concat([f_rk, f_fx.select(f_rk.columns)])
    print(f"{spec['name']} lane {lane}: train {f_tr.height:,} target {tg.height:,} ({time.time() - t0:.0f}s)", flush=True)
    pred, info = predict_fold(f_tr, tg, cfg)
    p = pred.filter(pl.col("MVT_ID_mvt") > 0).rename({"pred": "pred_orig"}).join(
        pred.filter(pl.col("MVT_ID_mvt") < 0).with_columns(-pl.col("MVT_ID_mvt")).rename({"pred": "pred_fix"}), on="MVT_ID_mvt")
    p = p.join(f_rk.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt")
    p.write_parquet(f"{OUT}/pred_{spec['name']}_{lane}_{os.environ.get('FIX', 'union')}.parquet")
    d = p["pred_fix"] - p["pred_orig"]
    print(f"done {time.time() - t0:.0f}s rows {p.height} mean delta {d.mean():+.2f} mean|delta| {d.abs().mean():.2f} "
          f"sum delta^2/344841 {float((d ** 2).sum()) / 344841:.1f}", {k: v for k, v in info.items() if 'round' in k})
    print(p.with_columns(d.alias("d")).group_by("ADEP_mvt").agg(pl.len(), pl.col("d").mean().round(1), pl.col("d").abs().mean().round(1).alias("absd")).sort("ADEP_mvt"))


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else None if sys.argv[1] == "copyq" else predict(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)


def copyq():
    """q = P(schedule copy) of the production LIRF matched-lane overlay, with original and with fixed ranking features."""
    import json

    import lightgbm as lgb
    os.environ.setdefault("APT", "LIRF")
    sys.path.insert(0, "analysis")
    import local_lirf_copy_overlay as C
    T, R, fn, cat, X = C.frames()
    it = int(json.load(open(os.path.join(C.S, "learn", "copy_LIRF.json")))["best_iter"] * 1.1)
    mdl = lgb.train(C.P, lgb.Dataset(X(T), T["copy"].to_numpy().astype(int), feature_name=fn, categorical_feature=cat), it)
    fix = pl.read_parquet(f"{OUT}/flrate_rank_fixed.parquet").rename({"fl_nm_rate_other": "_fix"})
    Rf = (R.join(fix, on="MVT_ID_mvt", how="left", maintain_order="left")
          .with_columns(pl.col("_fix").cast(R["fl_nm_rate_other"].dtype).alias("fl_nm_rate_other")).drop("_fix"))
    q0, q1 = mdl.predict(X(R)), mdl.predict(X(Rf))
    pl.DataFrame({"MVT_ID_mvt": R["MVT_ID_mvt"], "ms": R["mvt_minus_sched"], "q_orig": q0, "q_fix": q1}).write_parquet(f"{OUT}/copyq_LIRF.parquet")
    print("LIRF matched rows", R.height, "mean q orig/fix", round(float(q0.mean()), 4), round(float(q1.mean()), 4),
          "mean |dq|", round(float(abs(q1 - q0).mean()), 4))


if __name__ == "__main__" and sys.argv[1] == "copyq":
    copyq()
