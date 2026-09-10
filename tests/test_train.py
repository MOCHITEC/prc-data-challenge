import numpy as np
import polars as pl

from prc.cv import month_block_folds, paired_bootstrap_rmse_diff
from prc.features import build_departure_features
from prc.synth import generate_month
from prc.train import TrainConfig, predict_fold, run_cv


def test_folds():
    f = month_block_folds([1, 2, 7, 11, 12], "fast")
    names = [x.name for x in f]
    assert names == ["jan", "jul", "forward"]
    assert month_block_folds([1, 2, 3], "full")[0].valid_months == (1,)


def test_paired_bootstrap_direction():
    y = np.random.default_rng(0).normal(900, 300, 5000)
    a = y + np.random.default_rng(1).normal(0, 100, 5000)
    b = y + np.random.default_rng(2).normal(0, 50, 5000)
    r = paired_bootstrap_rmse_diff(y, a, b, n_boot=200)
    assert r["diff_rmse"] < 0 and r["ci95"][1] < 0


def test_cv_runs_and_beats_median_baseline(tmp_path):
    df = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=12) for m in (1, 6, 7)])
    feats = build_departure_features(df)
    cfg = TrainConfig(scheme="fast", num_rounds=200, early_stopping=20)
    rec = run_cv(feats, cfg, out_dir=tmp_path)
    assert set(rec["folds"]) == {"jan", "jul"}
    y = feats["TAXITIME_SEC_mvt"].to_numpy()
    base = float(np.sqrt(np.mean((y - np.median(y)) ** 2)))
    assert rec["overall"]["rmse_full"] < base
    assert (tmp_path / "experiments.jsonl").exists()


def test_predict_fold_handles_unlabelled_ranking_frame():
    tr = build_departure_features(generate_month(2025, 7, seed=3, deps_per_airport_day=10))
    rk = build_departure_features(generate_month(2026, 7, seed=4, deps_per_airport_day=4, blank_dep=True))
    p, _info = predict_fold(tr, rk, TrainConfig(num_rounds=50, early_stopping=10))
    assert len(p) == rk.height and np.isfinite(p).all() and (p >= 0).all()
