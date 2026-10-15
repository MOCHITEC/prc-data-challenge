"""NbrNet on synthetic data (prc.synth): the leakage rules R1, R3, R5, R6, R7 and an end-to-end smoke run.

    .venv/bin/python -m pytest -q experiments/nbrnet_2026-10-15/tests"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nbrnet"))

import nbr_eval
import nbr_query
import nbr_train
from nbr_model import NbrNet
from nbr_tokens import (
    ARR_DYNAMIC,
    ARR_STATIC,
    DEP_DYNAMIC,
    DEP_STATIC,
    WIN_AFTER_S,
    WIN_BEFORE_S,
    build_index,
    build_tables,
    gather,
)

from prc.features import build_departure_features
from prc.synth import generate_month

MONTHS = (1, 2, 11, 12)


@pytest.fixture(scope="module")
def raw() -> pl.DataFrame:
    frames = [generate_month(2025, m, 100 + m, deps_per_airport_day=30).with_columns(pl.lit(i, dtype=pl.Int16).alias("src")) for i, m in enumerate(MONTHS)]
    return pl.concat(frames, how="diagonal")


@pytest.fixture(scope="module")
def built(raw):
    tb = build_tables(raw)
    return tb, *build_index(tb)


def _all_tokens(tb, idx_d, idx_a):
    q = np.arange(len(tb.dep_t))
    return gather(tb, q, idx_d, idx_a, idx_d.shape[1], idx_a.shape[1])


def test_r1_other_departures_labels_never_enter_tokens(raw, built):
    """Overwrite every departure's BLOCK_TIME and TAXITIME (shifted, and nulled): tables, indices and every gathered
    token are bit-identical. This also covers the query's own label (R3) and any label-derived field (R4)."""
    tb, idx_d, idx_a, cov = built
    rng = np.random.default_rng(0)
    dep = pl.col("PHASE_mvt") == "DEP"
    shift = pl.Series(rng.integers(-5000, 5000, raw.height)).cast(pl.Int64)
    variants = [
        raw.with_columns(
            pl.when(dep).then(pl.col("BLOCK_TIME_UTC_mvt") + pl.duration(seconds=shift)).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"),
            pl.when(dep).then(pl.lit(rng.integers(0, 9999, raw.height))).otherwise(pl.col("TAXITIME_SEC_mvt")).cast(pl.Int32).alias("TAXITIME_SEC_mvt"),
        ),
        raw.with_columns(
            pl.when(dep).then(None).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"),
            pl.when(dep).then(None).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt"),
        ),
    ]
    ref = _all_tokens(tb, idx_d, idx_a)
    for v in variants:
        tb2 = build_tables(v)
        i2d, i2a, c2 = build_index(tb2)
        for k, arr in tb.__dict__.items():
            if isinstance(arr, np.ndarray):
                np.testing.assert_array_equal(arr, getattr(tb2, k), err_msg=k)
        np.testing.assert_array_equal(idx_d, i2d)
        np.testing.assert_array_equal(idx_a, i2a)
        np.testing.assert_array_equal(cov, c2)
        for x, y in zip(ref, _all_tokens(tb2, i2d, i2a)):
            np.testing.assert_array_equal(x, y)


def test_r1_arrival_labels_do_enter_tokens(raw, built):
    """The control for the test above: changing arrivals' in-block time does change arrival tokens (R2), so the
    identity above is not vacuous."""
    tb = built[0]
    arr = pl.col("PHASE_mvt") == "ARR"
    v = raw.with_columns(pl.when(arr).then(pl.col("BLOCK_TIME_UTC_mvt") + pl.duration(seconds=120)).otherwise(pl.col("BLOCK_TIME_UTC_mvt")).alias("BLOCK_TIME_UTC_mvt"))
    tb2 = build_tables(v)
    assert not np.array_equal(tb.arr_t_inblock, tb2.arr_t_inblock)


def test_r3_query_never_its_own_token(built):
    tb, idx_d, _, _ = built
    own = np.arange(len(tb.dep_t))[:, None]
    assert not np.any(idx_d == own)
    assert (idx_d >= 0).any()


def test_r6_r7_same_month_same_airport_inside_window_nearest_first(built):
    tb, idx_d, idx_a, cov = built
    q = np.arange(len(tb.dep_t))
    md, ma = idx_d >= 0, idx_a >= 0
    gq = tb.dep_group[q][:, None]
    assert np.all(tb.dep_group[np.where(md, idx_d, 0)][md] == np.broadcast_to(gq, md.shape)[md])
    assert np.all(tb.arr_group[np.where(ma, idx_a, 0)][ma] == np.broadcast_to(gq, ma.shape)[ma])
    t0 = tb.dep_t[q].astype(np.int64)[:, None]
    dd = tb.dep_t[np.where(md, idx_d, 0)].astype(np.int64) - t0
    da = tb.arr_key[np.where(ma, idx_a, 0)].astype(np.int64) - t0
    for d, m in ((dd, md), (da, ma)):
        assert np.all(d[m] >= -WIN_BEFORE_S) and np.all(d[m] <= WIN_AFTER_S)
        a = np.where(m, np.abs(d), np.iinfo(np.int64).max)
        assert np.all(np.diff(a, axis=1) >= 0), "not nearest-first"
        assert np.all(m[:, :-1] | ~m[:, 1:]), "padding is not at the end"
    # completeness against brute force on a sample: the selected set is the k nearest in the window
    rng = np.random.default_rng(1)
    for i in rng.choice(len(q), 300, replace=False):
        g = tb.dep_group[i]
        cand = np.flatnonzero((tb.dep_group == g) & (np.arange(len(q)) != i))
        dt = tb.dep_t[cand].astype(np.int64) - tb.dep_t[i]
        cand, dt = cand[(dt >= -WIN_BEFORE_S) & (dt <= WIN_AFTER_S)], dt[(dt >= -WIN_BEFORE_S) & (dt <= WIN_AFTER_S)]
        k = idx_d.shape[1]
        want = np.sort(np.abs(dt))[:k]
        got = np.abs(tb.dep_t[idx_d[i][idx_d[i] >= 0]].astype(np.int64) - tb.dep_t[i])
        np.testing.assert_array_equal(np.sort(got), want)
    assert np.all((cov >= 0) & (cov <= 1)) and (cov < 1).any()


def test_noctx_twin_ignores_tokens_and_equals_all_masked():
    torch.manual_seed(0)
    nd, na = len(DEP_STATIC) + len(DEP_DYNAMIC), len(ARR_STATIC) + len(ARR_DYNAMIC)
    net = NbrNet(30, nd, na, k_dep=8, k_arr=6).eval()
    B = 16
    xq = torch.randn(B, 30)
    dep, arr = torch.randn(B, 8, nd), torch.randn(B, 6, na)
    none_d, none_a = torch.zeros(B, 8, dtype=torch.bool), torch.zeros(B, 6, dtype=torch.bool)
    with torch.no_grad():
        p_noctx = net(xq)
        p_masked = net(xq, dep, none_d, arr, none_a, torch.zeros(B))
        p_masked2 = net(xq, torch.randn(B, 8, nd), none_d, torch.randn(B, 6, na), none_a, torch.zeros(B))
        p_ctx = net(xq, dep, torch.ones(B, 8, dtype=torch.bool), arr, torch.ones(B, 6, dtype=torch.bool), torch.ones(B))
    torch.testing.assert_close(p_noctx, p_masked, rtol=0, atol=1e-6)
    torch.testing.assert_close(p_noctx, p_masked2, rtol=0, atol=1e-6)
    assert not torch.allclose(p_noctx, p_ctx)
    assert torch.isfinite(p_masked).all()


@pytest.fixture(scope="module")
def fold_dirs(raw, built, tmp_path_factory):
    tb = built[0]
    member = build_departure_features(raw.drop("src"))
    _, cfg = nbr_query.load_cfg(Path(__file__).resolve().parents[3] / "configs/main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml")
    cfg = replace(cfg, fe_osm_dist=False, fe_osm_null_airports=[])  # the OSM tables need the network; the column is label-free either way
    root = tmp_path_factory.mktemp("nbr")
    tokdir = root / "tokens"
    tb.save(tokdir)
    for k, v in zip(("idx_d", "idx_a", "cov"), built[1:]):
        np.save(tokdir / f"{k}.npy", v)
    rec = nbr_query.prepare_fold(member, "forward", cfg, tb.dep_id, root / "folds" / "forward")
    return root, member, cfg, rec


def test_r5_training_side_does_not_depend_on_evaluation_labels(fold_dirs, built, tmp_path):
    root, member, cfg, rec = fold_dirs
    assert rec["n_tr"] > 0 and rec["n_va"] > 0 and rec["n_hold"] > 0
    ym = member["MVT_TIME_UTC_mvt"].dt.month()
    rng = np.random.default_rng(3)
    perm = pl.Series(rng.permutation(member.height))
    shuffled = member.with_columns(
        pl.when(ym >= 11).then(member["TAXITIME_SEC_mvt"].gather(perm)).otherwise(pl.col("TAXITIME_SEC_mvt")).alias("TAXITIME_SEC_mvt")
    )
    nbr_query.prepare_fold(shuffled, "forward", cfg, built[0].dep_id, tmp_path / "f2")
    a, b = nbr_train.load_fold(root / "folds" / "forward"), nbr_train.load_fold(tmp_path / "f2")
    for k in ("tr_Xn", "tr_C", "tr_y", "tr_proxy", "tr_row", "tr_hold", "va_Xn", "va_C", "va_row"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert nbr_train.target_stats(a["tr_y"] - a["tr_proxy"]) == nbr_train.target_stats(b["tr_y"] - b["tr_proxy"])


def test_end_to_end_synthetic(fold_dirs, tmp_path):
    root, _, _, rec = fold_dirs
    for mode in ("ctx", "noctx"):
        args = ["--fold", "forward", "--seed", "1", "--mode", mode, "--max-epochs", "3", "--batch", "256",
                "--folds-dir", str(root / "folds"), "--tokens", str(root / "tokens"), "--runs-dir", str(tmp_path / "runs"),
                "--exp-dir", str(tmp_path), "--no-git", "--ckpt-every", "1"]
        assert nbr_train.main(args) == 0
        p = pl.read_parquet(tmp_path / "preds" / f"forward_s1_{mode}.parquet")
        assert p.height == rec["n_va"] and p["nn_pred"].is_finite().all()
        assert (tmp_path / "ckpt" / f"forward_s1_{mode}.pt").exists()
        # a re-run resumes from the local checkpoint and ends with the same predictions
        assert nbr_train.main(args) == 0
        p2 = pl.read_parquet(tmp_path / "preds" / f"forward_s1_{mode}.parquet")
        np.testing.assert_array_equal(p["nn_pred"].to_numpy(), p2["nn_pred"].to_numpy())
    # evaluation on the synthetic fold, with a fake a2o OOF (the proxy plus the training median)
    meta = pl.read_parquet(root / "folds" / "forward" / "valid_meta.parquet")
    a2o_dir = tmp_path / "a2o"
    a2o_dir.mkdir()
    meta.select("MVT_ID_mvt", (pl.col("proxy").fill_null(600.0) + 60.0).alias("pred")).write_parquet(a2o_dir / "forward.parquet")
    df = nbr_eval.load_frame(["forward"], [1], root / "folds", a2o_dir, tmp_path / "preds")
    res = nbr_eval.evaluate(df, n_boot=50)
    assert set(res["rmse"]) >= {"a2o", "nn_ctx_alone", "blend_a2o75_ctx25"}
    assert isinstance(res["C1"]["adopt"], bool) and "stop" in res["futility"]


def test_projection_matches_lobt_bound():
    lobt = np.array([np.nan, 100.0, 5000.0, 5000.0, 3000.0])
    pred = np.array([-5.0, 9000.0, 100.0, 7000.0, 50.0])
    got = nbr_eval.project(pred, lobt)
    np.testing.assert_allclose(got, [0.0, 3706.0, 1394.0, 7000.0, 50.0])
