"""P10: CatBoost as a second model family behind `TrainConfig.model`.

docs/competitor_research_2026-09-21/plans/P10-catboost-rival.md. The comparison is only a family comparison if
the two arms differ in the fitter and in nothing else, and if the new fitter cannot see a label it must not.
Every test below is one of those two properties, or a guard against a CatBoost setting being silently dropped.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

import prc.train as T
from prc.features import build_departure_features
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import PROXY, TrainConfig, _cb_matrix, _inner_holdout_mask, predict_fold

LABEL = "TAXITIME_SEC_mvt"


def _cb(**kw) -> TrainConfig:
    """A small CatBoost config. Model B is sized like model A, which CatBoost requires (see __post_init__)."""
    base = {
        "model": "catboost",
        "num_rounds": 60,
        "early_stopping": 10,
        "unmatched_num_rounds": 60,
        "unmatched_num_leaves": 127,
        "unmatched_min_data_in_leaf": 50,
    }
    base.update(kw)
    return TrainConfig(**base)


def _lgb(**kw) -> TrainConfig:
    base = {
        "num_rounds": 60,
        "early_stopping": 10,
        "unmatched_num_rounds": 60,
        "unmatched_num_leaves": 127,
        "unmatched_min_data_in_leaf": 50,
    }
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def frames():
    """Four training months (so the inner holdout is non-empty) and enough unmatched rows for model B."""
    tr = pl.concat(
        [build_departure_features(generate_month(2025, m, seed=60 + m, deps_per_airport_day=16)) for m in (1, 2, 3, 4)]
    )
    tg = build_departure_features(generate_month(2025, 5, seed=70, deps_per_airport_day=8))
    assert tr.filter(pl.col(PROXY).is_null()).height >= 200, "model B must be fitted or half of this is untested"
    assert tg[LABEL].null_count() == 0
    return tr, tg


# ---------- configuration: nothing CatBoost cannot honour is accepted silently ----------


def test_the_default_is_lightgbm_and_archived_experiment_ids_are_unchanged():
    assert TrainConfig().model == "lightgbm" and TrainConfig().cb_params == {}
    # ids from experiments/experiments.jsonl, written before these fields existed
    for path, want in (
        ("configs/main_v15.yaml", "5c84fe33d3"),
        ("configs/c13_c1_2col_jf.yaml", "fe53c49187"),
        ("configs/c13_base_jf.yaml", "919d5fce5b"),
    ):
        spec = yaml.safe_load(Path(path).read_text())
        assert train_config_from_spec(spec).hash() == want, path
    # and setting the family is a different config
    assert _cb().hash() != replace(_cb(), model="lightgbm").hash()


def test_an_unknown_or_misspelled_recipe_key_raises():
    with pytest.raises(ValueError, match="unknown recipe keys"):
        train_config_from_spec({"name": "x", "model_familly": "catboost"})
    with pytest.raises(ValueError, match="unknown recipe keys"):
        train_config_from_spec({"name": "x", "cb_param": {"depth": 8}})
    # the keys main() reads itself are still accepted
    train_config_from_spec({"name": "x", "use_weather": True, "wind_dir": False, "note": "n"})


def test_every_recipe_in_configs_still_loads():
    import glob

    for p in sorted(glob.glob("configs/*.yaml")):
        train_config_from_spec(yaml.safe_load(Path(p).read_text()))


def test_lightgbm_only_settings_are_refused_under_catboost_not_ignored():
    with pytest.raises(ValueError, match="linear_tree"):
        _cb(use_linear_tree=True)
    with pytest.raises(ValueError, match="params are LightGBM"):
        _cb(params={**TrainConfig().params, "num_leaves": 31})
    with pytest.raises(ValueError, match="capacity override"):
        TrainConfig(model="catboost")  # the dataclass default sizes model B at 255 leaves
    with pytest.raises(ValueError, match="cb_params is set"):
        TrainConfig(cb_params={"depth": 8})
    with pytest.raises(ValueError, match="model must be"):
        TrainConfig(model="xgboost")
    for owned in ("iterations", "random_seed", "has_time", "od_wait", "cat_features"):
        with pytest.raises(ValueError, match="the fitter owns"):
            _cb(cb_params={owned: 1})
    # model B keeps its own round cap under CatBoost, exactly as under LightGBM
    assert _cb(unmatched_num_rounds=77).for_unmatched().num_rounds == 77


def test_max_ctr_complexity_is_explicit_by_default_and_the_resolved_value_is_logged(frames):
    """The library default moves from 1 to 4 with the iteration count, so it is never left to the library."""
    assert T.CB_DEFAULTS["max_ctr_complexity"] == 1
    tr, tg = frames
    for want in (1, 2):
        _, info = predict_fold(tr, tg, _cb(cb_params={"max_ctr_complexity": want}))
        assert info["model"] == "catboost"
        for lane in ("cb_matched", "cb_unmatched"):
            assert info[lane]["resolved"]["max_ctr_complexity"] == want, (lane, info[lane])
            assert info[lane]["resolved"]["has_time"] is False
            assert info[lane]["resolved"]["thread_count"] == 4


# ---------- the two arms are handed the same thing ----------


def test_both_families_are_handed_byte_identical_feature_and_categorical_lists(frames, monkeypatch):
    """Asserted at the matrix boundary of each fitter, lane by lane: an unequal feature list would make every
    comparison number meaningless, and it is the one thing a reviewer cannot see in an RMSE."""
    tr, tg = frames
    seen: dict[str, list] = {"lightgbm": [], "catboost": []}
    real_m, real_cb = T._matrix, T._cb_matrix

    def spy_m(df, f, c):
        seen["lightgbm"].append((list(f), list(c), df.height))
        return real_m(df, f, c)

    def spy_cb(df, f, c):
        seen["catboost"].append((list(f), list(c), df.height))
        return real_cb(df, f, c)

    monkeypatch.setattr(T, "_matrix", spy_m)
    monkeypatch.setattr(T, "_cb_matrix", spy_cb)
    p_l, _ = predict_fold(tr, tg, _lgb())
    p_c, _ = predict_fold(tr, tg, _cb())
    assert len(seen["lightgbm"]) == 2 and len(seen["catboost"]) == 2, seen  # one per lane, each family
    assert seen["lightgbm"] == seen["catboost"]
    assert all(len(c) == len(T.CATEGORICALS) for _, c, _ in seen["catboost"])
    # same rows, same order, same key
    assert p_l["MVT_ID_mvt"].to_list() == p_c["MVT_ID_mvt"].to_list() == tg["MVT_ID_mvt"].to_list()
    assert np.isfinite(p_c["pred"].to_numpy()).all() and (p_c["pred"].to_numpy() >= 0).all()
    # and they are two different models, not one fitter behind two names
    assert not np.allclose(p_l["pred"].to_numpy(), p_c["pred"].to_numpy())


def test_catboost_receives_the_categoricals_as_categoricals(frames, monkeypatch):
    """At the CatBoost boundary: every fit's Pool declares exactly the configured categorical columns, by name,
    after the numeric ones - and hiding them changes the model."""
    from catboost import CatBoostRegressor

    tr, tg = frames
    pools = []
    real_fit = CatBoostRegressor.fit

    def spy(self, X, *a, **k):
        pools.append((X.get_feature_names(), X.get_cat_feature_indices()))
        return real_fit(self, X, *a, **k)

    monkeypatch.setattr(CatBoostRegressor, "fit", spy)
    p_on, _ = predict_fold(tr, tg, _cb())
    assert pools and all(len(idx) == len(T.CATEGORICALS) for _, idx in pools)
    for names, idx in pools:
        assert [names[i] for i in idx] == list(T.CATEGORICALS)
        assert idx == list(range(len(names) - len(idx), len(names)))
    pools.clear()
    p_off, _ = predict_fold(tr, tg, _cb(use_categoricals=False))
    assert pools and all(idx == [] for _, idx in pools)
    assert not np.allclose(p_on["pred"].to_numpy(), p_off["pred"].to_numpy())


def test_the_seed_loop_and_the_fit_count_match_lightgbm(frames, monkeypatch):
    """One early-stopping fit plus n_seeds refits per lane, at seed, seed+1, ... - as `_fit` does."""
    from catboost import CatBoostRegressor

    tr, tg = frames
    seeds = []
    real_fit = CatBoostRegressor.fit

    def spy(self, *a, **k):
        seeds.append((self.get_params()["random_seed"], "eval_set" in k))
        return real_fit(self, *a, **k)

    monkeypatch.setattr(CatBoostRegressor, "fit", spy)
    p2, _ = predict_fold(tr, tg, _cb(n_seeds=2, seed=7))
    assert seeds == [(7, True), (7, False), (8, False)] * 2, seeds
    p1, _ = predict_fold(tr, tg, _cb(n_seeds=1, seed=7))
    assert not np.array_equal(p1["pred"].to_numpy(), p2["pred"].to_numpy()), "the second seed changed nothing"


def test_the_round_count_is_the_lightgbm_formula_on_the_catboost_best_count(frames):
    tr, tg = frames
    cfg = _cb(num_rounds=400, early_stopping=10)
    _, info = predict_fold(tr, tg, cfg)
    m_tr = tr.filter(pl.col(LABEL).is_not_null() & pl.col(PROXY).is_not_null())
    n, n_hold = m_tr.height, int(_inner_holdout_mask(m_tr, cfg.inner_holdout_months).sum())
    rec = info["cb_matched"]
    assert rec["es_best_count"] < cfg.num_rounds, "early stopping never fired; the test proves nothing"
    want = int(min(cfg.num_rounds, max(10, round(rec["es_best_count"] * n / (n - n_hold)))))
    assert rec["rounds"] == info["rounds_matched"] == want == rec["tree_count"]


# ---------- no label reaches a row it must not ----------


def test_target_labels_cannot_move_a_prediction_or_the_stopping_round(frames):
    """Blanking or shuffling the evaluation frame's labels must change nothing - not a prediction, not a round.

    Ordered target statistics read labels by design; this is the check that the only labels they read are the
    training side's. Early stopping must fire for the round half of this to mean anything."""
    tr, tg = frames
    cfg = _cb(num_rounds=400, early_stopping=10)
    p0, i0 = predict_fold(tr, tg, cfg)
    blank = tg.with_columns(pl.lit(None, dtype=pl.Float64).alias(LABEL))
    shuf = tg.with_columns(pl.Series(LABEL, np.random.default_rng(0).permutation(tg[LABEL].to_numpy())))
    for other in (blank, shuf):
        p1, i1 = predict_fold(tr, other, cfg)
        assert p0["pred"].to_numpy().tobytes() == p1["pred"].to_numpy().tobytes()
        assert (i0["rounds_matched"], i0["rounds_unmatched"]) == (i1["rounds_matched"], i1["rounds_unmatched"])
    assert i0["cb_matched"]["es_best_count"] < cfg.num_rounds


def test_a_prediction_does_not_depend_on_the_rows_it_is_predicted_with(frames):
    """A row's categorical statistics at apply time come from the fitted tables, not from its batch - so the
    same row predicted inside the full frame and inside half of it gets the same number. (LightGBM does not
    have this property here: its categorical codes are built over the stacked frame.)"""
    tr, tg = frames
    cfg = _cb()
    full, _ = predict_fold(tr, tg, cfg)
    half = tg.filter(pl.int_range(pl.len()) % 2 == 0)
    part, _ = predict_fold(tr, half, cfg)
    joined = part.join(full, on="MVT_ID_mvt", suffix="_full")
    assert joined.height == half.height
    assert np.array_equal(joined["pred"].to_numpy(), joined["pred_full"].to_numpy())


def test_a_training_label_does_reach_the_model(frames):
    """The positive control for the two tests above: shuffling the TRAINING labels must change predictions,
    or the invariance above is the invariance of a model that reads no labels at all."""
    tr, tg = frames
    cfg = _cb()
    p0, _ = predict_fold(tr, tg, cfg)
    shuf = tr.with_columns(pl.Series(LABEL, np.random.default_rng(1).permutation(tr[LABEL].to_numpy())))
    p1, _ = predict_fold(shuf, tg, cfg)
    assert not np.allclose(p0["pred"].to_numpy(), p1["pred"].to_numpy())


# ---------- determinism ----------


def test_two_catboost_runs_agree_bit_for_bit(frames):
    tr, tg = frames
    cfg = _cb(n_seeds=2)
    runs = [predict_fold(tr, tg, cfg)[0]["pred"].to_numpy().tobytes() for _ in range(2)]
    assert runs[0] == runs[1]


def test_the_categorical_block_is_a_function_of_the_values_not_of_the_chunking():
    """The G11 guard for the CatBoost route. The same column split into many chunks must give the same block."""
    rng = np.random.default_rng(3)
    vals = [None if x == 0 else f"S{x}" for x in rng.integers(0, 40, 5000)]
    df = pl.DataFrame({"a": rng.normal(size=5000).astype(np.float32), "STAND_mvt": vals})
    chunked = pl.concat([df.slice(i, 250) for i in range(0, 5000, 250)], rechunk=False)
    assert chunked.n_chunks() > 1
    n1, c1 = _cb_matrix(df, ["a"], ["STAND_mvt"])
    n2, c2 = _cb_matrix(chunked, ["a"], ["STAND_mvt"])
    assert n1.tobytes() == n2.tobytes() and n1.dtype == np.float32
    assert c1.tolist() == c2.tolist()
    # the null is the same literal level LightGBM's matrix uses, and every value is the level itself
    assert set(c1[:, 0].tolist()) == {s.encode() for s in ("__NA__", *[f"S{i}" for i in range(1, 40)])}
    want = [b"__NA__" if v is None else v.encode() for v in vals]
    assert c1[:, 0].tolist() == want


def test_lightgbm_never_touches_the_catboost_fitter(frames, monkeypatch):
    tr, tg = frames

    def boom(*a, **k):
        raise AssertionError("the LightGBM recipe reached _fit_catboost")

    monkeypatch.setattr(T, "_fit_catboost", boom)
    _, info = predict_fold(tr, tg, _lgb())
    assert "model" not in info and "cb_matched" not in info


def test_early_stopping_learns_on_the_non_holdout_rows_and_stops_on_the_holdout_only(frames, monkeypatch):
    """Pool sizes at the CatBoost boundary: the early-stopping fit learns on n - n_hold rows and is evaluated
    on the n_hold inner-holdout rows; every refit learns on all n. And the best COUNT carried forward is the
    library's 0-based best iteration plus one - the unit LightGBM's `best_iteration` is in."""
    from catboost import CatBoostRegressor

    tr, tg = frames
    cfg = _cb(num_rounds=400, early_stopping=10)
    calls = []
    real_fit = CatBoostRegressor.fit

    def spy(self, X, *a, **k):
        out = real_fit(self, X, *a, **k)
        ev = k.get("eval_set")
        calls.append((X.num_row(), None if ev is None else ev.num_row(), self.get_best_iteration() if ev else None))
        return out

    monkeypatch.setattr(CatBoostRegressor, "fit", spy)
    _, info = predict_fold(tr, tg, cfg)
    m_tr = tr.filter(pl.col(LABEL).is_not_null() & pl.col(PROXY).is_not_null())
    n, n_hold = m_tr.height, int(_inner_holdout_mask(m_tr, cfg.inner_holdout_months).sum())
    assert 0 < n_hold < n
    (n_learn, n_eval, best_it), (n_refit, no_eval, _) = calls[0], calls[1]
    assert (n_learn, n_eval) == (n - n_hold, n_hold)
    assert (n_refit, no_eval) == (n, None)
    assert info["cb_matched"]["es_best_count"] == best_it + 1


def test_catboost_honours_the_p2_fix1_holdout_block(frames):
    """With es_holdout_encodings on, the re-encoded holdout block is what early stopping reads, as for LightGBM."""
    tr, tg = frames
    cfg = _cb(num_rounds=400, early_stopping=10, es_holdout_encodings=True)
    _, info = predict_fold(tr, tg, cfg)
    assert info["es_encoder_matched"] and info["es_encoder_matched"][0]["rows"] > 0
    assert info["cb_matched"]["es_best_count"] < cfg.num_rounds
