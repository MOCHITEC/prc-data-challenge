"""The CatBoost matched-lane blend member (docs/competitor_research_2026-09-21/results/catboost-blend.md).

The member is a recipe, not new training code: configs/main_flseq_osm_nolirf_cb.yaml is its LightGBM base plus
the P10 C+E CatBoost settings. Nothing in src/ changes, so every existing recipe keeps its id (checked for all of
configs/ by tests/test_matched_target.py); here the new recipe is checked to be exactly its base plus the listed
keys, and the blend script's weight fit to recover known weights."""

import importlib.util
from pathlib import Path

import numpy as np
import yaml

from prc.pipeline import train_config_from_spec
from prc.train import CB_DEFAULTS

BASE = "configs/main_flseq_osm_nolirf.yaml"
CB = "configs/main_flseq_osm_nolirf_cb.yaml"
ADDED = {"model": "catboost", "folds": ["jan", "jul", "forward"]}
CHANGED = {"num_rounds": 5000, "n_seeds": 1}


def _load(p):
    return yaml.safe_load(Path(p).read_text())


def test_the_cb_recipe_is_its_base_plus_the_p10_settings():
    b, c = _load(BASE), _load(CB)
    for k, v in ADDED.items():
        assert k not in b and c.pop(k) == v, k
    for k, v in CHANGED.items():
        assert b[k] != v and c[k] == v, k
        b[k] = v
    assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}


def test_the_cb_recipe_resolves_to_p10_ce_catboost_anchored():
    cfg = train_config_from_spec(_load(CB))
    assert cfg.model == "catboost" and cfg.cb_params == {}
    assert cfg.catboost_params() == CB_DEFAULTS
    assert (CB_DEFAULTS["learning_rate"], CB_DEFAULTS["depth"], CB_DEFAULTS["max_ctr_complexity"]) == (0.2, 6, 1)
    assert cfg.matched_target == "anchored" and cfg.use_encodings and not cfg.es_holdout_encodings
    base = train_config_from_spec(_load(BASE))
    assert base.hash() == "926741d779"
    assert cfg.hash() != base.hash()
    # P10's C+E recipe differs in features and folds only; the family settings are the same.
    p10 = train_config_from_spec(_load("configs/p10_cb_e.yaml"))
    assert p10.catboost_params() == cfg.catboost_params()
    assert (p10.num_rounds, p10.early_stopping, p10.n_seeds) == (cfg.num_rounds, cfg.early_stopping, cfg.n_seeds)


def test_sum_to_one_weights_recover_known_weights():
    spec = importlib.util.spec_from_file_location("catboost_blend", Path("scripts/catboost_blend.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 3)) * 10
    w = np.array([0.5, 0.2, 0.3])
    y = X @ w
    got = mod.sum_to_one_weights(y, X)
    assert np.allclose(got, w) and abs(got.sum() - 1) < 1e-12
