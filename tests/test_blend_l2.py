"""Lever L2 in the plain-y and CatBoost blend members (docs/competitor_research_2026-09-21/results/blend-l2.md).

The two new recipes are their adopted-blend bases plus `fe_fwd_nm: true` and nothing else, and the blend script
builds P, F and the base from the members the pre-registration names."""

import importlib.util
from pathlib import Path

import yaml

from prc.pipeline import train_config_from_spec

NEW = {
    "main_flseq_osm_nolirf_plainy_l2.yaml": ("main_flseq_osm_nolirf_plainy.yaml", "b971f4b81b"),
    "main_flseq_osm_nolirf_cb_l2.yaml": ("main_flseq_osm_nolirf_cb.yaml", "6297d62f5f"),
}


def _load(name):
    return yaml.safe_load((Path("configs") / name).read_text())


def test_the_l2_member_recipes_are_their_base_plus_the_flag():
    for name, (base, base_id) in NEW.items():
        c, b = _load(name), _load(base)
        assert "fe_fwd_nm" not in b and c.pop("fe_fwd_nm") is True, name
        assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}, name
        cb, cc = train_config_from_spec(b), train_config_from_spec(_load(name))
        assert cb.hash() == base_id, base
        assert cc.hash() != base_id and cc.fe_fwd_nm and not cc.fe_fwd_nm_unmatched
        assert (cc.model, cc.matched_target, cc.folds, cc.n_seeds) == (cb.model, cb.matched_target, cb.folds, cb.n_seeds)


def test_the_blend_script_members_match_the_preregistration():
    spec = importlib.util.spec_from_file_location("blend_l2", Path("scripts/blend_l2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HYBRIDS == {
        "hybrid_plainy_cb": ("a", "p", "c"),
        "hybrid_blend_l2_F": ("a2", "p", "c"),
        "hybrid_blend_l2_P": ("a2", "p2", "c2"),
    }
    assert mod.ANCHORED_L2 == "main_flseq_osm_nolirf_l2-ef204b44ff"
    assert mod.ANCHORED == "main_flseq_osm_nolirf-926741d779" and mod.UNMATCHED == "main_flseq-8615e6a194"
    assert (mod.PLAIN, mod.CB) == ("main_flseq_osm_nolirf_plainy-b971f4b81b", "main_flseq_osm_nolirf_cb-6297d62f5f")
    for c in ("a2", "p2", "c2"):
        assert mod.FULL_FILES[c].endswith("_l2.parquet")
