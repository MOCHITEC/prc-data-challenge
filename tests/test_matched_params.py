"""TrainConfig.matched_params: LightGBM parameters for model A only (optuna-retune.md), and the retune script.

With matched_params empty (the default) every existing recipe keeps its experiment id and the resolved parameters of
both its models to the byte, and the default / plain-y / per-airport paths give byte-identical predictions on a fixed
synthetic fold (digests computed by the code at 548538c, before the field existed). Set, model A takes the override
and model B is byte-identical to the recipe without it."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from prc.cv import month_block_folds
from prc.features import build_departure_features
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import PROXY, TrainConfig, predict_fold

# (experiment id, digest of [model A params, model B params]) of every recipe in configs/ at 548538c.
BEFORE = {
    "main_flseq_osm_nolirf_l2_opt_gm.yaml": ("67f49a5d62", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2_opt_cmix.yaml": ("d9b7e7a288", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2_opt_l6.yaml": ("177174f530", "9f53bbe724"),
    # new-stand / month-scope / L6 recipes on claude/peaceful-babbage-md9l2x (2026-10-05)
    "ms_scope.yaml": ("4a3e399ba3", "9f53bbe724"),
    "ns_base.yaml": ("2c65506a2d", "9f53bbe724"),
    "ns_base_u.yaml": ("2d1a8167a5", "9f53bbe724"),
    "ns_drop.yaml": ("693275dd32", "9f53bbe724"),
    "ns_drop_u.yaml": ("f58349b6af", "9f53bbe724"),
    "ret_base.yaml": ("338abfb11c", "9f53bbe724"),
    "ret_l6.yaml": ("65265584c7", "9f53bbe724"),
    # recipes that exist only on claude/peaceful-babbage-md9l2x (ids checked: L5 = e1ee6f181e as logged)
    "clip1800.yaml": ("935169f2ee", "9f53bbe724"),
    "clip1800_fl.yaml": ("f55e2764d5", "9f53bbe724"),
    "flwin_both.yaml": ("33ed7358b9", "9f53bbe724"),
    "flwin_target.yaml": ("91b712363d", "9f53bbe724"),
    "main_flseq_osm_nolirf_l5.yaml": ("e1ee6f181e", "9f53bbe724"),
    "mb_b255.yaml": ("2b5e598a0b", "864be36cf6"),
    "mb_ctl.yaml": ("272df160d4", "9f53bbe724"),
    "mb_md5.yaml": ("f88e5ddbb0", "0f3ae1444a"),
    "mb_seed.yaml": ("ac32d9cd1f", "5a707d5e9a"),
    "opdi_noise2.yaml": ("b78076cdfc", "9f53bbe724"),
    "opdi_opdi.yaml": ("01270d4efc", "9f53bbe724"),
    "opdi_stress.yaml": ("6c7b50c739", "9f53bbe724"),
    "osm_full_base.yaml": ("2113efe1f1", "9f53bbe724"),
    "osm_full_dist.yaml": ("20f84df7d1", "9f53bbe724"),
    "osm_full_noise1.yaml": ("d86a669f35", "9f53bbe724"),
    "seqwin_both.yaml": ("0ff38bd918", "9f53bbe724"),
    "seqwin_target.yaml": ("69503b3795", "9f53bbe724"),
    "sw1.yaml": ("c0d223319b", "9f53bbe724"),
    "tc_ctrl.yaml": ("c217ddd030", "9f53bbe724"),
    "tc_drop.yaml": ("86a3693e18", "9f53bbe724"),
    "abl_no_categoricals.yaml": ("2d339105bd", "864be36cf6"),
    "abl_no_encodings.yaml": ("c9761a913d", "864be36cf6"),
    "abl_vanilla_params.yaml": ("66aa584f56", "f95eb3b33e"),
    "afe_forward_agg.yaml": ("c338f6a535", "9f53bbe724"),
    "afe_forward_base.yaml": ("3808ed13de", "9f53bbe724"),
    "afe_forward_cyc.yaml": ("379abdbf1e", "9f53bbe724"),
    "afe_forward_freq.yaml": ("81fe33b3a1", "9f53bbe724"),
    "afe_forward_meta.yaml": ("4524650838", "9f53bbe724"),
    "afe_forward_noise2.yaml": ("cd928d7121", "9f53bbe724"),
    "afe_forward_noise6.yaml": ("51cd031781", "9f53bbe724"),
    "afe_jan_agg.yaml": ("827a396b47", "9f53bbe724"),
    "afe_jan_base.yaml": ("1add616cd7", "9f53bbe724"),
    "afe_jan_cyc.yaml": ("42fb2c599c", "9f53bbe724"),
    "afe_jan_freq.yaml": ("bf563e37f2", "9f53bbe724"),
    "afe_jan_meta.yaml": ("6c18f00a5f", "9f53bbe724"),
    "afe_jan_noise2.yaml": ("2bbf657152", "9f53bbe724"),
    "afe_jan_noise6.yaml": ("967f104b20", "9f53bbe724"),
    "c13_base.yaml": ("e6806da206", "9f53bbe724"),
    "c13_base_jf.yaml": ("919d5fce5b", "9f53bbe724"),
    "c13_c1.yaml": ("84f03c27e2", "9f53bbe724"),
    "c13_c1_2col.yaml": ("99502d00ed", "9f53bbe724"),
    "c13_c1_2col_jf.yaml": ("fe53c49187", "9f53bbe724"),
    "c13_c3wind.yaml": ("f0658b5d2d", "9f53bbe724"),
    "c13_gust.yaml": ("69fd3a7c9e", "9f53bbe724"),
    "c13_noise2.yaml": ("e6ab01d48e", "9f53bbe724"),
    "c13_noise2_jf.yaml": ("dadd0e0efe", "9f53bbe724"),
    "c13_noise3.yaml": ("acca475972", "9f53bbe724"),
    "c13_noise4.yaml": ("b97278ddd3", "9f53bbe724"),
    "c13_refit_mem.yaml": ("096c3c01b9", "9f53bbe724"),
    "cb_all.yaml": ("c258423fda", "9f53bbe724"),
    "cb_base.yaml": ("aceed63e33", "9f53bbe724"),
    "cb_bk.yaml": ("ba8cb09a14", "9f53bbe724"),
    "cb_fl.yaml": ("b0344482c2", "9f53bbe724"),
    "cb_flseq.yaml": ("df2d40a22f", "9f53bbe724"),
    "cb_noise.yaml": ("777d757f54", "9f53bbe724"),
    "cb_occ.yaml": ("3614f14bec", "9f53bbe724"),
    "cb_seq.yaml": ("3d255f10d5", "9f53bbe724"),
    "cb_ta.yaml": ("b257927427", "9f53bbe724"),
    "det_a.yaml": ("c1fbab2941", "9f53bbe724"),
    "det_b.yaml": ("d3ef1634a6", "9f53bbe724"),
    "determinism_check.yaml": ("bbeeeedde8", "9f53bbe724"),
    "determinism_check_airport.yaml": ("2fe8f964fd", "9f53bbe724"),
    "f10_plannedflight.yaml": ("2c6be0bee2", "21300f5568"),
    "feat_base.yaml": ("37369fa81a", "9f53bbe724"),
    "feat_f1_queue.yaml": ("5f1ca09cf5", "9f53bbe724"),
    "feat_f2_bank.yaml": ("8f5239a856", "9f53bbe724"),
    "feat_f3_share.yaml": ("faede6d472", "9f53bbe724"),
    "feat_f4_noise.yaml": ("b9476b4005", "9f53bbe724"),
    "feat_f5_rwyact.yaml": ("73a6f3020d", "21300f5568"),
    "feat_f6_deparr.yaml": ("1a17a58778", "21300f5568"),
    "feat_f7_vsusual.yaml": ("b2ddf23c88", "21300f5568"),
    "feat_f8_opoffset.yaml": ("e9f1637c5a", "21300f5568"),
    "feat_f9_dist.yaml": ("7ce128b230", "21300f5568"),
    "ff1_base.yaml": ("f95441831c", "21300f5568"),
    "ff1_noise.yaml": ("8e90f0b161", "21300f5568"),
    "inv_baseline_forward.yaml": ("c6b9dbe79b", "62d486499b"),
    "inv_it10_leaf5.yaml": ("58170fdce1", "2be8ce0fe9"),
    "inv_it11_lineartree.yaml": ("5449aae42c", "cfd5cdff4e"),
    "inv_it23a_rounds600.yaml": ("65dc05a1b4", "62d486499b"),
    "inv_it23b_holdout2.yaml": ("83f6214ff9", "62d486499b"),
    "leak_check.yaml": ("84d69fbcc1", "9f53bbe724"),
    "main.yaml": ("0f859c881a", "9f53bbe724"),
    "main_flseq.yaml": ("8615e6a194", "9f53bbe724"),
    "main_flseq_a1.yaml": ("c14b85d14f", "9f53bbe724"),
    "main_flseq_l2b.yaml": ("e40f03b86d", "9f53bbe724"),
    "main_flseq_osm.yaml": ("bb204cd922", "9f53bbe724"),
    "main_flseq_osm_nolirf.yaml": ("926741d779", "9f53bbe724"),
    "main_flseq_osm_nolirf_a1.yaml": ("e0daec598e", "9f53bbe724"),
    "main_flseq_osm_nolirf_cb.yaml": ("6297d62f5f", "ef6028717f"),
    "main_flseq_osm_nolirf_cb_l2.yaml": ("4aa341b206", "ef6028717f"),
    "main_flseq_osm_nolirf_l1.yaml": ("50ebc5643d", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2.yaml": ("ef204b44ff", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2_lt.yaml": ("a9fb149ab5", "728861a8cb"),
    "main_flseq_osm_nolirf_l2_pa.yaml": ("85f815edf4", "9f53bbe724"),
    "main_flseq_osm_nolirf_l3.yaml": ("d08ab29d15", "9f53bbe724"),
    "main_flseq_osm_nolirf_plainy.yaml": ("b971f4b81b", "9f53bbe724"),
    "main_flseq_osm_nolirf_plainy_l2.yaml": ("c74a27239b", "9f53bbe724"),
    "main_flseq_osm_nolirf_plainy_l2_lt.yaml": ("af11759d5d", "728861a8cb"),
    "main_gap.yaml": ("517867a0de", "9f53bbe724"),
    "main_reparam.yaml": ("46b15a260c", "9f53bbe724"),
    "main_v15.yaml": ("5c84fe33d3", "9f53bbe724"),
    "osm_base_u.yaml": ("624ab3fdd0", "9f53bbe724"),
    "osm_dist.yaml": ("cbfea098b6", "9f53bbe724"),
    "osm_dist_u.yaml": ("7b834645f1", "9f53bbe724"),
    "osm_noise1.yaml": ("c6de0edd7c", "9f53bbe724"),
    "osm_noise1_u.yaml": ("6fbd71e29d", "9f53bbe724"),
    "p10_cb_e.yaml": ("eb80bbdcd1", "ef6028717f"),
    "p10_cb_ne.yaml": ("6f99a081d0", "ef6028717f"),
    "p10_cb_ne_ctr4.yaml": ("2cfc778cf4", "bfe4989136"),
    "p10_cb_ne_s43_jf.yaml": ("9e82398c28", "ef6028717f"),
    "p10_lgb_e.yaml": ("c431898be6", "9f53bbe724"),
    "p10_lgb_ne.yaml": ("c38c992d61", "9f53bbe724"),
    "p10_lgb_ne_s43.yaml": ("122d32a2de", "61011b43ed"),
    "p11_airport.yaml": ("688ef71567", "9f53bbe724"),
    "p11_airport_confirm.yaml": ("243c623581", "9f53bbe724"),
    "p11_base.yaml": ("6ac2c2eece", "9f53bbe724"),
    "p11_noise_shuffled.yaml": ("a68116c01a", "9f53bbe724"),
    "p2_fix1_base.yaml": ("40cfc6bb39", "9f53bbe724"),
    "p2_fix1_es.yaml": ("132b7a3ef0", "9f53bbe724"),
    "p4a_base.yaml": ("4a8e6ca3eb", "9f53bbe724"),
    "p4a_gap.yaml": ("72fed772d7", "9f53bbe724"),
    "p4a_noise.yaml": ("3799ecbb01", "9f53bbe724"),
    "p8_wx_off.yaml": ("f421484400", "9f53bbe724"),
    "p8_wx_on.yaml": ("f61e61a25a", "9f53bbe724"),
    "p8b_base.yaml": ("3dc917225f", "9f53bbe724"),
    "p8b_noise.yaml": ("b6b05e26c7", "9f53bbe724"),
    "p8b_wx.yaml": ("7f7be04073", "9f53bbe724"),
    "p8e_base.yaml": ("798baa7a90", "9f53bbe724"),
    "p8e_base2.yaml": ("a90601fef0", "9f53bbe724"),
    "p8e_ctrl.yaml": ("d7e31c32c6", "9f53bbe724"),
    "p8e_drop.yaml": ("30925f6d13", "9f53bbe724"),
    "pw_both.yaml": ("1bbd5f7f3a", "21300f5568"),
    "pw_noise.yaml": ("8a628982f8", "21300f5568"),
    "rep_jan_off.yaml": ("252fb4991a", "21300f5568"),
    "rep_jan_on.yaml": ("98f4ddb23b", "21300f5568"),
    "reparam_lirf.yaml": ("9140fe5b48", "62d486499b"),
    "repro_v2.yaml": ("eca534c1b4", "9f53bbe724"),
    "submit_v10_reparam.yaml": ("cb8bc19b97", "9f53bbe724"),
    "submitted_v4.yaml": ("69a3be9f36", "abab3846e4"),
    "weather_check.yaml": ("6c54f989fe", "9f53bbe724"),
    "wx2_B.yaml": ("135985b5fb", "9f53bbe724"),
    "wx2_C.yaml": ("71dc60918b", "9f53bbe724"),
    "wx2_base.yaml": ("00e088e604", "9f53bbe724"),
    "wx2_noise5.yaml": ("a6343e3753", "9f53bbe724"),
    "wx2_noise7.yaml": ("845c11d473", "9f53bbe724"),
}

# sha1(ids + preds)[:16] of predict_fold on the fixture below, computed by the code at 548538c.
PRED_BEFORE = {"default": "bd675c22ce30d32c", "plain": "7b6cfe30e8c6bd68", "pa": "a928a6226295514c"}

OVERRIDE = {
    "num_leaves": 63,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "lambda_l2": 5.0,
    "learning_rate": 0.08,
}


def _load(name):
    return yaml.safe_load((Path("configs") / name).read_text())


def _digest(c: TrainConfig) -> str:
    pa = c.lgb_params() if c.model == "lightgbm" else c.catboost_params()
    pb = c.for_unmatched().lgb_params() if c.model == "lightgbm" else c.for_unmatched().catboost_params()
    return hashlib.sha1(json.dumps([pa, pb], sort_keys=True, default=str).encode()).hexdigest()[:10]


def _fx(seed, deps):
    return build_departure_features(generate_month(2025, 7, seed=seed, deps_per_airport_day=deps))


def _pred_digest(p: pl.DataFrame) -> str:
    return hashlib.sha1(p["MVT_ID_mvt"].to_numpy().tobytes() + p["pred"].to_numpy().tobytes()).hexdigest()[:16]


def _base():
    return TrainConfig(num_rounds=30, early_stopping=10, unmatched_num_rounds=30, airport_min_group_rows=50, n_seeds=2)


def test_every_existing_recipe_keeps_its_id_and_both_models_params():
    assert TrainConfig().matched_params == {}
    names = {p.name for p in Path("configs").glob("*.yaml")}
    new = names - set(BEFORE)
    # *_mlp.yaml (mlp-member.md) are checked in tests/test_mlp_member.py.
    # *_cbopt.yaml (catboost-retune.md) are checked in tests/test_cb_matched_params.py.
    assert all(n.endswith(("_opt.yaml", "_mlp.yaml", "_cbopt.yaml")) for n in new), new
    for name, (want_id, want_params) in BEFORE.items():
        c = train_config_from_spec(_load(name))
        assert (c.hash(), _digest(c)) == (want_id, want_params), name
        assert c.matched_params == {} and c.for_matched() is c, name


def test_the_default_plainy_and_pa_paths_predict_byte_for_byte_as_before():
    tr, tg = _fx(43, 8), _fx(44, 5)
    base = _base()
    arms = {
        "default": base,
        "plain": replace(base, matched_target="plain"),
        "pa": replace(base, airport_models=True, airport_blend_weight=1.0, airport_specialist_only=True),
    }
    for name, c in arms.items():
        p, info = predict_fold(tr, tg, c)
        assert _pred_digest(p) == PRED_BEFORE[name], name
        assert "matched_params" not in info


def test_an_empty_override_is_the_same_recipe():
    c = TrainConfig(matched_params={})
    assert c.hash() == TrainConfig().hash() and c.for_matched() is c


@pytest.mark.parametrize("arm", ["default", "plain", "pa"])
def test_the_override_moves_model_a_only(arm):
    tr, tg = _fx(43, 8), _fx(44, 5)
    base = _base()
    base = {
        "default": base,
        "plain": replace(base, matched_target="plain"),
        "pa": replace(base, airport_models=True, airport_blend_weight=1.0, airport_specialist_only=True),
    }[arm]
    opt = replace(base, matched_params=dict(OVERRIDE))
    assert opt.hash() != base.hash()
    assert opt.for_unmatched().lgb_params() == base.for_unmatched().lgb_params()
    assert opt.for_matched().lgb_params() == {**base.lgb_params(), **OVERRIDE}
    assert opt.for_matched().matched_params == {}
    p0, i0 = predict_fold(tr, tg, base)
    p1, i1 = predict_fold(tr, tg, opt)
    assert p0["MVT_ID_mvt"].equals(p1["MVT_ID_mvt"])
    unm = tg.filter(pl.col(PROXY).is_null())["MVT_ID_mvt"]
    assert unm.len() > 0
    sel = pl.col("MVT_ID_mvt").is_in(unm.implode())
    assert p0.filter(sel).equals(p1.filter(sel))  # model B did not move
    assert i0["rounds_unmatched"] == i1["rounds_unmatched"]
    a0, a1 = p0.filter(~sel)["pred"].to_numpy(), p1.filter(~sel)["pred"].to_numpy()
    assert np.isfinite(a1).all() and not np.array_equal(a0, a1)  # model A did
    assert i1["matched_params"] == OVERRIDE


def test_the_override_equals_the_same_params_set_globally_on_model_a():
    """matched_params is exactly `params` for model A: same predictions on the matched rows."""
    tr, tg = _fx(43, 8), _fx(44, 5)
    base = _base()
    p1, _ = predict_fold(tr, tg, replace(base, matched_params=dict(OVERRIDE)))
    p2, _ = predict_fold(tr, tg, replace(base, params={**base.params, **OVERRIDE}))
    sel = pl.col("MVT_ID_mvt").is_in(tg.filter(pl.col(PROXY).is_not_null())["MVT_ID_mvt"].implode())
    assert p1.filter(sel).equals(p2.filter(sel))


def test_refusals():
    with pytest.raises(ValueError, match="matched_params"):
        TrainConfig(matched_params={"seed": 1})
    with pytest.raises(ValueError, match="matched_params"):
        TrainConfig(matched_params={"deterministic": False})
    with pytest.raises(ValueError, match="CatBoost"):
        TrainConfig(
            model="catboost", unmatched_num_leaves=127, unmatched_min_data_in_leaf=50, matched_params={"num_leaves": 31}
        )


def test_every_opt_recipe_is_its_base_plus_matched_params_only():
    for p in sorted(Path("configs").glob("*_opt.yaml")):
        base_name = p.name.replace("_opt.yaml", ".yaml")
        c, b = _load(p.name), _load(base_name)
        assert "matched_params" not in b
        mp = c.pop("matched_params")
        assert set(mp) <= {
            "num_leaves",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l2",
            "learning_rate",
        }
        assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}, p.name
        cc, cb = train_config_from_spec(_load(p.name)), train_config_from_spec(b)
        assert cc.for_unmatched().lgb_params() == cb.for_unmatched().lgb_params()
        assert cc.lgb_params() == cb.lgb_params()  # `params` untouched; only model A's override differs


def _script():
    spec = importlib.util.spec_from_file_location("retune_lgb", Path("scripts/retune_lgb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_retune_script_matches_the_preregistration():
    m = _script()
    assert m.BASE_CONFIG.name == "main_flseq_osm_nolirf_l2.yaml"
    assert m.sample(0) == {}
    assert m.effective_params(0) == {
        "num_leaves": 127,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "lambda_l2": 1.0,
        "learning_rate": 0.05,
    }
    assert set(m.SPACE) == {
        "num_leaves",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l2",
        "learning_rate",
    }
    for i in range(1, 200):
        s = m.sample(i)
        assert s == m.sample(i) and set(s) == set(m.SPACE)
        for k, (_, lo, hi) in m.SPACE.items():
            assert lo <= s[k] <= hi, (i, k, s[k])
        assert isinstance(s["num_leaves"], int) and isinstance(s["min_data_in_leaf"], int)
        TrainConfig(matched_params=s)  # accepted
    # The tuning fold validates on a month none of the comparison folds validates on, and trains on the months
    # before it only.
    comp = month_block_folds([202500 + k for k in range(1, 13)])
    assert [f.name for f in comp] == ["jan", "jul", "jul_fwd", "forward"]
    comp_valid = {x for f in comp for x in f.valid_months}
    assert m.TUNE_FOLD.valid_months == (202510,)
    assert not set(m.TUNE_FOLD.valid_months) & comp_valid
    assert m.TUNE_FOLD.train_months == tuple(range(202501, 202510))


def test_the_opt_blend_script_matches_the_preregistration():
    spec = importlib.util.spec_from_file_location("blend_pa_opt", Path("scripts/blend_pa_opt.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.PA_L2 == "main_flseq_osm_nolirf_l2_pa-85f815edf4"
    assert mod.HYBRIDS == {
        "hybrid_blend_l2_pa": ("a2", "pa", "p2", "c2"),
        "hybrid_blend_l2_pa_opt": ("a2o", "pao", "p2o", "c2"),
    }
    assert mod.BASE_FILE == "outputs/hybrid_blend_l2_pa.parquet"
    assert {k: mod.FULL_FILES[k] for k in ("a2o", "pao", "p2o", "c2", "u")} == {
        "a2o": "outputs/main_flseq_osm_nolirf_l2_opt.parquet",
        "pao": "outputs/main_flseq_osm_nolirf_l2_pa_opt.parquet",
        "p2o": "outputs/main_flseq_osm_nolirf_plainy_l2_opt.parquet",
        "c2": "outputs/main_flseq_osm_nolirf_cb_l2.parquet",
        "u": "outputs/main_flseq.parquet",
    }
    v = {"a2o": 1.0, "pao": 4.0, "p2o": 10.0, "c2": 100.0}
    assert mod.matched_value(mod.HYBRIDS["hybrid_blend_l2_pa_opt"], v.__getitem__) == pytest.approx(
        (0.5 * 1 + 0.5 * 4 + 10 + 100) / 3
    )
