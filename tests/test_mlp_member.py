"""TrainConfig.mlp_matched: model A as a scikit-learn MLP (prc.mlp), the matched-lane member M4 of mlp-member.md.

Off (the default) the flag is invisible: every recipe in configs/ keeps its experiment id and the resolved parameters
of both its models, and predict_fold's predictions and `info` on a fixed synthetic fold hash exactly as under the code
at f6b09ef, before the flag existed (default, plain-y, per-airport and matched_params arms). On, model B's predictions
are byte-identical to the same recipe without it and model A moves."""

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from prc import mlp
from prc.features import build_departure_features
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import LABEL, PROXY, TrainConfig, predict_fold

MLP_RECIPE = "main_flseq_osm_nolirf_l2_mlp.yaml"
MLP_BASE = "main_flseq_osm_nolirf_l2.yaml"

# (experiment id, digest of [model A params, model B params]) of every recipe in configs/ at f6b09ef.
BEFORE = {
    "main_flseq_osm_nolirf_plainy_l2_opt_l6.yaml": ("1f94f51232", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2_pa_opt_l6.yaml": ("523fcc4e8c", "9f53bbe724"),
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
    'abl_no_categoricals.yaml': ('2d339105bd', '864be36cf6'),
    'abl_no_encodings.yaml': ('c9761a913d', '864be36cf6'),
    'abl_vanilla_params.yaml': ('66aa584f56', 'f95eb3b33e'),
    'afe_forward_agg.yaml': ('c338f6a535', '9f53bbe724'),
    'afe_forward_base.yaml': ('3808ed13de', '9f53bbe724'),
    'afe_forward_cyc.yaml': ('379abdbf1e', '9f53bbe724'),
    'afe_forward_freq.yaml': ('81fe33b3a1', '9f53bbe724'),
    'afe_forward_meta.yaml': ('4524650838', '9f53bbe724'),
    'afe_forward_noise2.yaml': ('cd928d7121', '9f53bbe724'),
    'afe_forward_noise6.yaml': ('51cd031781', '9f53bbe724'),
    'afe_jan_agg.yaml': ('827a396b47', '9f53bbe724'),
    'afe_jan_base.yaml': ('1add616cd7', '9f53bbe724'),
    'afe_jan_cyc.yaml': ('42fb2c599c', '9f53bbe724'),
    'afe_jan_freq.yaml': ('bf563e37f2', '9f53bbe724'),
    'afe_jan_meta.yaml': ('6c18f00a5f', '9f53bbe724'),
    'afe_jan_noise2.yaml': ('2bbf657152', '9f53bbe724'),
    'afe_jan_noise6.yaml': ('967f104b20', '9f53bbe724'),
    'c13_base.yaml': ('e6806da206', '9f53bbe724'),
    'c13_base_jf.yaml': ('919d5fce5b', '9f53bbe724'),
    'c13_c1.yaml': ('84f03c27e2', '9f53bbe724'),
    'c13_c1_2col.yaml': ('99502d00ed', '9f53bbe724'),
    'c13_c1_2col_jf.yaml': ('fe53c49187', '9f53bbe724'),
    'c13_c3wind.yaml': ('f0658b5d2d', '9f53bbe724'),
    'c13_gust.yaml': ('69fd3a7c9e', '9f53bbe724'),
    'c13_noise2.yaml': ('e6ab01d48e', '9f53bbe724'),
    'c13_noise2_jf.yaml': ('dadd0e0efe', '9f53bbe724'),
    'c13_noise3.yaml': ('acca475972', '9f53bbe724'),
    'c13_noise4.yaml': ('b97278ddd3', '9f53bbe724'),
    'c13_refit_mem.yaml': ('096c3c01b9', '9f53bbe724'),
    'cb_all.yaml': ('c258423fda', '9f53bbe724'),
    'cb_base.yaml': ('aceed63e33', '9f53bbe724'),
    'cb_bk.yaml': ('ba8cb09a14', '9f53bbe724'),
    'cb_fl.yaml': ('b0344482c2', '9f53bbe724'),
    'cb_flseq.yaml': ('df2d40a22f', '9f53bbe724'),
    'cb_noise.yaml': ('777d757f54', '9f53bbe724'),
    'cb_occ.yaml': ('3614f14bec', '9f53bbe724'),
    'cb_seq.yaml': ('3d255f10d5', '9f53bbe724'),
    'cb_ta.yaml': ('b257927427', '9f53bbe724'),
    'det_a.yaml': ('c1fbab2941', '9f53bbe724'),
    'det_b.yaml': ('d3ef1634a6', '9f53bbe724'),
    'determinism_check.yaml': ('bbeeeedde8', '9f53bbe724'),
    'determinism_check_airport.yaml': ('2fe8f964fd', '9f53bbe724'),
    'f10_plannedflight.yaml': ('2c6be0bee2', '21300f5568'),
    'feat_base.yaml': ('37369fa81a', '9f53bbe724'),
    'feat_f1_queue.yaml': ('5f1ca09cf5', '9f53bbe724'),
    'feat_f2_bank.yaml': ('8f5239a856', '9f53bbe724'),
    'feat_f3_share.yaml': ('faede6d472', '9f53bbe724'),
    'feat_f4_noise.yaml': ('b9476b4005', '9f53bbe724'),
    'feat_f5_rwyact.yaml': ('73a6f3020d', '21300f5568'),
    'feat_f6_deparr.yaml': ('1a17a58778', '21300f5568'),
    'feat_f7_vsusual.yaml': ('b2ddf23c88', '21300f5568'),
    'feat_f8_opoffset.yaml': ('e9f1637c5a', '21300f5568'),
    'feat_f9_dist.yaml': ('7ce128b230', '21300f5568'),
    'ff1_base.yaml': ('f95441831c', '21300f5568'),
    'ff1_noise.yaml': ('8e90f0b161', '21300f5568'),
    'inv_baseline_forward.yaml': ('c6b9dbe79b', '62d486499b'),
    'inv_it10_leaf5.yaml': ('58170fdce1', '2be8ce0fe9'),
    'inv_it11_lineartree.yaml': ('5449aae42c', 'cfd5cdff4e'),
    'inv_it23a_rounds600.yaml': ('65dc05a1b4', '62d486499b'),
    'inv_it23b_holdout2.yaml': ('83f6214ff9', '62d486499b'),
    'leak_check.yaml': ('84d69fbcc1', '9f53bbe724'),
    'main.yaml': ('0f859c881a', '9f53bbe724'),
    'main_flseq.yaml': ('8615e6a194', '9f53bbe724'),
    'main_flseq_a1.yaml': ('c14b85d14f', '9f53bbe724'),
    'main_flseq_l2b.yaml': ('e40f03b86d', '9f53bbe724'),
    'main_flseq_osm.yaml': ('bb204cd922', '9f53bbe724'),
    'main_flseq_osm_nolirf.yaml': ('926741d779', '9f53bbe724'),
    'main_flseq_osm_nolirf_a1.yaml': ('e0daec598e', '9f53bbe724'),
    'main_flseq_osm_nolirf_cb.yaml': ('6297d62f5f', 'ef6028717f'),
    'main_flseq_osm_nolirf_cb_l2.yaml': ('4aa341b206', 'ef6028717f'),
    'main_flseq_osm_nolirf_l1.yaml': ('50ebc5643d', '9f53bbe724'),
    'main_flseq_osm_nolirf_l2.yaml': ('ef204b44ff', '9f53bbe724'),
    'main_flseq_osm_nolirf_l2_lt.yaml': ('a9fb149ab5', '728861a8cb'),
    'main_flseq_osm_nolirf_l2_opt.yaml': ('7244b3004f', '9f53bbe724'),
    'main_flseq_osm_nolirf_l2_pa.yaml': ('85f815edf4', '9f53bbe724'),
    'main_flseq_osm_nolirf_l2_pa_opt.yaml': ('8d54effc06', '9f53bbe724'),
    'main_flseq_osm_nolirf_l3.yaml': ('d08ab29d15', '9f53bbe724'),
    'main_flseq_osm_nolirf_plainy.yaml': ('b971f4b81b', '9f53bbe724'),
    'main_flseq_osm_nolirf_plainy_l2.yaml': ('c74a27239b', '9f53bbe724'),
    'main_flseq_osm_nolirf_plainy_l2_lt.yaml': ('af11759d5d', '728861a8cb'),
    'main_flseq_osm_nolirf_plainy_l2_opt.yaml': ('dd8b8e2c7c', '9f53bbe724'),
    'main_gap.yaml': ('517867a0de', '9f53bbe724'),
    'main_reparam.yaml': ('46b15a260c', '9f53bbe724'),
    'main_v15.yaml': ('5c84fe33d3', '9f53bbe724'),
    'osm_base_u.yaml': ('624ab3fdd0', '9f53bbe724'),
    'osm_dist.yaml': ('cbfea098b6', '9f53bbe724'),
    'osm_dist_u.yaml': ('7b834645f1', '9f53bbe724'),
    'osm_noise1.yaml': ('c6de0edd7c', '9f53bbe724'),
    'osm_noise1_u.yaml': ('6fbd71e29d', '9f53bbe724'),
    'p10_cb_e.yaml': ('eb80bbdcd1', 'ef6028717f'),
    'p10_cb_ne.yaml': ('6f99a081d0', 'ef6028717f'),
    'p10_cb_ne_ctr4.yaml': ('2cfc778cf4', 'bfe4989136'),
    'p10_cb_ne_s43_jf.yaml': ('9e82398c28', 'ef6028717f'),
    'p10_lgb_e.yaml': ('c431898be6', '9f53bbe724'),
    'p10_lgb_ne.yaml': ('c38c992d61', '9f53bbe724'),
    'p10_lgb_ne_s43.yaml': ('122d32a2de', '61011b43ed'),
    'p11_airport.yaml': ('688ef71567', '9f53bbe724'),
    'p11_airport_confirm.yaml': ('243c623581', '9f53bbe724'),
    'p11_base.yaml': ('6ac2c2eece', '9f53bbe724'),
    'p11_noise_shuffled.yaml': ('a68116c01a', '9f53bbe724'),
    'p2_fix1_base.yaml': ('40cfc6bb39', '9f53bbe724'),
    'p2_fix1_es.yaml': ('132b7a3ef0', '9f53bbe724'),
    'p4a_base.yaml': ('4a8e6ca3eb', '9f53bbe724'),
    'p4a_gap.yaml': ('72fed772d7', '9f53bbe724'),
    'p4a_noise.yaml': ('3799ecbb01', '9f53bbe724'),
    'p8_wx_off.yaml': ('f421484400', '9f53bbe724'),
    'p8_wx_on.yaml': ('f61e61a25a', '9f53bbe724'),
    'p8b_base.yaml': ('3dc917225f', '9f53bbe724'),
    'p8b_noise.yaml': ('b6b05e26c7', '9f53bbe724'),
    'p8b_wx.yaml': ('7f7be04073', '9f53bbe724'),
    'p8e_base.yaml': ('798baa7a90', '9f53bbe724'),
    'p8e_base2.yaml': ('a90601fef0', '9f53bbe724'),
    'p8e_ctrl.yaml': ('d7e31c32c6', '9f53bbe724'),
    'p8e_drop.yaml': ('30925f6d13', '9f53bbe724'),
    'pw_both.yaml': ('1bbd5f7f3a', '21300f5568'),
    'pw_noise.yaml': ('8a628982f8', '21300f5568'),
    'rep_jan_off.yaml': ('252fb4991a', '21300f5568'),
    'rep_jan_on.yaml': ('98f4ddb23b', '21300f5568'),
    'reparam_lirf.yaml': ('9140fe5b48', '62d486499b'),
    'repro_v2.yaml': ('eca534c1b4', '9f53bbe724'),
    'submit_v10_reparam.yaml': ('cb8bc19b97', '9f53bbe724'),
    'submitted_v4.yaml': ('69a3be9f36', 'abab3846e4'),
    'weather_check.yaml': ('6c54f989fe', '9f53bbe724'),
    'wx2_B.yaml': ('135985b5fb', '9f53bbe724'),
    'wx2_C.yaml': ('71dc60918b', '9f53bbe724'),
    'wx2_base.yaml': ('00e088e604', '9f53bbe724'),
    'wx2_noise5.yaml': ('a6343e3753', '9f53bbe724'),
    'wx2_noise7.yaml': ('845c11d473', '9f53bbe724'),
}

# [sha1(ids + preds)[:16], sha1(info json)[:16]] of predict_fold on `_fx(43, 8)` -> `_fx(44, 5)`, code at f6b09ef.
PRED_BEFORE = {
    'default': ('bd675c22ce30d32c', '54a56a10544c07c9'),
    'plain': ('7b6cfe30e8c6bd68', 'e256fc06be1999bd'),
    'pa': ('a928a6226295514c', '031c1ddecc548327'),
    'opt': ('03bb7a370d054c30', '1c583bbe1940868c'),
}

TINY = {"hidden": [16], "max_epochs": 4, "chunk_rows": 512, "batch_size": 64, "quantile_rows": 5000, "cat_min_count": 5}


def _load(name):
    return yaml.safe_load((Path("configs") / name).read_text())


def _digest(c: TrainConfig) -> str:
    pa = c.lgb_params() if c.model == "lightgbm" else c.catboost_params()
    pb = c.for_unmatched().lgb_params() if c.model == "lightgbm" else c.for_unmatched().catboost_params()
    return hashlib.sha1(json.dumps([pa, pb], sort_keys=True, default=str).encode()).hexdigest()[:10]


def _fx(seed, deps, month=7):
    return build_departure_features(generate_month(2025, month, seed=seed, deps_per_airport_day=deps))


def _base():
    return TrainConfig(num_rounds=30, early_stopping=10, unmatched_num_rounds=30, airport_min_group_rows=50, n_seeds=2)


def _two_months():
    """Two training months, so the inner holdout (the latest month) is not empty."""
    return pl.concat([_fx(43, 8, 6), _fx(45, 8, 7)], how="diagonal_relaxed"), _fx(44, 5, 8)


def test_every_existing_recipe_keeps_its_id_and_both_models_params():
    assert TrainConfig().mlp_matched is False and TrainConfig().mlp_params == {}
    names = {p.name for p in Path("configs").glob("*.yaml")}
    # *_cbopt.yaml (catboost-retune.md) are checked in tests/test_cb_matched_params.py.
    assert {n for n in names - set(BEFORE) if not n.endswith("_cbopt.yaml")} == {MLP_RECIPE}
    for name, (want_id, want_params) in BEFORE.items():
        c = train_config_from_spec(_load(name))
        assert (c.hash(), _digest(c)) == (want_id, want_params), name
        assert c.mlp_matched is False and c.mlp_params == {}, name


def test_existing_paths_predict_byte_for_byte_as_before():
    tr, tg = _fx(43, 8), _fx(44, 5)
    base = _base()
    arms = {
        "default": base,
        "plain": replace(base, matched_target="plain"),
        "pa": replace(base, airport_models=True, airport_blend_weight=1.0, airport_specialist_only=True),
        "opt": replace(base, matched_params={"num_leaves": 63, "learning_rate": 0.08}),
    }
    for name, c in arms.items():
        p, info = predict_fold(tr, tg, c)
        got = [
            hashlib.sha1(p["MVT_ID_mvt"].to_numpy().tobytes() + p["pred"].to_numpy().tobytes()).hexdigest()[:16],
            hashlib.sha1(json.dumps(info, sort_keys=True, default=str).encode()).hexdigest()[:16],
        ]
        assert tuple(got) == PRED_BEFORE[name], name
        assert "mlp_matched" not in info


def test_the_mlp_recipe_is_its_base_plus_the_flag_only():
    c, b = _load(MLP_RECIPE), _load(MLP_BASE)
    assert c.pop("mlp_matched") is True
    assert c.pop("name") == "main_flseq_osm_nolirf_l2_mlp"
    b.pop("name")
    assert c == b
    cc, cb = train_config_from_spec(_load(MLP_RECIPE)), train_config_from_spec(_load(MLP_BASE))
    assert cc.mlp_params == {} and cc.n_seeds == 3 and cc.inner_holdout_months == 1
    assert cc.for_unmatched().lgb_params() == cb.for_unmatched().lgb_params()
    assert cc.hash() != cb.hash()
    assert mlp.resolve(cc.mlp_params) == mlp.MLP_DEFAULTS


def test_preregistered_settings():
    assert mlp.MLP_DEFAULTS == {
        "hidden": [512, 256],
        "activation": "relu",
        "alpha": 1e-4,
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "max_epochs": 40,
        "patience": 5,
        "lr_halve_every": 2,
        "chunk_rows": 65536,
        "quantiles": 1000,
        "quantile_rows": 200000,
        "cat_min_count": 1000,
        "cat_max_levels": 64,
    }


def test_refusals():
    with pytest.raises(ValueError, match="mlp_params"):
        TrainConfig(mlp_params={"hidden": [8]})
    with pytest.raises(ValueError, match="unknown mlp_params"):
        TrainConfig(mlp_matched=True, mlp_params={"depth": 3})
    for bad in (
        {"matched_params": {"num_leaves": 31}},
        {"airport_models": True},
        {"es_holdout_encodings": True},
        {"use_linear_tree": True},
    ):
        with pytest.raises(ValueError, match="mlp_matched cannot"):
            TrainConfig(mlp_matched=True, **bad)
    assert TrainConfig(mlp_matched=True, mlp_params={"hidden": [8]}).hash() != TrainConfig(mlp_matched=True).hash()


def test_the_flag_moves_model_a_only():
    tr, tg = _two_months()
    base = _base()
    m = replace(base, mlp_matched=True, mlp_params=dict(TINY))
    p0, i0 = predict_fold(tr, tg, base)
    p1, i1 = predict_fold(tr, tg, m)
    assert p0["MVT_ID_mvt"].equals(p1["MVT_ID_mvt"])
    unm = tg.filter(pl.col(PROXY).is_null())["MVT_ID_mvt"]
    assert unm.len() > 0
    sel = pl.col("MVT_ID_mvt").is_in(unm.implode())
    assert p0.filter(sel).equals(p1.filter(sel))  # model B did not move
    assert i0["rounds_unmatched"] == i1["rounds_unmatched"]
    a0, a1 = p0.filter(~sel)["pred"].to_numpy(), p1.filter(~sel)["pred"].to_numpy()
    assert np.isfinite(a1).all() and (a1 >= 0).all() and not np.array_equal(a0, a1)
    rec = i1["mlp_matched"]
    assert rec["holdout_rows"] > 0 and rec["es_best_epoch"] >= 1 and rec["epochs"] == rec["es_best_epoch"]
    assert i1["rounds_matched"] == rec["epochs"]


def test_target_labels_and_batch_mates_cannot_move_a_prediction():
    tr, tg = _two_months()
    m = replace(_base(), mlp_matched=True, mlp_params=dict(TINY))
    p1, _ = predict_fold(tr, tg, m)
    p2, _ = predict_fold(tr, tg.with_columns(pl.lit(None, dtype=tg[LABEL].dtype).alias(LABEL)), m)
    assert p1.equals(p2)
    half = tg.head(tg.height // 2)
    p3, _ = predict_fold(tr, half, m)
    # Equal up to float32 summation order (BLAS blocks a matrix product by its row count), not by information.
    assert p3["MVT_ID_mvt"].equals(p1.head(half.height)["MVT_ID_mvt"])
    np.testing.assert_allclose(p3["pred"].to_numpy(), p1.head(half.height)["pred"].to_numpy(), rtol=0, atol=1e-3)


def test_preprocessor_unseen_levels_and_missing_indicators():
    fr = pl.DataFrame({"a": [1.0, None, 3.0, 4.0] * 10, "b": [1.0] * 40, "c": ["x", "y", "x", None] * 10})
    pre = mlp.Preprocessor({**mlp.MLP_DEFAULTS, "cat_min_count": 10, "quantile_rows": 100}, 42).fit(fr, ["a", "b"], ["c"])
    assert pre.summary() == {"n_num": 2, "n_nan_indicators": 1, "cat_levels": {"c": 3}, "input_width": 2 + 1 + 4}
    new = pl.DataFrame({"a": [None, 2.0], "b": [1.0, float("inf")], "c": ["zz", "y"]})
    D = pre.design(pre.num(new), pre.codes(new))
    assert D.shape == (2, 7) and np.isfinite(D).all()
    assert D[:, 2].tolist() == [1.0, 0.0]  # a missing
    assert D[0, 3 + 3] == 1.0 and D[0, 3:6].sum() == 0  # unseen level -> other
    assert D[1, 3:].sum() == 1.0


def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path, monkeypatch):
    tr, tg = _two_months()
    m = replace(_base(), mlp_matched=True, mlp_params=dict(TINY))
    monkeypatch.delenv("PRC_MLP_CKPT", raising=False)
    ref, _ = predict_fold(tr, tg, m)
    monkeypatch.setenv("PRC_MLP_CKPT", str(tmp_path))
    real, calls = mlp._epoch, {"n": 0}

    def dying(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("simulated restart")
        return real(*a, **k)

    monkeypatch.setattr(mlp, "_epoch", dying)
    with pytest.raises(KeyboardInterrupt):
        predict_fold(tr, tg, m)
    saved = list(tmp_path.glob("*.pkl"))
    assert len(saved) == 1 and pickle.loads(saved[0].read_bytes())["epoch"] == 2
    monkeypatch.setattr(mlp, "_epoch", real)
    got, info = predict_fold(tr, tg, m)
    assert got["MVT_ID_mvt"].equals(ref["MVT_ID_mvt"])
    np.testing.assert_allclose(got["pred"].to_numpy(), ref["pred"].to_numpy(), rtol=0, atol=1e-6)
    again, _ = predict_fold(tr, tg, m)  # everything is on disk now: the refits' predictions are read back
    assert again.equals(got)


def test_the_blend_script_matches_the_preregistration():
    import importlib.util

    spec = importlib.util.spec_from_file_location("blend_mlp", Path("scripts/blend_mlp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.A2O, mod.PAO, mod.P2O, mod.C2) == (
        "main_flseq_osm_nolirf_l2_opt-7244b3004f",
        "main_flseq_osm_nolirf_l2_pa_opt-8d54effc06",
        "main_flseq_osm_nolirf_plainy_l2_opt-dd8b8e2c7c",
        "main_flseq_osm_nolirf_cb_l2-4aa341b206",
    )
    assert mod.FULL_FILES == {
        "a2o": "outputs/main_flseq_osm_nolirf_l2_opt.parquet",
        "pao": "outputs/main_flseq_osm_nolirf_l2_pa_opt.parquet",
        "p2o": "outputs/main_flseq_osm_nolirf_plainy_l2_opt.parquet",
        "c2": "outputs/main_flseq_osm_nolirf_cb_l2.parquet",
        "m4": "outputs/main_flseq_osm_nolirf_l2_mlp.parquet",
        "u": "outputs/main_flseq.parquet",
    }
    assert mod.BASE_FILE == "outputs/hybrid_blend_l2_pa_opt.parquet"
    v = {"a2o": 1.0, "pao": 3.0, "p2o": 10.0, "c2": 100.0, "m4": 1000.0}
    assert mod.base_value(v.__getitem__) == pytest.approx((2.0 + 10 + 100) / 3)
    assert mod.candidate_value(v.__getitem__) == pytest.approx((2.0 + 10 + 100 + 1000) / 4)
