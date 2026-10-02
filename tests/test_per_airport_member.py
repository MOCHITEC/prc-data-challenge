"""The per-airport member PA (TrainConfig.airport_specialist_only) and the per-fold CV checkpoint (run_cv fold_cache).
docs/competitor_research_2026-09-21/results/per-airport-member.md.

With both off, every existing recipe keeps its experiment id and the LightGBM parameters of both its models, to the
byte, and the default and the P11 grouped paths give byte-identical predictions on a fixed synthetic fold (digests
computed by the code at f3ab74f, before either existed). With the flag on, model A is ten per-airport fits and no
global fit, and model B is untouched."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from prc.features import build_departure_features
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import PROXY, TrainConfig, predict_fold, run_cv

# (experiment id, digest of [model A lgb_params, model B lgb_params]) of every recipe in configs/, computed by the
# code at f3ab74f, before airport_specialist_only and fold_cache existed.
BEFORE = {
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
    "main_flseq_osm_nolirf_cb.yaml": ("6297d62f5f", "5000510713"),
    "main_flseq_osm_nolirf_cb_l2.yaml": ("4aa341b206", "5000510713"),
    "main_flseq_osm_nolirf_l1.yaml": ("50ebc5643d", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2.yaml": ("ef204b44ff", "9f53bbe724"),
    "main_flseq_osm_nolirf_l2_lt.yaml": ("a9fb149ab5", "728861a8cb"),
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
    "p10_cb_e.yaml": ("eb80bbdcd1", "5000510713"),
    "p10_cb_ne.yaml": ("6f99a081d0", "5000510713"),
    "p10_cb_ne_ctr4.yaml": ("2cfc778cf4", "5000510713"),
    "p10_cb_ne_s43_jf.yaml": ("9e82398c28", "5000510713"),
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

# sha1(ids + preds)[:16] of predict_fold on the fixture below, computed by the code at f3ab74f.
PRED_BEFORE = {"default": "957d912af59e4b35", "w05": "f05e05307f28a651", "w10": "db0df94db4569ee6"}

PA = "main_flseq_osm_nolirf_l2_pa.yaml"
BASE = "main_flseq_osm_nolirf_l2.yaml"
PA_KEYS = {"airport_models": True, "airport_blend_weight": 1.0, "airport_specialist_only": True}


def _load(name):
    return yaml.safe_load((Path("configs") / name).read_text())


def _digest(c: TrainConfig) -> str:
    pa = c.lgb_params() if c.model == "lightgbm" else {}
    pb = c.for_unmatched().lgb_params() if c.model == "lightgbm" else {}
    return hashlib.sha1(json.dumps([pa, pb], sort_keys=True, default=str).encode()).hexdigest()[:10]


def _fx(seed, deps):
    return build_departure_features(generate_month(2025, 7, seed=seed, deps_per_airport_day=deps))


def _pred_digest(p: pl.DataFrame) -> str:
    return hashlib.sha1(p["MVT_ID_mvt"].to_numpy().tobytes() + p["pred"].to_numpy().tobytes()).hexdigest()[:16]


def test_every_existing_recipe_keeps_its_id_and_its_lgb_params():
    assert TrainConfig().airport_specialist_only is False
    names = {p.name for p in Path("configs").glob("*.yaml")}
    # *_opt.yaml (optuna-retune.md) are checked against their bases in tests/test_matched_params.py.
    new = {n for n in names - set(BEFORE) if not n.endswith("_opt.yaml")}
    assert new == {PA}, "a new recipe needs adding to BEFORE (or this list)"
    for name, (want_id, want_params) in BEFORE.items():
        c = train_config_from_spec(_load(name))
        assert (c.hash(), _digest(c)) == (want_id, want_params), name
        assert not c.airport_specialist_only, name


def test_the_default_and_the_p11_grouped_paths_predict_byte_for_byte_as_before():
    tr, tg = _fx(43, 8), _fx(44, 5)
    base = TrainConfig(num_rounds=30, early_stopping=10, unmatched_num_rounds=30, airport_min_group_rows=50)
    arms = {
        "default": base,
        "w05": replace(base, airport_models=True, airport_blend_weight=0.5),
        "w10": replace(base, airport_models=True, airport_blend_weight=1.0),
    }
    for name, c in arms.items():
        p, _ = predict_fold(tr, tg, c)
        assert _pred_digest(p) == PRED_BEFORE[name], name


def test_the_pa_recipe_is_its_base_plus_the_three_keys():
    c, b = _load(PA), _load(BASE)
    for k, v in PA_KEYS.items():
        assert k not in b, k
        got = c.pop(k)
        assert got == v and type(got) is type(v), k
    assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}
    cp, cb = train_config_from_spec(_load(PA)), train_config_from_spec(b)
    assert cp.hash() != cb.hash()
    assert cp.lgb_params() == cb.lgb_params() and cp.for_unmatched().lgb_params() == cb.for_unmatched().lgb_params()
    assert (cp.num_rounds, cp.early_stopping, cp.n_seeds, cp.folds, cp.matched_target, cp.inner_holdout_months) == (
        cb.num_rounds,
        cb.early_stopping,
        cb.n_seeds,
        cb.folds,
        cb.matched_target,
        cb.inner_holdout_months,
    )
    assert cp.airport_group_key == "adep" and cp.airport_min_group_rows == 20_000


def test_the_flag_is_refused_without_airport_models_at_full_weight():
    with pytest.raises(ValueError, match="airport_specialist_only"):
        TrainConfig(airport_specialist_only=True)
    with pytest.raises(ValueError, match="airport_specialist_only"):
        TrainConfig(airport_specialist_only=True, airport_models=True, airport_blend_weight=0.5)


def test_specialist_only_fits_every_airport_alone_and_leaves_model_b_byte_identical():
    tr, tg = _fx(43, 8), _fx(44, 5)
    base = TrainConfig(num_rounds=30, early_stopping=10, unmatched_num_rounds=30, airport_min_group_rows=50)
    p0, i0 = predict_fold(tr, tg, base)
    p1, i1 = predict_fold(tr, tg, replace(base, **PA_KEYS))
    assert p0["MVT_ID_mvt"].equals(p1["MVT_ID_mvt"])
    unm = tg.filter(pl.col(PROXY).is_null())["MVT_ID_mvt"]
    assert unm.len() > 0
    sel = pl.col("MVT_ID_mvt").is_in(unm.implode())
    assert p0.filter(sel).equals(p1.filter(sel))  # model B did not move
    assert i0["rounds_unmatched"] == i1["rounds_unmatched"]
    a0, a1 = p0.filter(~sel)["pred"].to_numpy(), p1.filter(~sel)["pred"].to_numpy()
    assert np.isfinite(a1).all() and not np.array_equal(a0, a1)  # model A did
    airports = set(tr.filter(pl.col(PROXY).is_not_null())["ADEP_mvt"].unique().to_list())
    assert i1["airport_specialist_only"] is True
    assert set(i1["rounds_matched"]) == airports and all(r > 0 for r in i1["rounds_matched"].values())
    assert i1["airport_n_groups_fitted"] == len(airports) and i1["airport_ungrouped_target_rows"] == 0
    assert i1["airport_blended_rows"] == int(tg[PROXY].is_not_null().sum())
    assert all(r["dropped"] == ["ADEP_mvt"] for r in i1["airport_groups"].values())


def test_specialist_only_refuses_a_matched_row_no_group_predicts():
    tr, tg = _fx(43, 8), _fx(44, 5)
    cfg = TrainConfig(num_rounds=10, early_stopping=5, unmatched_num_rounds=10, airport_min_group_rows=50, **PA_KEYS)
    with pytest.raises(ValueError, match="fall in no fitted group"):
        predict_fold(tr, tg, replace(cfg, airport_min_group_rows=10**9))
    # an airport in the target with no training rows has no specialist either
    with pytest.raises(ValueError, match="fall in no fitted group"):
        predict_fold(tr.filter(pl.col("ADEP_mvt") != "EDDF"), tg, cfg)


def test_the_fold_cache_resumes_to_the_same_oof_and_record(tmp_path, monkeypatch):
    import prc.train as T

    df = pl.concat([generate_month(2025, m, seed=60 + m, deps_per_airport_day=4) for m in range(1, 13)])
    feats = build_departure_features(df)
    cfg = TrainConfig(name="fcache", num_rounds=20, early_stopping=5, unmatched_num_rounds=20)
    plain = run_cv(feats, cfg, out_dir=tmp_path / "a", log_name="t.jsonl")
    cache = tmp_path / "cache"
    first = run_cv(feats, cfg, out_dir=tmp_path / "b", log_name="t.jsonl", fold_cache=cache)
    folds = list(plain["folds"])
    assert sorted(p.name for p in (cache / plain["exp_id"]).iterdir()) == sorted(
        f"{f}.{x}" for f in folds for x in ("json", "parquet")
    )
    # Simulate a restart after the second fold: drop the later checkpoints, and make any refit of a cached fold fail.
    for f in folds[2:]:
        (cache / plain["exp_id"] / f"{f}.json").unlink()
    real, calls = T.predict_fold, []

    def counting(tr, va, c):
        calls.append(1)
        return real(tr, va, c)

    monkeypatch.setattr(T, "predict_fold", counting)
    resumed = run_cv(feats, cfg, out_dir=tmp_path / "c", log_name="t.jsonl", fold_cache=cache)
    assert len(calls) == len(folds) - 2
    oofs = [pl.read_parquet(tmp_path / d / f"oof_{plain['exp_id']}.parquet") for d in ("a", "b", "c")]
    assert oofs[0].equals(oofs[1]) and oofs[0].equals(oofs[2])

    def norm(rec):
        return json.loads(json.dumps({k: v for k, v in rec.items() if k != "train_seconds"}, default=str))

    assert norm(plain) == norm(first) == norm(resumed)


def test_the_blend_script_members_match_the_preregistration():
    import importlib.util

    spec = importlib.util.spec_from_file_location("blend_pa", Path("scripts/blend_pa.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.ANCHORED_L2, mod.PLAIN_L2, mod.CB_L2, mod.UNMATCHED) == (
        "main_flseq_osm_nolirf_l2-ef204b44ff",
        "main_flseq_osm_nolirf_plainy_l2-c74a27239b",
        "main_flseq_osm_nolirf_cb_l2-4aa341b206",
        "main_flseq-8615e6a194",
    )
    assert mod.BASE_FILE == "outputs/hybrid_blend_l2.parquet"
    assert mod.FULL_FILES == {
        "a2": "outputs/main_flseq_osm_nolirf_l2.parquet",
        "p2": "outputs/main_flseq_osm_nolirf_plainy_l2.parquet",
        "c2": "outputs/main_flseq_osm_nolirf_cb_l2.parquet",
        "pa": "outputs/main_flseq_osm_nolirf_l2_pa.parquet",
        "u": "outputs/main_flseq.parquet",
    }
    v = {"a2": 1.0, "pa": 4.0, "p2": 10.0, "c2": 100.0}
    get = v.__getitem__
    assert mod.matched_value(mod.HYBRIDS["hybrid_blend_l2_P"], get) == pytest.approx((1 + 10 + 100) / 3)
    assert mod.matched_value(mod.HYBRIDS["hybrid_blend_l2_pa"], get) == pytest.approx(
        (0.5 * 1 + 0.5 * 4 + 10 + 100) / 3
    )
    assert mod.matched_value(mod.HYBRIDS["hybrid_blend_l2_pa4"], get) == pytest.approx((1 + 4 + 10 + 100) / 4)
