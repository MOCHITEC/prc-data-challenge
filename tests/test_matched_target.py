"""Model A's plain-label target (TrainConfig.matched_target), the plain-y member of the matched-lane blend.

The flag must be invisible at its default: every recipe in configs/ keeps its experiment id, and the anchored
path's predictions and record do not change. With it on, only model A's target and add-back change: model B's
predictions do not move. docs/competitor_research_2026-09-21/results/plainy-blend.md."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from prc.features import build_departure_features
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import LABEL, PROXY, TrainConfig, predict_fold

# Experiment ids of every recipe in configs/ as computed by the code at f67a8c1, before matched_target existed.
IDS_BEFORE = {
    "abl_no_categoricals.yaml": "2d339105bd",
    "abl_no_encodings.yaml": "c9761a913d",
    "abl_vanilla_params.yaml": "66aa584f56",
    "afe_forward_agg.yaml": "c338f6a535",
    "afe_forward_base.yaml": "3808ed13de",
    "afe_forward_cyc.yaml": "379abdbf1e",
    "afe_forward_freq.yaml": "81fe33b3a1",
    "afe_forward_meta.yaml": "4524650838",
    "afe_forward_noise2.yaml": "cd928d7121",
    "afe_forward_noise6.yaml": "51cd031781",
    "afe_jan_agg.yaml": "827a396b47",
    "afe_jan_base.yaml": "1add616cd7",
    "afe_jan_cyc.yaml": "42fb2c599c",
    "afe_jan_freq.yaml": "bf563e37f2",
    "afe_jan_meta.yaml": "6c18f00a5f",
    "afe_jan_noise2.yaml": "2bbf657152",
    "afe_jan_noise6.yaml": "967f104b20",
    "c13_base.yaml": "e6806da206",
    "c13_base_jf.yaml": "919d5fce5b",
    "c13_c1.yaml": "84f03c27e2",
    "c13_c1_2col.yaml": "99502d00ed",
    "c13_c1_2col_jf.yaml": "fe53c49187",
    "c13_c3wind.yaml": "f0658b5d2d",
    "c13_gust.yaml": "69fd3a7c9e",
    "c13_noise2.yaml": "e6ab01d48e",
    "c13_noise2_jf.yaml": "dadd0e0efe",
    "c13_noise3.yaml": "acca475972",
    "c13_noise4.yaml": "b97278ddd3",
    "c13_refit_mem.yaml": "096c3c01b9",
    "cb_all.yaml": "c258423fda",
    "cb_base.yaml": "aceed63e33",
    "cb_bk.yaml": "ba8cb09a14",
    "cb_fl.yaml": "b0344482c2",
    "cb_flseq.yaml": "df2d40a22f",
    "cb_noise.yaml": "777d757f54",
    "cb_occ.yaml": "3614f14bec",
    "cb_seq.yaml": "3d255f10d5",
    "cb_ta.yaml": "b257927427",
    "det_a.yaml": "c1fbab2941",
    "det_b.yaml": "d3ef1634a6",
    "determinism_check.yaml": "bbeeeedde8",
    "determinism_check_airport.yaml": "2fe8f964fd",
    "f10_plannedflight.yaml": "2c6be0bee2",
    "feat_base.yaml": "37369fa81a",
    "feat_f1_queue.yaml": "5f1ca09cf5",
    "feat_f2_bank.yaml": "8f5239a856",
    "feat_f3_share.yaml": "faede6d472",
    "feat_f4_noise.yaml": "b9476b4005",
    "feat_f5_rwyact.yaml": "73a6f3020d",
    "feat_f6_deparr.yaml": "1a17a58778",
    "feat_f7_vsusual.yaml": "b2ddf23c88",
    "feat_f8_opoffset.yaml": "e9f1637c5a",
    "feat_f9_dist.yaml": "7ce128b230",
    "ff1_base.yaml": "f95441831c",
    "ff1_noise.yaml": "8e90f0b161",
    "inv_baseline_forward.yaml": "c6b9dbe79b",
    "inv_it10_leaf5.yaml": "58170fdce1",
    "inv_it11_lineartree.yaml": "5449aae42c",
    "inv_it23a_rounds600.yaml": "65dc05a1b4",
    "inv_it23b_holdout2.yaml": "83f6214ff9",
    "leak_check.yaml": "84d69fbcc1",
    "main.yaml": "0f859c881a",
    "main_flseq.yaml": "8615e6a194",
    "main_flseq_a1.yaml": "c14b85d14f",
    "main_flseq_osm.yaml": "bb204cd922",
    "main_flseq_osm_nolirf.yaml": "926741d779",
    "main_flseq_osm_nolirf_a1.yaml": "e0daec598e",
    "main_gap.yaml": "517867a0de",
    "main_reparam.yaml": "46b15a260c",
    "main_v15.yaml": "5c84fe33d3",
    "osm_base_u.yaml": "624ab3fdd0",
    "osm_dist.yaml": "cbfea098b6",
    "osm_dist_u.yaml": "7b834645f1",
    "osm_noise1.yaml": "c6de0edd7c",
    "osm_noise1_u.yaml": "6fbd71e29d",
    "p10_cb_e.yaml": "eb80bbdcd1",
    "p10_cb_ne.yaml": "6f99a081d0",
    "p10_cb_ne_ctr4.yaml": "2cfc778cf4",
    "p10_cb_ne_s43_jf.yaml": "9e82398c28",
    "p10_lgb_e.yaml": "c431898be6",
    "p10_lgb_ne.yaml": "c38c992d61",
    "p10_lgb_ne_s43.yaml": "122d32a2de",
    "p11_airport.yaml": "688ef71567",
    "p11_airport_confirm.yaml": "243c623581",
    "p11_base.yaml": "6ac2c2eece",
    "p11_noise_shuffled.yaml": "a68116c01a",
    "p2_fix1_base.yaml": "40cfc6bb39",
    "p2_fix1_es.yaml": "132b7a3ef0",
    "p4a_base.yaml": "4a8e6ca3eb",
    "p4a_gap.yaml": "72fed772d7",
    "p4a_noise.yaml": "3799ecbb01",
    "p8_wx_off.yaml": "f421484400",
    "p8_wx_on.yaml": "f61e61a25a",
    "p8b_base.yaml": "3dc917225f",
    "p8b_noise.yaml": "b6b05e26c7",
    "p8b_wx.yaml": "7f7be04073",
    "p8e_base.yaml": "798baa7a90",
    "p8e_base2.yaml": "a90601fef0",
    "p8e_ctrl.yaml": "d7e31c32c6",
    "p8e_drop.yaml": "30925f6d13",
    "pw_both.yaml": "1bbd5f7f3a",
    "pw_noise.yaml": "8a628982f8",
    "rep_jan_off.yaml": "252fb4991a",
    "rep_jan_on.yaml": "98f4ddb23b",
    "reparam_lirf.yaml": "9140fe5b48",
    "repro_v2.yaml": "eca534c1b4",
    "submit_v10_reparam.yaml": "cb8bc19b97",
    "submitted_v4.yaml": "69a3be9f36",
    "weather_check.yaml": "6c54f989fe",
    "wx2_B.yaml": "135985b5fb",
    "wx2_C.yaml": "71dc60918b",
    "wx2_base.yaml": "00e088e604",
    "wx2_noise5.yaml": "a6343e3753",
    "wx2_noise7.yaml": "845c11d473",
}


def test_every_existing_recipe_keeps_its_id_and_the_anchored_target():
    for name, want in IDS_BEFORE.items():
        cfg = train_config_from_spec(yaml.safe_load((Path("configs") / name).read_text()))
        assert cfg.hash() == want, name
        assert cfg.matched_target == "anchored", name
    assert TrainConfig().hash() == TrainConfig(matched_target="anchored").hash()


def test_the_plainy_recipe_is_its_base_plus_the_key():
    b, c = (
        yaml.safe_load(Path(p).read_text())
        for p in ("configs/main_flseq_osm_nolirf.yaml", "configs/main_flseq_osm_nolirf_plainy.yaml")
    )
    assert c.pop("matched_target") == "plain"
    assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}
    assert train_config_from_spec(c | {"matched_target": "plain"}).hash() != train_config_from_spec(b).hash()


def test_an_unknown_target_is_refused():
    with pytest.raises(ValueError, match="matched_target"):
        TrainConfig(matched_target="direct")


@pytest.fixture(scope="module")
def frames():
    raw = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in (1, 2, 3)], how="diagonal")
    feats = build_departure_features(raw)
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    return feats.filter(m < 3), feats.filter(m == 3)


def test_the_default_is_the_anchored_path_byte_for_byte(frames):
    tr, tg = frames
    kw = {"num_rounds": 30, "early_stopping": 5, "n_seeds": 2, "lobt_bound": True}
    p0, i0 = predict_fold(tr, tg, TrainConfig(**kw))
    p1, i1 = predict_fold(tr, tg, TrainConfig(**kw, matched_target="anchored"))
    assert p0.equals(p1)
    assert i0 == i1 and "matched_target" not in i0


def test_plain_changes_model_a_only(frames):
    tr, tg = frames
    kw = {"num_rounds": 30, "early_stopping": 5, "n_seeds": 1, "lobt_bound": True, "unmatched_sched_reparam": True}
    p0, _ = predict_fold(tr, tg, TrainConfig(**kw))
    p1, info = predict_fold(tr, tg, TrainConfig(**kw, matched_target="plain"))
    assert info["matched_target"] == "plain"
    assert p0["MVT_ID_mvt"].equals(p1["MVT_ID_mvt"])
    unm = tg.filter(pl.col(PROXY).is_null())["MVT_ID_mvt"]
    mat = tg.filter(pl.col(PROXY).is_not_null())
    assert unm.len() > 0 and mat.height > 0
    sel = pl.col("MVT_ID_mvt").is_in(unm.implode())
    assert p0.filter(sel).equals(p1.filter(sel))
    a = p0.filter(~sel)["pred"].to_numpy()
    b = p1.filter(~sel)["pred"].to_numpy()
    assert not np.allclose(a, b)
    # A plain fit is a fit of the label: its mean sits near the label mean, not at proxy + mean offset.
    y = tr.filter(pl.col(PROXY).is_not_null())[LABEL].drop_nulls().to_numpy().astype(float)
    assert abs(b.mean() - y.mean()) < 0.5 * y.std()
