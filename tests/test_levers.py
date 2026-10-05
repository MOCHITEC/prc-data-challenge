"""The public-code levers L1-L3 (prc.levers; TrainConfig.fe_plan_nm_taxi / fe_fwd_nm / fe_fwd_nm_unmatched /
fe_push_queue). docs/competitor_research_2026-09-21/results/feature-levers-L1-L4.md.

With every flag off an existing recipe keeps its experiment id, its design matrix and its predictions to the byte,
whether or not the lever columns happen to be in the frame. With a flag on, the columns reach only the lanes the
pre-registration names, and the other lane's predictions do not move."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from prc.features import build_departure_features
from prc.levers import L1_COLS, L2_COLS, L3_COLS, L6_COLS, LEVER_COLS, _in_system, attach_levers
from prc.pipeline import train_config_from_spec
from prc.synth import generate_month
from prc.train import PROXY, TrainConfig, predict_fold, unmatched_columns, visible_numeric_columns

# Experiment ids of every recipe in configs/ computed by the code at d8775f9, before prc.levers existed.
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
    "main_flseq_osm_nolirf_cb.yaml": "6297d62f5f",
    "main_flseq_osm_nolirf_plainy.yaml": "b971f4b81b",
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

LEVER_CONFIGS = {
    "main_flseq_osm_nolirf_l1.yaml": ("main_flseq_osm_nolirf.yaml", {"fe_plan_nm_taxi": True}),
    "main_flseq_osm_nolirf_l2.yaml": ("main_flseq_osm_nolirf.yaml", {"fe_fwd_nm": True}),
    "main_flseq_osm_nolirf_l3.yaml": ("main_flseq_osm_nolirf.yaml", {"fe_push_queue": True}),
    "main_flseq_l2b.yaml": ("main_flseq.yaml", {"fe_fwd_nm_unmatched": True}),
}


def test_every_existing_recipe_keeps_its_id():
    cfg = TrainConfig()
    assert not (cfg.fe_plan_nm_taxi or cfg.fe_fwd_nm or cfg.fe_fwd_nm_unmatched or cfg.fe_push_queue)
    for name, want in IDS_BEFORE.items():
        spec = yaml.safe_load((Path("configs") / name).read_text())
        assert train_config_from_spec(spec).hash() == want, name


def test_the_lever_configs_differ_from_their_base_only_by_the_flag():
    for name, (base, flags) in LEVER_CONFIGS.items():
        c = yaml.safe_load((Path("configs") / name).read_text())
        b = yaml.safe_load((Path("configs") / base).read_text())
        for k, v in flags.items():
            assert c.pop(k) is v, (name, k)
        assert {k: v for k, v in c.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}, name
        assert train_config_from_spec(c | flags).hash() != train_config_from_spec(b).hash()


def _t(h, m=0, s=0):
    return datetime(2025, 7, 1, h, m, s, tzinfo=UTC)


def _raw() -> pl.DataFrame:
    """Four departures at XXXX (runway R1 x3, R2 x1) and two arrivals.

    id 1: push 10:00, take-off 10:20   id 2: push 10:05, take-off 10:10   id 3: no AOBT_3, take-off 10:15
    id 4: push 10:02, take-off 10:30 (R2)
    arrivals: landed 09:55, in-block 10:03 ; landed 10:01, in-block 10:12"""
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0],
            "PHASE_mvt": ["DEP"] * 4,
            "ADEP_mvt": ["XXXX"] * 4,
            "ADES_mvt": ["AAAA", "AAAA", "AAAA", "BBBB"],
            "AIRCRAFT_TYPE_mvt": ["A320", "A320", "A320", "B738"],
            "RUNWAY_mvt": ["R1", "R1", "R1", "R2"],
            "MVT_TIME_UTC_mvt": [_t(10, 20), _t(10, 10), _t(10, 15), _t(10, 30)],
            "AOBT_3_flt": [_t(10), _t(10, 5), None, _t(10, 2)],
            "EOBT_1_flt": [_t(9), _t(9), _t(9), _t(9)],
            "ARVT_1_flt": [_t(11), _t(11, 10), _t(11, 20), _t(12)],
            "BLOCK_TIME_UTC_mvt": [_t(9, 59), _t(10, 4), _t(10), _t(10)],
        }
    )
    arr = pl.DataFrame(
        {
            "MVT_ID_mvt": [5.0, 6.0],
            "PHASE_mvt": ["ARR", "ARR"],
            "ADEP_mvt": ["CCCC", "CCCC"],
            "ADES_mvt": ["XXXX", "XXXX"],
            "AIRCRAFT_TYPE_mvt": ["A320", "A320"],
            "RUNWAY_mvt": ["R1", "R1"],
            "MVT_TIME_UTC_mvt": [_t(9, 55), _t(10, 1)],
            "AOBT_3_flt": [None, None],
            "EOBT_1_flt": [None, None],
            "ARVT_1_flt": [None, None],
            "BLOCK_TIME_UTC_mvt": [_t(10, 3), _t(10, 12)],
        },
        schema_overrides={c: pl.Datetime("us", "UTC") for c in ("AOBT_3_flt", "EOBT_1_flt", "ARVT_1_flt")},
    )
    return pl.concat([dep, arr], how="vertical_relaxed")


def _levers(raw, l1=True, l2=True, l3=True):
    f = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt").sort("MVT_ID_mvt", descending=True)
    a, _ = attach_levers(f, f, raw, raw, l1, l2, l3)
    return a.sort("MVT_ID_mvt")


def test_hand_built_values():
    out = _levers(_raw())
    # L1: plans 7,200 / 7,800 / 8,400 on (XXXX, AAAA, A320) and 10,800 on BBBB. The table is fitted on both
    # frames (the same rows twice here): 6 rows for AAAA meet PLAN_MIN_N = 5, the 2 for BBBB do not, at either level.
    assert out["plan_nm_taxi"].to_list() == pytest.approx([-600.0, 0.0, 600.0, None])
    # L2, airport, 15 min: row 2 (t 10:10) sees 3 (no AOBT_3, not counted) and 1 (proxy 1,200) -> mean 1,200, n 1;
    # row 3 (t 10:15) sees 1 (1,200) and 4 (1,680, at exactly t + 15 min: the window is closed on the right)
    assert out["fw_apt15_n"].to_list() == [1.0, 1.0, 2.0, 0.0]
    assert out["fw_apt15_mean"].to_list() == pytest.approx([1680.0, 1200.0, 1440.0, None])
    assert out["fw_apt30_n"].to_list() == [1.0, 2.0, 2.0, 0.0]
    assert out["fw_rwy30_n"].to_list() == [0.0, 1.0, 1.0, 0.0]
    assert out["fw_rwy30_mean"].to_list() == pytest.approx([None, 1200.0, 1200.0, None])
    # L3
    # at 10:00 nobody has pushed; at 10:05 rows 1 and 4 are pushed and not airborne; at 10:02 row 1 is
    assert out["qp_apt_at_push"].to_list() == [0.0, 2.0, None, 1.0]
    assert out["qp_rwy_at_push"].to_list() == [0.0, 1.0, None, 0.0]
    # row 1 taxis 10:00-10:20: pushes at 10:02 and 10:05; row 4 taxis 10:02-10:30: push at 10:05
    assert out["qp_pushed_after_me"].to_list() == [2.0, 0.0, None, 1.0]
    # row 1 taxis 10:00-10:20 on R1: take-offs 10:10 and 10:15 (row 3 counts: a take-off needs no AOBT_3)
    assert out["qp_rwy_tko_in_taxi"].to_list() == [2.0, 0.0, None, 0.0]
    assert out["qp_rwy_push_pre20"].to_list() == [0.0, 1.0, None, 0.0]
    # arrivals taxiing: 09:55-10:03 and 10:01-10:12
    assert out["qp_arr_taxiing_at_push"].to_list() == [1.0, 1.0, None, 2.0]


def test_no_departure_block_time_or_label_is_read():
    raw = _raw()
    moved = raw.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(None)
        .otherwise(pl.col("BLOCK_TIME_UTC_mvt"))
        .alias("BLOCK_TIME_UTC_mvt")
    )
    assert _levers(raw).equals(_levers(moved))


def test_in_system_matches_brute_force_with_ties_and_negative_offsets():
    rng = np.random.default_rng(3)
    a = np.round(rng.uniform(0, 5000, 400))
    t = a + np.round(rng.normal(600, 400, 400))  # some t <= a
    a[::37] = np.nan
    q = np.concatenate([np.where(np.isfinite(a), a, 0.0), np.round(rng.uniform(0, 5000, 50))])
    got = _in_system(a, t, q)
    ok = np.isfinite(a) & np.isfinite(t)
    want = np.array([((a[ok] < x) & (x < t[ok])).sum() for x in q], dtype=float)
    # a row's own pair never satisfies a_i < a_i, so the brute force excludes self exactly like the definition
    assert np.array_equal(got, want)


def test_attach_refuses_a_second_attach_and_keeps_the_frame():
    raw = _raw()
    f = raw.filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "ADEP_mvt")
    a, _ = attach_levers(f, f, raw, raw, True, True, True)
    assert a.columns == f.columns + list(L1_COLS + L2_COLS + L3_COLS)
    assert a.select(f.columns).equals(f)
    with pytest.raises(ValueError, match="already has"):
        attach_levers(a, a, raw, raw, True, False, False)
    b, _ = attach_levers(f, f, raw, raw, False, True, False)
    assert b.columns == f.columns + list(L2_COLS)
    assert attach_levers(f, f, raw, raw, False, False, False)[0] is f


@pytest.fixture(scope="module")
def frames():
    raw = pl.concat([generate_month(2025, m, seed=m, deps_per_airport_day=6) for m in (1, 2, 3)], how="diagonal")
    # The generator files every plan at exactly two hours, which would make L1 a constant; spread it by a
    # hash of the id so the column has something for the model to split on.
    raw = raw.with_columns(
        (pl.col("ARVT_1_flt") + pl.duration(seconds=(pl.col("MVT_ID_mvt").hash(seed=5) % 1800).cast(pl.Int64))).alias(
            "ARVT_1_flt"
        )
    )
    feats = build_departure_features(raw)
    with_l, _ = attach_levers(feats, feats, raw, raw, True, True, True, l5=True)
    m = feats["MVT_TIME_UTC_mvt"].dt.month()
    return feats, with_l, m


def _cfg(**kw):
    return TrainConfig(num_rounds=30, early_stopping=5, n_seeds=1, **kw)


def test_the_synthetic_frame_has_lever_values(frames):
    _, with_l, _ = frames
    for c in [c for c in LEVER_COLS if c not in L6_COLS]:  # the fixture does not ask for L6
        assert with_l[c].is_not_null().sum() > 0, c


def test_with_every_flag_off_the_columns_change_nothing(frames):
    feats, with_l, m = frames
    cfg = _cfg()
    assert visible_numeric_columns(with_l, cfg) == visible_numeric_columns(feats, cfg)
    p0, i0 = predict_fold(feats.filter(m < 3), feats.filter(m == 3), cfg)
    p1, i1 = predict_fold(with_l.filter(m < 3), with_l.filter(m == 3), cfg)
    assert p0.equals(p1)
    assert i0 == i1 and "lever_cols" not in i0


def _unmatched_ids(feats, m):
    unm = feats.filter(m == 3).filter(pl.col(PROXY).is_null())["MVT_ID_mvt"]
    assert unm.len() > 0
    return pl.col("MVT_ID_mvt").is_in(unm.implode())


@pytest.mark.parametrize(
    ("flags", "cols"),
    [({"fe_plan_nm_taxi": True}, L1_COLS), ({"fe_fwd_nm": True}, L2_COLS), ({"fe_push_queue": True}, L3_COLS)],
)
def test_a_model_a_lever_reaches_model_a_only(frames, flags, cols):
    feats, with_l, m = frames
    on = _cfg(**flags)
    num = visible_numeric_columns(with_l, on)
    assert set(cols) <= set(num)
    assert not (set(L1_COLS) | set(L3_COLS)) & set(unmatched_columns(num))
    p0, _ = predict_fold(feats.filter(m < 3), feats.filter(m == 3), _cfg())
    p1, info = predict_fold(with_l.filter(m < 3), with_l.filter(m == 3), on)
    assert info["lever_cols"] == list(cols)
    sel = _unmatched_ids(feats, m)
    assert p0.filter(sel).equals(p1.filter(sel))  # model B did not move
    assert not p0.filter(~sel).equals(p1.filter(~sel))  # model A did
    with pytest.raises(ValueError, match="lever flag is on"):
        predict_fold(feats.filter(m < 3), feats.filter(m == 3), on)


def test_l2_into_model_b_only_leaves_model_a_byte_identical(frames):
    feats, with_l, m = frames
    on = _cfg(fe_fwd_nm_unmatched=True)
    num = visible_numeric_columns(with_l, on)
    assert set(L2_COLS) <= set(unmatched_columns(num))
    assert not (set(L1_COLS) | set(L3_COLS)) & set(
        unmatched_columns(
            visible_numeric_columns(with_l, _cfg(fe_plan_nm_taxi=True, fe_push_queue=True, fe_fwd_nm_unmatched=True))
        )
    )
    p0, _ = predict_fold(feats.filter(m < 3), feats.filter(m == 3), _cfg())
    p1, info = predict_fold(with_l.filter(m < 3), with_l.filter(m == 3), on)
    assert info["lever_cols"] == list(L2_COLS)
    sel = _unmatched_ids(feats, m)
    assert p0.filter(~sel).equals(p1.filter(~sel))  # model A did not move
    assert info["n_features_unmatched"] > 0
