"""OSM stand geometry (docs/osm_stand_2026-09-28/plan.md)."""

import numpy as np
import polars as pl

from prc.osm import add_stand_runway_distance, normalise_stand, stand_table
from prc.train import TrainConfig, unseen_stand_mask


def test_stand_refs_normalise_across_spacing_case_and_leading_zeros():
    assert normalise_stand("b 41") == "B41"
    assert normalise_stand("01") == "1"
    assert normalise_stand("A01") == "A1"
    assert normalise_stand("237L") == "237L"
    assert normalise_stand("100") == "100"  # zeros inside a number are kept
    assert normalise_stand(None) is None


def test_a_multi_stand_ref_is_split_and_a_stand_drawn_twice_is_averaged():
    osm = {
        "EDDF": {
            "elements": [
                {"type": "node", "lat": 50.0, "lon": 8.0, "tags": {"aeroway": "parking_position", "ref": "2 2A"}},
                {"type": "node", "lat": 50.2, "lon": 8.2, "tags": {"aeroway": "parking_position", "ref": "A1"}},
                {
                    "type": "way",
                    "center": {"lat": 50.4, "lon": 8.4},
                    "tags": {"aeroway": "parking_position", "ref": "A1"},
                },
                {"type": "node", "lat": 50.9, "lon": 8.9, "tags": {"aeroway": "parking_position"}},  # no ref: skipped
                {"type": "way", "center": {"lat": 50.5, "lon": 8.5}, "tags": {"aeroway": "runway", "ref": "07C/25C"}},
            ]
        }
    }
    t = stand_table(osm).sort("stand")
    assert t["stand"].to_list() == ["2", "2A", "A1"]
    a1 = t.filter(pl.col("stand") == "A1")
    assert np.isclose(a1["lat"][0], 50.3) and a1["n_elem"][0] == 2


def test_distance_is_to_the_named_runway_end_and_null_when_either_side_is_unknown():
    stands = pl.DataFrame({"icao": ["EDDF"], "stand": ["A1"], "lat": [50.0], "lon": [8.0]})
    ends = pl.DataFrame({"icao": ["EDDF", "EDDF"], "runway": ["25C", "07C"], "rlat": [50.0, 50.0], "rlon": [8.1, 8.5]})
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1, 2, 3, 4],
            "ADEP_mvt": ["EDDF"] * 4,
            "STAND_mvt": ["A01", "A1", "Z9", "A1"],
            "RUNWAY_mvt": ["25C", "07C", "25C", "99"],
        }
    )
    out = add_stand_runway_distance(dep, stands, ends)
    km = out["osm_stand_rwy_km"].to_list()
    assert 7.0 < km[0] < 7.3 and 35.5 < km[1] < 36.2  # 0.1 and 0.5 degrees of longitude at 50 N
    assert km[2] is None and km[3] is None
    assert out["MVT_ID_mvt"].to_list() == [1, 2, 3, 4]


def test_the_unseen_stand_draw_is_stable_and_about_the_requested_share():
    df = pl.DataFrame({"ADEP_mvt": ["EDDF"] * 2000, "STAND_mvt": [f"S{i}" for i in range(2000)]})
    m1 = unseen_stand_mask(df, 0.2)
    m2 = unseen_stand_mask(df.reverse(), 0.2)[::-1]
    assert np.array_equal(m1, m2)
    assert 0.17 < m1.mean() < 0.23
    # every row of a held-out stand goes, not a random subset of its rows
    rep = pl.DataFrame({"ADEP_mvt": ["EDDF"] * 6, "STAND_mvt": ["S1"] * 3 + ["S2"] * 3})
    m = unseen_stand_mask(rep, 0.5)
    assert len(set(m[:3])) == 1 and len(set(m[3:])) == 1


def test_osm_distance_and_the_simulation_are_off_by_default():
    cfg = TrainConfig()
    assert cfg.fe_osm_dist is False and cfg.sim_unseen_stands == 0.0
