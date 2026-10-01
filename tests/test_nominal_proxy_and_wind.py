"""C1 (`proxy_is_nominal`) and C3 (wind direction relative to the assigned runway).

Both candidates come from docs/competitor_research_2026-09-21/results/delay-mechanisms.md §2.3, §2.4 and §5.

Every test here is written against a named mutant, because this repository has shipped three silent no-ops
and the most recent one passed every hand-written test: the code computed the corrected values and then
handed the model the uncorrected ones. The mutants these tests exist to kill, one per line:

C1
  * the mode table fitted per frame instead of on the union : test_the_mode_table_is_fitted_on_the_union
  * the table keyed on ADEP_mvt alone, not ADEP x RUNWAY    : test_the_mode_is_per_runway_not_per_airport
  * `_pm == mode` flipped, or the raw minute handed over    : test_the_three_columns_are_what_they_say
  * the columns computed and then not joined / not shown    : test_the_columns_reach_the_model_when_asked
  * `build_feature_frames` fitting one table per frame      : test_the_pair_builder_scores_both_frames_on_one_table
  * a per-frame table invented when none was handed in      : test_without_a_table_the_columns_are_null
  * a label read into the table or the columns              : test_c1_is_blind_to_the_blanked_label_columns

C3
  * the designator read as degrees rather than tens         : test_runway_heading_parses_the_designator
  * cos and sin swapped, or the headwind sign flipped       : test_wind_is_resolved_onto_the_assigned_runway
  * a null bearing silently becoming a zero angle           : test_a_missing_bearing_stays_missing
  * the raw bearing left in the frame as a model column     : test_the_raw_bearing_never_reaches_the_model
  * the flags defaulting to on, changing shipped recipes    : test_the_new_weather_columns_are_off_by_default
                                                              test_a_recipe_that_says_nothing_gets_the_old_frame
  * `gust_excess` left equal to `gust_filled`               : test_the_gust_repair_separates_calm_from_unseen
  * two recipes sharing one weather cache file              : test_every_weather_flag_is_in_the_cache_key
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from prc.features import (
    PROXY_MODE_COLS,
    build_departure_features,
    build_feature_frames,
    mask_departures,
    proxy_mode_table,
    proxy_nominal_features,
)
from prc.pipeline import cache_fingerprint, weather_cache_path, weather_flags
from prc.synth import generate_month
from prc.train import TrainConfig, visible_numeric_columns
from prc.weather import (
    hourly_weather_table,
    join_weather,
    parse_metar_csv,
    runway_heading_deg,
    wind_relative_columns,
)

# ---------------------------------------------------------------- C1


def _movements(rows: list[tuple[str, str, int, int]]) -> pl.DataFrame:
    """(airport, runway, offset minutes, n) -> a minimal movement frame with that offset distribution."""
    out = []
    t = dt.datetime(2025, 3, 1, 6, 0)
    for airport, runway, minutes, n in rows:
        for i in range(n):
            mvt = t + dt.timedelta(minutes=i)
            out.append(
                {
                    "MVT_ID_mvt": len(out) + 1,
                    "PHASE_mvt": "DEP",
                    "ADEP_mvt": airport,
                    "RUNWAY_mvt": runway,
                    "MVT_TIME_UTC_mvt": mvt,
                    "AOBT_3_flt": mvt - dt.timedelta(minutes=minutes),
                }
            )
    return pl.DataFrame(out)


def _rows(frame: pl.DataFrame) -> pl.DataFrame:
    """The row-side view `proxy_nominal_features` expects: ids, keys and the offset in seconds."""
    return frame.select(
        "MVT_ID_mvt",
        "ADEP_mvt",
        "RUNWAY_mvt",
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("mvt_minus_aobt3"),
    )


def test_the_mode_table_is_fitted_on_the_union():
    """The requirement the note states explicitly, and the one that turns a feature into a fold artefact.

    Training is dominated by the 10-minute allowance; the ranking month has more 20-minute rows than
    10-minute ones. Fitted per frame, the ranking rows would be scored against 20 and every 20-minute row
    would be flagged nominal - a column that means one thing in training and another in serving."""
    train = _movements([("LTFM", "36", 10, 100), ("LTFM", "36", 20, 5)])
    rank = _movements([("LTFM", "36", 20, 30), ("LTFM", "36", 10, 12)])

    union = proxy_mode_table([train, rank])
    assert union["_proxy_mode_min"].to_list() == [10.0]
    rank_only = proxy_mode_table([rank])  # what a per-frame fit would have produced
    assert rank_only["_proxy_mode_min"].to_list() == [20.0]

    got = proxy_nominal_features(_rows(rank), union)
    flag = got.join(_rows(rank), on="MVT_ID_mvt")
    at_mode = flag.filter(pl.col("proxy_at_mode") == 1)
    assert at_mode.height == 12  # the 10-minute rows, which is the union's mode
    assert set(at_mode["mvt_minus_aobt3"].to_list()) == {600.0}


def test_the_mode_is_per_runway_not_per_airport():
    """LTFM's two runway groups carry two different allowances (§2.3: 18-20 min off 36/35L, 23-25 off 18/17R).
    A table keyed on the airport alone would flag one group and miss the other."""
    frame = _movements([("LTFM", "36", 19, 60), ("LTFM", "17R", 24, 40)])
    table = proxy_mode_table([frame])
    got = proxy_nominal_features(_rows(frame), table).join(frame.select("MVT_ID_mvt", "RUNWAY_mvt"), on="MVT_ID_mvt")
    assert got.filter(pl.col("proxy_at_mode") == 1).height == 100  # every row is at its own runway's mode
    by_rwy = dict(table.select("RUNWAY_mvt", "_proxy_mode_min").iter_rows())
    assert by_rwy == {"36": 19.0, "17R": 24.0}


def test_the_three_columns_are_what_they_say():
    """Exact values, so a flipped comparison or a raw minute handed over in place of the flag fails here."""
    frame = _movements([("LEBL", "25R", 20, 50), ("LEBL", "25R", 22, 3)])
    # one row with a seconds part: 20 min 37 s, which is not on the whole minute
    frame = pl.concat(
        [
            frame,
            pl.DataFrame(
                [
                    {
                        "MVT_ID_mvt": 999,
                        "PHASE_mvt": "DEP",
                        "ADEP_mvt": "LEBL",
                        "RUNWAY_mvt": "25R",
                        "MVT_TIME_UTC_mvt": dt.datetime(2025, 3, 2, 9, 0, 37),
                        "AOBT_3_flt": dt.datetime(2025, 3, 2, 8, 40, 0),
                    }
                ]
            ),
        ]
    )
    got = proxy_nominal_features(_rows(frame), proxy_mode_table([frame])).sort("MVT_ID_mvt")
    first = got.row(0, named=True)
    assert first == {"MVT_ID_mvt": 1, "proxy_at_mode": 1, "proxy_minus_mode_min": 0.0, "proxy_sec_in_min": 0.0}
    late = got.filter(pl.col("MVT_ID_mvt") == 999).row(0, named=True)
    assert late["proxy_at_mode"] == 1 and late["proxy_sec_in_min"] == 37.0
    off = got.filter(pl.col("MVT_ID_mvt") > 50).filter(pl.col("MVT_ID_mvt") < 999)
    assert off["proxy_at_mode"].to_list() == [0, 0, 0]
    assert off["proxy_minus_mode_min"].to_list() == [2.0, 2.0, 2.0]


def test_without_a_table_the_columns_are_null():
    """No table, no column. The alternative - inventing one from whichever frame is in hand - is the defect."""
    df = generate_month(2025, 7, seed=101, deps_per_airport_day=3)
    feats = build_departure_features(df)
    for c in PROXY_MODE_COLS:
        assert feats[c].null_count() == feats.height


def test_proxy_mode_table_refuses_a_single_frame():
    df = generate_month(2025, 7, seed=102, deps_per_airport_day=2)
    with pytest.raises(TypeError, match="sequence of frames"):
        proxy_mode_table(df)


def test_the_columns_reach_the_model_when_asked():
    """The P8 trap in its general form: values computed correctly and then not handed to the model.

    Both directions are asserted - hidden by default like every other candidate column, and actually in the
    list `run_cv` builds its matrix from when the recipe names them."""
    train = generate_month(2025, 7, seed=103, deps_per_airport_day=3)
    rank = generate_month(2026, 7, seed=104, deps_per_airport_day=2, blank_dep=True)
    f_tr, f_rk = build_feature_frames(train, rank)
    for f in (f_tr, f_rk):
        for c in PROXY_MODE_COLS:
            assert c in f.columns
            assert f[c].null_count() < f.height  # real values, not a column of nulls
    assert not set(PROXY_MODE_COLS) & set(visible_numeric_columns(f_tr, TrainConfig()))
    shown = visible_numeric_columns(f_tr, TrainConfig(include_candidates=PROXY_MODE_COLS))
    assert set(PROXY_MODE_COLS) <= set(shown)


def _forced_offset(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Every departure's AOBT_3 set to take-off minus `minutes` - a whole frame on one nominal allowance."""
    return df.with_columns(
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=minutes))
        .otherwise(pl.col("AOBT_3_flt"))
        .alias("AOBT_3_flt")
    )


def test_the_pair_builder_scores_both_frames_on_one_table():
    """The same requirement as `test_the_mode_table_is_fitted_on_the_union`, asserted where it is actually
    used. The first test proves the fitting function can do it; this one proves the pair builder does.

    Training runs on a 10-minute allowance, the (much smaller) ranking month on a 20-minute one. The union's
    mode is 10, so no ranking row is at the mode and every one of them is 10 minutes away from it. Fitted per
    frame, every ranking row would be flagged nominal and sit at distance 0 - the same column meaning two
    different things on the two sides of the join."""
    train = _forced_offset(generate_month(2025, 7, seed=108, deps_per_airport_day=6), 10)
    rank = _forced_offset(generate_month(2026, 7, seed=109, deps_per_airport_day=2, blank_dep=True), 20)
    f_tr, f_rk = build_feature_frames(train, rank)
    assert f_tr["proxy_at_mode"].sum() == f_tr.height  # training sets the mode, so training is at it
    assert f_rk["proxy_at_mode"].sum() == 0
    assert f_rk["proxy_minus_mode_min"].unique().to_list() == [10.0]


def test_c1_is_blind_to_the_blanked_label_columns():
    """Parity: BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are blank on every scored row, so a column that moves
    when they are blanked would be unbuildable at serve time - and, here, a label in disguise."""
    train = generate_month(2025, 7, seed=105, deps_per_airport_day=4)
    blanked = mask_departures(train)
    assert proxy_mode_table([train, blanked]).equals(proxy_mode_table([blanked, blanked]))
    table = proxy_mode_table([train, blanked])
    a = build_departure_features(train, proxy_mode=table).select(["MVT_ID_mvt", *PROXY_MODE_COLS]).sort("MVT_ID_mvt")
    b = build_departure_features(blanked, proxy_mode=table).select(["MVT_ID_mvt", *PROXY_MODE_COLS]).sort("MVT_ID_mvt")
    assert a.equals(b)


# ---------------------------------------------------------------- C3

SAMPLE = """station,valid,tmpf,dwpf,relh,drct,sknt,gust,p01i,vsby,wxcodes,skyc1,feel
EHAM,2026-01-06 05:55,30.20,28.40,92.85,90.00,10.00,25.00,0.01,0.62,SN BR,OVC,20.10
LEBL,2026-07-15 12:00,86.00,60.80,44.00,90.00,10.00,M,0.00,9.94,M,FEW,88.00
"""


def _wind_frame(runways: list[str], drct: float | None = 90.0, wind_ms: float = 10.0 * 0.514444) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "MVT_ID_mvt": list(range(len(runways))),
            "RUNWAY_mvt": runways,
            "drct": [drct] * len(runways),
            "wind_ms": [wind_ms] * len(runways),
        }
    )


def test_runway_heading_parses_the_designator():
    got = pl.DataFrame({"RUNWAY_mvt": ["27L", "09", "36", "18C", "XX", "00", "37", None]}).select(runway_heading_deg())[
        "_rwy_heading"
    ]
    assert got.to_list() == [270.0, 90.0, 360.0, 180.0, None, None, None, None]


def test_wind_is_resolved_onto_the_assigned_runway():
    """A 10 kt wind from 090. Down runway 09 it is 10 kt of headwind and no crosswind; across runway 18 it is
    all crosswind; down runway 27 it is a 10 kt tailwind and the misalignment is at its maximum of 2."""
    got = wind_relative_columns(_wind_frame(["09", "18", "27"])).sort("MVT_ID_mvt")
    assert [round(v, 6) for v in got["headwind_kt"].to_list()] == [10.0, 0.0, -10.0]
    assert [round(v, 6) for v in got["crosswind_kt"].to_list()] == [0.0, 10.0, 0.0]
    assert [round(v, 6) for v in got["rwy_wind_misalign"].to_list()] == [0.0, 1.0, 2.0]
    assert got["wind_dir_deg"].to_list() == [90.0, 90.0, 90.0]


def test_a_missing_bearing_stays_missing():
    """`drct` is null on 10.0% of cached observations and 30.3% at LSZH. A null angle must stay null: LightGBM
    turns a serve-time NaN into 0.0 on a split whose training side had none, and 0 is a legal heading."""
    got = wind_relative_columns(_wind_frame(["09"], drct=None))
    for c in ("wind_dir_deg", "headwind_kt", "crosswind_kt", "rwy_wind_misalign"):
        assert got[c].null_count() == 1
    unparsed = wind_relative_columns(_wind_frame(["GRASS"]))
    assert unparsed["headwind_kt"].null_count() == 1 and unparsed["rwy_wind_misalign"].null_count() == 1


def test_the_raw_bearing_never_reaches_the_model():
    hourly = hourly_weather_table(parse_metar_csv(SAMPLE), wind_dir=True)
    dep = pl.DataFrame(
        {
            "MVT_ID_mvt": [1],
            "ADEP_mvt": ["LEBL"],
            "RUNWAY_mvt": ["09"],
            "MVT_TIME_UTC_mvt": [dt.datetime(2026, 7, 15, 12, 30)],
        },
        schema_overrides={"MVT_TIME_UTC_mvt": pl.Datetime("us")},
    )
    j = join_weather(dep, hourly)
    assert "drct" not in j.columns and "drct_prev1h" not in j.columns
    assert round(j["headwind_kt"].item(), 6) == 10.0


def test_the_new_weather_columns_are_off_by_default():
    """Every archived recipe has to rebuild to the frame it rebuilt before."""
    default = hourly_weather_table(parse_metar_csv(SAMPLE))
    assert "drct" not in default.columns
    assert not {"gust_filled", "gust_excess"} & set(default.columns)


def test_the_gust_repair_separates_calm_from_unseen():
    """METAR omits the gust group when there is no gust, so a null `gust_ms` means 'steady', not 'unknown'.
    The repair adds the disambiguating pair and leaves `gust_ms` itself alone."""
    hourly = hourly_weather_table(parse_metar_csv(SAMPLE), gust_repair=True).sort("station")
    eham = hourly.filter(pl.col("station") == "EHAM").row(0, named=True)
    lebl = hourly.filter(pl.col("station") == "LEBL").row(0, named=True)
    assert lebl["gust_ms"] is None  # no gust group in the report
    assert round(lebl["gust_filled"], 4) == round(lebl["wind_ms"], 4)
    assert lebl["gust_excess"] == 0.0
    assert eham["gust_excess"] > 0.0 and eham["gust_filled"] > eham["wind_ms"]


def test_a_recipe_that_says_nothing_gets_the_old_frame():
    """The flags as `main` reads them. A default that flips is a change to every recipe in the repository,
    and the recipes do not mention these keys at all."""
    assert weather_flags({}) == (False, False, False, False)
    assert weather_flags({"use_weather": True}) == (True, False, False, False)
    assert weather_flags({"use_weather": True, "wind_dir": True}) == (True, False, True, False)
    assert weather_flags({"use_weather": True, "wx_codes": True, "gust_repair": True}) == (True, True, False, True)


def test_every_weather_flag_is_in_the_cache_key(tmp_path):
    """One cache file per weather recipe, and one fingerprint per weather recipe. Sharing either is how a
    frame built without a column is handed to a run that asked for it."""
    paths = {weather_cache_path(wx, wd, gr) for wx in (False, True) for wd in (False, True) for gr in (False, True)}
    assert len(paths) == 8
    assert weather_cache_path(False).name == "metar_hourly.parquet"  # the two that already exist, unchanged
    assert weather_cache_path(True).name == "metar_hourly_wxcodes.parquet"

    train = generate_month(2025, 7, seed=106, deps_per_airport_day=2)
    rank = generate_month(2026, 7, seed=107, deps_per_airport_day=2, blank_dep=True)
    base = cache_fingerprint(train, rank, tmp_path, True)
    assert cache_fingerprint(train, rank, tmp_path, True, wind_dir=True) != base
    assert cache_fingerprint(train, rank, tmp_path, True, gust_repair=True) != base
    # ... and neither flag disturbs a recipe with no weather join to vary
    off = cache_fingerprint(train, rank, tmp_path, False)
    assert cache_fingerprint(train, rank, tmp_path, False, wind_dir=True, gust_repair=True) == off
