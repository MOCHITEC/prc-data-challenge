"""A1: the arrival mirror of LIRF's schedule fill, as model-A columns (TrainConfig.fe_arr_fill).

Specification: docs/competitor_research_2026-09-21/results/lirf-lfpg-mechanisms.md §2 and §5 "A1 - arrival-mirror
fill state". At LIRF the recording process writes the schedule as the block time, on arrivals as well as on
departures, and the arrival record is not blanked in ranking.parquet. When the inbound arrival that
`stand_arrival_gap` links to was filled, the departure is 2.6x as likely to be filled (0.399 against 0.156).

The builder is `build_arrival_fill` of scripts/lirf_lfpg_mechanisms.py, moved here unchanged except that the join
keys are cast to strings (a categorical ADEP would otherwise refuse the as-of join) and only the four columns the
note names are attached.

Parity. Only ARRIVAL rows' BLOCK_TIME_UTC_mvt and SCHED_TIME_UTC_mvt are read. ranking.parquet keeps arrival
BLOCK on every scored arrival (the same source `stand_arrival_gap` already reads); no departure's BLOCK or label
is read. Training departures are linked to training arrivals, ranking departures to ranking arrivals.

This module is deliberately not `features.py` / `features_extra.py`: those files are hashed into the feature-cache
fingerprint, and a recipe without the flag must reuse its cache and build a byte-identical design matrix."""

from __future__ import annotations

import numpy as np
import polars as pl

FILL_TOL = 12  # seconds; LIRF clocks sit within +-6 s of a whole minute (lirf-lfpg-mechanisms.md [A])
LINK_WINDOW_S = 21_600  # the latest same-stand arrival in-block before take-off, within 6 h
OPDAY_SHRINK_K = 5
ARR_FILL_COLS = ("arr_link_fill", "arr_link_bs_abs", "arr_link_same_op", "arr_opday_fill")


def arrivals_of(raw: pl.DataFrame) -> pl.DataFrame:
    """ARR rows of a raw movement frame. BLOCK is the arrival IN-block; ADES is the airport it arrived at."""
    return (
        raw.lazy()
        .filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            pl.col("ADES_mvt").cast(pl.Utf8).alias("ADEP_mvt"),
            pl.col("STAND_mvt").cast(pl.Utf8),
            pl.col("AIRCRAFT_OPERATOR_flt").cast(pl.Utf8).alias("_a_op"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("_a_block"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("_a_bs"),
        )
        .with_columns(
            (pl.col("_a_bs").abs() <= FILL_TOL).cast(pl.Int8).alias("_a_fill"),
            pl.col("_a_block").dt.date().alias("_date"),
        )
        .collect()
    )


def build_arrival_fill(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Arrival-side schedule-fill state for each departure, one row per `dep` row in `dep`'s order.

    arr_link_fill     the same-stand arrival that stand_arrival_gap links to (latest in-block before this
                      departure's take-off, within 6 h) had its in-block written as its schedule (|bs| <= 12 s)
    arr_link_bs_abs   |in-block - sched| of that arrival, seconds (continuous version of the flag)
    arr_link_same_op  that arrival is the same operator (the rotation is plausibly the same aircraft)
    arr_opday_fill    fill rate over this operator's arrivals at this airport on this UTC date, shrunk
                      toward the airport-day rate with k = 5"""
    a = arr.drop_nulls(["ADEP_mvt", "_a_block"])
    day = a.group_by("ADEP_mvt", "_date").agg(pl.col("_a_fill").mean().alias("arr_day_fill"))
    opd = (
        a.group_by("ADEP_mvt", "_a_op", "_date")
        .agg(pl.col("_a_fill").sum().alias("_k"), pl.len().alias("_n"))
        .join(day, on=["ADEP_mvt", "_date"], how="left")
        .with_columns(
            ((pl.col("_k") + OPDAY_SHRINK_K * pl.col("arr_day_fill")) / (pl.col("_n") + OPDAY_SHRINK_K)).alias(
                "arr_opday_fill"
            )
        )
        .select("ADEP_mvt", pl.col("_a_op").alias("AIRCRAFT_OPERATOR_flt"), "_date", "arr_opday_fill")
    )
    link_src = (
        a.drop_nulls("STAND_mvt")
        .sort("_a_block")
        .select("ADEP_mvt", "STAND_mvt", "_a_block", "_a_bs", "_a_fill", "_a_op")
    )
    base = dep.select(
        "MVT_ID_mvt",
        pl.col("ADEP_mvt").cast(pl.Utf8),
        pl.col("STAND_mvt").cast(pl.Utf8),
        pl.col("AIRCRAFT_OPERATOR_flt").cast(pl.Utf8),
        "MVT_TIME_UTC_mvt",
    ).with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_date"))
    linked = (
        base.drop_nulls("MVT_TIME_UTC_mvt")
        .sort("MVT_TIME_UTC_mvt")
        .join_asof(
            link_src,
            left_on="MVT_TIME_UTC_mvt",
            right_on="_a_block",
            by=["ADEP_mvt", "STAND_mvt"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(
            ((pl.col("MVT_TIME_UTC_mvt") - pl.col("_a_block")).dt.total_seconds() <= LINK_WINDOW_S).alias("_ok")
        )
        .select(
            "MVT_ID_mvt",
            pl.when("_ok").then(pl.col("_a_fill")).alias("arr_link_fill"),
            pl.when("_ok").then(pl.col("_a_bs").abs().cast(pl.Float64)).alias("arr_link_bs_abs"),
            pl.when("_ok")
            .then((pl.col("_a_op") == pl.col("AIRCRAFT_OPERATOR_flt")).cast(pl.Int8))
            .alias("arr_link_same_op"),
        )
    )
    out = (
        base.join(linked, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(day, on=["ADEP_mvt", "_date"], how="left", maintain_order="left")
        .join(opd, on=["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "_date"], how="left", maintain_order="left")
        .with_columns(pl.col("arr_opday_fill").fill_null(pl.col("arr_day_fill")))
        .select("MVT_ID_mvt", *ARR_FILL_COLS)
    )
    if out.height != dep.height or not out["MVT_ID_mvt"].equals(dep["MVT_ID_mvt"]):
        raise AssertionError("build_arrival_fill changed the row count or order")
    return out


PERMUTE_SEED = 11  # the seed scripts/lirf_lfpg_mechanisms.py used for its permuted controls


def permute_within_month(cols: pl.DataFrame, when: pl.Series, seed: int = PERMUTE_SEED) -> pl.DataFrame:
    """Each column shuffled independently within calendar month (UTC year-month of `when`): the note's
    arity-matched control, which keeps every column's per-month distribution and breaks its link to the row."""
    rng = np.random.default_rng(seed)
    ym = (when.dt.year().cast(pl.Int32) * 100 + when.dt.month().cast(pl.Int32)).fill_null(0).to_numpy()
    groups = [np.flatnonzero(ym == m) for m in np.unique(ym)]
    out = []
    for c in cols.columns:
        s = cols[c]
        idx = np.arange(len(s))
        for ix in groups:
            idx[ix] = ix[rng.permutation(len(ix))]
        out.append(s.gather(idx))
    return pl.DataFrame(out)


def attach_arrival_fill(feats: pl.DataFrame, raw: pl.DataFrame, permute: bool = False) -> pl.DataFrame:
    """`feats` (departure feature frame) with the four A1 columns appended, from `raw`'s arrival rows.
    `permute` (TrainConfig.fe_arr_fill_permute) shuffles each of them within calendar month."""
    if feats["MVT_ID_mvt"].n_unique() != feats.height:
        raise AssertionError("MVT_ID_mvt is not unique in the feature frame")
    clash = [c for c in ARR_FILL_COLS if c in feats.columns]
    if clash:
        raise ValueError(f"the feature frame already has {clash}")
    f = build_arrival_fill(feats, arrivals_of(raw)).drop("MVT_ID_mvt")
    if permute:
        f = permute_within_month(f, feats["MVT_TIME_UTC_mvt"])
    return feats.hstack(f)
