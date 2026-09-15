"""Local (polars/numpy) feature builder, v1.

Design rules (docs/README-§5, §7):
* Feature-availability parity: context features are computed from a *masked* view in which every departure's
  BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are null, for training and ranking alike. Only columns that survive
  in ranking.parquet are used for context (take-off times, arrivals, NM times, schedule).
* Windows are anchored on the known take-off time and exclude the row itself (half-open [t-W, t)).
* Label-dependent statistics (stand x runway reference, target encodings) are NOT built here; they are computed
  per fold in train.py from the training side only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .airports import add_destination_distance
from .schema import BLANKED_DEP_COLS

AIRPORT_TZ = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}
WINDOWS_MIN = (5, 10, 15, 30, 60)
CATEGORICALS = [
    "ADEP_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "ADES_mvt",
    "ADES_REGION",
]


def mask_departures(df: pl.DataFrame) -> pl.DataFrame:
    """Return the parity view: departures lose block time and taxi time (exactly like ranking.parquet)."""
    return df.with_columns(
        [pl.when(pl.col("PHASE_mvt") == "DEP").then(None).otherwise(pl.col(c)).alias(c) for c in BLANKED_DEP_COLS]
    )


def _sec(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def clock_and_static_features(dep: pl.DataFrame) -> pl.DataFrame:
    """Row-local features for departures (no other rows involved)."""
    out = dep.with_columns(
        [
            _sec("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_minus_aobt3"),
            _sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_minus_sched"),
            _sec("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_minus_eobt1"),
            _sec("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_minus_lobt"),
            _sec("MVT_TIME_UTC_mvt", "IOBT_flt").alias("mvt_minus_iobt"),
            _sec("LOBT_flt", "EOBT_1_flt").alias("lobt_minus_eobt1"),
            _sec("AOBT_3_flt", "EOBT_1_flt").alias("aobt3_minus_eobt1"),
            _sec("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt3_minus_sched"),
            pl.col("FLIGHT_ID_mvt").is_not_null().cast(pl.Int8).alias("nm_matched"),
            pl.col("AOBT_3_flt").is_null().cast(pl.Int8).alias("aobt3_null"),
            pl.col("LOBT_flt").is_null().cast(pl.Int8).alias("lobt_null"),
            pl.col("EOBT_1_flt").is_null().cast(pl.Int8).alias("eobt1_null"),
            pl.col("STAND_mvt").is_null().cast(pl.Int8).alias("stand_null"),
            pl.col("RUNWAY_mvt").is_null().cast(pl.Int8).alias("runway_null"),
            (pl.col("ADES_flt") != pl.col("ADES_FILED_flt")).fill_null(False).cast(pl.Int8).alias("diverted"),
            (pl.col("AIRCRAFT_TYPE_flt") != pl.col("AIRCRAFT_TYPE_mvt"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("actype_mismatch"),
            pl.col("ADES_mvt").str.slice(0, 2).alias("ADES_REGION"),
            pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour_utc"),
            (pl.col("MVT_TIME_UTC_mvt").dt.hour() * 60 + pl.col("MVT_TIME_UTC_mvt").dt.minute()).alias(
                "minute_of_day_utc"
            ),
        ]
    )
    # airport-local hour via IANA zones (docs/README: never rely on session time zone)
    parts = []
    for apt, tz in AIRPORT_TZ.items():
        p = out.filter(pl.col("ADEP_mvt") == apt)
        if p.height:
            local = p["MVT_TIME_UTC_mvt"].dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
            parts.append(p.with_columns(local.dt.hour().alias("hour_local")))
    rest = out.filter(~pl.col("ADEP_mvt").is_in(list(AIRPORT_TZ)).fill_null(False))
    if rest.height:
        parts.append(rest.with_columns(pl.col("hour_utc").alias("hour_local")))
    return pl.concat(parts) if parts else out.with_columns(pl.col("hour_utc").alias("hour_local"))


def _counts_in_windows(anchor: np.ndarray, events: np.ndarray, w_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor time t, count events in [t-w, t) (strictly before) and (t, t+w] (strictly after)."""
    events = np.sort(events)
    before = np.searchsorted(events, anchor, side="left") - np.searchsorted(events, anchor - w_sec, side="left")
    after = np.searchsorted(events, anchor + w_sec, side="right") - np.searchsorted(events, anchor, side="right")
    return before.astype(np.int32), after.astype(np.int32)


def _window_mean(
    anchor: np.ndarray, times: np.ndarray, values: np.ndarray, w_sec: int, lag_sec: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of `values` whose time is in [t-w-lag, t-lag) for each anchor t. NaN values are ignored."""
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    ok = ~np.isnan(values)
    t_ok, v_ok = times[ok], values[ok]
    csum = np.concatenate([[0.0], np.cumsum(v_ok)])
    hi = np.searchsorted(t_ok, anchor - lag_sec, side="left")
    lo = np.searchsorted(t_ok, anchor - lag_sec - w_sec, side="left")
    n = (hi - lo).astype(np.float64)
    s = csum[hi] - csum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
    return mean, n


def _headways(anchor: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seconds to the previous and next event (other rows) on the same runway/airport."""
    events = np.sort(events)
    i_prev = np.searchsorted(events, anchor, side="left") - 1
    i_next = np.searchsorted(events, anchor, side="right")
    prev = np.where(i_prev >= 0, anchor - events[np.clip(i_prev, 0, len(events) - 1)], np.nan)
    nxt = np.where(i_next < len(events), events[np.clip(i_next, 0, len(events) - 1)] - anchor, np.nan)
    return prev.astype(np.float64), nxt.astype(np.float64)


def context_features(masked: pl.DataFrame, dep_ids: pl.Series | None = None) -> pl.DataFrame:
    """Take-off-anchored congestion/context features for departures, computed from the masked view.

    Returns one row per departure MVT_ID with numeric feature columns."""
    m = masked.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t"))
    dep_all = m.filter(pl.col("PHASE_mvt") == "DEP")
    arr_all = m.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t_inblock")
    )
    if dep_ids is not None:
        dep_target = dep_all.filter(pl.col("MVT_ID_mvt").is_in(dep_ids.implode()))
    else:
        dep_target = dep_all
    frames = []
    for apt in dep_target["ADEP_mvt"].unique().to_list():
        # a null airport is its own group: `== None` yields null and would drop those rows silently
        dep_pred = pl.col("ADEP_mvt").is_null() if apt is None else (pl.col("ADEP_mvt") == apt)
        arr_pred = pl.col("ADES_mvt").is_null() if apt is None else (pl.col("ADES_mvt") == apt)
        d_apt = dep_all.filter(dep_pred)
        tgt = dep_target.filter(dep_pred)
        a_apt = arr_all.filter(arr_pred)
        t = tgt["_t"].to_numpy()
        anchor_ok = ~np.isnan(t)
        cols: dict[str, np.ndarray] = {"MVT_ID_mvt": tgt["MVT_ID_mvt"].to_numpy()}
        dep_times = d_apt["_t"].to_numpy()
        arr_land = a_apt["_t"].to_numpy()
        arr_inblock = a_apt["_t_inblock"].to_numpy()
        for w in WINDOWS_MIN:
            b, a = _counts_in_windows(t, dep_times, w * 60)
            cols[f"dep_tko_prev_{w}m"], cols[f"dep_tko_next_{w}m"] = b, a
            b, a = _counts_in_windows(t, arr_land, w * 60)
            cols[f"arr_land_prev_{w}m"], cols[f"arr_land_next_{w}m"] = b, a
            b, _ = _counts_in_windows(t, arr_inblock[~np.isnan(arr_inblock)], w * 60)
            cols[f"arr_inblock_prev_{w}m"] = b
        # same-runway take-offs and headways
        rwy_prev = np.full(len(t), np.nan)
        rwy_next = np.full(len(t), np.nan)
        rwy_cnt15 = np.zeros(len(t), dtype=np.int32)
        rwy_cnt60 = np.zeros(len(t), dtype=np.int32)
        for rwy in tgt["RUNWAY_mvt"].unique().to_list():
            sel = (
                (tgt["RUNWAY_mvt"] == rwy).fill_null(False).to_numpy()
                if rwy is not None
                else tgt["RUNWAY_mvt"].is_null().to_numpy()
            )
            ev = d_apt.filter(pl.col("RUNWAY_mvt") == rwy)["_t"].to_numpy() if rwy is not None else np.array([])
            if len(ev) == 0:
                continue
            p, n = _headways(t[sel], ev)
            rwy_prev[sel], rwy_next[sel] = p, n
            rwy_cnt15[sel] = _counts_in_windows(t[sel], ev, 15 * 60)[0]
            rwy_cnt60[sel] = _counts_in_windows(t[sel], ev, 60 * 60)[0]
        cols["rwy_headway_prev_s"], cols["rwy_headway_next_s"] = rwy_prev, rwy_next
        cols["anchor_ok"] = anchor_ok.astype(np.int8)
        # how much of the widest look-back window is actually inside the frame: counts near a frame edge are
        # structurally lower, and a hidden final set may cover only part of a month (docs/README)
        span_lo = np.nanmin(dep_times) if len(dep_times) else np.nan
        cols["window_coverage_60m"] = np.clip((t - span_lo) / 3600.0, 0.0, 1.0)
        cols["rwy_tko_prev_15m"], cols["rwy_tko_prev_60m"] = rwy_cnt15, rwy_cnt60
        # neighbour state from NM times of other departures (both survive in ranking); lag 60 s to avoid ties
        aobt_delay = (d_apt["AOBT_3_flt"] - d_apt["EOBT_1_flt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
        mvt_minus_aobt = (
            (d_apt["MVT_TIME_UTC_mvt"] - d_apt["AOBT_3_flt"]).dt.total_seconds().cast(pl.Float64).to_numpy()
        )
        for w in (30, 60):
            mean, n = _window_mean(t, dep_times, aobt_delay, w * 60, lag_sec=60)
            cols[f"nbr_aobt_delay_mean_{w}m"] = mean
            mean2, _ = _window_mean(t, dep_times, mvt_minus_aobt, w * 60, lag_sec=60)
            cols[f"nbr_mvt_minus_aobt_mean_{w}m"] = mean2
            cols[f"nbr_dep_n_{w}m"] = n
        # arrival taxi-in state (labels of arrivals are visible in ranking)
        taxi_in = a_apt["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
        for w in (30, 60):
            mean, n = _window_mean(t, arr_inblock, taxi_in, w * 60, lag_sec=0)
            cols[f"arr_taxi_in_mean_{w}m"] = mean
            cols[f"arr_taxi_in_n_{w}m"] = n
        frame = pl.DataFrame(cols)
        if (~anchor_ok).any():
            ctx_cols = [c for c in frame.columns if c not in ("MVT_ID_mvt", "anchor_ok")]
            frame = frame.with_columns(
                [pl.when(pl.col("anchor_ok") == 1).then(pl.col(c)).otherwise(None).alias(c) for c in ctx_cols]
            )
        frames.append(frame)
    if not frames:
        return pl.DataFrame({"MVT_ID_mvt": pl.Series([], dtype=pl.Int64)})
    ctx = pl.concat(frames, how="diagonal")
    ctx = ctx.with_columns(pl.col("anchor_ok").cast(pl.Int8))
    float_cols = [c for c in ctx.columns if c.startswith(("rwy_headway", "nbr_aobt", "nbr_mvt", "arr_taxi_in_mean"))]
    count_cols = [c for c in ctx.columns if c != "MVT_ID_mvt" and c not in float_cols]
    return ctx.with_columns(
        [pl.col(c).cast(pl.Float32) for c in float_cols] + [pl.col(c).cast(pl.Int32) for c in count_cols]
    )


WIDE_RANGE_COLS = (
    "mvt_minus_aobt3",
    "mvt_minus_sched",
    "mvt_minus_eobt1",
    "mvt_minus_lobt",
    "mvt_minus_iobt",
    "lobt_minus_eobt1",
    "aobt3_minus_eobt1",
    "aobt3_minus_sched",
    "rwy_headway_prev_s",
    "rwy_headway_next_s",
)
KEEP_RAW_COLS = (
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "ADES_REGION",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
)


def add_signed_log(df: pl.DataFrame, cols: tuple[str, ...] = WIDE_RANGE_COLS) -> pl.DataFrame:
    """sign(x)*log1p(|x|) copies so leaf-linear models cannot extrapolate wildly on the seconds-scale tails."""
    present = [c for c in cols if c in df.columns]
    return df.with_columns(
        [(pl.col(c).sign() * (pl.col(c).abs() + 1.0).log()).cast(pl.Float32).alias(f"{c}_slog") for c in present]
    )


# Candidate columns under test. They are computed always and hidden from the model with
# TrainConfig.exclude_features, so one feature can be measured at a time without rebuilding the feature cache
# (a rebuild costs about four minutes; a CV run costs five).
CANDIDATE_COLS = (
    "queue_ahead",
    "bank_position",
    "stand_rwy_share",
    "noise_control",
    "active_runways_60m",
    "dep_arr_ratio_30m",
    "congestion_vs_usual_30m",
    "dest_distance_km",
)


def _destination_distance(row: pl.DataFrame) -> pl.DataFrame:
    """Great-circle km from ADEP to ADES. Exogenous: the files say where the flight goes, not how far, and the
    distance stands in for the fuel load that decides how heavy the aircraft is on the ground. A missing
    coordinate cache degrades to a null column rather than taking the whole feature build down."""
    try:
        return add_destination_distance(row.select(["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt"])).select(
            ["MVT_ID_mvt", "dest_distance_km"]
        )
    except Exception:  # noqa: BLE001 - see the docstring
        return row.select("MVT_ID_mvt").with_columns(pl.lit(None, dtype=pl.Float64).alias("dest_distance_km"))


def candidate_features(row: pl.DataFrame, ctx: pl.DataFrame, ctx_src: pl.DataFrame) -> pl.DataFrame:
    """Hypothesis-driven columns, each computable from what the ranking set actually carries.

    queue_ahead     how many departures left the same airport between this flight's scheduled off-block and its
                    take-off. The literal queue it waited through, where dep_tko_prev_Nm is only a fixed window.
    bank_position   where in a departure bank the flight sits: 0 at the front, 1 at the back. Congestion counts
                    say how busy it is, not whether the queue is building or draining.
    stand_rwy_share how usual this stand -> runway pairing is at this airport. A rare pairing is a long or
                    unusual taxi route. Counts only, no labels, so nothing to leak."""
    dep_ctx = ctx_src.filter(pl.col("PHASE_mvt") == "DEP").select(
        ["ADEP_mvt", pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t")]
    )
    tgt = row.select(
        [
            "MVT_ID_mvt",
            "ADEP_mvt",
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_tko"),
            pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_sched"),
        ]
    )
    out = []
    for apt in tgt["ADEP_mvt"].unique().to_list():
        pred = pl.col("ADEP_mvt").is_null() if apt is None else (pl.col("ADEP_mvt") == apt)
        t = tgt.filter(pred)
        ev = np.sort(dep_ctx.filter(pred)["_t"].drop_nulls().to_numpy())
        lo = t["_sched"].to_numpy()
        hi = t["_tko"].to_numpy()
        n = np.full(len(lo), np.nan)
        ok = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
        if ev.size and ok.any():
            n[ok] = np.searchsorted(ev, hi[ok], "right") - np.searchsorted(ev, lo[ok], "left")
        out.append(pl.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"], "queue_ahead": n}))
    q = pl.concat(out) if out else pl.DataFrame({"MVT_ID_mvt": row["MVT_ID_mvt"].head(0), "queue_ahead": []})

    share = (
        row.group_by(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .len()
        .join(row.group_by(["ADEP_mvt", "STAND_mvt"]).len().rename({"len": "_stand_n"}), on=["ADEP_mvt", "STAND_mvt"])
        .with_columns((pl.col("len") / pl.col("_stand_n")).alias("stand_rwy_share"))
        .select(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "stand_rwy_share"])
    )
    # Batch 2. The first three asked about the queue; these ask about the airport's mode of operation.
    #   active_runways_60m      how many distinct runways the airport departed from in the hour around this
    #                           take-off. A configuration change rewrites every taxi route, and nothing in the
    #                           feature set says which configuration is running - only which runway this flight
    #                           used.
    #   dep_arr_ratio_30m       departures over departures+arrivals in the half hour. Mixed-mode operation puts
    #                           departures and arrivals on the same surface; the counts are there separately but
    #                           their balance is not.
    #   congestion_vs_usual_30m this flight's 30-minute departure count divided by the median count for its
    #                           airport and local hour. "Busier than normal for a Tuesday 08:00" rather than
    #                           "busy", which is what a queue actually responds to.
    rwy_act = []
    for apt in tgt["ADEP_mvt"].unique().to_list():
        pred = pl.col("ADEP_mvt").is_null() if apt is None else (pl.col("ADEP_mvt") == apt)
        t = tgt.filter(pred)
        ev = ctx_src.filter((pl.col("PHASE_mvt") == "DEP") & pred).select(
            [pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("_t"), "RUNWAY_mvt"]
        )
        ev = ev.drop_nulls("_t").sort("_t")
        et, er = ev["_t"].to_numpy(), ev["RUNWAY_mvt"].to_list()
        tk = t["_tko"].to_numpy()
        n = np.full(len(tk), np.nan)
        if et.size:
            lo = np.searchsorted(et, tk - 1800.0, "left")
            hi = np.searchsorted(et, tk + 1800.0, "right")
            for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
                if np.isfinite(tk[i]):
                    n[i] = len({r for r in er[a:b] if r is not None})
        rwy_act.append(pl.DataFrame({"MVT_ID_mvt": t["MVT_ID_mvt"], "active_runways_60m": n}))
    rw = (
        pl.concat(rwy_act)
        if rwy_act
        else pl.DataFrame({"MVT_ID_mvt": row["MVT_ID_mvt"].head(0), "active_runways_60m": []})
    )

    prev60, next60 = pl.col("dep_tko_prev_60m"), pl.col("dep_tko_next_60m")
    return (
        row.select(["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
        .join(q, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(share, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left", maintain_order="left")
        .join(rw, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(
            ctx.select(
                [
                    "MVT_ID_mvt",
                    "dep_tko_prev_60m",
                    "dep_tko_next_60m",
                    "dep_tko_prev_30m",
                    "arr_land_prev_30m",
                ]
            ),
            on="MVT_ID_mvt",
            how="left",
        )
        .join(row.select(["MVT_ID_mvt", "hour_local"]), on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns(
            pl.when((prev60 + next60) > 0).then(prev60 / (prev60 + next60)).otherwise(None).alias("bank_position")
        )
        .with_columns(
            pl.when((pl.col("dep_tko_prev_30m") + pl.col("arr_land_prev_30m")) > 0)
            .then(pl.col("dep_tko_prev_30m") / (pl.col("dep_tko_prev_30m") + pl.col("arr_land_prev_30m")))
            .otherwise(None)
            .alias("dep_arr_ratio_30m")
        )
        .with_columns(
            (
                pl.col("dep_tko_prev_30m")
                / pl.col("dep_tko_prev_30m").median().over(["ADEP_mvt", "hour_local"]).replace(0, None)
            ).alias("congestion_vs_usual_30m")
        )
        .with_columns(
            # A column that cannot carry signal, to price what adding *any* column costs. feature_fraction is
            # 0.8, so a new column changes which columns every split samples from; if a pure-noise column costs
            # as much as a real candidate, the comparison is measuring the reshuffle and not the feature.
            (pl.col("MVT_ID_mvt").hash(seed=20260915) % 1_000_000).cast(pl.Float64).alias("noise_control")
        )
        .join(_destination_distance(row), on="MVT_ID_mvt", how="left", maintain_order="left")
        .select(["MVT_ID_mvt", *CANDIDATE_COLS])
    )


def build_departure_features(movements: pl.DataFrame, context_source: pl.DataFrame | None = None) -> pl.DataFrame:
    """Full v1 feature table for the departures in `movements`.

    `movements` may contain labels (training) - they are kept as TAXITIME_SEC_mvt / BLOCK_TIME_UTC_mvt for the
    trainer but never used for features. `context_source` (default: the same frame) is masked before use so the
    same code path serves training and ranking."""
    ctx_src = mask_departures(context_source if context_source is not None else movements)
    dep = movements.filter(pl.col("PHASE_mvt") == "DEP")
    row = clock_and_static_features(dep)
    ctx = context_features(ctx_src, dep_ids=dep["MVT_ID_mvt"])
    keep = [c for c in KEEP_RAW_COLS if c in row.columns]
    derived = [c for c in row.columns if c not in KEEP_RAW_COLS and c not in ("_t",)]
    row = row.select(keep + derived)
    cand = candidate_features(row, ctx, ctx_src)
    feats = add_signed_log(
        row.join(ctx, on="MVT_ID_mvt", how="left", maintain_order="left").join(
            cand, on="MVT_ID_mvt", how="left", maintain_order="left"
        )
    )
    if feats.height != dep.height:
        raise AssertionError(f"feature rows {feats.height} != departure rows {dep.height}")
    if feats["MVT_ID_mvt"].n_unique() != feats.height:
        raise AssertionError("duplicate MVT_ID_mvt in feature frame")
    return feats


def numeric_feature_columns(feats: pl.DataFrame) -> list[str]:
    exclude = set(BLANKED_DEP_COLS) | {"MVT_ID_mvt", "FLIGHT_ID_mvt", "PHASE_mvt", "_t"}
    return [
        c
        for c, dt in feats.schema.items()
        if c not in exclude
        and dt in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32, pl.UInt8, pl.Int16)
    ]
