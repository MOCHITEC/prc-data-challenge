"""Link OPDI stand-exit events to the challenge departures and measure them (docs/opdi_events_2026-10-05/README.md).

    .venv/bin/python docs/opdi_events_2026-10-05/link_events.py   # writes data/work/opdi_exitpark.parquet + a report

For every departure (train and ranking) the observed off-block is the LAST `exit-parking_position` event at the same
airport, with the same callsign, in [MVT - 2 h, MVT - 30 s]. Matched rows link on CALLSIGN_flt; unmatched rows have no
callsign and link on FLIGHT_mvt when it looks like an ICAO callsign (3 letters + digits). The table holds only
event-derived columns (no label); the report reads labels on 2025 rows to measure accuracy."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import polars as pl

LABEL, PROXY = "TAXITIME_SEC_mvt", "mvt_minus_aobt3"
WORK = Path("data/work_flseq")
OUT = Path("data/work/opdi_exitpark.parquet")
WIN_S, MIN_S = 7200, 30


def events() -> pl.DataFrame:
    files = sorted(glob.glob("data/external/opdi_events/ground_*.parquet"))
    ev = pl.concat([pl.read_parquet(f) for f in files])
    return (
        ev.filter((pl.col("type") == "exit-parking_position") & pl.col("callsign").is_not_null())
        .select(
            pl.col("airport"),
            pl.col("callsign").alias("cs"),
            pl.col("event_time").dt.replace_time_zone(None).alias("t_exit"),
            pl.col("ref").alias("exit_stand"),
            pl.col("in_use_s").alias("exit_in_use_s"),
        )
        .unique()
        .sort("t_exit")
    )


def departures() -> pl.DataFrame:
    cols = [
        "MVT_ID_mvt",
        "ADEP_mvt",
        "CALLSIGN_flt",
        "FLIGHT_mvt",
        "MVT_TIME_UTC_mvt",
        "STAND_mvt",
        "AOBT_3_flt",
        PROXY,
    ]
    tr = pl.read_parquet(WORK / "feat_train.parquet", columns=[*cols, LABEL]).with_columns(pl.lit("train").alias("set"))
    rk = pl.read_parquet(WORK / "feat_rank.parquet", columns=cols).with_columns(
        pl.lit(None, dtype=pl.Int32).alias(LABEL), pl.lit("rank").alias("set")
    )
    d = pl.concat([tr, rk.select(tr.columns)], how="vertical_relaxed")
    fl = pl.col("FLIGHT_mvt").cast(pl.Utf8).str.strip_chars()
    return d.with_columns(
        pl.coalesce(
            pl.col("CALLSIGN_flt").cast(pl.Utf8).str.strip_chars(),
            pl.when(fl.str.contains(r"^[A-Z]{3}[0-9][0-9A-Z]{0,4}$")).then(fl),
        ).alias("cs"),
        pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("t_mvt"),
    )


def link(d: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """All stand exits at the departure airport with the same callsign in [MVT - 2 h, MVT - 30 s]; the exit at the
    departure's own stand (STAND_mvt, normalised) if any, else the earliest one."""
    norm = lambda c: pl.col(c).cast(pl.Utf8).str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "")
    j = (
        d.drop_nulls(["cs", "t_mvt"])
        .select("MVT_ID_mvt", "ADEP_mvt", "cs", "t_mvt", "STAND_mvt", "AOBT_3_flt")
        .join(ev.rename({"airport": "ADEP_mvt"}), on=["ADEP_mvt", "cs"], how="inner")
        .filter(
            (pl.col("t_exit") <= pl.col("t_mvt") - pl.duration(seconds=MIN_S))
            & (pl.col("t_exit") >= pl.col("t_mvt") - pl.duration(seconds=WIN_S))
        )
        .with_columns((norm("exit_stand") == norm("STAND_mvt")).fill_null(False).alias("_match"))
    )
    best = (
        j.sort(["MVT_ID_mvt", "_match", "t_exit"], descending=[False, True, False])
        .group_by("MVT_ID_mvt", maintain_order=True)
        .agg(
            pl.first("t_mvt"),
            pl.first("t_exit"),
            pl.first("_match"),
            pl.first("exit_in_use_s"),
            pl.first("AOBT_3_flt"),
            pl.len().alias("opdi_exit_n"),
        )
    )
    aobt = pl.col("AOBT_3_flt").cast(pl.Datetime("us")).dt.replace_time_zone(None)
    return best.select(
        "MVT_ID_mvt",
        (pl.col("t_mvt") - pl.col("t_exit")).dt.total_seconds().cast(pl.Float64).alias("opdi_exit_est_s"),
        (pl.col("t_exit") - aobt).dt.total_seconds().cast(pl.Float64).alias("opdi_exit_minus_aobt3_s"),
        pl.col("_match").cast(pl.Float64).alias("opdi_exit_stand_match"),
        pl.col("opdi_exit_n").cast(pl.Float64),
        pl.col("exit_in_use_s").cast(pl.Float64).alias("opdi_exit_in_use_s"),
    )


def main() -> None:
    ev = events()
    d = departures()
    t = link(d, ev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.write_parquet(OUT)
    d = d.join(t, on="MVT_ID_mvt", how="left")
    d = d.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
        pl.col(PROXY).is_not_null().alias("matched"),
        pl.col("opdi_exit_est_s").is_not_null().alias("has"),
    )
    print(f"exit-parking events {ev.height:,}; linked departures {t.height:,} of {d.height:,}")
    print("\n== coverage (share of departures with an observed stand exit), by airport and month ==")
    cov = (
        d.group_by("ADEP_mvt", "ym")
        .agg(pl.col("has").mean().round(3).alias("cov"))
        .pivot(on="ym", index="ADEP_mvt", values="cov")
        .sort("ADEP_mvt")
    )
    with pl.Config(tbl_cols=30, tbl_width_chars=300):
        print(cov.select(["ADEP_mvt", *sorted(c for c in cov.columns if c != "ADEP_mvt")]))
    rk = d.filter(pl.col("set") == "rank")
    print(
        "\nranking coverage by lane:",
        rk.group_by("matched").agg(pl.col("has").mean().round(3), pl.len()).sort("matched").to_dicts(),
    )
    tr = d.filter((pl.col("set") == "train") & pl.col("has") & pl.col(LABEL).is_finite())
    tr = tr.with_columns((pl.col(LABEL) - pl.col("opdi_exit_est_s")).alias("resid"))
    print("\n== 2025 accuracy: label - (MVT - exit_parking), by airport ==")
    rows = []
    for a, g in tr.group_by("ADEP_mvt"):
        r = g["resid"].to_numpy()
        med = float(np.median(r))
        e = r - med
        p = g[PROXY].to_numpy()
        y = g[LABEL].to_numpy().astype(float)
        mt = np.isfinite(p)
        rows.append(
            (
                a[0],
                len(r),
                med,
                float(np.mean(np.abs(e) < 60)),
                float(np.mean(np.abs(e) < 180)),
                float(np.sqrt(np.mean(np.clip(e, -1800, 1800) ** 2))),
                float(np.sqrt(np.mean(np.clip(y[mt] - p[mt], -1800, 1800) ** 2))) if mt.any() else float("nan"),
            )
        )
    rep = pl.DataFrame(
        rows,
        schema=["airport", "n", "bias_s", "within_60s", "within_180s", "rmse_clip_s", "proxy_rmse_clip_s"],
        orient="row",
    ).sort("n", descending=True)
    with pl.Config(tbl_rows=20):
        print(rep)


if __name__ == "__main__":
    main()
