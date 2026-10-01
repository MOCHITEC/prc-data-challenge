"""Data audit 2026-09-18 -- lens: timestamp integrity across the nine time columns.

No model training. POLARS_MAX_THREADS=1. Lazy scans, narrow column selection.

Stages (run as:  python timestamps.py <stage> [stage ...]  or  all):
  inv    -- inventory: rows, null rates, value ranges, sentinel/epoch/midnight values
  eq     -- exact-equality / near-equality census over all 36 column pairs,
            per set x phase x airport x month
  eqop   -- the same census restricted to the pairs flagged by `eq`, per operator
  ord    -- ordering violations for every pair whose sign is implied by physics
            or by the NM meaning of the field
  prec   -- rounding / precision: second-of-minute and minute-of-hour grids per
            column per airport per set
  span   -- implausible spans: quantiles of every pairwise difference, and rows
            whose timestamps lie far outside the flight's own window
  cross  -- cross-row integrity: DEP row vs the ARR row of the same FLIGHT_ID
  derived-- consistency of data/work/feat_*.parquet derived deltas with raw

Outputs land in <scratchpad>/ts/*.csv and are echoed to stdout.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
warnings.filterwarnings("ignore")

import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw"
WORK = REPO / "data" / "work"
OUT = Path(
    os.environ.get(
        "TS_OUT",
        "/tmp/claude-0/-home-user-prc-data-challenge/"
        "ca703460-62c0-5c84-91fe-d76c3d0e1e20/scratchpad/ts",
    )
)
OUT.mkdir(parents=True, exist_ok=True)

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(60)

# The nine timestamp columns, short names used throughout.
TS = {
    "MVT": "MVT_TIME_UTC_mvt",
    "BLOCK": "BLOCK_TIME_UTC_mvt",
    "SCHED": "SCHED_TIME_UTC_mvt",
    "LOBT": "LOBT_flt",
    "IOBT": "IOBT_flt",
    "EOBT1": "EOBT_1_flt",
    "AOBT3": "AOBT_3_flt",
    "ARVT1": "ARVT_1_flt",
    "ARVT3": "ARVT_3_flt",
}
NAMES = list(TS)
PAIRS = [(NAMES[i], NAMES[j]) for i in range(len(NAMES)) for j in range(i + 1, len(NAMES))]
assert len(PAIRS) == 36

KEEP = [
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "TAXITIME_SEC_mvt",
    *TS.values(),
]

TRAIN_GLOB = str(RAW / "training_*.parquet")
RANK = str(RAW / "ranking.parquet")


def base(source: str, tag: str) -> pl.LazyFrame:
    """Lazy frame with the audit's standard derived keys.

    apt   = the challenge airport of the row (ADEP for DEP rows, ADES for ARR rows)
    month = calendar month of MVT_TIME
    """
    lf = pl.scan_parquet(source).select(KEEP)
    return lf.with_columns(
        pl.lit(tag).alias("set"),
        pl.when(pl.col("PHASE_mvt") == "DEP")
        .then(pl.col("ADEP_mvt"))
        .otherwise(pl.col("ADES_mvt"))
        .alias("apt"),
        pl.col(TS["MVT"]).dt.strftime("%Y-%m").alias("month"),
    )


def both() -> list[tuple[str, pl.LazyFrame]]:
    return [("train", base(TRAIN_GLOB, "train")), ("rank", base(RANK, "rank"))]


def show(df: pl.DataFrame, name: str, head: int | None = None) -> None:
    df.write_csv(OUT / f"{name}.csv")
    print(f"\n===== {name} ({df.height} rows) -> {OUT / (name + '.csv')} =====")
    print(df.head(head) if head else df)


def dsec(a: str, b: str) -> pl.Expr:
    return (pl.col(TS[a]) - pl.col(TS[b])).dt.total_seconds()


# --------------------------------------------------------------------------- inv
def stage_inv() -> None:
    frames = []
    for tag, lf in both():
        aggs = [pl.len().alias("n")]
        for nm, col in TS.items():
            aggs += [
                pl.col(col).null_count().alias(f"{nm}__null"),
                pl.col(col).min().alias(f"{nm}__min"),
                pl.col(col).max().alias(f"{nm}__max"),
            ]
        frames.append(lf.group_by("set", "PHASE_mvt").agg(aggs).collect(engine="streaming"))
    d = pl.concat(frames)
    long = []
    for nm in TS:
        long.append(
            d.select(
                "set",
                "PHASE_mvt",
                "n",
                pl.lit(nm).alias("col"),
                pl.col(f"{nm}__null").alias("n_null"),
                (pl.col(f"{nm}__null") / pl.col("n")).round(6).alias("null_rate"),
                pl.col(f"{nm}__min").alias("min"),
                pl.col(f"{nm}__max").alias("max"),
            )
        )
    show(pl.concat(long).sort("set", "PHASE_mvt", "col"), "inv_nullrange")

    # sentinel / boundary values
    frames = []
    for tag, lf in both():
        aggs = [pl.len().alias("n")]
        for nm, col in TS.items():
            t = pl.col(col)
            aggs += [
                (t.dt.year() < 2024).sum().alias(f"{nm}__pre2024"),
                (t.dt.year() > 2026).sum().alias(f"{nm}__post2026"),
                ((t.dt.hour() == 0) & (t.dt.minute() == 0) & (t.dt.second() == 0))
                .sum()
                .alias(f"{nm}__midnight"),
                (t.dt.second() == 0).sum().alias(f"{nm}__sec0"),
                ((t.dt.second() == 0) & (t.dt.minute() % 5 == 0)).sum().alias(f"{nm}__m5"),
                (t.dt.microsecond() != 0).sum().alias(f"{nm}__subsec"),
            ]
        frames.append(lf.group_by("set", "PHASE_mvt").agg(aggs).collect(engine="streaming"))
    d = pl.concat(frames)
    long = []
    for nm in TS:
        long.append(
            d.select(
                "set",
                "PHASE_mvt",
                "n",
                pl.lit(nm).alias("col"),
                pl.col(f"{nm}__pre2024").alias("pre2024"),
                pl.col(f"{nm}__post2026").alias("post2026"),
                pl.col(f"{nm}__midnight").alias("midnight_exact"),
                (pl.col(f"{nm}__sec0") / pl.col("n")).round(5).alias("sec0_rate"),
                (pl.col(f"{nm}__m5") / pl.col("n")).round(5).alias("min5_sec0_rate"),
                pl.col(f"{nm}__subsec").alias("subsecond"),
            )
        )
    show(pl.concat(long).sort("set", "PHASE_mvt", "col"), "inv_sentinel")


# ---------------------------------------------------------------------------- eq
def _eq_aggs() -> list[pl.Expr]:
    aggs = [pl.len().alias("n")]
    for a, b in PAIRS:
        d = dsec(a, b)
        key = f"{a}~{b}"
        aggs += [
            d.is_not_null().sum().alias(f"{key}|both"),
            (d == 0).sum().alias(f"{key}|eq0"),
            (d.abs() <= 6).sum().alias(f"{key}|le6"),
            (d.abs() <= 60).sum().alias(f"{key}|le60"),
        ]
    return aggs


def _eq_long(d: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    out = []
    for a, b in PAIRS:
        key = f"{a}~{b}"
        out.append(
            d.select(
                *keys,
                pl.lit(a).alias("A"),
                pl.lit(b).alias("B"),
                pl.col(f"{key}|both").alias("n_both"),
                pl.col(f"{key}|eq0").alias("n_eq"),
                pl.col(f"{key}|le6").alias("n_le6"),
                pl.col(f"{key}|le60").alias("n_le60"),
            )
        )
    return (
        pl.concat(out)
        .with_columns(
            (pl.col("n_eq") / pl.col("n_both")).round(6).alias("eq_rate"),
            (pl.col("n_le6") / pl.col("n_both")).round(6).alias("le6_rate"),
            (pl.col("n_le60") / pl.col("n_both")).round(6).alias("le60_rate"),
        )
        .sort(*keys, "A", "B")
    )


def stage_eq() -> None:
    frames = []
    for tag, lf in both():
        frames.append(
            lf.group_by("set", "PHASE_mvt", "apt", "month")
            .agg(_eq_aggs())
            .collect(engine="streaming")
        )
    d = pl.concat(frames)
    cell = _eq_long(d, ["set", "PHASE_mvt", "apt", "month"])
    cell.write_parquet(OUT / "eq_cell.parquet")
    print(f"wrote {OUT / 'eq_cell.parquet'} ({cell.height} rows)")

    # roll-ups
    def roll(keys: list[str], name: str) -> None:
        r = (
            cell.group_by(*keys, "A", "B")
            .agg(pl.col("n_both").sum(), pl.col("n_eq").sum(), pl.col("n_le6").sum(),
                 pl.col("n_le60").sum())
            .with_columns(
                (pl.col("n_eq") / pl.col("n_both")).round(6).alias("eq_rate"),
                (pl.col("n_le6") / pl.col("n_both")).round(6).alias("le6_rate"),
                (pl.col("n_le60") / pl.col("n_both")).round(6).alias("le60_rate"),
            )
            .sort(*keys, "A", "B")
        )
        show(r, name)

    roll(["set", "PHASE_mvt"], "eq_by_set_phase")
    roll(["set", "PHASE_mvt", "apt"], "eq_by_apt")
    roll(["set", "PHASE_mvt", "month"], "eq_by_month")


# --------------------------------------------------------------------------- ord
# (A, B, rel) means the field meaning implies A rel B on the stated phase.
ORDER_DEP = [
    ("BLOCK", "MVT", "<=", "off-block before take-off"),
    ("AOBT3", "MVT", "<=", "actual off-block before take-off"),
    ("MVT", "ARVT3", "<=", "take-off before actual arrival"),
    ("MVT", "ARVT1", "<=", "take-off before planned arrival"),
    ("AOBT3", "ARVT3", "<=", "actual off-block before actual arrival"),
    ("EOBT1", "ARVT1", "<=", "planned off-block before planned arrival"),
    ("IOBT", "ARVT1", "<=", "initial off-block before planned arrival"),
    ("LOBT", "ARVT3", "<=", "LOBT before actual arrival"),
    ("BLOCK", "ARVT3", "<=", "off-block before actual arrival"),
    ("SCHED", "ARVT1", "<=", "scheduled off-block before planned arrival"),
]
ORDER_ARR = [
    ("MVT", "BLOCK", "<=", "landing before on-block"),
    ("ARVT3", "BLOCK", "<=", "actual arrival at/ before on-block"),
    ("AOBT3", "MVT", "<=", "off-block of the inbound leg before its landing"),
    ("AOBT3", "ARVT3", "<=", "actual off-block before actual arrival"),
    ("EOBT1", "ARVT1", "<=", "planned off-block before planned arrival"),
]


def stage_ord() -> None:
    rows = []
    for tag, lf in both():
        for phase, spec in (("DEP", ORDER_DEP), ("ARR", ORDER_ARR)):
            aggs = [pl.len().alias("n")]
            for a, b, _rel, _d in spec:
                d = dsec(a, b)
                k = f"{a}>{b}"
                aggs += [
                    d.is_not_null().sum().alias(f"{k}|both"),
                    (d > 0).sum().alias(f"{k}|viol"),
                    (d > 60).sum().alias(f"{k}|viol60"),
                    (d > 3600).sum().alias(f"{k}|viol1h"),
                    pl.when(d > 0).then(d).max().alias(f"{k}|max"),
                    pl.when(d > 0).then(d).median().alias(f"{k}|med"),
                ]
            d0 = (
                lf.filter(pl.col("PHASE_mvt") == phase)
                .group_by("set", "apt")
                .agg(aggs)
                .collect(engine="streaming")
            )
            for a, b, _rel, desc in spec:
                k = f"{a}>{b}"
                rows.append(
                    d0.select(
                        "set",
                        pl.lit(phase).alias("phase"),
                        "apt",
                        pl.lit(f"{a} <= {b}").alias("rule"),
                        pl.lit(desc).alias("meaning"),
                        pl.col(f"{k}|both").alias("n_both"),
                        pl.col(f"{k}|viol").alias("n_viol"),
                        pl.col(f"{k}|viol60").alias("n_viol_gt60s"),
                        pl.col(f"{k}|viol1h").alias("n_viol_gt1h"),
                        pl.col(f"{k}|med").alias("viol_median_s"),
                        pl.col(f"{k}|max").alias("viol_max_s"),
                    )
                )
    r = pl.concat(rows).with_columns(
        (pl.col("n_viol") / pl.col("n_both")).round(7).alias("viol_rate")
    )
    r.write_parquet(OUT / "ord_apt.parquet")
    show(r.sort("rule", "set", "phase", "apt"), "ord_by_apt")

    tot = (
        r.group_by("set", "phase", "rule", "meaning")
        .agg(pl.col("n_both").sum(), pl.col("n_viol").sum(),
             pl.col("n_viol_gt60s").sum(), pl.col("n_viol_gt1h").sum(),
             pl.col("viol_max_s").max())
        .with_columns((pl.col("n_viol") / pl.col("n_both")).round(7).alias("viol_rate"))
        .sort("rule", "set", "phase")
    )
    show(tot, "ord_total")

    # by month, for the rules that fire at all
    fired = tot.filter(pl.col("n_viol") > 0).select("phase", "rule").unique()
    rows = []
    for tag, lf in both():
        for phase, spec in (("DEP", ORDER_DEP), ("ARR", ORDER_ARR)):
            sub = [s for s in spec if f"{s[0]} <= {s[1]}" in
                   set(fired.filter(pl.col("phase") == phase)["rule"])]
            if not sub:
                continue
            aggs = [pl.len().alias("n")]
            for a, b, _rel, _d in sub:
                d = dsec(a, b)
                k = f"{a}>{b}"
                aggs += [d.is_not_null().sum().alias(f"{k}|both"), (d > 0).sum().alias(f"{k}|viol")]
            d0 = (
                lf.filter(pl.col("PHASE_mvt") == phase)
                .group_by("set", "month")
                .agg(aggs)
                .collect(engine="streaming")
            )
            for a, b, _rel, _desc in sub:
                k = f"{a}>{b}"
                rows.append(
                    d0.select(
                        "set", pl.lit(phase).alias("phase"), "month",
                        pl.lit(f"{a} <= {b}").alias("rule"),
                        pl.col(f"{k}|both").alias("n_both"),
                        pl.col(f"{k}|viol").alias("n_viol"),
                    )
                )
    m = pl.concat(rows).with_columns(
        (pl.col("n_viol") / pl.col("n_both")).round(7).alias("viol_rate")
    )
    show(m.sort("rule", "phase", "month"), "ord_by_month")


# -------------------------------------------------------------------------- prec
def stage_prec() -> None:
    frames = []
    for tag, lf in both():
        for nm, col in TS.items():
            g = (
                lf.filter(pl.col(col).is_not_null())
                .group_by("set", "PHASE_mvt", "apt", pl.col(col).dt.second().alias("sec"))
                .agg(pl.len().alias("n"))
                .with_columns(pl.lit(nm).alias("col"))
                .collect(engine="streaming")
            )
            frames.append(g)
    d = pl.concat(frames)
    d.write_parquet(OUT / "prec_sec.parquet")
    tot = d.group_by("set", "PHASE_mvt", "apt", "col").agg(pl.col("n").sum().alias("N"))
    j = d.join(tot, on=["set", "PHASE_mvt", "apt", "col"]).with_columns(
        (pl.col("n") / pl.col("N")).alias("p")
    )
    summ = (
        j.group_by("set", "PHASE_mvt", "apt", "col")
        .agg(
            pl.col("N").first(),
            pl.col("sec").n_unique().alias("n_distinct_sec"),
            pl.when(pl.col("sec") == 0).then(pl.col("p")).sum().round(5).alias("p_sec0"),
            pl.when((pl.col("sec") <= 6) | (pl.col("sec") >= 54))
            .then(pl.col("p"))
            .sum()
            .round(5)
            .alias("p_within6s_of_minute"),
            (-(pl.col("p") * pl.col("p").log())).sum().round(4).alias("entropy_nats"),
        )
        .sort("col", "set", "PHASE_mvt", "apt")
    )
    show(summ, "prec_sec_summary")

    # minute-of-hour grid (are estimates on a 5-minute grid?)
    frames = []
    for tag, lf in both():
        aggs = [pl.len().alias("n")]
        for nm, col in TS.items():
            t = pl.col(col)
            aggs += [
                t.is_not_null().sum().alias(f"{nm}|nn"),
                ((t.dt.second() == 0) & (t.dt.minute() % 5 == 0)).sum().alias(f"{nm}|m5"),
                ((t.dt.second() == 0) & (t.dt.minute() % 10 == 0)).sum().alias(f"{nm}|m10"),
                ((t.dt.second() == 0) & (t.dt.minute() % 15 == 0)).sum().alias(f"{nm}|m15"),
            ]
        frames.append(lf.group_by("set", "PHASE_mvt", "apt").agg(aggs).collect(engine="streaming"))
    d = pl.concat(frames)
    out = []
    for nm in TS:
        out.append(
            d.select(
                "set", "PHASE_mvt", "apt", pl.lit(nm).alias("col"),
                pl.col(f"{nm}|nn").alias("n_nonnull"),
                (pl.col(f"{nm}|m5") / pl.col(f"{nm}|nn")).round(5).alias("p_min_mult5"),
                (pl.col(f"{nm}|m10") / pl.col(f"{nm}|nn")).round(5).alias("p_min_mult10"),
                (pl.col(f"{nm}|m15") / pl.col(f"{nm}|nn")).round(5).alias("p_min_mult15"),
            )
        )
    show(pl.concat(out).sort("col", "set", "PHASE_mvt", "apt"), "prec_minute_grid")


# -------------------------------------------------------------------------- span
QS = [0.0, 0.001, 0.5, 0.999, 1.0]
# Curated pair list for the span stage: quantiles need a full sort, so this is
# limited to the differences whose magnitude has a physical reading.
SPAN_PAIRS = [
    ("MVT", "BLOCK"), ("MVT", "SCHED"), ("MVT", "AOBT3"), ("MVT", "LOBT"),
    ("MVT", "EOBT1"), ("MVT", "IOBT"), ("BLOCK", "SCHED"), ("BLOCK", "AOBT3"),
    ("AOBT3", "SCHED"), ("AOBT3", "EOBT1"), ("LOBT", "EOBT1"), ("SCHED", "EOBT1"),
    ("ARVT1", "ARVT3"), ("ARVT3", "MVT"), ("ARVT1", "EOBT1"), ("ARVT3", "AOBT3"),
    ("ARVT3", "BLOCK"),
]


def stage_span() -> None:
    rows = []
    for tag, lf in both():
        for phase in ("DEP", "ARR"):
            aggs = []
            for a, b in SPAN_PAIRS:
                d = dsec(a, b)
                k = f"{a}~{b}"
                aggs.append(d.is_not_null().sum().alias(f"{k}|n"))
                for q in QS:
                    aggs.append(d.quantile(q).alias(f"{k}|q{q}"))
                aggs.append((d.abs() > 86400 * 7).sum().alias(f"{k}|gt7d"))
            d0 = (
                lf.filter(pl.col("PHASE_mvt") == phase)
                .select(aggs)
                .collect(engine="streaming")
            )
            for a, b in SPAN_PAIRS:
                k = f"{a}~{b}"
                rows.append(
                    d0.select(
                        pl.lit(tag).alias("set"), pl.lit(phase).alias("phase"),
                        pl.lit(f"{a}-{b}").alias("diff"),
                        pl.col(f"{k}|n").alias("n"),
                        *[pl.col(f"{k}|q{q}").alias(f"q{q}") for q in QS],
                        pl.col(f"{k}|gt7d").alias("n_abs_gt_7d"),
                    )
                )
    r = pl.concat(rows)
    show(r.sort("diff", "phase", "set"), "span_quantiles")


# ------------------------------------------------------------------------- cross
def stage_cross() -> None:
    """DEP row vs the ARR row of the same FLIGHT_ID: are the _flt columns the same
    object, and does ARVT_3 on the DEP row agree with the ARR row's clocks?"""
    for tag, src in (("train", TRAIN_GLOB), ("rank", RANK)):
        lf = base(src, tag).filter(pl.col("FLIGHT_ID_mvt").is_not_null())
        dep = lf.filter(pl.col("PHASE_mvt") == "DEP").select(
            "FLIGHT_ID_mvt", "apt",
            *[pl.col(c).alias(f"d_{n}") for n, c in TS.items()],
        )
        arr = lf.filter(pl.col("PHASE_mvt") == "ARR").select(
            "FLIGHT_ID_mvt",
            pl.col("apt").alias("apt_arr"),
            *[pl.col(c).alias(f"a_{n}") for n, c in TS.items()],
        )
        j = dep.join(arr, on="FLIGHT_ID_mvt", how="inner")
        aggs = [pl.len().alias("n_pairs")]
        for n in TS:
            aggs.append(
                (pl.col(f"d_{n}") == pl.col(f"a_{n}")).sum().alias(f"{n}|same")
            )
            aggs.append(
                (pl.col(f"d_{n}").is_not_null() & pl.col(f"a_{n}").is_not_null())
                .sum()
                .alias(f"{n}|both")
            )
        aggs += [
            (pl.col("d_ARVT3") == pl.col("a_MVT")).sum().alias("depARVT3_eq_arrMVT"),
            (pl.col("d_ARVT3") == pl.col("a_BLOCK")).sum().alias("depARVT3_eq_arrBLOCK"),
            ((pl.col("d_ARVT3") - pl.col("a_MVT")).dt.total_seconds().abs() <= 60)
            .sum()
            .alias("depARVT3_near_arrMVT_60s"),
            ((pl.col("d_ARVT3") - pl.col("a_BLOCK")).dt.total_seconds().abs() <= 60)
            .sum()
            .alias("depARVT3_near_arrBLOCK_60s"),
            (pl.col("d_MVT") > pl.col("a_MVT")).sum().alias("dep_after_arr_landing"),
        ]
        r = j.select(aggs).collect(engine="streaming")
        show(r, f"cross_{tag}")


# ----------------------------------------------------------------------- derived
def stage_derived() -> None:
    cols = ["ADEP_mvt", "month", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt",
            "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt", "TAXITIME_SEC_mvt",
            "mvt_minus_aobt3", "mvt_minus_sched", "mvt_minus_eobt1",
            "mvt_minus_lobt", "mvt_minus_iobt", "lobt_minus_eobt1",
            "aobt3_minus_eobt1", "aobt3_minus_sched", "planned_flight_s",
            "nm_matched"]
    ident = {
        "mvt_minus_aobt3": ("MVT", "AOBT3"),
        "mvt_minus_sched": ("MVT", "SCHED"),
        "mvt_minus_eobt1": ("MVT", "EOBT1"),
        "mvt_minus_lobt": ("MVT", "LOBT"),
        "mvt_minus_iobt": ("MVT", "IOBT"),
        "lobt_minus_eobt1": ("LOBT", "EOBT1"),
        "aobt3_minus_eobt1": ("AOBT3", "EOBT1"),
        "aobt3_minus_sched": ("AOBT3", "SCHED"),
    }
    for tag, f in (("train", "feat_train.parquet"), ("rank", "feat_rank.parquet")):
        lf = pl.scan_parquet(WORK / f).select(cols)
        aggs = [pl.len().alias("n")]
        for feat, (a, b) in ident.items():
            d = dsec(a, b)
            aggs += [
                ((pl.col(feat) - d).abs() > 1e-6).sum().alias(f"{feat}|mismatch"),
                (pl.col(feat).is_null() != d.is_null()).sum().alias(f"{feat}|nullmismatch"),
            ]
        aggs += [
            ((pl.col("planned_flight_s")
              - (pl.col("ARVT_1_flt") - pl.col("EOBT_1_flt")).dt.total_seconds()).abs() > 1e-6)
            .sum().alias("planned_flight_s|mismatch"),
            (pl.col("planned_flight_s") < 0).sum().alias("planned_flight_s|negative"),
            (pl.col("planned_flight_s") == 0).sum().alias("planned_flight_s|zero"),
            pl.col("planned_flight_s").null_count().alias("planned_flight_s|null"),
        ]
        r = lf.select(aggs).collect(engine="streaming")
        show(r.unpivot(), f"derived_{tag}")



# ------------------------------------------------------------------------- spike
# Pairs where an exact-equality cluster was seen; for each, the histogram of the
# difference on the minute grid tells a copy (a spike at 0 far above the local
# density) from a coincidence of two independently minute-rounded clocks.
SPIKE_PAIRS = [
    ("EOBT1", "AOBT3"), ("IOBT", "AOBT3"), ("LOBT", "AOBT3"), ("SCHED", "AOBT3"),
    ("LOBT", "IOBT"), ("SCHED", "IOBT"), ("SCHED", "LOBT"), ("SCHED", "EOBT1"),
    ("LOBT", "EOBT1"), ("IOBT", "EOBT1"), ("BLOCK", "AOBT3"), ("BLOCK", "SCHED"),
    ("ARVT1", "ARVT3"), ("MVT", "AOBT3"),
]


def stage_spike() -> None:
    rows = []
    for tag, lf in both():
        for a, b in SPIKE_PAIRS:
            g = (
                lf.filter((pl.col("PHASE_mvt") == "DEP") & dsec(a, b).is_not_null()
                          & (dsec(a, b).abs() <= 900))
                .group_by("set", (dsec(a, b) / 60).floor().cast(pl.Int32).alias("dmin"))
                .agg(pl.len().alias("n"))
                .with_columns(pl.lit(a).alias("A"), pl.lit(b).alias("B"))
                .collect(engine="streaming")
            )
            rows.append(g)
    d = pl.concat(rows)
    d.write_parquet(OUT / "spike_hist.parquet")
    # excess at bin 0 against the mean of bins -3..-1 and +1..+3
    piv = d.pivot(on="dmin", index=["set", "A", "B"], values="n").fill_null(0)
    nb = [c for c in ["-3", "-2", "-1", "1", "2", "3"] if c in piv.columns]
    r = piv.select(
        "set", "A", "B",
        pl.col("0").alias("n_bin0"),
        (pl.sum_horizontal([pl.col(c) for c in nb]) / len(nb)).round(1).alias("nbr_mean"),
    ).with_columns(
        (pl.col("n_bin0") - pl.col("nbr_mean")).round(1).alias("excess"),
        (pl.col("n_bin0") / pl.col("nbr_mean")).round(2).alias("ratio"),
    ).sort("A", "B", "set")
    show(r, "spike_excess")


# ------------------------------------------------------------------------ origin
def stage_origin() -> None:
    """LOBT/IOBT/EOBT_1 are properties of the flight's ORIGIN, not of the row's
    airport; recompute the equality census keyed on ADEP_flt for both phases."""
    rows = []
    for tag, lf in both():
        aggs = [pl.len().alias("n")]
        for a, b in [("LOBT", "EOBT1"), ("IOBT", "EOBT1"), ("LOBT", "IOBT"),
                     ("SCHED", "EOBT1"), ("EOBT1", "AOBT3")]:
            d = dsec(a, b)
            k = f"{a}~{b}"
            aggs += [d.is_not_null().sum().alias(f"{k}|both"), (d == 0).sum().alias(f"{k}|eq")]
        g = (
            lf.group_by("set", "PHASE_mvt", pl.col("ADEP_mvt").alias("origin"))
            .agg(aggs)
            .collect(engine="streaming")
        )
        for a, b in [("LOBT", "EOBT1"), ("IOBT", "EOBT1"), ("LOBT", "IOBT"),
                     ("SCHED", "EOBT1"), ("EOBT1", "AOBT3")]:
            k = f"{a}~{b}"
            rows.append(
                g.select("set", "PHASE_mvt", "origin", pl.lit(f"{a}=={b}").alias("pair"),
                         pl.col(f"{k}|both").alias("n_both"),
                         pl.col(f"{k}|eq").alias("n_eq"))
            )
    r = pl.concat(rows).with_columns(
        (pl.col("n_eq") / pl.col("n_both")).round(6).alias("eq_rate")
    )
    r.write_parquet(OUT / "origin_eq.parquet")
    show(r.filter(pl.col("n_both") >= 500).sort("pair", "set", "PHASE_mvt", "eq_rate",
                                                descending=[False, False, False, True]),
         "origin_eq", head=120)


# --------------------------------------------------------------------------- eqop
EQOP_PAIRS = [("EOBT1", "AOBT3"), ("LOBT", "EOBT1"), ("SCHED", "EOBT1"),
              ("SCHED", "IOBT"), ("LOBT", "IOBT"), ("BLOCK", "SCHED"),
              ("BLOCK", "AOBT3"), ("ARVT1", "ARVT3")]


def stage_eqop() -> None:
    rows = []
    for tag, lf in both():
        aggs = [pl.len().alias("n")]
        for a, b in EQOP_PAIRS:
            d = dsec(a, b)
            k = f"{a}~{b}"
            aggs += [d.is_not_null().sum().alias(f"{k}|both"), (d == 0).sum().alias(f"{k}|eq")]
        g = (
            lf.filter(pl.col("PHASE_mvt") == "DEP")
            .group_by("set", "AIRCRAFT_OPERATOR_flt")
            .agg(aggs)
            .collect(engine="streaming")
        )
        for a, b in EQOP_PAIRS:
            k = f"{a}~{b}"
            rows.append(
                g.select("set", pl.col("AIRCRAFT_OPERATOR_flt").str.slice(0, 10).alias("op"),
                         pl.lit(f"{a}=={b}").alias("pair"),
                         pl.col(f"{k}|both").alias("n_both"),
                         pl.col(f"{k}|eq").alias("n_eq"))
            )
    r = pl.concat(rows).with_columns(
        (pl.col("n_eq") / pl.col("n_both")).round(6).alias("eq_rate")
    )
    r.write_parquet(OUT / "eqop.parquet")
    big = r.filter(pl.col("n_both") >= 2000)
    show(
        big.group_by("set", "pair").agg(
            pl.len().alias("n_operators"),
            pl.col("eq_rate").min().round(4).alias("min"),
            pl.col("eq_rate").quantile(0.5).round(4).alias("median"),
            pl.col("eq_rate").max().round(4).alias("max"),
            (pl.col("eq_rate") > 0.98).sum().alias("n_ops_gt_98pct"),
            (pl.col("eq_rate") < 0.02).sum().alias("n_ops_lt_2pct"),
        ).sort("pair", "set"),
        "eqop_summary",
    )
    show(big.filter(pl.col("pair") == "EOBT1==AOBT3").sort("eq_rate", descending=True),
         "eqop_eobt1_aobt3", head=60)



def stage_spike2() -> None:
    """Exact-copy test that does not assume a minute grid: the count at d == 0
    exactly, against the counts at d == +-60, +-120, +-180 s exactly."""
    rows = []
    for tag, lf in both():
        for a, b in SPIKE_PAIRS:
            d = dsec(a, b)
            aggs = [(d == 0).sum().alias("d0")]
            for k in (60, 120, 180, 240, 300):
                aggs += [(d == k).sum().alias(f"p{k}"), (d == -k).sum().alias(f"m{k}")]
            aggs += [d.is_not_null().sum().alias("n_both")]
            g = (
                lf.filter(pl.col("PHASE_mvt") == "DEP")
                .select(aggs)
                .collect(engine="streaming")
                .with_columns(pl.lit(tag).alias("set"), pl.lit(a).alias("A"), pl.lit(b).alias("B"))
            )
            rows.append(g)
    d = pl.concat(rows)
    nb = ["p60", "m60", "p120", "m120", "p180", "m180"]
    r = d.select(
        "set", "A", "B", "n_both", pl.col("d0").alias("n_exact_eq"),
        (pl.sum_horizontal([pl.col(c) for c in nb]) / 6).round(1).alias("nbr_mean_60_180"),
        *[pl.col(c) for c in nb],
    ).with_columns(
        (pl.col("n_exact_eq") / pl.col("nbr_mean_60_180")).round(2).alias("copy_ratio"),
        (pl.col("n_exact_eq") - pl.col("nbr_mean_60_180")).round(0).alias("copy_excess"),
    ).sort("A", "B", "set")
    show(r, "spike2_exact")


def stage_distinct() -> None:
    """How many distinct values the four off-block-estimate stamps actually carry."""
    fams = {
        "SCHED/LOBT/IOBT/EOBT1": ["SCHED", "LOBT", "IOBT", "EOBT1"],
    }
    rows = []
    for tag, lf in both():
        for fname, cols in fams.items():
            e = pl.concat_list([pl.col(TS[c]).dt.epoch("s") for c in cols]).list.n_unique()
            g = (
                lf.filter(pl.col("PHASE_mvt") == "DEP")
                .filter(pl.all_horizontal([pl.col(TS[c]).is_not_null() for c in cols]))
                .group_by("set", "apt", e.alias("n_distinct"))
                .agg(pl.len().alias("n"))
                .collect(engine="streaming")
                .with_columns(pl.lit(fname).alias("family"))
            )
            rows.append(g)
    d = pl.concat(rows)
    tot = d.group_by("set", "apt", "family").agg(pl.col("n").sum().alias("N"))
    r = (
        d.join(tot, on=["set", "apt", "family"])
        .with_columns((pl.col("n") / pl.col("N")).round(5).alias("share"))
        .pivot(on="n_distinct", index=["family", "set", "apt"], values="share")
        .sort("family", "set", "apt")
    )
    show(r, "distinct_offblock_family")



def stage_spike3() -> None:
    """Copy test with a +-6 s tolerance, for pairs where one side is not on the
    minute grid (BLOCK, MVT, ARVT_3): bucket k = rows with |d - 60k| <= 6.
    Count at k = 0 against k = +-1, +-2, +-3.  Reported per airport as well."""
    def tol(d: pl.Expr, k: int) -> pl.Expr:
        return ((d - 60 * k).abs() <= 6)

    rows = []
    for tag, lf in both():
        for a, b in SPIKE_PAIRS:
            d = dsec(a, b)
            aggs = [d.is_not_null().sum().alias("n_both"), tol(d, 0).sum().alias("k0")]
            for k in (1, 2, 3):
                aggs += [tol(d, k).sum().alias(f"kp{k}"), tol(d, -k).sum().alias(f"km{k}")]
            g = (
                lf.filter(pl.col("PHASE_mvt") == "DEP")
                .group_by("set", "apt")
                .agg(aggs)
                .collect(engine="streaming")
                .with_columns(pl.lit(a).alias("A"), pl.lit(b).alias("B"))
            )
            rows.append(g)
    d = pl.concat(rows)
    nb = ["kp1", "km1", "kp2", "km2", "kp3", "km3"]
    lin = ["kp1", "km1"]
    r = d.with_columns(
        (pl.sum_horizontal([pl.col(c) for c in nb]) / 6).alias("nbr6"),
        (pl.sum_horizontal([pl.col(c) for c in lin]) / 2).alias("nbr2"),
    ).select(
        "set", "A", "B", "apt", "n_both", pl.col("k0").alias("n_eq6"),
        pl.col("nbr6").round(1), pl.col("nbr2").round(1),
        (pl.col("k0") / pl.col("nbr6")).round(2).alias("ratio6"),
        (pl.col("k0") / pl.col("nbr2")).round(2).alias("ratio2"),
        (pl.col("k0") - pl.col("nbr2")).round(0).alias("excess2"),
    ).sort("A", "B", "set", "apt")
    r.write_parquet(OUT / "spike3.parquet")
    tot = (
        d.group_by("set", "A", "B")
        .agg([pl.col(c).sum() for c in ["n_both", "k0"] + nb])
        .with_columns(
            (pl.sum_horizontal([pl.col(c) for c in nb]) / 6).alias("nbr6"),
            ((pl.col("kp1") + pl.col("km1")) / 2).alias("nbr2"),
        )
        .select("set", "A", "B", "n_both", pl.col("k0").alias("n_eq6"),
                pl.col("nbr6").round(1), pl.col("nbr2").round(1),
                (pl.col("k0") / pl.col("nbr6")).round(2).alias("ratio6"),
                (pl.col("k0") / pl.col("nbr2")).round(2).alias("ratio2"),
                (pl.col("k0") - pl.col("nbr2")).round(0).alias("excess2"))
        .sort("A", "B", "set")
    )
    show(tot, "spike3_total")
    show(r.filter(pl.col("A").is_in(["BLOCK", "MVT"]) | (pl.col("B") == "AOBT3")),
         "spike3_by_apt", head=200)



def stage_extreme() -> None:
    """Day-scale anomalies: counts per set x phase x airport x month, plus the
    row-level listing of the handful of worst cases."""
    conds = {
        "DEP |MVT-SCHED| > 1d": ("DEP", dsec("MVT", "SCHED").abs() > 86400),
        "DEP MVT-SCHED < -1h": ("DEP", dsec("MVT", "SCHED") < -3600),
        "DEP MVT-AOBT3 > 1d": ("DEP", dsec("MVT", "AOBT3") > 86400),
        "DEP MVT-AOBT3 < 0": ("DEP", dsec("MVT", "AOBT3") < 0),
        "DEP ARVT3 < MVT": ("DEP", dsec("ARVT3", "MVT") < 0),
        "DEP |SCHED-EOBT1| > 1d": ("DEP", dsec("SCHED", "EOBT1").abs() > 86400),
        "DEP MVT-BLOCK > 1d": ("DEP", dsec("MVT", "BLOCK") > 86400),
        "DEP MVT-BLOCK < 0": ("DEP", dsec("MVT", "BLOCK") < 0),
        "ARR BLOCK-MVT > 1h": ("ARR", dsec("BLOCK", "MVT") > 3600),
        "ARR |MVT-SCHED| > 1d": ("ARR", dsec("MVT", "SCHED").abs() > 86400),
        "ARR ARVT3 > BLOCK": ("ARR", dsec("ARVT3", "BLOCK") > 0),
    }
    rows = []
    for tag, lf in both():
        for phase in ("DEP", "ARR"):
            sub = {k: c for k, (p_, c) in conds.items() if p_ == phase}
            if not sub:
                continue
            g = (
                lf.filter(pl.col("PHASE_mvt") == phase)
                .group_by("set", "apt", "month")
                .agg(pl.len().alias("n"), *[c.sum().alias(k) for k, c in sub.items()])
                .collect(engine="streaming")
            )
            for k in sub:
                rows.append(g.select("set", "apt", "month", "n",
                                     pl.lit(k).alias("cond"), pl.col(k).alias("n_hit")))
    r = pl.concat(rows)
    r.write_parquet(OUT / "extreme_cell.parquet")
    show(
        r.group_by("set", "cond").agg(pl.col("n").sum(), pl.col("n_hit").sum())
        .with_columns((pl.col("n_hit") / pl.col("n")).alias("rate"))
        .sort("cond", "set"),
        "extreme_total",
    )
    show(
        r.group_by("set", "cond", "apt").agg(pl.col("n").sum(), pl.col("n_hit").sum())
        .with_columns((pl.col("n_hit") / pl.col("n")).round(7).alias("rate"))
        .filter(pl.col("n_hit") > 0).sort("cond", "set", "n_hit", descending=[False, False, True]),
        "extreme_by_apt", head=200,
    )
    show(
        r.group_by("set", "cond", "month").agg(pl.col("n").sum(), pl.col("n_hit").sum())
        .with_columns((pl.col("n_hit") / pl.col("n")).round(7).alias("rate"))
        .sort("cond", "month"),
        "extreme_by_month", head=200,
    )

    # row-level listing of the worst arrival outliers
    for tag, src in (("train", TRAIN_GLOB), ("rank", RANK)):
        lf = base(src, tag).filter(pl.col("PHASE_mvt") == "ARR")
        worst = (
            lf.filter(dsec("BLOCK", "MVT") > 3600)
            .select("set", "apt", "month", "TAXITIME_SEC_mvt",
                    *[pl.col(c).alias(n) for n, c in TS.items()])
            .sort("TAXITIME_SEC_mvt", descending=True)
            .head(15)
            .collect(engine="streaming")
        )
        show(worst, f"extreme_arr_taxiin_{tag}")



def stage_arrfill() -> None:
    """The arrival side of the sched-filled BLOCK: on ARR rows BLOCK_TIME is the
    on-block time and is NOT nulled in the ranking set, so the fill is observable
    there.  Same copy test as spike3, plus the taxi-in distribution it produces."""
    def tol(d, k):
        return (d - 60 * k).abs() <= 6

    rows, dist = [], []
    for tag, lf in both():
        d = dsec("BLOCK", "SCHED")
        y = pl.col("TAXITIME_SEC_mvt")
        aggs = [pl.len().alias("n"), tol(d, 0).sum().alias("k0")]
        for k in (1, 2, 3):
            aggs += [tol(d, k).sum().alias(f"kp{k}"), tol(d, -k).sum().alias(f"km{k}")]
        aggs += [
            (tol(d, 0) & (y > 3600)).sum().alias("k0_y_gt1h"),
            (~tol(d, 0) & (y > 3600)).sum().alias("nk0_y_gt1h"),
            (y > 3600).sum().alias("y_gt1h"),
            (y > 7200).sum().alias("y_gt2h"),
            (y > 43200).sum().alias("y_gt12h"),
            y.max().alias("y_max"),
            y.median().alias("y_med"),
            y.mean().round(1).alias("y_mean"),
            pl.when(y <= 3600).then(y).mean().round(1).alias("y_mean_capped1h"),
        ]
        g = (
            lf.filter(pl.col("PHASE_mvt") == "ARR")
            .group_by("set", "apt")
            .agg(aggs)
            .collect(engine="streaming")
        )
        rows.append(g)
    d0 = pl.concat(rows)
    r = d0.with_columns(
        ((pl.col("kp1") + pl.col("km1")) / 2).alias("nbr2"),
    ).select(
        "set", "apt", "n", pl.col("k0").alias("n_eq6"), pl.col("nbr2").round(1),
        (pl.col("k0") / pl.col("nbr2")).round(2).alias("copy_ratio"),
        (pl.col("k0") - pl.col("nbr2")).round(0).alias("copy_excess"),
        "k0_y_gt1h", "nk0_y_gt1h", "y_gt1h", "y_gt2h", "y_gt12h",
        "y_max", "y_med", "y_mean", "y_mean_capped1h",
    ).sort("apt", "set")
    show(r, "arrfill_by_apt")

    # by month, LIRF only
    rows = []
    for tag, lf in both():
        d = dsec("BLOCK", "SCHED")
        aggs = [pl.len().alias("n"), tol(d, 0).sum().alias("k0"),
                tol(d, 1).sum().alias("kp1"), tol(d, -1).sum().alias("km1"),
                (pl.col("TAXITIME_SEC_mvt") > 3600).sum().alias("y_gt1h")]
        g = (
            lf.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("apt") == "LIRF"))
            .group_by("set", "month")
            .agg(aggs)
            .collect(engine="streaming")
        )
        rows.append(g)
    m = pl.concat(rows).with_columns(
        (pl.col("k0") / pl.col("n")).round(5).alias("eq6_rate"),
        ((pl.col("kp1") + pl.col("km1")) / 2).alias("nbr2"),
    ).with_columns((pl.col("k0") / pl.col("nbr2")).round(2).alias("copy_ratio")).sort("month")
    show(m, "arrfill_lirf_by_month")



def stage_extras() -> None:
    """Follow-ups on the cells the earlier stages flagged."""
    c = pl.read_parquet(OUT / "eq_cell.parquet")

    # (a) LFPG: the SCHED/LOBT-vs-EOBT_1 equality rate by month
    show(
        c.filter((pl.col("apt") == "LFPG") & (pl.col("PHASE_mvt") == "DEP")
                 & (pl.col("B") == "EOBT1") & pl.col("A").is_in(["SCHED", "LOBT", "IOBT"]))
        .pivot(on="A", index=["set", "month"], values="eq_rate").sort("month"),
        "extras_lfpg_eobt1_by_month",
    )
    # (b) EGLL: the AOBT_3 == EOBT_1 cluster by month
    show(
        c.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("A") == "EOBT1")
                 & (pl.col("B") == "AOBT3"))
        .pivot(on="apt", index=["set", "month"], values="eq_rate").sort("month"),
        "extras_eobt1_aobt3_by_month",
    )
    # (c) LTFM: LOBT == EOBT_1 by month, both phases (already 1.0 on DEP)
    show(
        c.filter((pl.col("apt") == "LTFM") & (pl.col("A") == "LOBT") & (pl.col("B") == "EOBT1"))
        .select("set", "PHASE_mvt", "month", "n_both", "n_eq", "eq_rate").sort("PHASE_mvt", "month"),
        "extras_ltfm_lobt_eobt1_by_month",
    )
    # (d) ARVT_3 second-of-minute regime by month, ARR rows
    rows = []
    for tag, lf in both():
        g = (
            lf.filter(pl.col("PHASE_mvt") == "ARR")
            .group_by("set", "apt", "month")
            .agg(
                pl.col(TS["ARVT3"]).is_not_null().sum().alias("n"),
                ((pl.col(TS["ARVT3"]).dt.second() <= 6)
                 | (pl.col(TS["ARVT3"]).dt.second() >= 54)).sum().alias("w6"),
            )
            .collect(engine="streaming")
        )
        rows.append(g)
    r = pl.concat(rows).with_columns((pl.col("w6") / pl.col("n")).round(4).alias("p_within6s"))
    show(r.pivot(on="apt", index=["set", "month"], values="p_within6s").sort("month"),
         "extras_arvt3_grid_by_month")

    # (e) which operator has LOBT == EOBT_1 on every row, and is it just LTFM
    op = pl.read_parquet(OUT / "eqop.parquet")
    show(op.filter((pl.col("pair") == "LOBT==EOBT1") & (pl.col("eq_rate") > 0.97))
         .sort("n_both", descending=True), "extras_lobt_eobt1_ops")



def stage_grid() -> None:
    """(a) the explicit second-of-minute grid of BLOCK and MVT at the grid
    airports; (b) whether every monthly file holds only its own month."""
    rows = []
    for tag, src in (("train", TRAIN_GLOB), ("rank", RANK)):
        for nm in ("MVT", "BLOCK"):
            g = (
                base(src, tag)
                .filter(pl.col("PHASE_mvt") == "DEP")
                .filter(pl.col(TS[nm]).is_not_null())
                .group_by("set", "apt", pl.col(TS[nm]).dt.second().alias("sec"))
                .agg(pl.len().alias("n"))
                .collect(engine="streaming")
                .with_columns(pl.lit(nm).alias("col"))
            )
            rows.append(g)
    d = pl.concat(rows)
    tot = d.group_by("set", "apt", "col").agg(pl.col("n").sum().alias("N"))
    j = d.join(tot, on=["set", "apt", "col"]).with_columns(
        (pl.col("n") / pl.col("N")).round(5).alias("p")
    )
    show(
        j.filter(pl.col("apt").is_in(["LIRF", "LTFM", "EDDM", "EGLL", "LFPG"]))
        .pivot(on=["col", "apt"], index=["set", "sec"], values="p").sort("set", "sec"),
        "grid_sec_share", head=140,
    )

    # file vs month
    rows = []
    for f in sorted(RAW.glob("training_*.parquet")):
        g = (
            pl.scan_parquet(f)
            .select(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("month"))
            .group_by("month").agg(pl.len().alias("n"))
            .collect(engine="streaming")
            .with_columns(pl.lit(f.name).alias("file"))
        )
        rows.append(g)
    show(pl.concat(rows).sort("file", "month"), "grid_file_month", head=40)


STAGES = {
    "inv": stage_inv,
    "eq": stage_eq,
    "ord": stage_ord,
    "prec": stage_prec,
    "span": stage_span,
    "cross": stage_cross,
    "derived": stage_derived,
    "spike": stage_spike,
    "origin": stage_origin,
    "eqop": stage_eqop,
    "spike2": stage_spike2,
    "distinct": stage_distinct,
    "spike3": stage_spike3,
    "extreme": stage_extreme,
    "arrfill": stage_arrfill,
    "extras": stage_extras,
    "grid": stage_grid,
}

if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    if want == ["all"]:
        want = list(STAGES)
    for s in want:
        print(f"\n######## stage {s} ########", flush=True)
        STAGES[s]()
