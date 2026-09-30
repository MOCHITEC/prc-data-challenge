"""Coverage-parity gate: does every column the model trains on still look the same at serve time?

    python -m prc.coverage --config configs/main_gap.yaml \
        --train data/work/feat_train.parquet --rank data/work/feat_rank.parquet

Why this module exists (docs/competitor_research_2026-09-21/plans/P8-weather-and-coverage-gate.md §5).
A competitor shipped a feed whose columns went 100.0% -> 0.0% non-null at one airport in one scored month.
LightGBM's numerical decision converts NaN to 0.0 when the split's feature had no missing value in training,
so every serve-time row routed to a branch nobody trained for; dropping the feed was worth -13.14 s and
median-imputing it instead was +58 s. That is a deployment failure, not an information failure, and nothing
in a CV number can see it: the training side is intact, so cross-validation is silent.

Our own precedent is the same shape. A weather join once created a column named `hour_utc`, which was also
the integer hour-of-day feature; the timestamp overwrote the feature and was then dropped, so two
submissions trained without it and nobody noticed (weather.py:165-169).

Two rules here are ours and are not in the competitor's gate:

  * DEAD. Their gate compares a training share against a serving share, so it catches a feed that *decays*
    and is blind to one that was never alive: `precip_mm` is non-null on 100% of rows and the literal 0.00 on
    100% of rows, in both years, so every drop it computes is exactly 0 points and it passes. A column that
    is constant on the training side is dead weight - measured at +-0.5 s of model jitter each
    (docs/features_log.md:44-52) - and fails here.
  * Rises are reported, never failed. A one-sided test is blind to a share that goes up, and their log
    contains such a case that nobody caught; but a rise is not the NaN->0 failure mode, and failing on it
    would have us fighting ordinary weather variation (`deicing_gate` at LSZH rising in January is the real
    world, and it is the entire reason weather might help at all).

The column set is what the model is given - `visible_numeric_columns(frame, cfg)` plus `CATEGORICALS` - and
not the width of the frame. `train.py:526-528` already makes that distinction; a gate that reported on the
frame would report on columns nobody trains on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import yaml

from .features import CATEGORICALS
from .train import TrainConfig, visible_numeric_columns

# The competitor's threshold, kept: 20 points of share (check_coverage.py:70-73).
DROP_POINTS = 20.0
RISE_POINTS = 20.0
# Ours, and in two tiers. The plan (§5.3) writes one rule, "non-zero share < 0.1% on the training side ->
# fail as dead weight", and then its own table marks `stand_null` (0.0013%) and `diverted` (0.0993%) as
# warnings rather than failures while calling only five columns dead. Those two statements disagree: 0.0993
# is below 0.1. Failing a build on `diverted` would be wrong - 0.0993% of 2,085,047 rows is about 2,070 rows
# that genuinely carry information, just rarely - so the fail tier is set where the plan's §5.5 acceptance
# criterion is, at columns that are identically zero, and the plan's 0.1% becomes the warn tier.
DEAD_NONZERO_PCT = 0.001  # fail: below ~1 informative row in 100,000, i.e. the column is a constant
NEAR_DEAD_NONZERO_PCT = 0.1  # report only: rare but real
ALL_AIRPORTS = "ALL"
ALL_PERIODS = "ALL"
TIME_COL = "MVT_TIME_UTC_mvt"
AIRPORT_COL = "ADEP_mvt"

FAIL_RULES = ("MISSING", "NONNULL", "NONZERO", "DEAD")


def model_visible_columns(frame: pl.DataFrame, cfg: TrainConfig) -> tuple[list[str], list[str]]:
    """(numeric, categorical) exactly as `run_cv` would hand them to LightGBM."""
    numeric = visible_numeric_columns(frame, cfg)
    cats = [c for c in CATEGORICALS if c in frame.columns and c not in set(cfg.exclude_features)]
    return numeric, cats


def _period_expr() -> pl.Expr:
    """year*100+month, never the bare `month` column: a month number alone mixes January 2025 with 2026."""
    return (pl.col(TIME_COL).dt.year() * 100 + pl.col(TIME_COL).dt.month()).cast(pl.Int32).alias("_period")


def coverage_frame(
    frame: pl.DataFrame,
    numeric: list[str],
    categorical: list[str] = (),
    airport_col: str = AIRPORT_COL,
) -> pl.DataFrame:
    """Long table: one row per (column, airport, period) with the two shares, in percent.

    `nonnull_pct` is the share of rows where the value is present. `nonzero_pct` is the share where it is
    present *and* not zero - a null is counted as a zero here on purpose, because that is what LightGBM does
    to it at serve time when the training side had no missing value. `nonzero_pct` is null for a categorical.

    Every (airport, period) cell is also rolled up to the `ALL` airport and the `ALL` period by weighting each
    cell by its row count, which is exact because the cells are disjoint.
    """
    cols = list(numeric) + [c for c in categorical if c not in set(numeric)]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"columns absent from the frame: {missing}")
    num = set(numeric)
    aggs: list[pl.Expr] = [pl.len().alias("_n")]
    for c in cols:
        aggs.append(pl.col(c).is_not_null().mean().alias(f"nn::{c}"))
        if c in num:
            aggs.append((pl.col(c).fill_null(0) != 0).mean().alias(f"nz::{c}"))
    wide = frame.with_columns(_period_expr()).group_by([airport_col, "_period"]).agg(aggs)

    recs = []
    for row in wide.iter_rows(named=True):
        n = row["_n"]
        for c in cols:
            recs.append(
                {
                    "column": c,
                    "airport": row[airport_col],
                    "period": int(row["_period"]),
                    "n": int(n),
                    # `or 0.0` would be wrong here: it maps a genuine None to 0.0 and hides the difference
                    # between "this group has no rows to measure" and "this group is all zeros".
                    "nonnull_pct": _pct(row[f"nn::{c}"]),
                    "nonzero_pct": _pct(row[f"nz::{c}"]) if c in num else None,
                }
            )
    cov = pl.DataFrame(
        recs,
        schema={
            "column": pl.Utf8,
            "airport": pl.Utf8,
            "period": pl.Int32,
            "n": pl.Int64,
            "nonnull_pct": pl.Float64,
            "nonzero_pct": pl.Float64,
        },
    )
    return pl.concat([cov.cast({"period": pl.Utf8}), _rollups(cov)], how="vertical")


def _pct(v) -> float | None:
    return None if v is None else 100.0 * float(v)


def _rollups(cov: pl.DataFrame) -> pl.DataFrame:
    """n-weighted roll-ups over airports, over periods, and over both."""
    out = []
    for keys, airport, period in (
        (["column", "period"], ALL_AIRPORTS, None),
        (["column", "airport"], None, ALL_PERIODS),
        (["column"], ALL_AIRPORTS, ALL_PERIODS),
    ):
        g = cov.group_by(keys).agg(
            [
                pl.col("n").sum().alias("n"),
                ((pl.col("nonnull_pct") * pl.col("n")).sum() / pl.col("n").sum()).alias("nonnull_pct"),
                # Null-aware on purpose: `nonzero_pct` is null for a categorical, and polars' sum treats a
                # null as 0, so the naive weighted mean would report every categorical as 0.0% non-zero and
                # the DEAD rule would fire on all ten of them.
                pl.when(pl.col("nonzero_pct").is_not_null().any())
                .then((pl.col("nonzero_pct") * pl.col("n")).sum() / pl.col("n").sum())
                .otherwise(pl.lit(None, pl.Float64))
                .alias("nonzero_pct"),
            ]
        )
        if airport is not None:
            g = g.with_columns(pl.lit(airport).alias("airport"))
        if period is not None:
            g = g.with_columns(pl.lit(period).alias("period"))
        out.append(g.with_columns(pl.col("period").cast(pl.Utf8)).select(cov.columns))
    return pl.concat(out, how="vertical")


def _lookup(cov: pl.DataFrame) -> dict:
    return {(r["column"], r["airport"], r["period"]): r for r in cov.iter_rows(named=True)}


def _month_matched(period: str) -> str:
    """The scored period's counterpart one year earlier: 202601 -> 202501. Absorbs seasonality."""
    return str(int(period) - 100)


def _waived(allow: dict, rule: str, column: str, airport: str, period: str) -> str | None:
    """A waiver is explicit or it does not exist: the key names what is excused and the value says why."""
    for key in (f"{rule}:{column}@{airport}@{period}", f"{rule}:{column}", column):
        if key in allow:
            reason = str(allow[key] or "").strip()
            if not reason:
                raise ValueError(f"coverage_allow['{key}'] needs a written reason, not an empty one")
            return reason
    return None


def evaluate(
    train_cov: pl.DataFrame,
    rank_cov: pl.DataFrame,
    columns: list[str],
    rank_columns: list[str] | None = None,
    allow: dict | None = None,
    drop_points: float = DROP_POINTS,
    rise_points: float = RISE_POINTS,
    dead_nonzero_pct: float = DEAD_NONZERO_PCT,
    near_dead_nonzero_pct: float = NEAR_DEAD_NONZERO_PCT,
) -> dict:
    """Run the rules. Returns {failures, warnings, waived}; `failures` non-empty means do not ship."""
    allow = dict(allow or {})
    tr, rk = _lookup(train_cov), _lookup(rank_cov)
    have_rank = set(rank_columns if rank_columns is not None else {c for c, _, _ in rk})
    failures, warnings, waived = [], [], []

    def record(rule, column, airport, period, detail, **extra):
        rec = {"rule": rule, "column": column, "airport": airport, "period": period, "detail": detail, **extra}
        if rule in FAIL_RULES:
            reason = _waived(allow, rule, column, airport, period)
            if reason is not None:
                waived.append({**rec, "waiver_reason": reason})
                return
            failures.append(rec)
        else:
            warnings.append(rec)

    periods = sorted({p for _, a, p in rk if a != ALL_AIRPORTS and p != ALL_PERIODS})
    airports = sorted({a for _, a, _ in rk if a != ALL_AIRPORTS})

    for c in columns:
        # (a) absent from the ranking frame entirely.
        if c not in have_rank:
            record("MISSING", c, ALL_AIRPORTS, ALL_PERIODS, "column is absent from the ranking frame")
            continue

        # (d) dead on the training side. Ours, not the competitor's: a constant column's drop is 0 points,
        # so every share comparison passes it. Evaluated on all training rows, once per column.
        base = tr.get((c, ALL_AIRPORTS, ALL_PERIODS))
        if base is not None and base["nonzero_pct"] is not None and base["nonzero_pct"] < near_dead_nonzero_pct:
            dead = base["nonzero_pct"] < dead_nonzero_pct
            record(
                "DEAD" if dead else "NEAR_DEAD",
                c,
                ALL_AIRPORTS,
                ALL_PERIODS,
                f"non-zero on {base['nonzero_pct']:.4f}% of {base['n']:,} training rows "
                + (
                    f"(< {dead_nonzero_pct}%): the model is given a constant"
                    if dead
                    else f"(< {near_dead_nonzero_pct}%): rare but not constant - reported, not failed"
                ),
                train_nonzero_pct=base["nonzero_pct"],
            )

        for period in periods:
            matched_period = _month_matched(period)
            for airport in airports:
                r = rk.get((c, airport, period))
                b = tr.get((c, airport, ALL_PERIODS))
                if r is None or b is None:
                    continue
                # (b) non-null share collapse vs the whole training year.
                nn_drop = b["nonnull_pct"] - r["nonnull_pct"]
                if nn_drop > drop_points:
                    record(
                        "NONNULL",
                        c,
                        airport,
                        period,
                        f"non-null {b['nonnull_pct']:.1f}% (all-2025) -> {r['nonnull_pct']:.1f}%: "
                        f"-{nn_drop:.1f} pts on {r['n']:,} rows",
                        drop_pts=nn_drop,
                        n=r["n"],
                    )
                if r["nonzero_pct"] is None or b["nonzero_pct"] is None:
                    continue
                # (c) non-zero share collapse, against the whole year AND the month-matched month.
                m = tr.get((c, airport, matched_period))
                nz_drop = b["nonzero_pct"] - r["nonzero_pct"]
                nz_drop_m = (m["nonzero_pct"] - r["nonzero_pct"]) if m is not None else None
                detail = (
                    f"non-zero {b['nonzero_pct']:.1f}% (all-2025) / "
                    f"{'n/a' if m is None else format(m['nonzero_pct'], '.1f') + '%'} ({matched_period}) -> "
                    f"{r['nonzero_pct']:.1f}%: -{nz_drop:.1f} pts vs all-2025, "
                    f"{'n/a' if nz_drop_m is None else format(-nz_drop_m, '+.1f') + ' pts month-matched'}"
                )
                both = nz_drop > drop_points and nz_drop_m is not None and nz_drop_m > drop_points
                either = nz_drop > drop_points or (nz_drop_m is not None and nz_drop_m > drop_points)
                if both:
                    record(
                        "NONZERO", c, airport, period, detail, drop_pts=nz_drop, drop_pts_matched=nz_drop_m, n=r["n"]
                    )
                elif either:
                    record(
                        "NONZERO_PARTIAL",
                        c,
                        airport,
                        period,
                        detail + " (one reference only - reported, not failed)",
                        drop_pts=nz_drop,
                        drop_pts_matched=nz_drop_m,
                        n=r["n"],
                    )
                # Rises: reported, never failed.
                if nz_drop_m is not None and -nz_drop_m > rise_points:
                    record(
                        "RISE",
                        c,
                        airport,
                        period,
                        f"non-zero {m['nonzero_pct']:.1f}% ({matched_period}) -> {r['nonzero_pct']:.1f}%: "
                        f"+{-nz_drop_m:.1f} pts month-matched (reported, not failed)",
                        rise_pts=-nz_drop_m,
                        n=r["n"],
                    )
    return {"failures": failures, "warnings": warnings, "waived": waived}


def check_coverage(
    train: pl.DataFrame,
    rank: pl.DataFrame,
    cfg: TrainConfig,
    allow: dict | None = None,
    **kw,
) -> dict:
    """The whole gate on two feature frames. Returns the report; the caller decides what to do with it."""
    numeric, cats = model_visible_columns(train, cfg)
    columns = numeric + [c for c in cats if c not in set(numeric)]
    rank_cols = [c for c in columns if c in rank.columns]
    rank_numeric = [c for c in numeric if c in rank.columns]
    train_cov = coverage_frame(train, numeric, cats)
    rank_cov = coverage_frame(rank, rank_numeric, [c for c in cats if c in rank.columns])
    rep = evaluate(train_cov, rank_cov, columns, rank_columns=rank_cols, allow=allow, **kw)
    rep["n_columns"] = len(columns)
    rep["n_numeric"] = len(numeric)
    rep["n_categorical"] = len(cats)
    return rep


def format_report(rep: dict) -> list[str]:
    lines = [
        (
            f"coverage gate: {rep['n_columns']} model-visible columns "
            f"({rep['n_numeric']} numeric + {rep['n_categorical']} categorical)"
        ),
        f"  {len(rep['failures'])} failure(s), {len(rep['warnings'])} report-only, {len(rep['waived'])} waived",
    ]
    for title, key in (("FAILURES", "failures"), ("WAIVED", "waived"), ("REPORT-ONLY", "warnings")):
        if not rep[key]:
            continue
        lines += ["", f"  {title}:"]
        for r in rep[key]:
            where = f"@{r['airport']} {r['period']}" if r["airport"] != ALL_AIRPORTS else ""
            suffix = f"   [waived: {r['waiver_reason']}]" if "waiver_reason" in r else ""
            lines.append(f"    {r['rule']:<16} {r['column']:<26} {where:<18} {r['detail']}{suffix}")
    return lines


def gate(
    train: pl.DataFrame, rank: pl.DataFrame, cfg: TrainConfig, allow: dict | None = None, strict: bool = True
) -> dict:
    """`check_coverage`, printed, raising on a failure when `strict`."""
    rep = check_coverage(train, rank, cfg, allow=allow)
    print("\n".join(format_report(rep)))
    if strict and rep["failures"]:
        raise SystemExit(f"coverage gate: {len(rep['failures'])} failure(s) - see the list above")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="recipe whose column visibility the gate should use")
    ap.add_argument("--train", type=Path, default=Path("data/work/feat_train.parquet"))
    ap.add_argument("--rank", type=Path, default=Path("data/work/feat_rank.parquet"))
    ap.add_argument("--out", type=Path, help="write the JSON report here (nothing is written without this)")
    ap.add_argument("--no-strict", action="store_true", help="report only; do not exit 1 on a failure")
    a = ap.parse_args(argv)

    cfg = TrainConfig()
    allow: dict = {}
    if a.config:
        spec = yaml.safe_load(a.config.read_text())
        fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
        if "params" in fields:
            fields["params"] = {**TrainConfig().params, **fields["params"]}
        for k in ("folds", "exclude_features", "include_candidates", "unmatched_sched_reparam_airports"):
            if isinstance(fields.get(k), list):
                fields[k] = tuple(fields[k])
        cfg = TrainConfig(**fields)
        allow = dict(spec.get("coverage_allow") or {})

    rep = check_coverage(pl.read_parquet(a.train), pl.read_parquet(a.rank), cfg, allow=allow)
    print("\n".join(format_report(rep)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"\nwrote {a.out}")
    return 1 if rep["failures"] and not a.no_strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
