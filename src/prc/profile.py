"""Data profiler: answers the open questions in docs/01 section 7 from the real files, in one pass.

Run right after the official parquet files land:  python -m prc.profile
Writes docs/data_profile.md and prints the three numbers that decide the model design (docs/04 section 4):
the ranking set's AOBT_3 coverage, the information content of that clock, and the reporting-failure rates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .config import data_dir
from .schema import ALL_COLS, TIMESTAMP_COLS

OFFICIAL_TRAINING_MOVEMENTS = 4_167_797  # docs/01 section 2.1
EXPECTED_RANKING_DEPARTURES = 344_841


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def file_inventory(raw_dir: Path) -> tuple[list[dict], list[str]]:
    """Row counts, parquet timestamp unit / UTC flag and schema drift across files (docs/01 U13)."""
    rows, notes = [], []
    files = sorted(raw_dir.rglob("*.parquet"))
    first_schema = None
    for f in files:
        md = pq.read_metadata(f)
        schema = pq.read_schema(f)
        if first_schema is None:
            first_schema = schema
        elif schema.names != first_schema.names:
            notes.append(
                f"schema names differ in {f.name}: "
                f"+{sorted(set(schema.names) - set(first_schema.names))} "
                f"-{sorted(set(first_schema.names) - set(schema.names))}"
            )
        ts_info = []
        for name in TIMESTAMP_COLS:
            if name in schema.names:
                t = schema.field(name).type
                ts_info.append(f"{name}:{t}")
        rows.append(
            {
                "file": f.name,
                "rows": md.num_rows,
                "size_mb": round(f.stat().st_size / 1e6, 1),
                "timestamps": "; ".join(ts_info[:2]) or "-",
            }
        )
    if first_schema is not None:
        missing = sorted(set(ALL_COLS) - set(first_schema.names))
        extra = sorted(set(first_schema.names) - set(ALL_COLS))
        if missing:
            notes.append(f"columns documented but absent: {missing}")
        if extra:
            notes.append(f"columns present but undocumented: {extra}")
        notes.append(
            "timestamp types: "
            + ", ".join(f"{n}={first_schema.field(n).type}" for n in TIMESTAMP_COLS if n in first_schema.names)
        )
    return rows, notes


def load(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    train = pl.concat([pl.read_parquet(p) for p in sorted(raw_dir.rglob("training_*.parquet"))], how="diagonal")
    rank_files = sorted(raw_dir.rglob("ranking.parquet"))
    rank = pl.read_parquet(rank_files[0]) if rank_files else pl.DataFrame()
    sub_files = sorted(raw_dir.rglob("submitting.parquet"))
    sub = pl.read_parquet(sub_files[0]) if sub_files else None
    for df in (train, rank):
        for c in TIMESTAMP_COLS:
            if c in df.columns and df[c].dtype == pl.Datetime(time_zone="UTC"):
                df = df.with_columns(pl.col(c).dt.replace_time_zone(None))
    return train, rank, sub


def _null_rates(df: pl.DataFrame, cols: list[str]) -> list[dict]:
    n = max(1, df.height)
    return [{"column": c, "null_rate": df[c].null_count() / n} for c in cols if c in df.columns]


def target_profile(dep: pl.DataFrame) -> tuple[dict, list[dict]]:
    y = dep["TAXITIME_SEC_mvt"]
    n = max(1, dep.height)
    overall = {
        "rows": dep.height,
        "null_rate": y.null_count() / n,
        "negative_rate": float((y < 0).sum() or 0) / n,
        "gt_2h_rate": float((y > 7200).sum() or 0) / n,
        "p01": y.quantile(0.01),
        "p10": y.quantile(0.10),
        "median": y.median(),
        "p90": y.quantile(0.90),
        "p99": y.quantile(0.99),
        "max": y.max(),
        "sd": y.std(),
    }
    by_airport = (
        dep.group_by("ADEP_mvt")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("TAXITIME_SEC_mvt").median().alias("median"),
                pl.col("TAXITIME_SEC_mvt").std().alias("sd"),
                pl.col("TAXITIME_SEC_mvt").quantile(0.99).alias("p99"),
                (pl.col("TAXITIME_SEC_mvt") > 7200).mean().alias("gt_2h_rate"),
                pl.col("TAXITIME_SEC_mvt").is_null().mean().alias("null_rate"),
            ]
        )
        .sort("ADEP_mvt")
        .to_dicts()
    )
    return overall, by_airport


def clock_profile(dep: pl.DataFrame) -> list[dict]:
    """How much the NM off-block clock tells us (docs/01 U1). A tiny spread of MVT - AOBT_3 would mean AOBT_3 is
    itself derived from the take-off time and carries no independent information."""
    d = dep.filter(pl.col("AOBT_3_flt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    if not d.height:
        return []
    d = d.with_columns(
        [
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("block_minus_aobt3"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("mvt_minus_aobt3"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().alias("block_minus_lobt"),
        ]
    )
    return (
        d.group_by("ADEP_mvt")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("block_minus_aobt3").median().alias("med_block_minus_aobt3"),
                (pl.col("block_minus_aobt3").abs() <= 60).mean().alias("aobt3_within_60s"),
                pl.col("mvt_minus_aobt3").std().alias("sd_mvt_minus_aobt3"),
                pl.col("block_minus_lobt").median().alias("med_block_minus_lobt"),
                (pl.col("block_minus_lobt").abs() <= 300).mean().alias("lobt_within_5min"),
            ]
        )
        .sort("ADEP_mvt")
        .to_dicts()
    )


def regime_profile(dep: pl.DataFrame) -> list[dict]:
    """Reporting-failure regimes: block time reported as the schedule, or a whole-day offset (docs/03 section 8).

    The day-offset columns measure the label, not BLOCK - SCHED. An earlier version tested (BLOCK - SCHED) +
    86400, which is a different phenomenon and the wrong sign besides: what actually drives the RMSE tail is a
    TAXITIME of about a day (docs/07), i.e. BLOCK and MVT landing on different dates."""
    d = dep.filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null() & pl.col("SCHED_TIME_UTC_mvt").is_not_null())
    if not d.height:
        return []
    d = d.with_columns((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("bs"))
    return (
        d.group_by("ADEP_mvt")
        .agg(
            [
                pl.len().alias("rows"),
                (pl.col("bs").abs() <= 1).mean().alias("block_eq_sched_rate"),
                # A day offset in either direction: BLOCK ahead of SCHED by a day is as plausible as behind.
                ((pl.col("bs").abs() - 86400).abs() <= 60).mean().alias("sched_day_offset_rate"),
                # The tail that dominates the metric: a taxi time of a day or more.
                (pl.col("TAXITIME_SEC_mvt") >= 86400 - 3600).mean().alias("day_offset_rate"),
                pl.col("FLIGHT_ID_mvt").is_null().mean().alias("nm_unmatched_rate"),
                ((pl.col("TAXITIME_SEC_mvt") > 14400).mean()).alias("gt_4h_rate"),
            ]
        )
        .sort("ADEP_mvt")
        .to_dicts()
    )


def coverage_profile(train_dep: pl.DataFrame, rank_dep: pl.DataFrame) -> list[dict]:
    """Share of ranking rows whose category level was seen in training (unseen stands appear in July 2026 at
    EDDF/EDDM per docs/03 section 9, so a hierarchical fallback is mandatory)."""
    out = []
    for col in ("STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADES_mvt", "FLIGHT_mvt"):
        if col not in train_dep.columns or col not in rank_dep.columns:
            continue
        seen = set(train_dep[col].drop_nulls().unique().to_list())
        vals = rank_dep[col].to_list()
        known = sum(1 for v in vals if v in seen)
        out.append(
            {
                "column": col,
                "levels_train": len(seen),
                "levels_rank": rank_dep[col].n_unique(),
                "rank_rows_covered": known / max(1, len(vals)),
            }
        )
    return out


def baselines(train_dep: pl.DataFrame) -> list[dict]:
    """In-sample RMSE of predictors any team can reproduce - the yardstick for the first submissions."""
    d = train_dep.filter(pl.col("TAXITIME_SEC_mvt").is_not_null())
    if not d.height:
        return []
    y = d["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    res = [
        {"predictor": "global mean", "rmse": float(np.sqrt(np.mean((y - y.mean()) ** 2)))},
        {"predictor": "global median", "rmse": float(np.sqrt(np.mean((y - np.median(y)) ** 2)))},
    ]
    for keys, name in (
        (["ADEP_mvt"], "airport mean"),
        (["ADEP_mvt", "RUNWAY_mvt"], "airport x runway mean"),
        (["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"], "airport x runway x stand mean"),
    ):
        g = d.group_by(keys).agg(pl.col("TAXITIME_SEC_mvt").mean().alias("_p"))
        j = d.join(g, on=keys, how="left")
        p = j["_p"].fill_null(float(y.mean())).to_numpy().astype(float)
        res.append(
            {
                "predictor": name,
                "rmse": float(np.sqrt(np.mean((j["TAXITIME_SEC_mvt"].to_numpy().astype(float) - p) ** 2))),
            }
        )
    nm = d.filter(pl.col("AOBT_3_flt").is_not_null())
    if nm.height:
        p = (nm["MVT_TIME_UTC_mvt"] - nm["AOBT_3_flt"]).dt.total_seconds().to_numpy().astype(float)
        yy = nm["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        res.append(
            {
                "predictor": f"MVT - AOBT_3 (NM rows, {nm.height / d.height:.1%} of rows)",
                "rmse": float(np.sqrt(np.mean((yy - p) ** 2))),
            }
        )
    return res


def build_report(raw_dir: Path) -> str:
    inv, notes = file_inventory(raw_dir)
    train, rank, sub = load(raw_dir)
    t_dep = train.filter(pl.col("PHASE_mvt") == "DEP")
    r_dep = rank.filter(pl.col("PHASE_mvt") == "DEP") if rank.height else pl.DataFrame()
    parts: list[str] = [
        "# データプロファイル\n",
        "`python -m prc.profile` の出力。docs/01 §7 の未確認事項に対する実測の回答。\n",
    ]
    parts += ["## 1. ファイル構成と型（U13）\n", _md_table(inv), "\n"]
    parts += ["\n".join(f"- {n}" for n in notes) + "\n"]

    parts += ["\n## 2. 行数・一意性・空港（U5, U7）\n"]
    total = train.height
    parts.append(
        _md_table(
            [
                {
                    "metric": "training rows",
                    "value": total,
                    "note": f"公式記載 {OFFICIAL_TRAINING_MOVEMENTS:,} との差 {total - OFFICIAL_TRAINING_MOVEMENTS:+,}",
                },
                {
                    "metric": "unique MVT_ID (training)",
                    "value": train["MVT_ID_mvt"].n_unique(),
                    "note": "行数と一致すべき",
                },
                {"metric": "training departures", "value": t_dep.height, "note": ""},
                {"metric": "ranking rows", "value": rank.height, "note": ""},
                {
                    "metric": "ranking departures",
                    "value": r_dep.height if r_dep.height else 0,
                    "note": f"期待 {EXPECTED_RANKING_DEPARTURES:,}（API の usedPairs）",
                },
                {"metric": "submitting rows", "value": sub.height if sub is not None else 0, "note": ""},
                {
                    "metric": "departure airports (training)",
                    "value": t_dep["ADEP_mvt"].n_unique(),
                    "note": ", ".join(sorted(x for x in t_dep["ADEP_mvt"].unique().to_list() if x)),
                },
            ],
            ["metric", "value", "note"],
        )
    )

    parts += ["\n## 3. 目的変数（U6）\n"]
    overall, by_apt = target_profile(t_dep)
    parts.append(_md_table([{"metric": k, "value": v} for k, v in overall.items()], ["metric", "value"]))
    parts += ["\n空港別:\n", _md_table(by_apt)]
    ident = t_dep.filter(pl.col("TAXITIME_SEC_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
    if ident.height:
        diff = (
            (ident["MVT_TIME_UTC_mvt"] - ident["BLOCK_TIME_UTC_mvt"]).dt.total_seconds() - ident["TAXITIME_SEC_mvt"]
        ).abs()
        parts.append(f"\n定義の整合: `|(MVT − BLOCK) − TAXITIME| <= 1s` の割合 = {float((diff <= 1).mean()):.6f}\n")

    parts += ["\n## 4. Ranking で何が残っているか（U1, U11）\n"]
    cols = [
        "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "TAXITIME_SEC_mvt",
        "SCHED_TIME_UTC_mvt",
        "RUNWAY_mvt",
        "STAND_mvt",
        "FLIGHT_ID_mvt",
        "AOBT_3_flt",
        "LOBT_flt",
        "EOBT_1_flt",
        "IOBT_flt",
        "ARVT_3_flt",
    ]
    if r_dep.height:
        rank_rates = {r["column"]: r["null_rate"] for r in _null_rates(r_dep, cols)}
        train_rates = {r["column"]: r["null_rate"] for r in _null_rates(t_dep, cols)}
        parts.append(
            _md_table(
                [
                    {
                        "column": c,
                        "null_rate_ranking_dep": rank_rates.get(c),
                        "null_rate_training_dep": train_rates.get(c),
                    }
                    for c in cols
                ]
            )
        )
    parts += ["\n## 5. NM の時計の情報量（U1）\n", _md_table(clock_profile(t_dep))]
    parts += ["\n## 6. 報告エラー体制（尾部の主因）\n", _md_table(regime_profile(t_dep))]
    if r_dep.height:
        parts += ["\n## 7. カテゴリ被覆率（未知スタンド対策）\n", _md_table(coverage_profile(t_dep, r_dep))]
        arr_ids = set(rank.filter(pl.col("PHASE_mvt") == "ARR")["FLIGHT_ID_mvt"].drop_nulls().to_list())
        dep_ids = r_dep["FLIGHT_ID_mvt"].to_list()
        share = sum(1 for i in dep_ids if i is not None and i in arr_ids) / max(1, len(dep_ids))
        parts.append(f"\n域内便（同一 FLIGHT_ID の到着行が Ranking 内にある出発便）の割合: {share:.4f}（U1b）\n")
    parts += ["\n## 8. 素朴な予測器のベンチマーク（2025 in-sample）\n", _md_table(baselines(t_dep))]
    parts += ["\n## 9. 便名・コールサインの匿名化（U8）\n"]
    samp = t_dep.select(
        [c for c in ("FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt") if c in t_dep.columns]
    ).head(10)
    parts.append("```\n" + str(samp) + "\n```\n")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=data_dir() / "raw")
    ap.add_argument("--out", type=Path, default=Path("docs/data_profile.md"))
    a = ap.parse_args(argv)
    report = build_report(a.raw_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(report)
    print(report[:4000])
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
