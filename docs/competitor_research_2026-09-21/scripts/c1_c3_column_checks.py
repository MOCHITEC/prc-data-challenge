"""What C1's and C3's columns actually contain, and whether they clear the P8-D coverage gate.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/c1_c3_column_checks.py \
        [--work data/work_c1c3]

Sections, each of which prints the numbers the results note cites:

  K1  C1's mode table: how many (airport, runway) keys, and on how many of them a table fitted per frame
      would disagree with the table fitted on the union. This is the construction requirement the note
      states explicitly, priced on the real data rather than asserted.
  K2  C1's three columns: the minute histogram and the spike ratio per airport (delay-mechanisms.md §2.3),
      the share of rows at the mode, and the null and non-zero shares the coverage gate reads.
  K3  C3's columns: `drct` coverage overall, per station and per scored month, and the same null/non-zero
      shares. The note's caveat is 10.0% null overall and 30.3% at LSZH.
  K4  The coverage gate itself (src/prc/coverage.py), run three times: on the base arm's column set, on the
      base plus C1, and on the base plus C3 and the gust repair. The base count is the control - this gate
      already fails on five identically-zero columns that predate both candidates (P8-D), so a raw failure
      count means nothing without it.

Read-only: nothing is written to outputs/, data/, src/ or configs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from prc.coverage import check_coverage
from prc.features import PROXY_MODE_COLS, proxy_mode_table
from prc.train import TrainConfig

WIND = ["wind_dir_deg", "headwind_kt", "crosswind_kt", "rwy_wind_misalign"]
GUST = ["gust_filled", "gust_excess", "gust_filled_prev1h", "gust_excess_prev1h"]
BASE_EXCLUDE = tuple(WIND + GUST)


def _pct(x: float, n: int) -> str:
    return f"{100.0 * x / n:.2f}%"


def K1(raw_train: pl.DataFrame, raw_rank: pl.DataFrame) -> None:
    union = proxy_mode_table([raw_train, raw_rank])
    tr_only = proxy_mode_table([raw_train])
    rk_only = proxy_mode_table([raw_rank])
    j = (
        union.rename({"_proxy_mode_min": "union"})
        .join(tr_only.rename({"_proxy_mode_min": "train_only"}), on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
        .join(rk_only.rename({"_proxy_mode_min": "rank_only"}), on=["ADEP_mvt", "RUNWAY_mvt"], how="left")
    )
    served = j.filter(pl.col("rank_only").is_not_null())
    print(f"K1 mode table: {union.height} (airport, runway) keys on the union of train and rank")
    print(f"K1   keys also present in the ranking frame: {served.height}")
    print(
        f"K1   a per-frame fit would put a DIFFERENT minute behind the same column name on "
        f"{served.filter(pl.col('union') != pl.col('rank_only')).height} of them"
    )
    print(f"K1   train-only vs union disagree on {j.filter(pl.col('union') != pl.col('train_only')).height} keys")
    print(served.filter(pl.col("union") != pl.col("rank_only")).head(30))


def K2(tr: pl.DataFrame, rk: pl.DataFrame) -> None:
    for nm, f in (("train", tr), ("rank", rk)):
        n = f.height
        print(f"\nK2 {nm}: {n:,} rows")
        for c in PROXY_MODE_COLS:
            nz = int((f[c].fill_null(0) != 0).sum())
            print(f"K2   {c:<22} null {_pct(f[c].null_count(), n):>7}   non-zero {_pct(nz, n):>7}")
    print("\nK2 per airport, on the training frame: the modal minute, its share, and the spike ratio")
    print("K2 (spike = rows in the modal minute / median rows in the ten neighbouring minute bins)")
    a = tr.filter(pl.col("mvt_minus_aobt3").is_not_null()).with_columns(
        (pl.col("mvt_minus_aobt3") // 60).cast(pl.Int64).alias("_pm")
    )
    rows = []
    for apt in sorted(a["ADEP_mvt"].unique().drop_nulls().to_list()):
        g = a.filter(pl.col("ADEP_mvt") == apt)
        h = g.group_by("_pm").len().sort("len", descending=True)
        mode_min, mode_n = h.row(0)[0], h.row(0)[1]
        counts = dict(g.group_by("_pm").len().iter_rows())
        nbrs = [counts.get(mode_min + d, 0) for d in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]
        nbrs.sort()
        med = (nbrs[4] + nbrs[5]) / 2
        share = g.filter(pl.col("_pm") == mode_min).height / g.height
        at_mode = g["proxy_at_mode"].sum()
        rows.append((apt, mode_min, mode_n, share, mode_n / med if med else float("inf"), at_mode / g.height))
    print(f"{'ADEP':>6} {'mode_min':>9} {'rows':>9} {'share':>7} {'spike':>8} {'proxy_at_mode':>14}")
    for r in sorted(rows, key=lambda r: -r[4]):
        print(f"{r[0]:>6} {r[1]:>9} {r[2]:>9,} {r[3]:>7.1%} {r[4]:>8.1f}x {r[5]:>14.1%}")


def K3(tr: pl.DataFrame, rk: pl.DataFrame) -> None:
    for nm, f in (("train", tr), ("rank", rk)):
        n = f.height
        print(f"\nK3 {nm}: {n:,} rows")
        for c in WIND + GUST:
            if c not in f.columns:
                print(f"K3   {c:<22} ABSENT from the frame")
                continue
            nz = int((f[c].fill_null(0) != 0).sum())
            print(f"K3   {c:<22} null {_pct(f[c].null_count(), n):>7}   non-zero {_pct(nz, n):>7}")
    print("\nK3 wind_dir_deg null share per airport (train / each scored period)")
    per = []
    for f, tag in ((tr, "train-2025"), (rk, "rank")):
        d = f.with_columns(
            (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("_p")
        )
        keys = ["ADEP_mvt"] if tag == "train-2025" else ["ADEP_mvt", "_p"]
        g = d.group_by(keys).agg(
            pl.len().alias("n"), (100.0 * pl.col("wind_dir_deg").is_null().mean()).alias("null_pct")
        )
        for row in g.sort(keys).iter_rows(named=True):
            per.append((tag if tag == "train-2025" else str(row["_p"]), row["ADEP_mvt"], row["n"], row["null_pct"]))
    print(f"{'period':>10} {'ADEP':>6} {'n':>9} {'null%':>7}")
    for p, apt, n, q in sorted(per, key=lambda r: (r[1], r[0])):
        print(f"{p:>10} {apt:>6} {n:>9,} {q:>7.1f}")


def K4(tr: pl.DataFrame, rk: pl.DataFrame) -> None:
    base = TrainConfig(include_candidates=("stand_arrival_gap",), exclude_features=BASE_EXCLUDE)
    arms = {
        "base (control: what already fails)": base,
        "base + C1": TrainConfig(
            include_candidates=("stand_arrival_gap", *PROXY_MODE_COLS), exclude_features=BASE_EXCLUDE
        ),
        "base + C3 wind": TrainConfig(include_candidates=("stand_arrival_gap",), exclude_features=tuple(GUST)),
        "base + gust repair": TrainConfig(include_candidates=("stand_arrival_gap",), exclude_features=tuple(WIND)),
        "base + C1 + C3 + gust": TrainConfig(include_candidates=("stand_arrival_gap", *PROXY_MODE_COLS)),
    }
    new = set(PROXY_MODE_COLS) | set(WIND) | set(GUST)
    for name, cfg in arms.items():
        rep = check_coverage(tr, rk, cfg)
        mine = [f for f in rep["failures"] if f["column"] in new]
        print(
            f"\nK4 {name}: {rep['n_columns']} model-visible columns, "
            f"{len(rep['failures'])} failure(s), {len(mine)} of them on a new column"
        )
        for f in rep["failures"]:
            mark = "NEW " if f["column"] in new else "    "
            print(f"K4   {mark}{f['rule']:<16} {f['column']:<24} @{f['airport']} {f['period']}  {f['detail']}")
        for w in rep["warnings"]:
            if w["column"] in new:
                print(f"K4   NEW {w['rule']:<16} {w['column']:<24} @{w['airport']} {w['period']}  {w['detail']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("data/work_c1c3"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--sections", nargs="*", default=["K1", "K2", "K3", "K4"])
    a = ap.parse_args()

    if "K1" in a.sections:
        cols = ["PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
        tr_raw = pl.concat(
            [pl.read_parquet(p, columns=cols) for p in sorted(a.raw.glob("training_*.parquet"))], how="diagonal"
        )
        K1(tr_raw, pl.read_parquet(a.raw / "ranking.parquet", columns=cols))
        del tr_raw

    if set(a.sections) & {"K2", "K3", "K4"}:
        tr = pl.read_parquet(a.work / "feat_train.parquet")
        rk = pl.read_parquet(a.work / "feat_rank.parquet")
        for s in ("K2", "K3", "K4"):
            if s in a.sections:
                print("\n" + "=" * 28 + f" {s} " + "=" * 28)
                globals()[s](tr, rk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
