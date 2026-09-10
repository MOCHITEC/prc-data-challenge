"""Error anatomy: where the squared error actually sits (docs/06).

    python -m prc.anatomy                      # uses experiments/oof_<latest>.parquet
    python -m prc.anatomy --oof path.parquet

Writes docs/error_anatomy_generated.md and prints the two tables that decide what to build next: the share of
squared error per clock source, and the oracle gain from routing each source to its known candidate value."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .clocks import CANDIDATE_VALUE, CANDIDATES, GENUINE, MATCH_TOL_S, ROLLOVER, with_candidates, with_clock_source
from .profile import _md_table
from .train import LABEL

FEAT_TRAIN = Path("data/work/feat_train.parquet")
FEAT_RANK = Path("data/work/feat_rank.parquet")
NEEDED = [
    "MVT_ID_mvt",
    LABEL,
    "ADEP_mvt",
    "nm_matched",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    *CANDIDATES.values(),
]


def latest_oof(exp_dir: Path = Path("experiments")) -> Path:
    cands = sorted(exp_dir.glob("oof_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no oof_*.parquet in {exp_dir}; run the pipeline first")
    return cands[-1]


def load_joined(oof_path: Path, feat_path: Path = FEAT_TRAIN) -> pl.DataFrame:
    """OOF predictions joined to the features they were made from, labelled rows only."""
    cols = sorted(set(NEEDED))
    feats = with_candidates(pl.read_parquet(feat_path, columns=cols))
    oof = pl.read_parquet(oof_path)
    pred_col = "oof" if "oof" in oof.columns else "pred"
    oof = oof.select(["MVT_ID_mvt", pl.col(pred_col).alias("pred")]).drop_nulls("pred")
    j = oof.join(feats, on="MVT_ID_mvt", how="inner").drop_nulls(LABEL)
    j = with_clock_source(j, MATCH_TOL_S).rename({"clock_src": "src"})
    return j.with_columns(y=pl.col(LABEL).cast(pl.Float64)).with_columns(se=(pl.col("y") - pl.col("pred")) ** 2)


def clock_copy_rates(feat_path: Path = FEAT_TRAIN) -> list[dict]:
    """How often BLOCK_TIME is an exact copy of each candidate clock (docs/06 section 1)."""
    f = pl.read_parquet(feat_path, columns=sorted(set(NEEDED))).drop_nulls(LABEL)
    n = f.height
    rows = []
    for name, col in CANDIDATES.items():
        d = (f["BLOCK_TIME_UTC_mvt"] - f[col]).dt.total_seconds().abs()
        row = {"clock": f"{name} ({col})", "non_null_pct": 100 * (1 - d.null_count() / n)}
        for tol in (0, 1, 5, 60, 300):
            row[f"|d|<={tol}s %"] = 100 * float((d <= tol).sum()) / n
        rows.append(row)
    return rows


def source_mix(feat_path: Path = FEAT_TRAIN) -> list[dict]:
    f = with_clock_source(pl.read_parquet(feat_path, columns=sorted(set(NEEDED))).drop_nulls(LABEL), MATCH_TOL_S)
    f = f.rename({"clock_src": "src"}).with_columns(y=pl.col(LABEL).cast(pl.Float64))
    n = f.height
    return (
        f.group_by("src")
        .agg(
            n=pl.len(),
            pct=(100 * pl.len() / n),
            med_y=pl.col("y").median(),
            sd_y=pl.col("y").std(),
            max_y=pl.col("y").max(),
        )
        .sort("n", descending=True)
        .to_dicts()
    )


def sse_by_source(j: pl.DataFrame) -> tuple[list[dict], dict]:
    """Share of squared error per clock source, plus the oracle-routing RMSE (docs/06 section 3)."""
    tot = float(j["se"].sum())
    exact = pl.coalesce([pl.when(pl.col("src") == name).then(pl.col(col)) for name, col in CANDIDATE_VALUE.items()])
    g = j.with_columns(exact=exact)
    g = g.with_columns(se_oracle=(pl.col("y") - pl.col("exact")) ** 2)
    rows = (
        g.group_by("src")
        .agg(
            n=pl.len(),
            pct_rows=(100 * pl.len() / g.height),
            pct_sse=(100 * pl.col("se").sum() / tot),
            model_rmse=pl.col("se").mean().sqrt(),
            oracle_rmse=pl.col("se_oracle").mean().sqrt(),
        )
        .sort("pct_sse", descending=True)
        .to_dicts()
    )
    routed = g.select(
        pl.when(pl.col("src").is_in(list(CANDIDATES))).then(pl.col("exact")).otherwise(pl.col("pred")).alias("m")
    )["m"].to_numpy()
    y = g["y"].to_numpy()
    summary = {
        "n": g.height,
        "model_rmse": float(np.sqrt(np.mean((y - g["pred"].to_numpy()) ** 2))),
        "oracle_routed_rmse": float(np.sqrt(np.nanmean((y - routed) ** 2))),
    }
    return rows, summary


def concentration(j: pl.DataFrame) -> list[dict]:
    se = np.sort(j["se"].to_numpy())[::-1]
    tot = se.sum()
    return [
        {"top_k_rows": k, "pct_of_sse": 100 * se[:k].sum() / tot, "pct_of_rows": 100 * k / len(se)}
        for k in (20, 50, 100, 200, 500, 1000, 5000)
        if k <= len(se)
    ]


def by_airport(j: pl.DataFrame) -> list[dict]:
    tot = float(j["se"].sum())
    return (
        j.group_by("ADEP_mvt")
        .agg(
            n=pl.len(),
            pct_sse=(100 * pl.col("se").sum() / tot),
            rmse=pl.col("se").mean().sqrt(),
            no_nm_pct=(100 * (pl.col("nm_matched") == 0).mean()),
            y_gt3h_pct=(100 * (pl.col("y") > 10800).mean()),
        )
        .sort("pct_sse", descending=True)
        .to_dicts()
    )


def by_nm(j: pl.DataFrame) -> list[dict]:
    tot = float(j["se"].sum())
    return (
        j.group_by("nm_matched")
        .agg(
            n=pl.len(),
            pct_rows=(100 * pl.len() / j.height),
            pct_sse=(100 * pl.col("se").sum() / tot),
            rmse=pl.col("se").mean().sqrt(),
            med_y=pl.col("y").median(),
            p99_y=pl.col("y").quantile(0.99),
        )
        .sort("nm_matched")
        .to_dicts()
    )


def rank_side(feat_path: Path = FEAT_RANK) -> list[dict]:
    """The ranking frame cannot be labelled, but its NM coverage decides how much the no-NM layer matters."""
    if not feat_path.exists():
        return []
    r = pl.read_parquet(feat_path, columns=["ADEP_mvt", "nm_matched"])
    return (
        r.group_by("ADEP_mvt")
        .agg(n=pl.len(), no_nm_pct=(100 * (pl.col("nm_matched") == 0).mean()))
        .sort("no_nm_pct", descending=True)
        .to_dicts()
    )


def build_report(oof_path: Path) -> str:
    j = load_joined(oof_path)
    src_rows, summary = sse_by_source(j)
    out = [
        "# Error anatomy (generated)",
        "",
        (
            f"Source: `{oof_path}` joined to `{FEAT_TRAIN}`. Labelled OOF rows: {summary['n']:,}. "
            f"Model RMSE **{summary['model_rmse']:.2f}**. Conclusions live in `docs/06_error_anatomy.md`."
        ),
        "",
        "## BLOCK_TIME as a copy of a known clock",
        "",
        _md_table(clock_copy_rates()),
        "## Clock-source mix (all labelled rows)",
        "",
        _md_table(source_mix()),
        f"## Squared error by clock source (tolerance +-{MATCH_TOL_S}s)",
        "",
        _md_table(src_rows),
        (
            f"Oracle routing (known candidate for a matched source, current model for `{GENUINE}`/`{ROLLOVER}`): "
            f"**RMSE {summary['oracle_routed_rmse']:.2f}** vs {summary['model_rmse']:.2f} now."
        ),
        "",
        "## Concentration",
        "",
        _md_table(concentration(j)),
        "## By airport",
        "",
        _md_table(by_airport(j)),
        "## By Network Manager coverage",
        "",
        _md_table(by_nm(j)),
        "## Ranking-side NM coverage",
        "",
        _md_table(rank_side()),
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path, default=None, help="OOF parquet (default: newest in experiments/)")
    ap.add_argument("--out", type=Path, default=Path("docs/error_anatomy_generated.md"))
    args = ap.parse_args(argv)
    oof = args.oof or latest_oof()
    report = build_report(oof)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
