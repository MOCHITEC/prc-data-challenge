"""P8-B: cut the three-arm `wxcodes` comparison on the rows the repair is supposed to act on.

    .venv/bin/python scripts/p8b_wxcodes_breakdown.py \
        --base p8b_base-<hash> --noise p8b_noise-<hash> --wx p8b_wx-<hash>

`make compare` gives the adopt/reject triple for a pair and is not re-implemented. This adds the three cuts
that decide what the repair actually did:

  * **base vs noise** - the price of arity alone, four signal-free columns. Anything the repair wins that is
    smaller than this is the reshuffle, not the weather.
  * **wx vs noise** - the arity-controlled question, and the one to quote.
  * the same difference on the rows the new columns are non-zero on (`wet_wx`, `shower_wx`), on the rows the
    repaired `deicing_gate` fires on, and per fold - because the whole argument for the repair is that it is
    live in July, where every column the old gate depended on is flat.

Negative = the second arm better, throughout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from prc.compare import group_diff, load_pair, robustness_curve, verdict
from prc.cv import rmse

SCORED_WEIGHTS = {"forward": 0.443, "jul_fwd": 0.557}
CUT_COLS = ("wet_wx", "shower_wx", "obscuration_wx", "deicing_gate")


def _fmt(label, r, extra=""):
    return (
        f"  {label:<40} n={r['n']:>7,} A={r['rmse_a']:9.2f} B={r['rmse_b']:9.2f} "
        f"{r['diff']:+8.2f} [{r['ci95'][0]:+8.2f},{r['ci95'][1]:+8.2f}]  P(B)={r['b_better_prob']:.3f}{extra}"
    )


def _pair(name_a, name_b, exp_a, exp_b, work, exp_dir, cuts, n_boot):
    d = load_pair(exp_a, exp_b, work, exp_dir)
    folds = pl.read_parquet(exp_dir / f"oof_{exp_a}_folds.parquet")
    d = d.join(folds, on="MVT_ID_mvt", how="left").join(cuts, on="MVT_ID_mvt", how="left")
    d = d.filter(pl.col("fold").is_not_null())
    y = d["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    p_a, p_b = d["p_a"].to_numpy(), d["p_b"].to_numpy()
    fold = d["fold"].to_numpy()

    print(f"\n{'=' * 110}\nA = {name_a} ({exp_a})\nB = {name_b} ({exp_b})   rows = {len(y):,}")
    curve = robustness_curve(y, p_a, p_b, n_boot=n_boot)
    for r in curve:
        print(
            f"  K={r['dropped']:<5} A={r['rmse_a']:9.2f} B={r['rmse_b']:9.2f} {r['diff']:+8.2f} "
            f"[{r['ci95'][0]:+7.2f},{r['ci95'][1]:+7.2f}]  P(B)={r['b_better_prob']:.3f}"
        )
    ok, why = verdict(curve)
    print(f"  VERDICT: {'ADOPT' if ok else 'REJECT'} - {why}")

    print("  per fold:")
    for f in sorted(set(fold.tolist())):
        m = fold == f
        print(_fmt(f"fold {f}", group_diff(y, p_a, p_b, m, n_boot=n_boot)))
    print("  matched lane per fold:")
    matched = d["mvt_minus_aobt3"].is_not_null().to_numpy()
    for f in sorted(set(fold.tolist())):
        m = (fold == f) & matched
        print(_fmt(f"fold {f} matched", group_diff(y, p_a, p_b, m, n_boot=n_boot)))
    print("  rows the repaired columns are live on (both folds pooled):")
    for c in CUT_COLS:
        if c not in d.columns:
            continue
        m = (d[c].fill_null(0) == 1).to_numpy()
        if m.sum() >= 50:
            print(_fmt(f"{c} = 1", group_diff(y, p_a, p_b, m, n_boot=n_boot)))
    w = np.ones(len(y))
    for f, target in SCORED_WEIGHTS.items():
        m = fold == f
        if m.sum():
            w[m] = target / m.sum()
    ra = float(np.sqrt(np.average((y - p_a) ** 2, weights=w)))
    rb = float(np.sqrt(np.average((y - p_b) ** 2, weights=w)))
    print(f"  re-weighted 44.3% Jan / 55.7% Jul: A={ra:.2f} B={rb:.2f}  {rb - ra:+.2f}")
    print(f"  pooled                           : A={rmse(y, p_a):.2f} B={rmse(y, p_b):.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--wx", required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument(
        "--wx-work-dir",
        type=Path,
        default=Path("data/work_p8bwx"),
        help="the repaired feature frame, which is where the wet_wx / shower_wx cut columns live",
    )
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)

    have = [c for c in CUT_COLS if c in pl.read_parquet(a.wx_work_dir / "feat_train.parquet", n_rows=1).columns]
    cuts = pl.read_parquet(a.wx_work_dir / "feat_train.parquet", columns=["MVT_ID_mvt", *have])
    for name_a, ea, name_b, eb in (
        ("base", a.base, "noise (4 signal-free columns)", a.noise),
        ("base", a.base, "wxcodes repair", a.wx),
        ("noise (arity-matched control)", a.noise, "wxcodes repair", a.wx),
    ):
        _pair(name_a, name_b, ea, eb, a.work_dir, a.exp_dir, cuts, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
