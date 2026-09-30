"""The verdicts for the C1/C3 arms, under the repository's decision rule.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/c1_c3_verdicts.py

The rule is `prc.compare.verdict` and nothing here changes it: adopt only when the full-data 95% CI and the
CI with the worst 200 rows removed both exclude zero on the improving side (docs/features_log.md, corrected
2026-09-25). Every pair below is run through `prc.compare`'s own functions rather than a re-implementation,
so what is printed is what `make compare` would print.

Each candidate is read twice: against the base arm, which is the production column set, and against an
arity-matched noise arm, because adding any column to this model is worth +/-0.5 s on its own
(docs/features_log.md, the three-loss analysis of 2026-09-15). The second reading is the honest one; the
first is reported because the decision rule is written about it.

Read-only. Reads experiments/oof_*.parquet and data/work_c1c3/feat_train.parquet; writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prc.compare import load_pair, month_bootstrap_rmse_diff, robustness_curve, verdict
from prc.cv import rmse, year_month_key
from prc.train import LABEL, PROXY

# docs/competitor_research_2026-09-21/results/model-a-ceiling.md §8.2, n = 1.
D_BOARD_D_OOF = 0.747

PAIRS = [
    ("c13_base", "c13_noise3", "what three signal-free columns cost"),
    ("c13_base", "c13_c1", "C1 against the production column set"),
    ("c13_noise3", "c13_c1", "C1 against three noise columns - the honest reading"),
    ("c13_base", "c13_noise4", "what four signal-free columns cost"),
    ("c13_base", "c13_c3wind", "C3 against the production column set"),
    ("c13_noise4", "c13_c3wind", "C3 against four noise columns - the honest reading"),
    ("c13_base", "c13_gust", "the gust repair against the production column set"),
    ("c13_noise4", "c13_gust", "the gust repair against four noise columns"),
]


def exp_ids(log: Path) -> dict[str, str]:
    """The most recent experiment id per config name."""
    out = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["config"]["name"]] = rec["exp_id"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path("data/work_c1c3"))
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()

    ids = exp_ids(a.exp_dir / "experiments.jsonl")
    for name, exp in sorted(ids.items()):
        if name.startswith("c13_"):
            print(f"# {name:<12} {exp}")
    print()

    for a_name, b_name, why in PAIRS:
        if a_name not in ids or b_name not in ids:
            print(f"{a_name} -> {b_name}: missing an arm, skipped")
            continue
        d = load_pair(ids[a_name], ids[b_name], a.work_dir, a.exp_dir)
        y = d[LABEL].to_numpy().astype(float)
        p_a, p_b = d["p_a"].to_numpy(), d["p_b"].to_numpy()
        curve = robustness_curve(y, p_a, p_b, n_boot=a.n_boot)
        mb = month_bootstrap_rmse_diff(y, p_a, p_b, year_month_key(d).to_numpy(), n_boot=a.n_boot)
        ok, why_v = verdict(curve)
        matched = d[PROXY].is_not_null().to_numpy()
        lane_a = rmse(y[matched], p_b[matched]) - rmse(y[matched], p_a[matched])
        full = next(r for r in curve if r["dropped"] == 0)

        print("=" * 100)
        print(f"A = {a_name} ({ids[a_name]})   B = {b_name} ({ids[b_name]})   n = {len(y):,}")
        print(f"    {why}")
        print(f"{'dropped':>8} {'A':>10} {'B':>10} {'diff':>8} {'CI95':>22} {'P(B)':>6}")
        for r in curve:
            ci = f"[{r['ci95'][0]:+7.3f},{r['ci95'][1]:+7.3f}]"
            print(
                f"{r['dropped']:>8} {r['rmse_a']:10.4f} {r['rmse_b']:10.4f} {r['diff']:+8.3f} {ci:>22} "
                f"{r['b_better_prob']:6.3f}"
            )
        print(
            f"    month bootstrap ({mb['n_months']} months): {mb['diff_rmse']:+.3f} "
            f"[{mb['ci95'][0]:+.3f}, {mb['ci95'][1]:+.3f}]  P(B better) = {mb['b_better_prob']:.3f}"
        )
        print(f"    lane A (matched, n={int(matched.sum()):,}): {lane_a:+.3f} s")
        print(f"    implied board at dBoard/dOOF = {D_BOARD_D_OOF}: {D_BOARD_D_OOF * full['diff']:+.3f} s")
        print(f"    VERDICT: {'ADOPT' if ok else 'REJECT'} - {why_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
