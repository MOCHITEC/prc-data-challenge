"""The verdicts for the two-column C1 arm, and a control that prices the K=200 leg itself.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/c1_2col_verdicts.py

Sibling of `c1_c3_verdicts.py`, which is left exactly as it was so `results/c1-c3-verdicts.txt` still
reproduces verbatim. The rule is still `prc.compare.verdict` and nothing here changes it: adopt only when the
full-data 95% CI and the CI with the worst 200 rows removed both exclude zero on the improving side. Every
pair is run through `prc.compare`'s own functions rather than a re-implementation.

Three blocks:

  A. the two-column C1 (`proxy_at_mode`, `proxy_minus_mode_min`) against the base arm and against its
     arity-matched control, `c13_noise2` - two noise columns, because a two-column candidate read against a
     three-column control is measuring the column count as well as the column;
  B. the three-column C1 re-read beside it on the same fold, so the two versions are directly comparable,
     and the two C1 arms against each other, which is what `proxy_sec_in_min` is worth;
  C. `c13_noise3` against `c13_noise4` - two arms whose only difference is which signal-free hash columns
     they were shown. Whatever this pair reads is what the instrument returns for a pure refit redraw, so it
     is the scale against which every number above has to be read. It is a control on the *instrument*, not
     on a feature, and it needs no new fit: both arms are already in experiments/.

Read-only. Reads experiments/oof_*.parquet and data/work_c1c3/feat_train.parquet; writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prc.compare import load_pair, month_bootstrap_rmse_diff, robustness_curve, verdict
from prc.cv import rmse, year_month_key
from prc.train import LABEL, PROXY

# docs/competitor_research_2026-09-21/results/model-a-ceiling.md §8.2, n = 1.
D_BOARD_D_OOF = 0.747

PAIRS = [
    # A. the two-column C1
    ("c13_base", "c13_noise2", "A1. what two signal-free columns cost"),
    ("c13_base", "c13_c1_2col", "A2. two-column C1 against the production column set - the rule's comparison"),
    ("c13_noise2", "c13_c1_2col", "A3. two-column C1 against two noise columns - the arity-matched reading"),
    # B. beside the three-column version, on the same fold and the same frame
    ("c13_base", "c13_c1", "B1. three-column C1 against the base (the 2026-09-25 reading, recomputed)"),
    ("c13_noise3", "c13_c1", "B2. three-column C1 against three noise columns"),
    ("c13_c1", "c13_c1_2col", "B3. dropping proxy_sec_in_min: B is the two-column arm"),
    # C. the instrument's own control
    ("c13_noise3", "c13_noise4", "C1. noise against noise - what the instrument returns for a pure redraw"),
    ("c13_noise2", "c13_noise3", "C2. the same, two noise columns against three"),
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
            print(f"# {name:<14} {exp}")
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

        print("=" * 104)
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
