"""The two-column C1 on `jul_fwd`, alone and pooled with the forward fold (C1-two-column-remeasured.md §9).

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/c1_2col_julfwd_verdicts.py

Sibling of `c1_2col_verdicts.py`, which is untouched. Same rule (`prc.compare.verdict`), same functions, same
500 resamples. The forward arms (`c13_base`, `c13_c1_2col`, `c13_noise2`) and their `jul_fwd` twins
(`*_jf`) are separate experiments because each ran one fold; the two folds validate disjoint months (Nov-Dec
and July), so the pooled OOF is their concatenation - which is what a single `folds: [forward, jul_fwd]` run
would write, since `run_cv` fits each fold independently and seeds by `cfg.seed + k`, not by fold.

Also prints, per fold, the harness checks pre-registered in §9.0 (H1: base reproduces `p8e_base`'s jul_fwd
fold; H2: lane B identical between base and C1) and the lane-A effect of each arm against base.

Read-only. Reads experiments/*.jsonl, experiments/oof_*.parquet and data/work_c1c3/feat_train.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from prc.compare import month_bootstrap_rmse_diff, robustness_curve, verdict
from prc.cv import rmse, year_month_key
from prc.train import LABEL, PROXY

ARMS = ("c13_base", "c13_c1_2col", "c13_noise2")
PAIRS = [
    ("c13_base", "c13_noise2", "what two signal-free columns cost"),
    ("c13_base", "c13_c1_2col", "two-column C1 against the production column set - the rule's comparison"),
    ("c13_noise2", "c13_c1_2col", "two-column C1 against two noise columns - the arity-matched reading"),
]


def records(log: Path) -> dict[str, dict]:
    out = {}
    for line in log.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["config"]["name"]] = rec
    return out


def oof(exp_id: str, exp_dir: Path, col: str) -> pl.DataFrame:
    return pl.read_parquet(exp_dir / f"oof_{exp_id}.parquet").rename({"oof": col})


def frame(ids: dict[str, list[str]], work_dir: Path, exp_dir: Path) -> pl.DataFrame:
    """Labels joined to every arm's OOF, with each arm's folds concatenated."""
    cols = ["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt"]
    d = pl.read_parquet(work_dir / "feat_train.parquet", columns=cols)
    for arm, exps in ids.items():
        parts = [oof(e, exp_dir, arm) for e in exps]
        o = pl.concat(parts)
        assert o["MVT_ID_mvt"].is_unique().all(), f"{arm}: folds overlap"
        d = d.join(o, on="MVT_ID_mvt", how="inner")
    return d.filter(pl.col(LABEL).is_not_null() & pl.all_horizontal(pl.col(a).is_not_null() for a in ids))


def report(d: pl.DataFrame, label: str, n_boot: int) -> None:
    y = d[LABEL].to_numpy().astype(float)
    matched = d[PROXY].is_not_null().to_numpy()
    month = year_month_key(d).to_numpy()
    print("#" * 104)
    print(
        f"# {label}: n = {len(y):,} (lane A {int(matched.sum()):,}, lane B {int((~matched).sum()):,}), "
        f"months {sorted(set(month.tolist()))}"
    )
    for arm in ARMS:
        p = d[arm].to_numpy()
        print(
            f"#   {arm:<12} full {rmse(y, p):.4f}  lane A {rmse(y[matched], p[matched]):.4f}  "
            f"lane B {rmse(y[~matched], p[~matched]):.4f}"
        )
    for a_name, b_name, why in PAIRS:
        p_a, p_b = d[a_name].to_numpy(), d[b_name].to_numpy()
        curve = robustness_curve(y, p_a, p_b, n_boot=n_boot)
        mb = month_bootstrap_rmse_diff(y, p_a, p_b, month, n_boot=n_boot)
        ok, why_v = verdict(curve)
        lane_a = rmse(y[matched], p_b[matched]) - rmse(y[matched], p_a[matched])
        lane_b = rmse(y[~matched], p_b[~matched]) - rmse(y[~matched], p_a[~matched])
        n_b_diff = int((p_a[~matched] != p_b[~matched]).sum())
        print("=" * 104)
        print(f"[{label}] A = {a_name}   B = {b_name}   ({why})")
        print(f"{'dropped':>8} {'A':>10} {'B':>10} {'diff':>8} {'CI95':>22} {'P(B)':>6}")
        for r in curve:
            ci = f"[{r['ci95'][0]:+7.3f},{r['ci95'][1]:+7.3f}]"
            print(
                f"{r['dropped']:>8} {r['rmse_a']:10.4f} {r['rmse_b']:10.4f} {r['diff']:+8.3f} {ci:>22} "
                f"{r['b_better_prob']:6.3f}"
            )
        if mb["n_months"]:
            print(
                f"    month bootstrap ({mb['n_months']} months): {mb['diff_rmse']:+.3f} "
                f"[{mb['ci95'][0]:+.3f}, {mb['ci95'][1]:+.3f}]  P(B better) = {mb['b_better_prob']:.3f}"
            )
        else:
            print("    month bootstrap: n/a (one month)")
        print(
            f"    lane A: {lane_a:+.3f} s   lane B: {lane_b:+.4f} s ({n_b_diff} of {int((~matched).sum())} "
            f"lane-B rows differ)"
        )
        print(f"    VERDICT: {'ADOPT' if ok else 'REJECT'} - {why_v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path("data/work_c1c3"))
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()

    recs = records(a.exp_dir / "experiments.jsonl")
    fwd = {arm: recs[arm]["exp_id"] for arm in ARMS}
    jf = {arm: recs[f"{arm}_jf"]["exp_id"] for arm in ARMS}
    for arm in ARMS:
        print(f"# {arm:<12} forward {fwd[arm]:<26} jul_fwd {jf[arm]}")

    # Harness checks (§9.0 H1, H2), from the runs' own records.
    print()
    ref = recs["p8e_base"]["folds"]["jul_fwd"]
    for arm in ARMS:
        f = recs[f"{arm}_jf"]["folds"]["jul_fwd"]
        print(
            f"# jul_fwd {arm:<12} full {f['rmse_full']:.6f}  A {f['matched']['rmse_full']:.6f}  "
            f"B {f['unmatched']['rmse_full']:.6f}  rounds A/B {f['rounds_matched']}/{f['rounds_unmatched']}  "
            f"n {f['n']:,}  train {f['train_rows']:,}  n_feat_B {f.get('n_features_unmatched')}"
        )
    print(
        f"# jul_fwd p8e_base     full {ref['rmse_full']:.6f}  A {ref['matched']['rmse_full']:.6f}  "
        f"B {ref['unmatched']['rmse_full']:.6f}  rounds A/B {ref['rounds_matched']}/{ref['rounds_unmatched']}"
    )
    print()

    d_j = frame({arm: [jf[arm]] for arm in ARMS}, a.work_dir, a.exp_dir)
    report(d_j, "jul_fwd", a.n_boot)
    d_f = frame({arm: [fwd[arm]] for arm in ARMS}, a.work_dir, a.exp_dir)
    report(d_f, "forward (recomputed, must match results/c1-2col-verdicts.txt)", a.n_boot)
    d_p = frame({arm: [fwd[arm], jf[arm]] for arm in ARMS}, a.work_dir, a.exp_dir)
    assert d_p.height == d_j.height + d_f.height
    report(d_p, "pooled [forward, jul_fwd]", a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
