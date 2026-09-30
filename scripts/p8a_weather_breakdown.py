"""P8-A: cut the paired weather on/off OOF frames the way the plan's §2.2 asks for.

    .venv/bin/python scripts/p8a_weather_breakdown.py \
        --exp-a p8_wx_off-<hash> --exp-b p8_wx_on-<hash>

`make compare A=... B=...` already reports the adopt/reject triple and the tail structure, and none of that
is re-implemented here. This script reports only what the verdict does not look at, and what P8-A's plan
calls the deliverable rather than the headline: the same paired difference cut per fold, per month inside
`forward`, per `deicing_gate`, per airport within the gate hours, per matched/unmatched lane, and re-weighted
to the composition of the set we are actually scored on.

Negative is B (weather on) better, throughout - the same sign convention as `prc.compare`.

Read the re-weighting line before the pooled one. The scored set is 44.3% January / 55.7% July (152,719 /
192,122 rows). `forward` validates on Nov+Dec 2025 and is the only honest winter proxy we can build - there
is no `jan_fwd` fold and there cannot be one (cv.py:39-42) - while `jul_fwd` validates on Jul 2025. A pooled
CV delta over the two folds weights them by their row counts, which is not the composition of the scored
set, and a de-icing effect measured on a winter fold applies to at most 44.3% of the scored rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from prc.compare import group_diff, load_pair, robustness_curve, verdict
from prc.cv import rmse

SCORED_WEIGHTS = {"forward": 0.443, "jul_fwd": 0.557}  # 152,719 / 344,841 and 192,122 / 344,841


def _paired_boot_diff(y, p_a, p_b, strata=None, n_boot=500, seed=0, weights=None):
    """RMSE(B) - RMSE(A) with a paired row bootstrap, resampling inside `strata` when given.

    Row-level and therefore optimistic about the CI - rows inside a month share weather and schedule. It is
    used here only for the cuts `prc.compare` does not produce; the adopt/reject statistic is unchanged and
    comes from `prc.compare`."""
    se_a, se_b = (y - p_a) ** 2, (y - p_b) ** 2
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    groups = [np.arange(len(y))] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    point = float(np.sqrt(np.average(se_b, weights=w)) - np.sqrt(np.average(se_a, weights=w)))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        diffs[i] = np.sqrt(np.average(se_b[pick], weights=w[pick])) - np.sqrt(np.average(se_a[pick], weights=w[pick]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": len(y), "diff": point, "ci95": [float(lo), float(hi)], "b_better_prob": float((diffs < 0).mean())}


def _row(label, r, rmse_a=None, rmse_b=None):
    a = "" if rmse_a is None else f"{rmse_a:9.2f}"
    b = "" if rmse_b is None else f"{rmse_b:9.2f}"
    return (
        f"  {label:<34} n={r['n']:>7,} {a:>9} {b:>9} "
        f"{r['diff']:+8.2f} [{r['ci95'][0]:+8.2f},{r['ci95'][1]:+8.2f}]  P(B)={r['b_better_prob']:.3f}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-a", required=True, help="baseline: weather OFF")
    ap.add_argument("--exp-b", required=True, help="candidate: weather ON")
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)

    d = load_pair(a.exp_a, a.exp_b, a.work_dir, a.exp_dir)
    # Fold membership is written per experiment; both arms ran the same fold definitions, so either file
    # serves. Taking A's and asserting B's agrees is cheaper than trusting they do.
    folds = pl.read_parquet(a.exp_dir / f"oof_{a.exp_a}_folds.parquet")
    folds_b = pl.read_parquet(a.exp_dir / f"oof_{a.exp_b}_folds.parquet")
    if not folds.sort("MVT_ID_mvt").equals(folds_b.sort("MVT_ID_mvt")):
        raise SystemExit("the two arms did not run the same folds - the comparison is not paired")
    # `deicing_gate` is a weather column, so it exists only in the weather-on feature frame. The gate cut is
    # a property of the row's hour, not of either model, which is exactly why it can be read from one side.
    gate = pl.read_parquet(a.work_dir / "feat_train.parquet", columns=["MVT_ID_mvt", "deicing_gate"])
    d = d.join(folds, on="MVT_ID_mvt", how="left").join(gate, on="MVT_ID_mvt", how="left")
    unfolded = int(d["fold"].is_null().sum())
    if unfolded:
        print(f"note: {unfolded:,} scored rows carry no fold label and are dropped from the per-fold cuts")
        d = d.filter(pl.col("fold").is_not_null())

    y = d["TAXITIME_SEC_mvt"].to_numpy().astype(float)
    p_a, p_b = d["p_a"].to_numpy(), d["p_b"].to_numpy()
    fold = d["fold"].to_numpy()
    airport = d["ADEP_mvt"].to_numpy()
    ym = (d["MVT_TIME_UTC_mvt"].dt.year() * 100 + d["MVT_TIME_UTC_mvt"].dt.month()).to_numpy()
    matched = d["mvt_minus_aobt3"].is_not_null().to_numpy()
    gate1 = (d["deicing_gate"].fill_null(0) == 1).to_numpy()

    print(f"A (weather OFF) = {a.exp_a}\nB (weather ON)  = {a.exp_b}\nrows = {len(y):,}")
    print(f"\n{'cut':<36} {'n':>9} {'A':>9} {'B':>9} {'diff':>8} {'CI95':>20}")

    print("\nper fold (never pooled: a pooled number mixes a regime that exists with one that does not)")
    for f in sorted(set(fold.tolist())):
        m = fold == f
        r = _paired_boot_diff(y[m], p_a[m], p_b[m], n_boot=a.n_boot)
        print(_row(f"fold {f}", r, rmse(y[m], p_a[m]), rmse(y[m], p_b[m])))
        c = robustness_curve(y[m], p_a[m], p_b[m], n_boot=a.n_boot)
        ok, why = verdict(c)
        k200 = next((x for x in c if x["dropped"] == 200), None)
        if k200:
            print(
                f"      K=200 drop: {k200['diff']:+.2f} [{k200['ci95'][0]:+.2f}, {k200['ci95'][1]:+.2f}]"
                f"   -> {'ADOPT' if ok else 'REJECT'}: {why}"
            )

    print("\nper month inside forward")
    for m_key in sorted(set(ym[fold == "forward"].tolist())):
        m = (fold == "forward") & (ym == m_key)
        print(_row(f"{m_key}", group_diff(y, p_a, p_b, m, n_boot=a.n_boot), rmse(y[m], p_a[m]), rmse(y[m], p_b[m])))

    print("\nper de-icing gate (this is where the old -8.00 s gate-row figure lived)")
    for name, m in (("gate=0", ~gate1), ("gate=1", gate1)):
        if m.sum():
            print(_row(name, group_diff(y, p_a, p_b, m, n_boot=a.n_boot), rmse(y[m], p_a[m]), rmse(y[m], p_b[m])))
    print("\n  gate=1, per airport (the old pair was not uniform: EDDF -22.9 ... EHAM +6.7, LTFM +19.0)")
    for ap_name in sorted(set(airport[gate1].tolist())):
        m = gate1 & (airport == ap_name)
        if m.sum() >= 20:
            print(_row(f"  gate=1 @{ap_name}", group_diff(y, p_a, p_b, m, n_boot=a.n_boot)))

    print("\nper lane")
    for name, m in (("matched", matched), ("unmatched", ~matched)):
        if m.sum():
            print(_row(name, group_diff(y, p_a, p_b, m, n_boot=a.n_boot), rmse(y[m], p_a[m]), rmse(y[m], p_b[m])))

    print("\npooled over both folds, and the same thing re-weighted to the scored composition")
    pooled = _paired_boot_diff(y, p_a, p_b, strata=fold, n_boot=a.n_boot)
    print(_row("pooled (row counts as they fall)", pooled, rmse(y, p_a), rmse(y, p_b)))
    w = np.ones(len(y))
    for f, target in SCORED_WEIGHTS.items():
        m = fold == f
        if m.sum():
            w[m] = target / m.sum()
    rw = _paired_boot_diff(y, p_a, p_b, strata=fold, n_boot=a.n_boot, weights=w)
    print(
        _row(
            "re-weighted 44.3% Jan / 55.7% Jul",
            rw,
            float(np.sqrt(np.average((y - p_a) ** 2, weights=w))),
            float(np.sqrt(np.average((y - p_b) ** 2, weights=w))),
        )
    )
    print(
        "\n  The re-weighted line treats `forward` (Nov+Dec 2025) as the January proxy and `jul_fwd` "
        "(Jul 2025)\n  as the July proxy, because no honest January fold exists. It is an estimate of the "
        "scored-set\n  effect, not a measurement of it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
