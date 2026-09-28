"""P10 verdicts: CatBoost against LightGBM, per cell of the 2x2, pooled and per fold, lane by lane.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/p10_verdicts.py

The rule is `prc.compare.verdict` and nothing here changes it (full-data CI and K=200 CI both excluding zero on
the same side). Every pair goes through `prc.compare`'s own functions. B is always the second arm named.

Per pair, three scopes: pooled over the two folds' rows, `forward` (Nov-Dec) alone, `jul_fwd` (July) alone - the
two folds' validation months do not overlap, so each OOF row belongs to exactly one fold. Per scope: the drop
curve and the verdict; the month bootstrap (pooled only - one fold of one month has none); lane A and lane B as
separate paired differences with CIs; and lane B's Kish effective sample size of the squared error, because a
lane whose error is carried by a handful of rows cannot decide a family comparison and should be seen not to.

Plus the 2x2 interaction, (C+E - C-E) - (L+E - L-E), with a paired row bootstrap over all four OOF vectors.

Read-only. Reads experiments/oof_*.parquet and data/work/feat_train.parquet; writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from prc.compare import group_diff, month_bootstrap_rmse_diff, robustness_curve, verdict
from prc.cv import year_month_key
from prc.train import LABEL, PROXY

D_BOARD_D_OOF = 0.747  # results/model-a-ceiling.md §8.2, n = 1
SCOPES = {"pooled": None, "forward": (202511, 202512), "jul_fwd": (202507,)}

PAIRS = [
    ("p10_lgb_ne", "p10_lgb_ne_s43", "CONTROL. LightGBM seed 42 -> seed 43, no encodings: the refit redraw"),
    ("p10_lgb_ne", "p10_cb_ne", "FAMILY, P2-invariant cell: LightGBM -> CatBoost, no encodings"),
    ("p10_lgb_e", "p10_cb_e", "FAMILY, production cell: LightGBM -> CatBoost, with encodings"),
    ("p10_lgb_ne", "p10_lgb_e", "ENCODINGS for LightGBM: off -> on"),
    ("p10_cb_ne", "p10_cb_e", "ENCODINGS for CatBoost: off -> on"),
    ("p10_lgb_e", "p10_cb_ne", "CROSS: production LightGBM -> CatBoost with no encodings"),
    ("p10_cb_ne", "p10_cb_ne_ctr4", "CTR: CatBoost max_ctr_complexity 1 -> 4, no encodings"),
]


def exp_ids(log: Path) -> dict[str, str]:
    out = {}
    for line in log.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["config"]["name"]] = rec["exp_id"]
    return out


def kish(se: np.ndarray) -> tuple[float, float]:
    """(effective sample size, share of the largest row) of a vector of squared errors."""
    s = float(se.sum())
    return (s * s / float((se * se).sum()) if s > 0 else float("nan")), (float(se.max()) / s if s > 0 else float("nan"))


def load(names: list[str], ids: dict[str, str], work: Path, exp: Path) -> pl.DataFrame:
    d = pl.read_parquet(
        work / "feat_train.parquet", columns=["MVT_ID_mvt", LABEL, PROXY, "MVT_TIME_UTC_mvt", "ADEP_mvt"]
    )
    for n in names:
        o = pl.read_parquet(exp / f"oof_{ids[n]}.parquet").rename({"oof": n})
        d = d.join(o, on="MVT_ID_mvt", how="inner")
    d = d.filter(pl.col(LABEL).is_not_null() & pl.all_horizontal([pl.col(n).is_not_null() for n in names]))
    return d.with_columns(year_month_key(d).alias("ym"))


def report_pair(d: pl.DataFrame, a: str, b: str, n_boot: int) -> dict:
    out = {}
    for scope, months in SCOPES.items():
        s = d if months is None else d.filter(pl.col("ym").is_in(list(months)))
        if s.height == 0:
            continue
        y = s[LABEL].to_numpy().astype(float)
        pa, pb = s[a].to_numpy(), s[b].to_numpy()
        curve = robustness_curve(y, pa, pb, n_boot=n_boot)
        ok, why = verdict(curve)
        m = s[PROXY].is_not_null().to_numpy()
        la, lb = group_diff(y, pa, pb, m, n_boot=n_boot), group_diff(y, pa, pb, ~m, n_boot=n_boot)
        mb = month_bootstrap_rmse_diff(y, pa, pb, s["ym"].to_numpy(), n_boot=n_boot) if months is None else None
        out[scope] = {
            "n": len(y),
            "curve": curve,
            "verdict": ("ADOPT" if ok else "REJECT", why),
            "month_boot": mb,
            "lane_a": la,
            "lane_b": lb,
            "kish_b": {"A": kish((y[~m] - pa[~m]) ** 2), "B": kish((y[~m] - pb[~m]) ** 2)},
        }
    return out


def interaction(d: pl.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """(C+E - C-E) - (L+E - L-E) in RMSE, per scope and lane, with a paired row bootstrap. Negative = the
    encodings help CatBoost more than they help LightGBM."""
    arms = ["p10_lgb_ne", "p10_lgb_e", "p10_cb_ne", "p10_cb_e"]
    out = {}
    for scope, months in SCOPES.items():
        s = d if months is None else d.filter(pl.col("ym").is_in(list(months)))
        y = s[LABEL].to_numpy().astype(float)
        m = s[PROXY].is_not_null().to_numpy()
        for lane, sel in (("all", np.ones(len(y), bool)), ("lane_a", m), ("lane_b", ~m)):
            se = {k: (y[sel] - s[k].to_numpy()[sel]) ** 2 for k in arms}

            def stat(idx, se=se):
                r = {k: np.sqrt(v[idx].mean()) for k, v in se.items()}
                return (r["p10_cb_e"] - r["p10_cb_ne"]) - (r["p10_lgb_e"] - r["p10_lgb_ne"])

            n = int(sel.sum())
            rng = np.random.default_rng(seed)
            boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
            r = {k: float(np.sqrt(v.mean())) for k, v in se.items()}
            out[f"{scope}/{lane}"] = {
                "n": n,
                "rmse": r,
                "enc_effect_lgb": r["p10_lgb_e"] - r["p10_lgb_ne"],
                "enc_effect_cb": r["p10_cb_e"] - r["p10_cb_ne"],
                "interaction": stat(np.arange(n)),
                "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            }
    return out


def fmt_ci(r: dict) -> str:
    return f"{r['diff']:+8.3f} [{r['ci95'][0]:+8.3f}, {r['ci95'][1]:+8.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--json", type=Path, default=None, help="also dump everything as JSON here")
    a = ap.parse_args()
    ids = exp_ids(a.exp_dir / "experiments.jsonl")
    ids = {k: v for k, v in ids.items() if k.startswith("p10_")}
    for k in sorted(ids):
        print(f"# {k:<16} {ids[k]}")
    dump = {"ids": ids, "pairs": {}}
    for arm_a, arm_b, why in PAIRS:
        if arm_a not in ids or arm_b not in ids:
            print(f"\n{arm_a} -> {arm_b}: missing an arm, skipped")
            continue
        d = load([arm_a, arm_b], ids, a.work_dir, a.exp_dir)
        rep = report_pair(d, arm_a, arm_b, a.n_boot)
        dump["pairs"][f"{arm_a}->{arm_b}"] = rep
        print("\n" + "=" * 110)
        print(f"A = {arm_a} ({ids[arm_a]})   B = {arm_b} ({ids[arm_b]})")
        print(f"    {why}")
        for scope, r in rep.items():
            print(f"  --- {scope}: n = {r['n']:,}")
            print(f"  {'dropped':>8} {'A':>10} {'B':>10} {'diff':>8} {'CI95':>22} {'P(B)':>6}")
            for c in r["curve"]:
                ci = f"[{c['ci95'][0]:+7.3f},{c['ci95'][1]:+7.3f}]"
                print(
                    f"  {c['dropped']:>8} {c['rmse_a']:10.4f} {c['rmse_b']:10.4f} {c['diff']:+8.3f} {ci:>22} "
                    f"{c['b_better_prob']:6.3f}"
                )
            if r["month_boot"] is not None:
                mb = r["month_boot"]
                print(
                    f"      month bootstrap ({mb['n_months']} months): {mb['diff_rmse']:+.3f} "
                    f"[{mb['ci95'][0]:+.3f}, {mb['ci95'][1]:+.3f}]  P(B better) = {mb['b_better_prob']:.3f}"
                )
            la, lb = r["lane_a"], r["lane_b"]
            print(f"      lane A n={la['n']:>7,}  A={la['rmse_a']:9.4f}  B={la['rmse_b']:9.4f}  {fmt_ci(la)}")
            print(f"      lane B n={lb['n']:>7,}  A={lb['rmse_a']:9.4f}  B={lb['rmse_b']:9.4f}  {fmt_ci(lb)}")
            ka, kb = r["kish_b"]["A"], r["kish_b"]["B"]
            print(
                f"      lane B Kish ESS of squared error: A {ka[0]:.2f} (top row {ka[1]:.1%}), "
                f"B {kb[0]:.2f} (top row {kb[1]:.1%})"
            )
            full = next(c for c in r["curve"] if c["dropped"] == 0)
            print(f"      implied board at dBoard/dOOF = {D_BOARD_D_OOF}: {D_BOARD_D_OOF * full['diff']:+.3f} s")
            print(f"      VERDICT: {r['verdict'][0]} - {r['verdict'][1]}")
    four = ["p10_lgb_ne", "p10_lgb_e", "p10_cb_ne", "p10_cb_e"]
    if all(k in ids for k in four):
        d = load(four, ids, a.work_dir, a.exp_dir)
        inter = interaction(d, a.n_boot)
        dump["interaction"] = inter
        print("\n" + "=" * 110)
        print("2x2 INTERACTION  (C+E - C-E) - (L+E - L-E); negative = encodings help CatBoost more")
        print(
            f"  {'scope/lane':<18} {'n':>8} {'L-E':>9} {'L+E':>9} {'C-E':>9} {'C+E':>9} {'enc(L)':>8} {'enc(C)':>8} {'inter':>8} {'CI95':>20}"
        )
        for k, v in inter.items():
            r = v["rmse"]
            print(
                f"  {k:<18} {v['n']:>8,} {r['p10_lgb_ne']:9.3f} {r['p10_lgb_e']:9.3f} {r['p10_cb_ne']:9.3f} "
                f"{r['p10_cb_e']:9.3f} {v['enc_effect_lgb']:+8.3f} {v['enc_effect_cb']:+8.3f} {v['interaction']:+8.3f} "
                f"[{v['ci95'][0]:+7.3f},{v['ci95'][1]:+7.3f}]"
            )
    if a.json:
        a.json.write_text(json.dumps(dump, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
