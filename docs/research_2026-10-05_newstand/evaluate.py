"""Verdicts 1 and 2 of docs/research_2026-10-05_newstand/plan.md: the simulation arms on the unseen-stand rows.

    .venv/bin/python docs/research_2026-10-05_newstand/evaluate.py <ns_base_u_id> <ns_drop_u_id>

Read-only. Unseen-stand rows = forward-fold validation rows whose (airport, stand) falls in unseen_stand_mask(0.2),
the stands both simulation models never saw in training. Verdict 3 is `prc.compare ns_base ns_drop`."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from prc.compare import robustness_curve
from prc.cv import month_block_folds, year_month_key
from prc.train import LABEL, PROXY, unseen_stand_mask

SHARE = 0.2


def main(argv: list[str]) -> int:
    base_id, drop_id = argv[:2]
    cols = ["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt"]
    feats = pl.read_parquet("data/work/feat_train.parquet", columns=cols)
    months = year_month_key(feats)
    fold = next(f for f in month_block_folds(months.unique().to_list()) if f.name == "forward")
    va = feats.filter(months.is_in(list(fold.valid_months)) & pl.col(LABEL).is_not_null())
    for name, eid in (("base_u", base_id), ("drop_u", drop_id)):
        va = va.join(pl.read_parquet(f"experiments/oof_{eid}.parquet").rename({"oof": name}), on="MVT_ID_mvt")
    va = va.drop_nulls(["base_u", "drop_u"])
    unseen = unseen_stand_mask(va, SHARE)
    y = va[LABEL].to_numpy().astype(float)
    a, b = va["base_u"].to_numpy().astype(float), va["drop_u"].to_numpy().astype(float)
    matched = va[PROXY].is_not_null().to_numpy()
    r = lambda p, m: float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    print(f"forward validation rows {va.height:,}; unseen-stand rows {unseen.sum():,} ({unseen.mean():.1%})")
    verdict = {}
    for label, m in (("unseen stands", unseen), ("unseen, matched", unseen & matched), ("seen stands", ~unseen)):
        c = robustness_curve(y[m], a[m], b[m], drop_levels=(0, 20, 200))
        cells = "  ".join(f"K={x['dropped']}: {x['diff']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]" for x in c)
        print(f"\n## {label}: n={m.sum():,}  RMSE base_u {r(a, m):.2f} -> drop_u {r(b, m):.2f}\n  {cells}")
        if label == "unseen stands":
            verdict = {x["dropped"]: x for x in c}
    ok1 = verdict[200]["ci95"][1] < 0
    ok2 = verdict[0]["diff"] <= 0
    print(f"\nverdict 1 (unseen, K=200 CI excludes 0 on the improving side): {'PASS' if ok1 else 'FAIL'}")
    print(f"verdict 2 (unseen, K=0 diff <= 0): {'PASS' if ok2 else 'FAIL'}")
    print("\n## unseen stands by airport (RMSE base_u -> drop_u)")
    apt = va["ADEP_mvt"].to_numpy()
    for ap in sorted(set(apt[unseen])):
        m = unseen & (apt == ap)
        print(f"  {ap}: n={m.sum():6,}  {r(a, m):7.1f} -> {r(b, m):7.1f}  ({r(b, m) - r(a, m):+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
