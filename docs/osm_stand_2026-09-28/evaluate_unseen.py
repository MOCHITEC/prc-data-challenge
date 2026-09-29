"""Verdicts 1 and 2 of docs/osm_stand_2026-09-28/plan.md §3: the simulation arms, read on the unseen-stand rows.

    .venv/bin/python docs/osm_stand_2026-09-28/evaluate_unseen.py <base_u_id> <noise1_u_id> <dist_u_id>

Read-only. The unseen-stand rows are the forward-fold validation rows whose (airport, stand) falls in the held-out
share: the exact rows whose stands the three simulation models never saw in training (same CRC32 draw)."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from prc.compare import robustness_curve, verdict
from prc.cv import month_block_folds, year_month_key
from prc.osm import add_stand_runway_distance
from prc.train import LABEL, PROXY, _osm_tables, unseen_stand_mask

SHARE = 0.2


def curve_line(name: str, y, a, b) -> None:
    c = robustness_curve(y, a, b, drop_levels=(0, 20, 200))
    ok, why = verdict(c)
    cells = "  ".join(f"K={r['dropped']}: {r['diff']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]" for r in c)
    print(f"  {name:22s} n={len(y):6,}  {cells}  -> {'ADOPT' if ok else 'REJECT'} ({why[:60]})")


def main(argv: list[str]) -> int:
    base_id, noise_id, dist_id = argv[:3]
    cols = ["MVT_ID_mvt", LABEL, PROXY, "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
    feats = pl.read_parquet("data/work/feat_train.parquet", columns=cols)
    months = year_month_key(feats)
    fold = next(f for f in month_block_folds(months.unique().to_list()) if f.name == "forward")
    va = feats.filter(months.is_in(list(fold.valid_months)) & pl.col(LABEL).is_not_null())
    for name, eid in (("base_u", base_id), ("noise1_u", noise_id), ("dist_u", dist_id)):
        oof = pl.read_parquet(f"experiments/oof_{eid}.parquet").rename({"oof": name})
        va = va.join(oof, on="MVT_ID_mvt", how="inner")
    va = va.drop_nulls(["base_u", "noise1_u", "dist_u"])
    stands, ends = _osm_tables()
    va = add_stand_runway_distance(va, stands, ends)
    unseen = unseen_stand_mask(va, SHARE)
    y = va[LABEL].to_numpy().astype(float)
    p = {k: va[k].to_numpy().astype(float) for k in ("base_u", "noise1_u", "dist_u")}
    matched = va[PROXY].is_not_null().to_numpy()
    has_dist = va["osm_stand_rwy_km"].is_not_null().to_numpy()
    print(
        f"forward validation rows {va.height:,}; unseen-stand rows {unseen.sum():,} ({unseen.mean():.1%}), "
        f"of which OSM-matched {has_dist[unseen].mean():.1%}"
    )
    for label, m in (
        ("unseen stands", unseen),
        ("unseen, matched lane", unseen & matched),
        ("unseen, OSM-matched", unseen & has_dist),
        ("seen stands", ~unseen),
    ):
        print(
            f"\n## {label}: RMSE base_u {np.sqrt(np.mean((y[m] - p['base_u'][m]) ** 2)):.2f}  noise1_u "
            f"{np.sqrt(np.mean((y[m] - p['noise1_u'][m]) ** 2)):.2f}  dist_u {np.sqrt(np.mean((y[m] - p['dist_u'][m]) ** 2)):.2f}"
        )
        curve_line("base_u -> dist_u", y[m], p["base_u"][m], p["dist_u"][m])
        curve_line("noise1_u -> dist_u", y[m], p["noise1_u"][m], p["dist_u"][m])
        curve_line("base_u -> noise1_u", y[m], p["base_u"][m], p["noise1_u"][m])
    print("\n## unseen stands by airport (RMSE base_u -> dist_u)")
    apt = va["ADEP_mvt"].to_numpy()
    for a in sorted(set(apt[unseen])):
        m = unseen & (apt == a)
        rb, rd = np.sqrt(np.mean((y[m] - p["base_u"][m]) ** 2)), np.sqrt(np.mean((y[m] - p["dist_u"][m]) ** 2))
        print(f"  {a}: n={m.sum():6,}  OSM-matched {has_dist[m].mean():6.1%}  {rb:7.1f} -> {rd:7.1f}  ({rd - rb:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
