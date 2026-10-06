"""Build the residual's per-day ADS-B features (scripts/adsb_learn_features.py, unchanged) and the appear-tier columns
(scripts/adsb_appear.py) for the days whose stage-1 joined file exists, then concatenate per year.

    SCRATCH=... POLARS_MAX_THREADS=2 .venv/bin/python docs/appear_tier_2026-10-05/build_features.py 2025 [2026]

adsb_learn_features' own `build`/`concat` need all 124 days; this one takes the days that are there (a missing day
only means fewer ADS-B rows: its departures keep the base) and prints which are missing."""

import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import adsb_appear as A
import adsb_learn_features as L
import adsb_stage1_feasibility as s1


# "2025nd": Nov+Dec 2025, for the 4-month re-measure (plan.md section 6.2); its files are feat_2025nd / appear_2025nd.
DAYS = {**L.DAYS, "2025nd": [f"2025-{m:02d}-{d:02d}" for m in (11, 12) for d in range(1, 32) if not (m == 11 and d == 31)]}
# "2025m8": the other eight months of 2025 (Feb-Jun, Aug-Oct), for the 12-month re-measure (plan_months12.md).
_ML = {2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 8: 31, 9: 30, 10: 31}
DAYS["2025m8"] = [f"2025-{m:02d}-{d:02d}" for m, n in _ML.items() for d in range(1, n + 1)]


def main(years: list[str]) -> None:
    os.makedirs(L.OUT, exist_ok=True)
    ST = A.stands()
    for y in years:
        have = [d for d in DAYS[y] if os.path.exists(os.path.join(s1.SCRATCH, f"{L.tag_of(d)}_joined.parquet"))]
        print(f"{y}: {len(have)} of {len(DAYS[y])} days; missing {sorted(set(DAYS[y]) - set(have))}", flush=True)
        for d in have:
            fp, ap = os.path.join(L.OUT, f"feat_{d}.parquet"), os.path.join(L.OUT, f"appear_{d}.parquet")
            if not os.path.exists(fp):
                L.day_features(d).write_parquet(fp)
            if not os.path.exists(ap):
                A.day_appear(d, ST).write_parquet(ap)
        for kind in ("feat", "appear"):
            D = pl.concat([pl.read_parquet(os.path.join(L.OUT, f"{kind}_{d}.parquet")) for d in have],
                          how="diagonal_relaxed")
            assert D["MVT_ID_mvt"].n_unique() == D.height, f"{kind}: a departure in two day files"
            assert not (set(D.columns) & L.LABEL_COLS)
            D.write_parquet(os.path.join(L.OUT, f"{kind}_{y}.parquet"))
            print(f"{y} {kind}: {D.shape}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["2025"])
