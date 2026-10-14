"""adsbaudit step 2: per-feature 2025-vs-2026 shift of the learned ADS-B residual's design matrix (local, read-only).

Rows: ground-fix rows with an ADS-B feature row, LTFM and LIRF dropped (production applies the current residual
elsewhere only). 2025 = the 12 training months, b = the OOF base each month was trained with
(cv_rows_months12_v2.parquet). 2026 = feat_2026 with b recovered from v40 (rescat rank_delta_all.parquet).
Per airport x lane: null share and p10/p50/p90 per feature; single-feature adversarial AUC from a 40-bin
quantile histogram (null = own bin), 2025 vs 2026 and the closest-coverage pairs Nov+Dec 2025 vs Jan 2026,
Jul 2025 vs Jul 2026. Writes $SCRATCH/agents/adsbaudit/dist_auc.parquet, dist_q.parquet.

    POLARS_MAX_THREADS=2 .venv/bin/python analysis/local_adsbaudit_dist.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import adsb_learn_features as L  # noqa: E402

S = os.environ["SCRATCH"]
OUT = os.path.join(S, "agents", "adsbaudit")
LD = os.path.join(S, "learn")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(30)


def load():
    F25 = pl.concat([pl.read_parquet(os.path.join(LD, f)) for f in
                     ("feat_2025.parquet", "feat_2025novdec.parquet", "feat_2025aso.parquet", "feat_2025fmj.parquet")],
                    how="diagonal_relaxed")
    R = pl.read_parquet(os.path.join(LD, "cv_rows_months12_v2.parquet"), columns=["MVT_ID_mvt", "ym", "b"])
    A = F25.join(R, on="MVT_ID_mvt").with_columns(pl.col("ym").alias("per"))
    F26 = pl.read_parquet(os.path.join(LD, "feat_2026.parquet"))
    B = F26.join(pl.read_parquet(os.path.join(S, "agents", "rescat", "rank_delta_all.parquet"), columns=["MVT_ID_mvt", "b"]),
                 on="MVT_ID_mvt").with_columns(pl.col("day").str.slice(0, 7).alias("per"))
    D = pl.concat([A, B], how="diagonal_relaxed").filter(pl.col("ground_fix").fill_null(False)
                                                        & ~pl.col("ADEP_mvt").is_in(["LTFM", "LIRF"]))
    X, feats, cats = L.design(D, "b")
    X = X.with_columns(pl.when(pl.col("t_aobt3").is_not_null()).then(pl.lit("A")).otherwise(pl.lit("B")).alias("lane"))
    return X.select(["MVT_ID_mvt", "ADEP_mvt", "lane", "per"] + feats), feats


def auc_binned(x1, x2, nb=40):
    """AUC of the histogram classifier P(2026 | bin of x), bins = pooled quantiles, null its own bin."""
    x = np.concatenate([x1, x2])
    lab = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
    nn = ~np.isnan(x)
    bins = np.full(len(x), -1)
    if nn.sum() > 0:
        edges = np.unique(np.quantile(x[nn], np.linspace(0, 1, nb + 1)[1:-1]))
        bins[nn] = np.searchsorted(edges, x[nn], side="right")
    ub, inv = np.unique(bins, return_inverse=True)
    n0 = np.bincount(inv, weights=1 - lab)
    n1 = np.bincount(inv, weights=lab)
    sc = n1 / (n0 + n1)
    o = np.argsort(sc)
    n0, n1 = n0[o], n1[o]
    # Mann-Whitney with ties inside a bin counted half
    cum0 = np.cumsum(n0) - n0
    auc = (n1 * (cum0 + 0.5 * n0)).sum() / (n0.sum() * n1.sum())
    return float(auc)


def main():
    X, feats = load()
    print("rows", X.group_by("per").len().sort("per").rows())
    pairs = {"12m_vs_26": (None, None), "ND25_vs_J26": (["2025-11", "2025-12"], ["2026-01"]),
             "JUL25_vs_JUL26": (["2025-07"], ["2026-07"]), "JAN25_vs_J26": (["2025-01"], ["2026-01"])}
    rows, qrows = [], []
    for apt in sorted(X["ADEP_mvt"].unique()):
        for lane in ("A",):
            Z = X.filter((pl.col("ADEP_mvt") == apt) & (pl.col("lane") == lane))
            is26 = Z["per"].str.starts_with("2026").to_numpy()
            per = Z["per"].to_numpy()
            for f in feats:
                v = Z[f].cast(pl.Float64).to_numpy()
                rec = {"apt": apt, "lane": lane, "feat": f}
                for k, (p1, p2) in pairs.items():
                    m1 = ~is26 if p1 is None else np.isin(per, p1)
                    m2 = is26 if p2 is None else np.isin(per, p2)
                    if m1.sum() < 200 or m2.sum() < 200:
                        rec[k] = None
                        continue
                    a = auc_binned(v[m1], v[m2])
                    rec[k] = round(max(a, 1 - a), 4)
                rows.append(rec)
                for lab, m in (("2025", ~is26), ("2025-01", per == "2025-01"), ("2025-07", per == "2025-07"),
                               ("2025-ND", np.isin(per, ["2025-11", "2025-12"])), ("2026-01", per == "2026-01"),
                               ("2026-07", per == "2026-07")):
                    w = v[m]
                    if len(w) == 0:
                        continue
                    nn = w[~np.isnan(w)]
                    qrows.append({"apt": apt, "lane": lane, "feat": f, "per": lab, "n": len(w),
                                  "null": round(float(np.isnan(w).mean()), 4),
                                  **({f"q{int(q * 100)}": round(float(np.quantile(nn, q)), 1) for q in (0.1, 0.5, 0.9)}
                                     if len(nn) else {})})
    A = pl.DataFrame(rows)
    Q = pl.DataFrame(qrows)
    A.write_parquet(os.path.join(OUT, "dist_auc.parquet"))
    Q.write_parquet(os.path.join(OUT, "dist_q.parquet"))
    print("\nmean single-feature AUC over the 9 airports (lane A), sorted by ND25_vs_J26")
    M = A.group_by("feat").agg([pl.col(k).mean().round(3) for k in pairs] + [pl.col("ND25_vs_J26").max().round(3).alias("ND_max")])
    print(M.sort("ND25_vs_J26", descending=True))
    print("\ntop 30 airport x feature by ND25_vs_J26")
    print(A.sort("ND25_vs_J26", descending=True, nulls_last=True).head(30))
    print("\ntop 30 airport x feature by JUL25_vs_JUL26")
    print(A.sort("JUL25_vs_JUL26", descending=True, nulls_last=True).head(30))


if __name__ == "__main__":
    main()
