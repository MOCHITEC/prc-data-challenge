"""P1 stage 2: can a per-row LIRF fill probability be learned well enough to gate stages 4 and 5?

Read-only; nothing enters the pipeline. The label is `f10 = |BLOCK_TIME - SCHED_TIME| <= 10 s`,
which is observable on training rows only. BLOCK_TIME is in KEEP_RAW_COLS and is excluded from the
feature set by BLANKED_DEP_COLS, so using it as a LABEL here is safe and using it as a FEATURE is
already impossible.

The hypothesis: training on LIRF MATCHED rows as well, where the fill is directly observable on
159,216 rows instead of 1,488, lifts held-out LIRF-unmatched AUC materially above the 0.9177 our
unmatched-only recipe reached. That extra training population is the one ingredient all three of
merry-quicksand, javidmardanov and kind-mango have and we do not.

PRE-REGISTERED GATE (plans/P1-unmatched-lane.md, Stage 2): held-out LIRF-unmatched AUC of recipe
(c) must reach 0.93. Below that, stages 4 and 5 are recorded CLOSED with the AUC as the reason.
Two further failure conditions: (b)->(c) driven by the stake weight with the prefix features
contributing nothing; or a permuted-prefix control keeping more than ~20% of the AUC lift.

Run: .venv/bin/python docs/competitor_research_2026-09-21/scripts/p1_stage2_fill_auc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from prc.features import CATEGORICALS, numeric_feature_columns  # noqa: E402
from prc.train import LABEL, PROXY, SCHED_OFFSET, _matrix  # noqa: E402

GATE_AUC = 0.93
SEED = 42
PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbose": -1, "num_threads": 4, "deterministic": True, "force_row_wise": True,
}
ROUNDS = 400


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC; ties get the average rank."""
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> int:
    feats = pl.read_parquet(ROOT / "data/work/feat_train.parquet")
    # The feature list must be taken BEFORE the label and the helper columns are attached:
    # numeric_feature_columns() reads whatever numeric columns the frame happens to carry, so
    # attaching f10 first puts the LABEL in the feature set. That is what the first run of this
    # script did, and it reported AUC 1.0000 on every recipe including the permuted-feature
    # negative control - which is the signature of a leak, not of a good classifier.
    _num_all = numeric_feature_columns(feats)
    lirf = feats.filter(pl.col("ADEP_mvt") == "LIRF").with_columns(
        [
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
            .cast(pl.Int8).alias("f10"),
            pl.col(PROXY).is_null().alias("unm"),
            (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"),
            pl.col("FLIGHT_mvt").cast(pl.Utf8).str.slice(0, 3).alias("flight_prefix"),
            pl.col("FLIGHT_mvt").cast(pl.Utf8).alias("flight_identity"),
        ]
    )
    u, m = lirf.filter(pl.col("unm")), lirf.filter(~pl.col("unm"))
    print(f"LIRF: matched {m.height:,} (fill {m['f10'].mean():.4f})   unmatched {u.height:,} (fill {u['f10'].mean():.4f})")
    print(f"flight_prefix null rate on unmatched: {u['flight_prefix'].null_count() / u.height:.4%}\n")

    # model B's feature set: the same exclusion predict_fold applies
    helpers = {"f10", "unm", "ym", "flight_prefix", "flight_identity"}
    num = [
        c for c in _num_all
        if c not in helpers
        and not c.startswith(("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt"))
    ]
    assert "f10" not in num and LABEL not in num, "label leaked into the feature set"
    base_cats = [c for c in CATEGORICALS if c in lirf.columns]
    months = sorted(lirf["ym"].unique().to_list())
    print(f"{len(num)} numeric + {len(base_cats)} categorical features, {len(months)} months\n")

    def run(name: str, use_matched: bool, w_unm: float, stake: bool, prefix: bool, permute: bool = False):
        rows, oof_s, oof_y, oof_ms = [], [], [], []
        cats = base_cats + (["flight_prefix", "flight_identity"] if prefix else [])
        for mo in months:
            tr = lirf.filter((pl.col("ym") != mo) & (pl.col("unm") | use_matched))
            va = u.filter(pl.col("ym") == mo)
            if va.height < 20 or tr["f10"].sum() < 10:
                continue
            if permute:
                rng = np.random.default_rng(SEED)
                for c in ("flight_prefix", "flight_identity"):
                    tr = tr.with_columns(pl.Series(c, rng.permutation(tr[c].to_numpy())))
            stacked = pl.concat([tr.select(num + cats), va.select(num + cats)], how="vertical_relaxed")
            X, cols = _matrix(stacked, num, cats)
            ci = [cols.index(c) for c in cats]
            w = np.where(tr["unm"].to_numpy(), w_unm, 1.0)
            if stake:
                ms = np.nan_to_num(tr[SCHED_OFFSET].to_numpy().astype(float))
                w = w * np.clip(1 + (np.maximum(ms, 0) / 3600.0) ** 2, 1, 25)
            d = lgb.Dataset(X[: tr.height], label=tr["f10"].to_numpy(), weight=w, categorical_feature=ci, params=PARAMS)
            b = lgb.train(PARAMS, d, num_boost_round=ROUNDS)
            s = b.predict(X[tr.height :])
            oof_s.append(s); oof_y.append(va["f10"].to_numpy()); oof_ms.append(np.nan_to_num(va[SCHED_OFFSET].to_numpy().astype(float)))
            rows.append((mo, va.height, auc(va["f10"].to_numpy(), s)))
        s, y, ms = np.concatenate(oof_s), np.concatenate(oof_y), np.concatenate(oof_ms)
        a, ll = auc(y, s), logloss(y, s)
        far = ms >= 7200
        a_far = auc(y[far], s[far]) if far.sum() > 20 else float("nan")
        print(f"{name}")
        print(f"   pooled LOMO AUC {a:.4f}   logloss {ll:.4f}   n={len(y):,} pos={int(y.sum()):,}")
        print(f"   AUC on ms >= 2h: {a_far:.4f}  (n={int(far.sum()):,}, pos={int(y[far].sum()):,})")
        per = [f"{mo % 100}:{v:.3f}" for mo, _, v in rows if v == v]
        print(f"   per month: {' '.join(per)}")
        q = np.quantile(s, np.linspace(0, 1, 7))
        bins = np.clip(np.digitize(s, q[1:-1]), 0, 5)
        cal = "  ".join(f"{y[bins == k].mean():.3f}" for k in range(6) if (bins == k).sum())
        print(f"   calibration by sextile (observed fill rate): {cal}\n")
        return a

    a = run("(a) unmatched rows only, our current recipe", False, 1.0, False, False)
    b = run("(b) + LIRF matched rows, unmatched up-weighted x8", True, 8.0, False, False)
    c = run("(c) + stake weight + flight prefix/identity", True, 8.0, True, True)
    c_ns = run("(c-) (c) without the stake weight, to isolate the prefix", True, 8.0, False, True)
    c_np = run("(c~) (c) with prefix/identity PERMUTED (negative control)", True, 8.0, True, True, permute=True)

    print("=" * 78)
    print(f"GATE: recipe (c) held-out LIRF-unmatched AUC {c:.4f} vs threshold {GATE_AUC}")
    print(f"  -> {'PASS: stages 4 and 5 may start' if c >= GATE_AUC else 'FAIL: record stages 4 and 5 CLOSED'}")
    lift = c - b
    print(f"\nlift (b)->(c) = {lift:+.4f}")
    print(f"  prefix alone (c- vs b)      = {c_ns - b:+.4f}")
    print(f"  permuted-prefix control (c~) = {c_np:.4f}, keeps {100 * (c_np - b) / lift if abs(lift) > 1e-9 else float('nan'):.0f}% of the lift")
    print(f"  (control keeping >20% of the lift is a stated failure condition)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
