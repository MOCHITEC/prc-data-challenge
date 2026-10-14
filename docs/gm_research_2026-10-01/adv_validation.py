"""Adversarial validation: which of model A's columns tell 2025 from 2026 in the same months (Jan + Jul)?

    .venv/bin/python docs/gm_research_2026-10-01/adv_validation.py

Train rows of Jan and Jul 2025 (label 0) against every ranking row (Jan and Jul 2026, label 1), on the numeric
columns production's model sees (configs/main_flseq.yaml; fold-time target encodings are not in the cache and are
left out). A LightGBM classifier on a random 50/50 split; AUC 0.5 = no drift. Reports the AUC, the columns the
classifier leans on, and each top column's 2025 -> 2026 shift. Diagnostic only; nothing here changes a model."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

from prc.train import TrainConfig, visible_numeric_columns

WORK = Path("data/work_flseq")


def main() -> None:
    spec = yaml.safe_load(open("configs/main_flseq.yaml"))
    cfg = TrainConfig(**{k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__})
    probe = pl.read_parquet(WORK / "feat_train.parquet", n_rows=2000)
    cols = [c for c in visible_numeric_columns(probe, cfg) if c in pl.read_parquet_schema(WORK / "feat_rank.parquet")]
    need = ["MVT_TIME_UTC_mvt", "ADEP_mvt", *cols]
    tr = pl.read_parquet(WORK / "feat_train.parquet", columns=need).filter(
        pl.col("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7])
    )
    rk = pl.read_parquet(WORK / "feat_rank.parquet", columns=need)
    d = pl.concat([tr.with_columns(pl.lit(0).alias("is26")), rk.with_columns(pl.lit(1).alias("is26"))])
    X = d.select(cols).cast(pl.Float32).to_numpy()
    y = d["is26"].to_numpy()
    rng = np.random.default_rng(20261001)
    half = rng.random(len(y)) < 0.5
    params = dict(
        objective="binary",
        learning_rate=0.1,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
        num_threads=4,
        seed=1,
    )
    m = lgb.train(params, lgb.Dataset(X[half], y[half], feature_name=cols), num_boost_round=300)
    auc = roc_auc_score(y[~half], m.predict(X[~half]))
    print(f"rows 2025 Jan+Jul {tr.height:,} vs 2026 {rk.height:,}; {len(cols)} columns; held-out AUC = {auc:.4f}")
    gain = m.feature_importance("gain")
    order = np.argsort(-gain)
    tot = gain.sum()
    print(
        f"\n{'column':>34} {'gain share':>10} {'2025 mean':>12} {'2026 mean':>12} {'shift/sd':>9} {'null25':>7} {'null26':>7}"
    )
    for i in order[:30]:
        c = cols[i]
        a, b = tr[c].cast(pl.Float64), rk[c].cast(pl.Float64)
        sd = float(pl.concat([a, b]).std() or 0.0)
        shift = (float(b.mean() or 0.0) - float(a.mean() or 0.0)) / sd if sd else float("nan")
        print(
            f"{c:>34} {gain[i] / tot:10.3f} {float(a.mean() or 0):12.2f} {float(b.mean() or 0):12.2f} {shift:+9.2f} "
            f"{a.null_count() / len(a):7.3f} {b.null_count() / len(b):7.3f}"
        )
    # Without the top column, again and again: how many columns carry the separation?
    drop: list[str] = []
    for _ in range(5):
        drop.append(cols[int(order[0])])
        keep = [c for c in cols if c not in drop]
        Xk = d.select(keep).cast(pl.Float32).to_numpy()
        mk = lgb.train(params, lgb.Dataset(Xk[half], y[half], feature_name=keep), num_boost_round=300)
        auc_k = roc_auc_score(y[~half], mk.predict(Xk[~half]))
        g = mk.feature_importance("gain")
        order = np.array([cols.index(keep[j]) for j in np.argsort(-g)])
        print(f"without {drop}: AUC {auc_k:.4f}; next top column {keep[int(np.argmax(g))]}")


if __name__ == "__main__":
    main()
