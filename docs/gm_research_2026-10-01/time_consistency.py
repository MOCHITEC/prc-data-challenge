"""plan_time_consistency.md §A: one-column models trained on Jan-Jun 2025, scored in time and out of time.

    .venv/bin/python docs/gm_research_2026-10-01/time_consistency.py > docs/gm_research_2026-10-01/time_consistency.txt

Model A's target g = y - proxy on matched rows, evaluated on g clipped to +-1,800 s (full-value numbers alongside).
A column is time-unstable when its in-time gain is > 0 and its out-of-time gain is <= 0 or < 25 % of the in-time gain."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

from prc.train import PROXY, TrainConfig, visible_numeric_columns

LABEL, WORK, CLIP = "TAXITIME_SEC_mvt", Path("data/work_flseq"), 1800.0
PARAMS = dict(
    objective="regression", learning_rate=0.1, num_leaves=31, min_data_in_leaf=500, verbose=-1, num_threads=4, seed=7
)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    spec = yaml.safe_load(open("configs/main_flseq.yaml"))
    cfg = TrainConfig(**{k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__})
    probe = pl.read_parquet(WORK / "feat_train.parquet", n_rows=2000)
    cols = visible_numeric_columns(probe, cfg)
    d = pl.read_parquet(WORK / "feat_train.parquet", columns=["MVT_TIME_UTC_mvt", LABEL, *cols]).filter(
        pl.col(PROXY).is_not_null() & pl.col(LABEL).is_finite()
    )
    d = d.with_columns((pl.col(LABEL) - pl.col(PROXY)).alias("_g"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("_m"))
    rng = np.random.default_rng(20261001)
    early = d.filter(pl.col("_m") <= 6)
    late = d.filter(pl.col("_m") >= 7)
    e_idx = rng.permutation(early.height)
    tr, it = early[e_idx[:400_000]], early[e_idx[400_000:500_000]]
    ot = late[rng.permutation(late.height)[:200_000]]
    fw = ot["_m"].to_numpy() >= 11
    g_tr, g_it, g_ot = (f["_g"].to_numpy().astype(float) for f in (tr, it, ot))
    c_it, c_ot = np.clip(g_it, -CLIP, CLIP), np.clip(g_ot, -CLIP, CLIP)
    const = float(np.mean(g_tr))
    base = {
        "in": rmse(c_it, np.full_like(c_it, const)),
        "out": rmse(c_ot, np.full_like(c_ot, const)),
        "fwd": rmse(c_ot[fw], np.full(fw.sum(), const)),
        "out_full": rmse(g_ot, np.full_like(g_ot, const)),
    }
    print(
        f"{len(cols)} columns; train {tr.height:,} (Jan-Jun), in-time {it.height:,}, out-of-time {ot.height:,} "
        f"(forward {int(fw.sum()):,}); constant RMSE (clipped) in {base['in']:.2f} out {base['out']:.2f}"
    )
    rows = []
    for c in cols:
        x_tr = tr[c].cast(pl.Float64).to_numpy().reshape(-1, 1)
        m = lgb.train(PARAMS, lgb.Dataset(x_tr, g_tr), num_boost_round=100)
        p_it = m.predict(it[c].cast(pl.Float64).to_numpy().reshape(-1, 1))
        p_ot = m.predict(ot[c].cast(pl.Float64).to_numpy().reshape(-1, 1))
        gi = base["in"] - rmse(c_it, p_it)
        go = base["out"] - rmse(c_ot, p_ot)
        gf = base["fwd"] - rmse(c_ot[fw], p_ot[fw])
        gfull = base["out_full"] - rmse(g_ot, p_ot)
        unstable = gi > 0 and (go <= 0 or go < 0.25 * gi)
        rows.append((c, gi, go, gf, gfull, unstable))
    rows.sort(key=lambda r: -r[1])
    print(f"\n{'column':>34} {'gain in':>8} {'gain out':>9} {'out/in':>7} {'gain fwd':>9} {'out full':>9}  flag")
    for c, gi, go, gf, gfull, u in rows:
        ratio = go / gi if gi > 0 else float("nan")
        print(f"{c:>34} {gi:8.3f} {go:9.3f} {ratio:7.2f} {gf:9.3f} {gfull:9.3f}  {'UNSTABLE' if u else ''}")
    flagged = [r[0] for r in rows if r[5]]
    print(f"\nflagged {len(flagged)}: {flagged}")
    pl.DataFrame(
        rows, schema=["column", "gain_in", "gain_out", "gain_fwd", "gain_out_full", "unstable"], orient="row"
    ).write_csv("docs/gm_research_2026-10-01/time_consistency.csv")


if __name__ == "__main__":
    main()
