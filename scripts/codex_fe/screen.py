"""Same-model residual screening, selection lock, and independent confirmation."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

OUT = Path("outputs/codex_fe")
RES = Path("docs/codex_fe_results")
ID = "MVT_ID_mvt"
LABEL = "TAXITIME_SEC_mvt"
CONTEXT = [
    "ADEP_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "AIRCRAFT_TYPE_mvt",
    "hour_local",
    "mvt_minus_sched",
    "mvt_minus_aobt3",
    "aobt3_minus_eobt1",
    "temp_c",
    "dep_tko_prev_30m",
    "arr_taxi_in_mean_30m",
    "pred",
]


def load():
    f = (
        pl.scan_parquet("data/work/feat_train.parquet")
        .filter(pl.col("month") >= 9)
        .select(ID, LABEL, "month", "MVT_TIME_UTC_mvt", *set(CONTEXT) - {"pred"})
        .collect()
    )
    f = f.join(pl.read_parquet(OUT / "baseline.parquet"), on=ID, maintain_order="left")
    f = f.join(pl.read_parquet(OUT / "candidates_train.parquet"), on=ID, maintain_order="left")
    return f.with_columns(
        (pl.col(LABEL) - pl.col("pred")).alias("residual"),
        pl.col("mvt_minus_aobt3").is_null().cast(pl.Int8).alias("unmatched"),
    )


def category(train, test, c):
    if c == "airport_only":
        return np.zeros(train.height, dtype=int), np.zeros(test.height, dtype=int)
    if train[c].dtype == pl.String or train[c].dtype == pl.Boolean:
        return train[c].cast(pl.String).fill_null("<null>").to_numpy(), test[c].cast(pl.String).fill_null(
            "<null>"
        ).to_numpy()
    a = train[c].cast(pl.Float64).to_numpy()
    b = test[c].cast(pl.Float64).to_numpy()
    finite = a[np.isfinite(a)]
    if len(np.unique(finite)) <= 20:
        cuts = np.unique(finite)[:-1]
    else:
        cuts = np.unique(np.quantile(finite, np.arange(0.1, 1, 0.1))) if len(finite) else np.array([])

    def bin_(v):
        return np.where(np.isfinite(v), np.searchsorted(cuts, v, side="left"), -1)

    return bin_(a), bin_(b)


def correction(train, test, c):
    a, b = category(train, test, c)
    keys = ["ADEP_mvt", "unmatched", "bin"]
    ta = train.select("ADEP_mvt", "unmatched", "residual").with_columns(pl.Series("bin", a))
    tb = test.select("ADEP_mvt", "unmatched").with_columns(pl.Series("bin", b))
    table = ta.group_by(keys).agg((pl.col("residual").sum() / (pl.len() + 100)).alias("correction"))
    return tb.join(table, on=keys, how="left", maintain_order="left")["correction"].fill_null(0).to_numpy()


def evaluate(frame, p, alpha=0.05, boot=2000):
    y = frame[LABEL].to_numpy()
    b = frame["pred"].to_numpy()
    p = np.maximum(0, p)
    se0 = (y - b) ** 2
    se1 = (y - p) ** 2
    dates = frame["MVT_TIME_UTC_mvt"].dt.date().to_numpy()
    _, day = np.unique(dates, return_inverse=True)
    n = np.bincount(day)
    a = np.bincount(day, weights=se0)
    z = np.bincount(day, weights=se1)
    rng = np.random.default_rng(20260918)
    w = rng.multinomial(len(n), np.ones(len(n)) / len(n), size=boot)
    diffs = np.sqrt((w @ z) / (w @ n)) - np.sqrt((w @ a) / (w @ n))
    delta = lambda m: float(np.sqrt(se1[m].mean()) - np.sqrt(se0[m].mean())) if np.any(m) else None
    trim = np.ones(len(y), bool)
    trim[np.argsort(se0)[-200:]] = False
    u = frame["unmatched"].to_numpy().astype(bool)
    return {
        "n": len(y),
        "days": len(n),
        "baseline": float(np.sqrt(se0.mean())),
        "rmse": float(np.sqrt(se1.mean())),
        "delta": delta(np.ones(len(y), bool)),
        "ci": np.quantile(diffs, [alpha / 2, 1 - alpha / 2]).tolist(),
        "matched": delta(~u),
        "unmatched": delta(u),
        "baseline_worst200_removed": delta(trim),
        "delta_sse": float((se1 - se0).sum()),
        "max_day_sse_fraction": float(np.max(np.abs(z - a)) / max(np.sum(np.abs(z - a)), 1e-12)),
    }


def joint(train, test, novel):
    cols = CONTEXT + novel
    stacked = pl.concat([train.select(cols), test.select(cols)])
    cats = [c for c in cols if stacked[c].dtype == pl.String]
    for c in cats:
        stacked = stacked.with_columns(pl.col(c).fill_null("<null>").cast(pl.Categorical).to_physical().alias(c))
    x = stacked.cast(pl.Float32).to_numpy()
    a = x[: train.height]
    b = x[train.height :]
    p = np.zeros(test.height)
    for regime in [0, 1]:
        tr = train["unmatched"].to_numpy() == regime
        te = test["unmatched"].to_numpy() == regime
        params = {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 2,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_data_in_leaf": 300 if regime == 0 else 100,
            "lambda_l2": 100,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "deterministic": True,
            "force_col_wise": True,
            "seed": 42,
        }
        data = lgb.Dataset(
            a[tr],
            label=train["residual"].to_numpy()[tr],
            categorical_feature=[cols.index(c) for c in cats],
            params=params,
        )
        model = lgb.train(params, data, num_boost_round=120)
        p[te] = model.predict(b[te], num_threads=2)
    return np.maximum(0, test["pred"].to_numpy() + p)


def main(mode):
    f = load()
    descriptions = json.loads((RES / "candidates.json").read_text())
    novel = [c for c in descriptions if not c.startswith("noise_")]
    if mode == "screen":
        tr = f.filter(pl.col("month") == 9)
        te = f.filter(pl.col("month") == 10)
        rows = []
        for c in ["airport_only", *descriptions, "hour_local", "mvt_minus_sched", "mvt_minus_aobt3"]:
            result = evaluate(te, te["pred"].to_numpy() + correction(tr, te, c))
            result["candidate"] = c
            rows.append(result)
            print(c, round(result["delta"], 4), result["ci"], flush=True)
        (RES / "screen.json").write_text(json.dumps(rows, indent=2))
        control = min(
            r["delta"] for r in rows if r["candidate"].startswith("noise_") or r["candidate"] == "airport_only"
        )
        selected = [
            r["candidate"]
            for r in sorted(rows, key=lambda x: x["delta"])
            if r["candidate"] in novel and r["ci"][1] < 0 and r["delta"] < control and r["matched"] < 0
        ][:5]
        (RES / "selection.json").write_text(
            json.dumps(
                {"selected": selected, "control_threshold": control, "novel_count": len(novel), "joint_novel": novel},
                indent=2,
            )
        )
        results = []
        for name, cols in [
            ("joint_context", []),
            ("joint_noise", [f"noise_{k}" for k in range(4)]),
            ("joint_novel", novel),
        ]:
            p = joint(tr, te, cols)
            r = evaluate(te, p)
            r["candidate"] = name
            results.append(r)
        (RES / "joint_screen.json").write_text(json.dumps(results, indent=2))
        print("SELECTED", selected, flush=True)
    else:
        selection = json.loads((RES / "selection.json").read_text())
        tr = f.filter(pl.col("month").is_in([9, 10]))
        te = f.filter(pl.col("month").is_in([11, 12]))
        # Selected scalar hypotheses plus predeclared joint-novel test; controls descriptive.
        alpha = 0.05 / (len(selection["selected"]) + 1)
        results = []
        predictions = {}
        for c in selection["selected"] + ["airport_only", "noise_0"]:
            predictions[c] = np.maximum(0, te["pred"].to_numpy() + correction(tr, te, c))
        for name, cols in [
            ("joint_context", []),
            ("joint_noise", [f"noise_{k}" for k in range(4)]),
            ("joint_novel", novel),
        ]:
            predictions[name] = joint(tr, te, cols)
        for c, p in predictions.items():
            r = evaluate(te, p, alpha=alpha, boot=10000)
            r["candidate"] = c
            r["by_month"] = {
                str(m): evaluate(te.filter(pl.col("month") == m), p[te["month"].to_numpy() == m]) for m in [11, 12]
            }
            r["by_airport"] = {
                apt: evaluate(te.filter(pl.col("ADEP_mvt") == apt), p[te["ADEP_mvt"].to_numpy() == apt])
                for apt in te["ADEP_mvt"].unique().sort().to_list()
            }
            if c == "joint_novel":
                for ctrl in ["joint_context", "joint_noise"]:
                    r["vs_" + ctrl] = evaluate(
                        te.with_columns(pl.Series("pred", predictions[ctrl])), p, alpha=alpha, boot=10000
                    )
            results.append(r)
            print(c, r["delta"], r["ci"], flush=True)
        (RES / "confirmation.json").write_text(json.dumps({"alpha": alpha, "results": results}, indent=2))
        te.select(ID).with_columns([pl.Series(c, p) for c, p in predictions.items()]).write_parquet(
            OUT / "correction_confirmation.parquet"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["screen", "confirm"])
    args = parser.parse_args()
    main(args.mode)
