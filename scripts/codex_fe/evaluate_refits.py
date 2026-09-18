"""Paired confirmation of frozen feature-addition arms, including tail diagnostics."""

import os

os.environ["POLARS_MAX_THREADS"] = "2"
import json
from pathlib import Path

import numpy as np
import polars as pl
from screen import evaluate, load

OUT = Path("outputs/codex_fe")
RES = Path("docs/codex_fe_results")


def week_ci(f, p, alpha=0.0125):
    """Circular moving 7-day block bootstrap, truncated to original day count."""
    y = f["TAXITIME_SEC_mvt"].to_numpy()
    b = f["pred"].to_numpy()
    _, day = np.unique(f["MVT_TIME_UTC_mvt"].dt.date().to_numpy(), return_inverse=True)
    n = np.bincount(day)
    a = np.bincount(day, weights=(y - b) ** 2)
    z = np.bincount(day, weights=(y - p) ** 2)
    rng = np.random.default_rng(20260919)
    starts = rng.integers(0, len(n), (10000, int(np.ceil(len(n) / 7))))
    idx = ((starts[:, :, None] + np.arange(7)) % len(n)).reshape(10000, -1)[:, : len(n)]
    ds = np.sqrt(z[idx].sum(axis=1) / n[idx].sum(axis=1)) - np.sqrt(a[idx].sum(axis=1) / n[idx].sum(axis=1))
    return np.quantile(ds, [alpha / 2, 1 - alpha / 2]).tolist()


def details(f, p):
    r = evaluate(f, p, alpha=0.0125, boot=10000)
    r["week_ci"] = week_ci(f, p)
    r["by_month"] = {
        str(m): evaluate(f.filter(pl.col("month") == m), p[f["month"].to_numpy() == m])
        for m in f["month"].unique().sort().to_list()
    }
    r["by_airport"] = {
        a: evaluate(f.filter(pl.col("ADEP_mvt") == a), p[f["ADEP_mvt"].to_numpy() == a])
        for a in f["ADEP_mvt"].unique().sort().to_list()
    }
    y = f["TAXITIME_SEC_mvt"].to_numpy()
    b = f["pred"].to_numpy()
    a = (y - b) ** 2
    z = (y - p) ** 2
    masks = {
        "matched": f["unmatched"].to_numpy() == 0,
        "unmatched": f["unmatched"].to_numpy() == 1,
        "label_ge_2h": y >= 7200,
        "label_ge_12h": y >= 43200,
        "prediction_ge_2h": (b >= 7200) | (p >= 7200),
    }
    r["strata"] = {
        key: {
            "n": int(m.sum()),
            "baseline_sse": float(a[m].sum()),
            "candidate_sse": float(z[m].sum()),
            "delta_sse": float((z - a)[m].sum()),
        }
        for key, m in masks.items()
    }
    r["tail_prediction_counts"] = {
        str(t): {"baseline": int((b >= t).sum()), "candidate": int((p >= t).sum())} for t in [7200, 14400, 43200]
    }
    diff = z - a
    order = np.argsort(np.abs(diff))[-10:][::-1]
    r["largest10_delta_sse"] = [
        {
            "id": float(f["MVT_ID_mvt"][int(i)]),
            "label": float(y[i]),
            "baseline": float(b[i]),
            "candidate": float(p[i]),
            "delta_sse": float(diff[i]),
        }
        for i in order
    ]
    return r


def main():
    f = load()
    arms = {}
    for name in ["noise", "selected", "all"]:
        path = OUT / f"refit_{name}.parquet"
        if path.exists():
            arms[name] = (
                f.select("MVT_ID_mvt")
                .join(pl.read_parquet(path), on="MVT_ID_mvt", maintain_order="left")["pred"]
                .to_numpy()
            )
    results = {}
    for period, mask in [("confirmation", f["month"].to_numpy() >= 11), ("october", f["month"].to_numpy() == 10)]:
        te = f.filter(pl.Series(mask))
        results[period] = {}
        for name, p in arms.items():
            results[period][name + "_vs_baseline"] = details(te, p[mask])
            if name != "noise":
                results[period][name + "_vs_noise"] = details(
                    te.with_columns(pl.Series("pred", arms["noise"][mask])), p[mask]
                )
    (RES / "refit_evaluation.json").write_text(json.dumps(results, indent=2))
    for name, r in results["confirmation"].items():
        print(name, r["delta"], r["ci"], "week", r["week_ci"], flush=True)


if __name__ == "__main__":
    main()
