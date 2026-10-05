"""Random-search retune of the anchored CatBoost+L2 member M3's model A (main_flseq_osm_nolirf_cb_l2).

    .venv/bin/python scripts/retune_cb.py trial  --index I       # one trial, in this process
    .venv/bin/python scripts/retune_cb.py study  --n N           # trials 0..N-1, each in a gated subprocess
    .venv/bin/python scripts/retune_cb.py summary                # table of the finished trials, best first
    .venv/bin/python scripts/retune_cb.py best                   # the best trial's cb_matched_params as JSON

Pre-registration and results: docs/competitor_research_2026-09-21/results/catboost-retune.md. The design is
optuna-retune.md's (scripts/retune_lgb.py), reused: Optuna is not installed, so this is a plain random search.

Tuning split (TUNE_FOLD): validate on 2025-10, train on 2025-01..2025-09. October is validated by none of the
comparison folds (jan, jul, forward = 2025-01, 2025-07, 2025-11+12), so the trial selection never reads a row the
confirmation is scored on. The objective is model A's RMSE on October's matched rows (`mvt_minus_aobt3`
non-null); model B is not fitted (no unmatched target rows are passed).

Each trial's settings are a pure function of its index (trial 0 = the current settings, i.e. no override), so a
restarted study samples the same points. Each finished trial is written atomically to
logs/cbretune/trials/trial_NNN.json before the next one starts; the study skips indices already there. A trial
interrupted by a restart leaves no record and is rerun from scratch (CatBoost on CPU with a fixed seed and thread
count is bit-identical, so the rerun is the same trial).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retune_lgb import TUNE_FOLD, n_adsb, total_rss_mb, train_features  # noqa: E402

from prc.cv import year_month_key  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import LABEL, PROXY, predict_fold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs/main_flseq_osm_nolirf_cb_l2.yaml"
OUT = ROOT / "logs/cbretune"
TRIALS = OUT / "trials"
SEED_BASE = 20_261_005
# Pre-registered search space (catboost-retune.md §0.3), drawn in this order from one uniform per key.
# ("int", lo, hi) uniform integer | ("int_log", lo, hi) | ("log", lo, hi) | ("uniform", lo, hi) | ("choice", values)
SPACE = {
    "depth": ("int", 4, 10),
    "learning_rate": ("log", 0.08, 0.4),
    "l2_leaf_reg": ("log", 0.3, 30.0),
    "random_strength": ("log", 0.1, 10.0),
    "subsample": ("uniform", 0.5, 1.0),
    "border_count": ("int_log", 32, 254),
    "one_hot_max_size": ("choice", (2, 4, 16)),
    "max_ctr_complexity": ("choice", (1, 2)),
}
# The current settings of every searched key (CB_DEFAULTS + CatBoost's resolved library defaults, as logged in
# experiments.jsonl for main_flseq_osm_nolirf_cb_l2-4aa341b206: l2_leaf_reg 3, border_count 254, MVS subsample
# 0.8, one_hot_max_size 2; random_strength's library default is 1).
CURRENT = {
    "depth": 6,
    "learning_rate": 0.2,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "subsample": 0.8,
    "border_count": 254,
    "one_hot_max_size": 2,
    "max_ctr_complexity": 1,
}
MEM_KILL_MB = 13_000
GATE_RSS_MB = 2_500


def sample(index: int) -> dict:
    """The trial's model-A override. Trial 0 is the current recipe (no override)."""
    if index == 0:
        return {}
    rng = np.random.default_rng(SEED_BASE + index)
    out = {}
    for k, spec in SPACE.items():
        kind = spec[0]
        u = float(rng.uniform())
        if kind == "choice":
            vals = spec[1]
            out[k] = vals[min(int(u * len(vals)), len(vals) - 1)]
            continue
        lo, hi = spec[1], spec[2]
        if kind == "int":
            out[k] = int(min(lo + int(u * (hi - lo + 1)), hi))
        elif kind == "uniform":
            out[k] = round(lo + u * (hi - lo), 4)
        else:
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            out[k] = int(round(v)) if kind == "int_log" else float(f"{v:.4g}")
    return out


def effective_params(index: int) -> dict:
    return dict(CURRENT) | sample(index)


def run_trial(index: int) -> dict:
    t0 = time.time()
    spec = yaml.safe_load(BASE_CONFIG.read_text())
    override = sample(index)
    if override:
        spec["cb_matched_params"] = override
    cfg = train_config_from_spec(spec)
    feats = train_features(spec)
    months = year_month_key(feats)
    tr = feats.filter(months.is_in(list(TUNE_FOLD.train_months)))
    va = feats.filter(months.is_in(list(TUNE_FOLD.valid_months)) & pl.col(PROXY).is_not_null())
    del feats, months
    t_fit = time.time()
    p, info = predict_fold(tr, va, cfg)
    j = va.select(["MVT_ID_mvt", LABEL, "ADEP_mvt"]).join(p, on="MVT_ID_mvt", how="inner")
    if j.height != va.height:
        raise AssertionError(f"{va.height - j.height} matched October rows without a prediction")
    y, pv = j[LABEL].to_numpy().astype(float), j["pred"].to_numpy().astype(float)
    if not np.isfinite(pv).all():
        raise AssertionError("non-finite prediction")
    by_apt = {
        a: float(np.sqrt(np.mean((y[m] - pv[m]) ** 2)))
        for a in sorted(set(j["ADEP_mvt"].to_list()))
        for m in [j["ADEP_mvt"].to_numpy() == a]
    }
    cbm = info.get("cb_matched", {})
    rec = {
        "index": index,
        "status": "ok",
        "cb_matched_params": override,
        "params": effective_params(index),
        "resolved": cbm.get("resolved"),
        "es_best_count": cbm.get("es_best_count"),
        "rmse_matched": float(np.sqrt(np.mean((y - pv) ** 2))),
        "n_valid": int(j.height),
        "n_train": int(info["train_rows"]),
        "rounds_matched": info.get("rounds_matched"),
        "by_airport": by_apt,
        "fit_seconds": round(time.time() - t_fit, 1),
        "seconds": round(time.time() - t0, 1),
        "exp_id": f"{cfg.name}-{cfg.hash()}",
    }
    pred_dir = OUT / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    p.write_parquet(pred_dir / f"trial_{index:03d}.parquet")
    return rec


def write_record(index: int, rec: dict) -> None:
    """Atomic: a temp file, fsynced, then renamed over the record."""
    TRIALS.mkdir(parents=True, exist_ok=True)
    path = TRIALS / f"trial_{index:03d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    with open(tmp) as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)


def gate(log) -> None:
    """Wait (polling every 60 s) until no adsb_learn_ or prc.pipeline process runs and total_rss < 2,500 MB."""
    waited = 0
    while True:
        rss, n = total_rss_mb(), n_adsb()
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        npipe = sum(1 for line in ps.splitlines() if "prc.pipeline" in line)
        if n == 0 and npipe == 0 and max(rss) < GATE_RSS_MB:
            if waited:
                log(f"gate open after {waited} s (rss {rss} MB)")
            return
        if waited % 600 == 0:
            log(f"gate closed: adsb_learn_ {n}, prc.pipeline {npipe}, total_rss {rss} MB; polling every 60 s")
        time.sleep(60)
        waited += 60


def study(n: int, max_hours: float) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "study.log", "a")

    def log(msg: str) -> None:
        logf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
        logf.flush()

    log(f"study start n={n} max_hours={max_hours} pid={os.getpid()}")
    spent = (
        sum(json.loads(p.read_text()).get("wall_s", 0) for p in TRIALS.glob("trial_*.json")) if TRIALS.exists() else 0
    )
    for i in range(n):
        if (TRIALS / f"trial_{i:03d}.json").exists():
            continue
        if spent > max_hours * 3600:
            log(f"stop: cumulative trial wall {spent:.0f} s exceeds {max_hours} h before trial {i}")
            break
        gate(log)
        log(f"trial {i} start params={sample(i)}")
        start = time.time()
        with open(OUT / f"trial_{i:03d}.log", "w") as tl:
            proc = subprocess.Popen(
                [sys.executable, __file__, "trial", "--index", str(i)],
                stdout=tl,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                start_new_session=True,
            )
            peak, killed = [0, 0], False
            while proc.poll() is None:
                rss = total_rss_mb()
                peak = [max(a, b) for a, b in zip(peak, rss)]
                if max(rss) > MEM_KILL_MB:
                    os.killpg(proc.pid, 9)
                    killed = True
                    proc.wait()
                    break
                time.sleep(10)
        wall = round(time.time() - start, 1)
        spent += wall
        res = OUT / f"trial_{i:03d}.result.json"
        if proc.returncode == 0 and res.exists() and not killed:
            rec = json.loads(res.read_text())
            rec.update(wall_s=wall, peak_total_rss_mb=peak)
            write_record(i, rec)
            res.unlink()
            log(f"trial {i} ok rmse_matched={rec['rmse_matched']:.3f} wall={wall}s peak={peak}MB")
        elif killed:
            write_record(
                i,
                {
                    "index": i,
                    "status": "killed_memory",
                    "cb_matched_params": sample(i),
                    "wall_s": wall,
                    "peak_total_rss_mb": peak,
                },
            )
            log(f"trial {i} KILLED at total_rss > {MEM_KILL_MB} MB (peak {peak}); recorded, not retried")
        else:
            log(f"trial {i} FAILED rc={proc.returncode}; see trial_{i:03d}.log; stopping the study")
            return 1
    log("study end")
    return 0


def load_trials() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(TRIALS.glob("trial_*.json"))]


def best() -> dict:
    ok = [r for r in load_trials() if r["status"] == "ok"]
    return min(ok, key=lambda r: r["rmse_matched"])


def summary() -> None:
    rows = sorted(load_trials(), key=lambda r: r.get("rmse_matched", math.inf))
    base = next((r["rmse_matched"] for r in rows if r["index"] == 0 and r["status"] == "ok"), math.nan)
    print(f"{len(rows)} trials; trial 0 (current params) {base:.3f}")
    print(
        "idx  rmse_m    d_vs_0  depth  lr      l2      rs      subs   border  ohe  ctr  es_best rounds  wall_s  peak_MB"
    )
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['index']:>3}  {r['status']}  {r['cb_matched_params']}")
            continue
        p = r["params"]
        print(
            f"{r['index']:>3}  {r['rmse_matched']:.3f}  {r['rmse_matched'] - base:+7.3f}  {p['depth']:>5}  "
            f"{p['learning_rate']:<6.4g}  {p['l2_leaf_reg']:<6.4g}  {p['random_strength']:<6.4g}  "
            f"{p['subsample']:.3f}  {p['border_count']:>6}  {p['one_hot_max_size']:>3}  {p['max_ctr_complexity']:>3}  "
            f"{r.get('es_best_count') or 0:>7}  {r['rounds_matched']:>6}  {r.get('wall_s', 0):>6.0f}  "
            f"{max(r.get('peak_total_rss_mb', [0]))}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["trial", "study", "summary", "best", "sample"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--max-hours", type=float, default=16.0)
    a = ap.parse_args()
    if a.what == "trial":
        rec = run_trial(a.index)
        res = OUT / f"trial_{a.index:03d}.result.json"
        res.write_text(json.dumps(rec, indent=1, default=str))
        print(json.dumps({k: v for k, v in rec.items() if k != "by_airport"}, default=str))
        return 0
    if a.what == "study":
        return study(a.n, a.max_hours)
    if a.what == "summary":
        summary()
        return 0
    if a.what == "best":
        print(json.dumps(best()["cb_matched_params"]))
        return 0
    for i in range(a.n or 10):
        print(i, sample(i))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
