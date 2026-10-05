"""Random-search retune of the pooled anchored LGB+L2 member's model A (main_flseq_osm_nolirf_l2).

    .venv/bin/python scripts/retune_lgb.py trial  --index I       # one trial, in this process
    .venv/bin/python scripts/retune_lgb.py study  --n N           # trials 0..N-1, each in a gated subprocess
    .venv/bin/python scripts/retune_lgb.py summary                # table of the finished trials, best first
    .venv/bin/python scripts/retune_lgb.py best                   # the best trial's matched_params as JSON

Pre-registration and results: docs/competitor_research_2026-09-21/results/optuna-retune.md. Optuna is not
installed in .venv and is not in requirements-lock.txt, so this is a plain random search (§0.3 there).

Tuning split (TUNE_FOLD): validate on 2025-10, train on 2025-01..2025-09 (forward shape). October is validated by
none of the four comparison folds (jan, jul, jul_fwd, forward = 2025-01, 2025-07, 2025-11+12), so the trial
selection never reads a row the confirmation is scored on. The objective is model A's RMSE on October's matched
rows (`mvt_minus_aobt3` non-null); model B is not fitted (no unmatched target rows are passed).

Each trial's params are a pure function of its index (trial 0 = the current params, i.e. no override), so a
restarted study samples the same points. Each finished trial is written atomically to
logs/retune/trials/trial_NNN.json before the next one starts; the study skips indices already there.
"""

from __future__ import annotations

import argparse
import gc
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

from prc.cv import Fold, year_month_key
from prc.pipeline import build_features, drop_hidden_candidates, load_raw, train_config_from_spec, weather_flags
from prc.train import LABEL, PROXY, predict_fold

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs/main_flseq_osm_nolirf_l2.yaml"
WORK = ROOT / "data/work_flseq"
RAW = ROOT / "data/raw"
OUT = ROOT / "logs/retune"
TRIALS = OUT / "trials"
TUNE_FOLD = Fold("tune_oct", (202510,), tuple(202500 + m for m in range(1, 10)))
SEED_BASE = 20_261_002
# Pre-registered search space (optuna-retune.md §0.3). ("int_log", lo, hi) | ("log", lo, hi) | ("uniform", lo, hi).
SPACE = {
    "num_leaves": ("int_log", 31, 511),
    "min_data_in_leaf": ("int_log", 10, 500),
    "feature_fraction": ("uniform", 0.4, 1.0),
    "bagging_fraction": ("uniform", 0.5, 1.0),
    "lambda_l2": ("log", 0.01, 100.0),
    "learning_rate": ("log", 0.03, 0.12),
}
MEM_KILL_MB = 13_000
GATE_RSS_MB = 2_500


def sample(index: int) -> dict:
    """The trial's model-A override. Trial 0 is the current recipe (no override)."""
    if index == 0:
        return {}
    rng = np.random.default_rng(SEED_BASE + index)
    out = {}
    for k, (kind, lo, hi) in SPACE.items():
        u = float(rng.uniform())
        if kind == "uniform":
            out[k] = round(lo + u * (hi - lo), 4)
        else:
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            out[k] = int(round(v)) if kind == "int_log" else float(f"{v:.4g}")
    return out


def effective_params(index: int) -> dict:
    cfg = train_config_from_spec(yaml.safe_load(BASE_CONFIG.read_text()))
    return {k: cfg.params[k] for k in SPACE} | sample(index)


def train_features(spec: dict) -> pl.DataFrame:
    """The labelled feature frame `prc.pipeline.main` builds for this recipe (same calls, same order)."""
    cfg = train_config_from_spec(spec)
    use_weather, wx_codes, wind_dir, gust_repair = weather_flags(spec)
    train, rank, _ = load_raw(RAW)
    f_tr, f_rk = build_features(train, rank, use_weather, WORK, RAW, wx_codes, wind_dir, gust_repair)
    if cfg.fe_arr_fill:
        raise SystemExit("fe_arr_fill is not handled here")
    if cfg.fe_plan_nm_taxi or cfg.fe_fwd_nm or cfg.fe_fwd_nm_unmatched or cfg.fe_push_queue:
        from prc.levers import attach_levers

        f_tr, f_rk = attach_levers(
            f_tr,
            f_rk,
            train,
            rank,
            l1=cfg.fe_plan_nm_taxi,
            l2=cfg.fe_fwd_nm or cfg.fe_fwd_nm_unmatched,
            l3=cfg.fe_push_queue,
        )
    del train, rank, f_rk
    gc.collect()
    return drop_hidden_candidates(f_tr, cfg)


def run_trial(index: int) -> dict:
    t0 = time.time()
    spec = yaml.safe_load(BASE_CONFIG.read_text())
    override = sample(index)
    if override:
        spec["matched_params"] = override
    cfg = train_config_from_spec(spec)
    feats = train_features(spec)
    months = year_month_key(feats)
    tr = feats.filter(months.is_in(list(TUNE_FOLD.train_months)))
    va = feats.filter(months.is_in(list(TUNE_FOLD.valid_months)) & pl.col(PROXY).is_not_null())
    del feats, months
    gc.collect()
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
    rec = {
        "index": index,
        "status": "ok",
        "matched_params": override,
        "params": effective_params(index),
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


def _write(index: int, rec: dict) -> None:
    TRIALS.mkdir(parents=True, exist_ok=True)
    path = TRIALS / f"trial_{index:03d}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    with open(tmp) as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)


def _cgroups() -> list[Path]:
    rel = next(
        line.split(":", 2)[2].strip()
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.split(":")[1] == "memory"
    )
    own = Path("/sys/fs/cgroup/memory") / rel.lstrip("/")
    return [own, own.parent]


def total_rss_mb() -> list[int]:
    out = []
    for cg in _cgroups():
        stat = dict(line.split() for line in (cg / "memory.stat").read_text().splitlines())
        out.append(int(stat["total_rss"]) // 1_048_576)
    return out


def n_adsb() -> int:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return sum(1 for line in ps.splitlines() if "adsb_learn_" in line)


def gate(log) -> None:
    """Wait (polling every 60 s) until no adsb_learn_ process runs and total_rss < 2,500 MB."""
    waited = 0
    while True:
        rss, n = total_rss_mb(), n_adsb()
        if n == 0 and max(rss) < GATE_RSS_MB:
            if waited:
                log(f"gate open after {waited} s (rss {rss} MB)")
            return
        if waited % 600 == 0:
            log(f"gate closed: adsb_learn_ procs {n}, total_rss {rss} MB; polling every 60 s")
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
            _write(i, rec)
            res.unlink()
            log(f"trial {i} ok rmse_matched={rec['rmse_matched']:.3f} wall={wall}s peak={peak}MB")
        elif killed:
            _write(
                i,
                {
                    "index": i,
                    "status": "killed_memory",
                    "matched_params": sample(i),
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
    print("idx  rmse_m    d_vs_0  leaves  min_leaf  ff     bf     l2       lr      rounds  wall_s  peak_MB")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['index']:>3}  {r['status']}  {r['matched_params']}")
            continue
        p = r["params"]
        print(
            f"{r['index']:>3}  {r['rmse_matched']:.3f}  {r['rmse_matched'] - base:+7.3f}  {p['num_leaves']:>6}  "
            f"{p['min_data_in_leaf']:>8}  {p['feature_fraction']:.3f}  {p['bagging_fraction']:.3f}  "
            f"{p['lambda_l2']:<7.4g}  {p['learning_rate']:<6.4g}  {r['rounds_matched']:>6}  {r.get('wall_s', 0):>6.0f}  "
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
        print(json.dumps(best()["matched_params"]))
        return 0
    for i in range(a.n or 10):
        print(i, sample(i))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
