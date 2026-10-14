"""Run a recipe's CV one fold per process, through the pipeline's fold cache (memory only; same results).

    .venv/bin/python scripts/cv_per_fold.py fold --config C --name F --fold-cache DIR   # one fold, this process
    .venv/bin/python scripts/cv_per_fold.py run  --config C --fold-cache DIR [--log-dir D] # every fold, gated

catboost-retune.md §2. `prc.pipeline` fits all folds in one process that also holds the full 2.08M-row feature
frame, and with the retuned CatBoost settings that peaks above the container's memory. Here each fold runs in its
own process: it builds the labelled feature frame exactly as `prc.pipeline.main` does (scripts/retune_lgb.py
`train_features`, the same calls in the same order), slices the fold with the same `month_block_folds` /
`cfg.folds` selection as `run_cv`, drops the full frame, and fits through `prc.train._cached_fold` with the same
exp id, so the fold lands in `<fold-cache>/<exp_id>/<fold>.parquet|.json` exactly as `run_cv` would write it.
Afterwards the ordinary `prc.pipeline --skip-predict --fold-cache DIR` loads every fold from the cache and writes the
OOF and the record (run_cv's documented resume path), so nothing downstream changes.

`run` starts each missing fold only when no prc.pipeline / adsb_learn_ / retune process runs and total_rss over all
session cgroups is < 2,500 MB, samples total_rss every 10 s (own cgroup and all sessions) and kills the fold above
13,000 MB. PRC_CB_SNAPSHOT_DIR is set per fold (<fold-cache>/snapshots/<fold>), so a fold interrupted by a restart
resumes from CatBoost's snapshot; the snapshots of a fold are deleted once its cache entry exists.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retune_lgb import train_features  # noqa: E402

from prc.cv import month_block_folds, year_month_key  # noqa: E402
from prc.pipeline import train_config_from_spec  # noqa: E402
from prc.train import _cached_fold, predict_fold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEM_KILL_MB = 13_000
GATE_RSS_MB = 2_500


def fold_list(cfg, months) -> list:
    folds = month_block_folds(months.unique().to_list(), named=cfg.folds or ())
    if cfg.folds:
        folds = [f for f in folds if f.name in set(cfg.folds)]
    return folds


def slice_fold(feats, cfg, name: str):
    """The (train, valid) frames `run_cv` builds for fold `name` (same fold list, same month filters)."""
    months = year_month_key(feats)
    (f,) = [x for x in fold_list(cfg, months) if x.name == name]
    return feats.filter(months.is_in(list(f.train_months))), feats.filter(months.is_in(list(f.valid_months)))


def fit_fold(tr, va, cfg, name: str, fold_cache: Path):
    """`run_cv`'s per-fold call: `_cached_fold(<cache>/<exp_id>, fold, exp_id, predict_fold(tr, va, cfg))`."""
    exp_id = f"{cfg.name}-{cfg.hash()}"
    return _cached_fold(Path(fold_cache) / exp_id, name, exp_id, lambda: predict_fold(tr, va, cfg))


def run_fold(config: Path, name: str, fold_cache: Path) -> int:
    spec = yaml.safe_load(config.read_text())
    cfg = train_config_from_spec(spec)
    feats = train_features(spec)
    tr, va = slice_fold(feats, cfg, name)
    del feats  # the point of this script: only the fold's frames stay resident
    gc.collect()
    t0 = time.time()
    p, _info = fit_fold(tr, va, cfg, name, fold_cache)
    print(f"fold {name}: {p.height:,} predictions, {time.time() - t0:.0f}s, exp_id {cfg.name}-{cfg.hash()}", flush=True)
    return 0


def _all_rss_mb() -> tuple[int, int]:
    rel = next(
        line.split(":", 2)[2].strip()
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.split(":")[1] == "memory"
    )
    own = Path("/sys/fs/cgroup/memory") / rel.lstrip("/")

    def rss(d: Path) -> int:
        stat = dict(line.split() for line in (d / "memory.stat").read_text().splitlines())
        return int(stat["total_rss"]) // 1_048_576

    total = sum(rss(d) for d in Path("/sys/fs/cgroup/memory/process_api").iterdir() if (d / "memory.stat").exists())
    return rss(own), total


def _busy() -> int:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=False).stdout
    pats = ("prc.pipeline", "adsb_learn_", "retune_cb.py", "retune_lgb.py", "cv_per_fold.py fold")
    return sum(1 for line in ps.splitlines() if any(p in line for p in pats))


def run_all(config: Path, fold_cache: Path, log_dir: Path) -> int:
    spec = yaml.safe_load(config.read_text())
    cfg = train_config_from_spec(spec)
    exp_id = f"{cfg.name}-{cfg.hash()}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logf = open(log_dir / "per_fold.log", "a")  # noqa: SIM115

    def log(msg: str) -> None:
        logf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
        logf.flush()

    log(f"run start {exp_id} pid={os.getpid()} cgroup={Path('/proc/self/cgroup').read_text().strip()}")
    names = list(cfg.folds) if cfg.folds else None
    if names is None:
        raise SystemExit("this driver needs the recipe's folds key")
    for name in names:
        done = (fold_cache / exp_id / f"{name}.json").exists()
        if done:
            log(f"fold {name}: cached")
            continue
        waited = 0
        while True:
            own, tot = _all_rss_mb()
            if _busy() == 0 and tot < GATE_RSS_MB:
                break
            if waited % 600 == 0:
                log(f"gate closed (busy {_busy()}, rss own {own} / all {tot} MB)")
            time.sleep(60)
            waited += 60
        snap = fold_cache / "snapshots" / name
        env = {**os.environ, "PRC_CB_SNAPSHOT_DIR": str(snap), "PYTHONUNBUFFERED": "1"}
        log(
            f"fold {name}: start (snapshots {snap}, resuming {len(list(snap.rglob('*.cbsnap'))) if snap.exists() else 0})"
        )
        start, peak, killed = time.time(), [0, 0], False
        with open(log_dir / f"fold_{name}.log", "a") as fl:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "fold",
                    "--config",
                    str(config),
                    "--name",
                    name,
                    "--fold-cache",
                    str(fold_cache),
                ],
                stdout=fl,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
            with open(log_dir / f"fold_{name}.mem", "a") as fm:
                while proc.poll() is None:
                    own, tot = _all_rss_mb()
                    peak = [max(peak[0], own), max(peak[1], tot)]
                    fm.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {own} {tot}\n")
                    fm.flush()
                    if max(own, tot) > MEM_KILL_MB:
                        os.killpg(proc.pid, 9)
                        killed = True
                        proc.wait()
                        break
                    time.sleep(10)
        wall = round(time.time() - start)
        rec = {"fold": name, "rc": proc.returncode, "killed": killed, "wall_s": wall, "peak_own_all_mb": peak}
        log(f"fold {name}: end {json.dumps(rec)}")
        if killed:
            log(f"fold {name}: KILLED above {MEM_KILL_MB} MB; stopping")
            return 2
        if proc.returncode != 0 or not (fold_cache / exp_id / f"{name}.json").exists():
            log(f"fold {name}: FAILED; stopping")
            return 1
        shutil.rmtree(snap, ignore_errors=True)
    log("run end: every fold cached")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["fold", "run"])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--name")
    ap.add_argument("--fold-cache", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=Path("logs/cbretune"))
    a = ap.parse_args()
    if a.what == "fold":
        return run_fold(a.config, a.name, a.fold_cache)
    return run_all(a.config, a.fold_cache, a.log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
