"""What LightGBM's determinism flags cost, and what turning them off would cost instead.

Produces every number in `docs/competitor_research_2026-09-21/results/determinism-cost.md`.

    python docs/competitor_research_2026-09-21/scripts/determinism_cost.py bench    --out <dir>
    python docs/competitor_research_2026-09-21/scripts/determinism_cost.py matrix   --out <dir>
    python docs/competitor_research_2026-09-21/scripts/determinism_cost.py fits     --out <dir> --n 12
    python docs/competitor_research_2026-09-21/scripts/determinism_cost.py analyse  --out <dir>

Nothing here writes to `outputs/`, `experiments/` or `data/`: every artefact lands under `--out`, which has
no default so a stray run cannot land in the repository. The only production file this reads is
`data/work/feat_train.parquet`.

Why each mode exists:

* `bench`   - the wall-clock claim. LightGBM alone, one flag corner per SUBPROCESS, several row counts, a
              fixed round count and a fixed thread count. Repeated, because the difference being measured
              turned out to be a couple of percent and one timing cannot see that.
* `matrix`  - G11, independently of the flags. Hashes `_matrix`'s output for the forward fold's model-A
              training frame in fresh processes. The design matrix must be identical whether the flags are on
              or off; if it is not, nothing downstream separates the two causes of nondeterminism.
* `fits`    - the spread. One forward-fold fit per subprocess, flags OFF, for two arms (`p4a_base` and
              `p4a_gap`), at `n_seeds=1` and `n_seeds=3`, plus two flags-ON control runs per arm. Separate
              processes matter: thread arrival order is a property of the process, so two calls inside one
              process are not two draws.
* `analyse` - the distribution of fold RMSE, the verdict simulation against `prc.compare.verdict`, the
              seed-averaging comparison and the flags-ON byte check.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import lightgbm as lgb  # noqa: E402

from prc.compare import robustness_curve, verdict  # noqa: E402
from prc.cv import rmse, year_month_key  # noqa: E402
from prc.features import CATEGORICALS  # noqa: E402
from prc.train import (  # noqa: E402
    LABEL,
    PROXY,
    TrainConfig,
    _matrix,
    attach_encodings,
    fit_encodings,
    predict_fold,
    visible_numeric_columns,
)

WORK = REPO / "data" / "work" / "feat_train.parquet"
FORWARD_VALID = (202511, 202512)
THREADS = 4  # TrainConfig.params["num_threads"]; the box has 4 CPUs
# The two arms of P4 stage A, unchanged: configs/main_reparam.yaml + folds:[forward], differing only in the
# one candidate column. Their archived, flags-ON records are p4a_base-58b999e0b7 and p4a_gap-ca4297619f.
ARMS = ("p4a_base", "p4a_gap")


# ---------------------------------------------------------------- shared loading


def load_cfg(arm: str) -> tuple[TrainConfig, dict]:
    """The same config-to-TrainConfig mapping `prc.pipeline.main` uses, so the arms are the shipped recipes."""
    spec = yaml.safe_load((REPO / "configs" / f"{arm}.yaml").read_text())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates", "unmatched_sched_reparam_airports"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    return TrainConfig(**fields), spec


def forward_split(feats: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    m = year_month_key(feats)
    return feats.filter(~m.is_in(list(FORWARD_VALID))), feats.filter(m.is_in(list(FORWARD_VALID)))


def model_a_training_matrix(cfg: TrainConfig) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Model A's design matrix on the forward fold's training side - what `bench` times and `matrix` hashes."""
    feats = pl.read_parquet(WORK)
    tr, _ = forward_split(feats)
    tr = tr.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    tr = attach_encodings(tr, fit_encodings(tr, cfg)).filter(pl.col(PROXY).is_not_null())
    num = visible_numeric_columns(tr, cfg)
    cols_a = [c for c in num if c != PROXY] + [PROXY]
    cats = [c for c in CATEGORICALS if c in tr.columns]
    X, cols = _matrix(tr, cols_a, cats)
    y = tr[LABEL].to_numpy().astype(float) - tr[PROXY].to_numpy().astype(float)
    return X, cols, y, cats


# ---------------------------------------------------------------- 1. the wall-clock bench

# Both flags, one flag, neither. `det_alone` and `row_only` are here because the claim under test came from
# one bench, and "the flags cost 10x" could only be true of a corner; measuring all of them says which.
CORNERS = {
    "on": {"deterministic": True, "force_row_wise": True},
    "det_alone": {"deterministic": True},
    "det_col": {"deterministic": True, "force_col_wise": True},
    "off": {},
    "row_only": {"force_row_wise": True},
}


def bench_one(corner: str, n_rows: int, rounds: int, threads: int) -> dict:
    cfg, _ = load_cfg("p4a_base")
    X, cols, y, cats = model_a_training_matrix(cfg)
    if n_rows > 0:
        X, y = X[:n_rows], y[:n_rows]
    p = {k: v for k, v in cfg.params.items() if k not in ("deterministic", "force_row_wise")}
    p.update(CORNERS[corner])
    p["num_threads"] = threads
    p["seed"] = 42
    cat_idx = [cols.index(c) for c in cats]
    d = lgb.Dataset(X, label=y, categorical_feature=cat_idx, params=p)
    t0 = time.time()
    lgb.train(p, d, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    wall = time.time() - t0
    return {
        "corner": corner,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "threads": threads,
        "rounds": rounds,
        "wall_s": wall,
        "s_per_round": wall / rounds,
    }


def cmd_bench(out: Path, repeats: int) -> None:
    """One subprocess per (corner, row count, repeat). A process per cell, because process start is where the
    thread pool is created and that is what the flags are alleged to interact with."""
    plan = [(110_000, 100), (0, 50)]  # 0 = every matched training row of the forward fold
    rows = []
    for n_rows, rounds in plan:
        for rep in range(repeats):
            for corner in CORNERS:
                r = _child(["bench-one", "--corner", corner, "--rows", str(n_rows), "--rounds", str(rounds)])
                r["repeat"] = rep
                rows.append(r)
                print(
                    f"  rows={r['rows']:>9,} {corner:<10} rep{rep} "
                    f"{r['wall_s']:7.2f}s  {r['s_per_round']:.4f} s/round",
                    flush=True,
                )
    (out / "bench.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------- 2. G11, independently of the flags


def cmd_matrix(out: Path, repeats: int) -> None:
    """The design matrix must be a function of the data alone, whichever way the flags are set.

    This is the claim `configs/determinism_check.yaml` was written for, isolated from LightGBM: if these
    hashes ever disagree, the spread measured below is not the spread of thread scheduling."""
    rows = []
    for det in (True, False):
        for rep in range(repeats):
            r = _child(["matrix-one", "--det", str(det)])
            r.update(det=det, repeat=rep)
            rows.append(r)
            print(f"  lgb_deterministic={det!s:<5} rep{rep}  sha256={r['sha256']}  rows={r['rows']:,}", flush=True)
    (out / "matrix.json").write_text(json.dumps(rows, indent=1))


def matrix_one(det: bool) -> dict:
    cfg, _ = load_cfg("p4a_base")
    cfg = replace(cfg, lgb_deterministic=det)
    X, cols, _y, _c = model_a_training_matrix(cfg)
    return {"sha256": hashlib.sha256(X.tobytes()).hexdigest()[:32], "rows": int(X.shape[0]), "cols": len(cols)}


# ---------------------------------------------------------------- 3. the spread, on the real pipeline


def fit_one(arm: str, det: bool, n_seeds: int, seed: int) -> dict:
    """One forward-fold fit of one arm. Returns the fold's metrics; writes its predictions beside them."""
    cfg, _ = load_cfg(arm)
    cfg = replace(cfg, lgb_deterministic=det, n_seeds=n_seeds, seed=seed)
    feats = pl.read_parquet(WORK)
    tr, va = forward_split(feats)
    t0 = time.time()
    pred, info = predict_fold(tr, va, cfg)
    wall = time.time() - t0
    joined = va.select(["MVT_ID_mvt", LABEL, "ADEP_mvt", PROXY]).join(pred, on="MVT_ID_mvt", how="inner")
    y = joined[LABEL].to_numpy().astype(float)
    p = joined["pred"].to_numpy().astype(float)
    matched = joined[PROXY].is_not_null().to_numpy()
    return {
        "arm": arm,
        "lgb_deterministic": det,
        "n_seeds": n_seeds,
        "seed": seed,
        "pid": os.getpid(),
        "wall_s": wall,
        "n": int(len(y)),
        "rmse_full": rmse(y, p),
        "rmse_matched": rmse(y[matched], p[matched]),
        "rmse_unmatched": rmse(y[~matched], p[~matched]),
        "rounds_matched": info.get("rounds_matched"),
        "rounds_unmatched": info.get("rounds_unmatched"),
        "pred_sha256": hashlib.sha256(p.tobytes()).hexdigest()[:32],
        "_ids": joined["MVT_ID_mvt"].to_numpy(),
        "_y": y,
        "_pred": p,
        "_matched": matched,
        "_airport": joined["ADEP_mvt"].to_numpy().astype(str),
        "_month": year_month_key(joined).to_numpy(),
    }


def cmd_fits(out: Path, n: int, seeds_list: tuple[int, ...]) -> None:
    """The runs. Each one is a separate process; each writes an .npz and appends a JSON line."""
    log = out / "fits.jsonl"
    jobs = []
    for arm in ARMS:
        # the flags-ON control: two runs of the same thing must be byte-identical
        jobs += [(arm, True, 3, 0), (arm, True, 3, 1)]
        for ns in seeds_list:
            jobs += [(arm, False, ns, r) for r in range(n)]
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done.add((r["arm"], r["lgb_deterministic"], r["n_seeds"], r["run"]))
    for arm, det, ns, run in jobs:
        if (arm, det, ns, run) in done:
            continue
        tag = f"{arm}_det{int(det)}_s{ns}_r{run}"
        t0 = time.time()
        r = _child(["fit-one", "--arm", arm, "--det", str(det), "--n-seeds", str(ns), "--npz", str(out / tag)])
        r["run"] = run
        with open(log, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(
            f"  {tag:<28} rmse={r['rmse_full']:8.3f}  rounds={r['rounds_matched']:>5}/{r['rounds_unmatched']:<4}"
            f"  {r['wall_s']:6.0f}s  (+{time.time() - t0 - r['wall_s']:.0f}s load)",
            flush=True,
        )


# ---------------------------------------------------------------- 4. analysis


def _load(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "fits.jsonl").read_text().splitlines()]


def _npz(out: Path, r: dict) -> dict:
    tag = f"{r['arm']}_det{int(r['lgb_deterministic'])}_s{r['n_seeds']}_r{r['run']}"
    z = np.load(out / f"{tag}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _sd_ci(x: np.ndarray) -> tuple[float, float, float]:
    """SD with a chi-square 95% interval on it. The point of choosing N is that this interval is reportable."""
    from scipy import stats  # noqa: PLC0415

    n = len(x)
    s = float(np.std(x, ddof=1))
    lo = s * np.sqrt((n - 1) / stats.chi2.ppf(0.975, n - 1))
    hi = s * np.sqrt((n - 1) / stats.chi2.ppf(0.025, n - 1))
    return s, float(lo), float(hi)


def spread_table(runs: list[dict]) -> list[dict]:
    out = []
    for arm, ns in itertools.product(ARMS, sorted({r["n_seeds"] for r in runs if not r["lgb_deterministic"]})):
        sel = [r for r in runs if r["arm"] == arm and not r["lgb_deterministic"] and r["n_seeds"] == ns]
        if len(sel) < 3:
            continue
        x = np.array([r["rmse_full"] for r in sel])
        s, lo, hi = _sd_ci(x)
        rd = np.array([r["rounds_matched"] for r in sel], dtype=float)
        out.append(
            {
                "arm": arm,
                "n_seeds": ns,
                "n_runs": len(sel),
                "mean": float(x.mean()),
                "sd": s,
                "sd_ci95": [lo, hi],
                "min": float(x.min()),
                "max": float(x.max()),
                "range": float(x.max() - x.min()),
                "rounds_mean": float(rd.mean()),
                "rounds_sd": float(rd.std(ddof=1)),
                "rounds_min": int(rd.min()),
                "rounds_max": int(rd.max()),
                "wall_mean_s": float(np.mean([r["wall_s"] for r in sel])),
            }
        )
    return out


def verdict_sim(out: Path, runs: list[dict], n_seeds: int, half: bool) -> dict:
    """Re-decide `stand_arrival_gap` under the noisy regime: every (base run i, gap run j) pair.

    `half` replaces the gap arm's predictions with `p_gap_j - 0.5 * mean_effect`, where the mean effect is the
    row-wise mean over the gap runs minus the row-wise mean over the base runs. That keeps one full run's
    worth of noise on the candidate and halves its expected effect, which is the question asked: a smaller
    effect measured with the same instrument, not the same effect measured better. It is a construction, not
    an observed arm."""
    base = [r for r in runs if r["arm"] == "p4a_base" and not r["lgb_deterministic"] and r["n_seeds"] == n_seeds]
    gap = [r for r in runs if r["arm"] == "p4a_gap" and not r["lgb_deterministic"] and r["n_seeds"] == n_seeds]
    B = [_npz(out, r) for r in base]
    G = [_npz(out, r) for r in gap]
    ids = B[0]["ids"]
    for z in B + G:
        assert np.array_equal(z["ids"], ids), "runs disagree about which rows they scored"
    y = B[0]["y"]
    eff = np.mean([z["pred"] for z in G], axis=0) - np.mean([z["pred"] for z in B], axis=0)
    rows = []
    for i, zb in enumerate(B):
        for j, zg in enumerate(G):
            pb = zb["pred"]
            pg = zg["pred"] - 0.5 * eff if half else zg["pred"]
            curve = robustness_curve(y, pb, pg, n_boot=200, seed=0)
            ok, why = verdict(curve)
            full = next(c for c in curve if c["dropped"] == 0)
            k200 = next(c for c in curve if c["dropped"] == 200)
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "adopt": bool(ok),
                    "why": why,
                    "diff": full["diff"],
                    "ci95": full["ci95"],
                    "diff200": k200["diff"],
                    "ci200": k200["ci95"],
                }
            )
    d = np.array([r["diff"] for r in rows])
    return {
        "n_seeds": n_seeds,
        "half": half,
        "pairs": len(rows),
        "adopt_rate": float(np.mean([r["adopt"] for r in rows])),
        "diff_mean": float(d.mean()),
        "diff_min": float(d.min()),
        "diff_max": float(d.max()),
        "rows": rows,
    }


def cmd_analyse(out: Path) -> None:
    runs = _load(out)
    res: dict = {"spread": spread_table(runs)}

    # the flags-ON control: same arm, same config, two processes
    ctl = {}
    for arm in ARMS:
        sel = [r for r in runs if r["arm"] == arm and r["lgb_deterministic"]]
        ctl[arm] = {
            "n": len(sel),
            "pred_sha256": sorted({r["pred_sha256"] for r in sel}),
            "rmse": sorted({round(r["rmse_full"], 10) for r in sel}),
            "rounds_matched": sorted({r["rounds_matched"] for r in sel}),
            "wall_mean_s": float(np.mean([r["wall_s"] for r in sel])) if sel else float("nan"),
            "identical": len({r["pred_sha256"] for r in sel}) == 1 and len(sel) > 1,
        }
    res["flags_on_control"] = ctl

    seeds_seen = sorted({r["n_seeds"] for r in runs if not r["lgb_deterministic"]})
    res["verdicts"] = [verdict_sim(out, runs, ns, h) for ns in seeds_seen for h in (False, True)]
    if (out / "bench.json").exists():
        res["bench"] = json.loads((out / "bench.json").read_text())
    if (out / "matrix.json").exists():
        res["matrix"] = json.loads((out / "matrix.json").read_text())
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("bench", "matrix")}, indent=1, default=str)[:6000])


# ---------------------------------------------------------------- subprocess plumbing


def _child(args: list[str]) -> dict:
    """Run one measurement in a fresh interpreter and read its single JSON line back.

    A fresh process per measurement is the whole design: LightGBM's thread pool, and the arrival order that
    decides the summation order, belong to the process."""
    cp = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args, "--out", "/dev/null"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    return json.loads(cp.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["bench", "bench-one", "matrix", "matrix-one", "fits", "fit-one", "analyse"])
    ap.add_argument("--out", type=Path, required=True, help="scratch directory; never inside outputs/")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n", type=int, default=12, help="runs per (arm, n_seeds) cell")
    ap.add_argument("--seeds-list", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--corner", choices=list(CORNERS))
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--det", default="False")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--npz", type=Path)
    a = ap.parse_args(argv)
    det = a.det.lower() in ("1", "true", "yes")

    if a.mode == "bench-one":
        print(json.dumps(bench_one(a.corner, a.rows, a.rounds, THREADS)))
        return 0
    if a.mode == "matrix-one":
        print(json.dumps(matrix_one(det)))
        return 0
    if a.mode == "fit-one":
        r = fit_one(a.arm, det, a.n_seeds, seed=42)
        arrays = {k[1:]: r.pop(k) for k in [k for k in list(r) if k.startswith("_")]}
        if a.npz:
            np.savez_compressed(a.npz.with_suffix(".npz"), **arrays)
        print(json.dumps(r))
        return 0

    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "bench":
        cmd_bench(a.out, a.repeats)
    elif a.mode == "matrix":
        cmd_matrix(a.out, a.repeats)
    elif a.mode == "fits":
        cmd_fits(a.out, a.n, tuple(a.seeds_list))
    elif a.mode == "analyse":
        cmd_analyse(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
