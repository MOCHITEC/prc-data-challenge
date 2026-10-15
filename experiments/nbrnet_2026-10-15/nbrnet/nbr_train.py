"""Train NbrNet (ctx) or its control twin (noctx) on one fold, or micro-benchmark it (PREREG.md §3, §6).

    python experiments/nbrnet_2026-10-15/nbrnet/nbr_train.py --fold forward --seed 1 --mode ctx
    python experiments/nbrnet_2026-10-15/nbrnet/nbr_train.py --fold forward --seed 1 --mode ctx --bench-steps 200

Reads the fold arrays from nbr_query.py and the token tables/index from nbr_tokens.py. Writes
  data/nbrnet/runs/<run>/ckpt.pt         full checkpoint (model, optimiser, epoch, RNG, best weights, history)
  <exp>/ckpt/<run>.pt                    the same when <= 2 MB, otherwise a light copy (no optimiser state, fp16
                                         weights) when that fits; committed and pushed every 2 epochs
  <exp>/preds/<run>.parquet              the best epoch's predictions on the evaluation fold's matched rows
  <exp>/runs/<run>.json                  settings, target statistics, per-epoch history, timings"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import polars as pl
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from nbr_model import NbrNet
from nbr_tokens import ARR_DYNAMIC, ARR_STATIC, DEP_DYNAMIC, DEP_STATIC, Tables, gather

GIT_CKPT_MAX = 2 * 1024 * 1024
WINSOR = (0.0005, 0.9995)


def load_fold(d: Path) -> dict:
    f = {k: np.load(d / f"{k}.npy") for k in ("tr_Xn", "tr_C", "tr_y", "tr_proxy", "tr_row", "tr_id", "tr_hold", "va_Xn", "va_C", "va_y", "va_proxy", "va_row", "va_id")}
    with open(d / "pre.pkl", "rb") as fh:
        f["pre"] = pickle.load(fh)
    f["meta"] = json.loads((d / "fold.json").read_text())
    return f


def target_stats(g_train: np.ndarray) -> dict:
    """Winsor bounds and standardisation, from the fold's training rows only (R5)."""
    lo, hi = np.quantile(g_train, WINSOR)
    gw = np.clip(g_train, lo, hi)
    return {"lo": float(lo), "hi": float(hi), "mu": float(gw.mean()), "sd": float(gw.std() or 1.0)}


class Batcher:
    """Builds (xq, tokens...) torch tensors for a set of query rows (indices into the fold arrays)."""

    def __init__(self, Xn, C, qrow, pre, tb, idx_d, idx_a, cov, k_dep, k_arr, ctx: bool):
        self.Xn, self.C, self.qrow, self.pre = Xn, C, qrow, pre
        self.tb, self.idx_d, self.idx_a, self.cov = tb, idx_d, idx_a, cov
        self.k_dep, self.k_arr, self.ctx = k_dep, k_arr, ctx
        self.t_gather = 0.0

    def __call__(self, rows: np.ndarray):
        t = time.perf_counter()
        xq = torch.from_numpy(self.pre.design(self.Xn[rows], self.C[rows]))
        if not self.ctx:
            self.t_gather += time.perf_counter() - t
            return (xq,)
        q = self.qrow[rows]
        dep, md, arr, ma = gather(self.tb, q, self.idx_d, self.idx_a, self.k_dep, self.k_arr)
        out = (
            xq,
            torch.from_numpy(dep),
            torch.from_numpy(md),
            torch.from_numpy(arr),
            torch.from_numpy(ma),
            torch.from_numpy(np.asarray(self.cov[q], dtype=np.float32)),
        )
        self.t_gather += time.perf_counter() - t
        return out


def predict(model, batcher, rows: np.ndarray, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = np.empty(len(rows), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            out[s : s + bs] = model(*batcher(rows[s : s + bs])).numpy()
    model.train()
    return out


def _git_push(paths: list[Path], msg: str) -> str:
    trailer = (
        "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01U4r4yp4haN4XVEVVfmBd5H"
    )
    rel = [str(p.relative_to(ROOT)) for p in paths]
    for attempt in range(5):
        try:
            subprocess.run(["git", "-C", str(ROOT), "add", "-f", *rel], check=True, capture_output=True)
            r = subprocess.run(check=False, args=["git", "-C", str(ROOT), "commit", "-q", "-m", msg + trailer, "--", *rel], capture_output=True, text=True)
            if r.returncode != 0 and "nothing" not in (r.stdout + r.stderr):
                raise RuntimeError(r.stderr.strip()[:200])
            p = subprocess.run(check=False, args=["git", "-C", str(ROOT), "push", "-q", "origin", "HEAD"], capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(p.stderr.strip()[:200])
            return "pushed"
        except Exception as e:  # noqa: BLE001 - a failed push must never kill a training run
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return f"push failed: {err}"


def save_ckpt(path: Path, state: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path.stat().st_size


def light_state(state: dict) -> dict:
    half = lambda sd: {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
    out = {k: v for k, v in state.items() if k not in ("optim", "model", "best_state")}
    out["model_fp16"] = half(state["model"])
    out["best_state_fp16"] = half(state["best_state"]) if state["best_state"] is not None else None
    out["light"] = True
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="forward")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["ctx", "noctx"], default="ctx")
    ap.add_argument("--k-dep", type=int, default=64)
    ap.add_argument("--k-arr", type=int, default=48)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--ckpt-every", type=int, default=2)
    ap.add_argument("--bench-steps", type=int, default=0, help="time this many training steps and stop")
    ap.add_argument("--folds-dir", type=Path, default=ROOT / "data/nbrnet/folds")
    ap.add_argument("--tokens", type=Path, default=ROOT / "data/nbrnet/tokens")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "data/nbrnet/runs")
    ap.add_argument("--exp-dir", type=Path, default=EXP, help="where ckpt/, preds/, runs/, bench/ go")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    t_start = time.time()
    exp = a.exp_dir
    run = f"{a.fold}_s{a.seed}_{a.mode}" + (f"_{a.tag}" if a.tag else "")
    ctx = a.mode == "ctx"

    F = load_fold(a.folds_dir / a.fold)
    tb = Tables.load(a.tokens)
    idx_d = np.load(a.tokens / "idx_d.npy", mmap_mode="r")
    idx_a = np.load(a.tokens / "idx_a.npy", mmap_mode="r")
    cov = np.load(a.tokens / "cov.npy")
    pre = F["pre"]
    g_tr = F["tr_y"] - F["tr_proxy"]
    st = target_stats(g_tr)
    g_std = ((np.clip(g_tr, st["lo"], st["hi"]) - st["mu"]) / st["sd"]).astype(np.float32)
    hold = F["tr_hold"]
    fit_rows, hold_rows = np.flatnonzero(~hold), np.flatnonzero(hold)
    mk = lambda pfx: Batcher(F[f"{pfx}_Xn"], F[f"{pfx}_C"], F[f"{pfx}_row"], pre, tb, idx_d, idx_a, cov, a.k_dep, a.k_arr, ctx)
    btr, bva = mk("tr"), mk("va")

    torch.manual_seed(a.seed)
    model = NbrNet(pre.width, len(DEP_STATIC) + len(DEP_DYNAMIC), len(ARR_STATIC) + len(ARR_DYNAMIC), d=a.d, k_dep=a.k_dep, k_arr=a.k_arr)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    n_params = sum(p.numel() for p in model.parameters())
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    rec = {"run": run, "settings": settings, "target": st, "n_params": n_params, "query_width": int(pre.width),
           "n_fit": len(fit_rows), "n_hold": len(hold_rows), "n_valid": len(F["va_y"]),
           "fold": F["meta"].get("fold"), "hold_months": F["meta"].get("hold_months"), "torch": torch.__version__,
           "threads": torch.get_num_threads()}
    print(json.dumps(rec), flush=True)
    lossf = torch.nn.MSELoss()
    steps_per_epoch = int(np.ceil(len(fit_rows) * a.subsample / a.batch))

    if a.bench_steps:
        perm = np.random.default_rng([a.seed, 1]).permutation(len(fit_rows))
        t0, btr.t_gather = time.perf_counter(), 0.0
        for i in range(a.bench_steps):
            rows = fit_rows[perm[i * a.batch : (i + 1) * a.batch]]
            batch = btr(rows)
            loss = lossf(model(*batch), torch.from_numpy(g_std[rows]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dt = time.perf_counter() - t0
        te = time.perf_counter()
        nh = min(len(hold_rows), 20 * 4096)
        predict(model, btr, hold_rows[:nh])
        dt_eval = (time.perf_counter() - te) * len(hold_rows) / max(1, nh)
        per_step = dt / a.bench_steps
        epoch_min = (per_step * steps_per_epoch + dt_eval) / 60
        out = {**rec, "bench_steps": a.bench_steps, "sec_per_step": per_step, "gather_share": btr.t_gather / dt,
               "steps_per_epoch": steps_per_epoch, "holdout_eval_s": dt_eval, "epoch_min_est": epoch_min,
               "last_loss": loss.item(), "rss_mb": _rss_mb()}
        print("BENCH " + json.dumps(out), flush=True)
        (exp / "bench").mkdir(exist_ok=True)
        (exp / "bench" / f"{run}.json").write_text(json.dumps(out, indent=1))
        return 0

    run_dir = a.runs_dir / run
    ck_path = run_dir / "ckpt.pt"
    state = {"epoch": 0, "hist": [], "best": (float("inf"), 0), "bad": 0, "best_state": None}
    if ck_path.exists():
        saved = torch.load(ck_path, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optim"])
        torch.set_rng_state(saved["rng_torch"])
        state = {k: saved[k] for k in state}
        print(f"resumed {run} after epoch {state['epoch']}", flush=True)
    elif (exp / "ckpt" / f"{run}.pt").exists():
        saved = torch.load(exp / "ckpt" / f"{run}.pt", weights_only=False)
        if saved.get("light"):
            model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in saved["model_fp16"].items()})
            bs = saved["best_state_fp16"]
            saved["best_state"] = None if bs is None else {k: v.float() if v.is_floating_point() else v for k, v in bs.items()}
            rec["resumed_from_light_ckpt"] = int(saved["epoch"])  # optimiser state re-initialised: not an exact resume
        else:
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optim"])
        torch.set_rng_state(saved["rng_torch"])
        state = {k: saved[k] for k in state}
        print(f"resumed {run} from the git checkpoint after epoch {state['epoch']}", flush=True)

    g_hold = g_tr[hold_rows]
    while state["epoch"] < a.max_epochs and state["bad"] < a.patience:
        e = state["epoch"] + 1
        te = time.time()
        perm = np.random.default_rng([a.seed, e]).permutation(len(fit_rows))
        if a.subsample < 1.0:
            perm = perm[: int(len(fit_rows) * a.subsample)]
        btr.t_gather, tot = 0.0, 0.0
        for i, s in enumerate(range(0, len(perm), a.batch)):
            rows = fit_rows[perm[s : s + a.batch]]
            batch = btr(rows)
            loss = lossf(model(*batch), torch.from_numpy(g_std[rows]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(rows)
            if i % 500 == 0:
                print(f"  epoch {e} step {i}/{steps_per_epoch} loss {float(loss):.4f} {time.time() - te:.0f}s rss {_rss_mb()}MB", flush=True)
        t_train = time.time() - te
        ph = predict(model, btr, hold_rows) * st["sd"] + st["mu"]
        r = float(np.sqrt(np.mean((ph - g_hold) ** 2)))
        improved = r < state["best"][0]
        if improved:
            state["best"], state["bad"] = (r, e), 0
            state["best_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            state["bad"] += 1
        state["epoch"] = e
        state["hist"].append({"epoch": e, "train_mse_std": tot / len(perm), "holdout_rmse": r, "train_s": round(t_train, 1),
                              "epoch_s": round(time.time() - te, 1), "gather_s": round(btr.t_gather, 1), "rss_mb": _rss_mb()})
        print(f"epoch {e}: holdout {r:.3f} best {state['best'][0]:.3f}@{state['best'][1]} ({time.time() - te:.0f}s)", flush=True)
        stop = state["bad"] >= a.patience or e >= a.max_epochs
        if e % a.ckpt_every == 0 and not stop:
            full = {**state, "model": model.state_dict(), "optim": opt.state_dict(), "rng_torch": torch.get_rng_state(), "target": st, "settings": settings}
            size = save_ckpt(ck_path, full)
            gpath = exp / "ckpt" / f"{run}.pt"
            if size <= GIT_CKPT_MAX:
                save_ckpt(gpath, full)
            else:
                buf = io.BytesIO()
                torch.save(light_state(full), buf)
                if buf.tell() <= GIT_CKPT_MAX:
                    save_ckpt(gpath, light_state(full))
                else:
                    gpath = None
            msg = "no git checkpoint (too large)" if gpath is None else ("skipped (--no-git)" if a.no_git else _git_push([gpath], f"NbrNet checkpoint {run} epoch {e}"))
            print(f"  checkpoint epoch {e}: local {size / 1e6:.2f} MB; git {msg}", flush=True)

    model.load_state_dict(state["best_state"])
    t_pred = time.time()
    va_rows = np.arange(len(F["va_y"]))
    gv = predict(model, bva, va_rows) * st["sd"] + st["mu"]
    pred = F["va_proxy"] + gv
    (exp / "preds").mkdir(exist_ok=True)
    pl.DataFrame({"MVT_ID_mvt": F["va_id"], "nn_g": gv.astype(np.float32), "nn_pred": pred.astype(np.float32)}).write_parquet(
        exp / "preds" / f"{run}.parquet", compression="zstd", compression_level=19
    )
    rec.update(hist=state["hist"], best_epoch=state["best"][1], best_holdout_rmse=state["best"][0],
               epochs_run=state["epoch"], predict_s=round(time.time() - t_pred, 1), wall_s=round(time.time() - t_start, 1),
               valid_matched_rmse_nn=float(np.sqrt(np.mean((pred - F["va_y"]) ** 2))))
    (exp / "runs").mkdir(exist_ok=True)
    (exp / "runs" / f"{run}.json").write_text(json.dumps(rec, indent=1))
    print("DONE " + json.dumps({k: rec[k] for k in ("run", "best_epoch", "best_holdout_rmse", "epochs_run", "wall_s", "valid_matched_rmse_nn")}), flush=True)
    return 0


def _rss_mb() -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


if __name__ == "__main__":
    raise SystemExit(main())
