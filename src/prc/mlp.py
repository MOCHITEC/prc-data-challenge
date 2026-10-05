"""Model A as a multilayer perceptron (scikit-learn), for the matched-lane blend member M4.

Pre-registration and results: docs/competitor_research_2026-09-21/results/mlp-member.md.

Only `predict_fold` calls this, and only when `TrainConfig.mlp_matched` is on. It takes the place of `_fit` for
model A and keeps `_fit`'s contract:

1. The fold's training rows are the only rows any statistic or fit reads. The preprocessing (quantiles, the
   categorical level lists, which columns get a missing indicator) is label-free and fitted on the training
   rows; the target's mean and scale are fitted on the rows the stage trains on.
2. Early stopping reads the inner holdout of the training side (`_inner_holdout_mask`, the latest training
   month), never the evaluation fold. It picks the epoch count E* and the learning-rate schedule. The final
   models are refitted on all training rows for E* epochs with that schedule replayed, one per seed
   (`seed + k`, k < n_seeds), and their predictions are averaged.
3. Predictions are returned in model space (the anchored target g), like `_fit`'s.

Checkpoints. With the environment variable PRC_MLP_CKPT set to a directory, every epoch of every stage is
written there (atomically), keyed by a digest of the stage's inputs, and a re-run of the same stage resumes after
its last finished epoch. Unset, nothing is read or written. The epoch's row order is a function of (seed, epoch)
alone, so a resumed stage sees the same batches it would have seen uninterrupted."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# Every setting explicit. Fixed in the pre-registration (mlp-member.md §0.3); a recipe may override any key
# through `TrainConfig.mlp_params`, and the resolved dict is part of the experiment id.
MLP_DEFAULTS: dict = {
    "hidden": [512, 256],
    "activation": "relu",
    "alpha": 1e-4,  # scikit-learn's L2 penalty
    "batch_size": 1024,
    "learning_rate": 1e-3,  # Adam, scikit-learn's default betas and epsilon
    "max_epochs": 40,
    "patience": 5,  # stop after this many epochs without a new best holdout RMSE
    "lr_halve_every": 2,  # halve the rate each time the epochs-since-best count reaches a multiple of this
    "chunk_rows": 65536,  # rows per partial_fit call; one call is one pass over the chunk in mini-batches
    "quantiles": 1000,
    "quantile_rows": 200000,
    "cat_min_count": 1000,
    "cat_max_levels": 64,
}
MLP_KEYS = frozenset(MLP_DEFAULTS)


def resolve(params: dict) -> dict:
    unknown = sorted(set(params) - MLP_KEYS)
    if unknown:
        raise ValueError(f"unknown mlp_params keys {unknown}; allowed: {sorted(MLP_KEYS)}")
    return {**MLP_DEFAULTS, **params}


# ---------- preprocessing ----------


class Preprocessor:
    """Numeric: inf -> NaN, then a normal-output quantile transform fitted on a fixed subsample of the training
    rows. NaN after the transform -> 0 (the median), plus a 0/1 indicator for every column that has a NaN on
    the training rows. Categorical: the string value (null -> "__NA__"); one-hot over the levels with at least
    `cat_min_count` training rows, at most `cat_max_levels` of the most frequent (ties by name), and one
    "other" column for everything else, unseen levels included."""

    def __init__(self, p: dict, seed: int):
        self.p, self.seed = p, seed

    def fit(self, frame: pl.DataFrame, num_cols: list[str], cat_cols: list[str]) -> Preprocessor:
        from sklearn.preprocessing import QuantileTransformer

        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        X = self._raw_num(frame)
        n = X.shape[0]
        k = min(n, int(self.p["quantile_rows"]))
        sub = np.sort(np.random.default_rng(self.seed).choice(n, size=k, replace=False))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-null column: its quantiles are NaN, so is its output
            self.qt = QuantileTransformer(
                n_quantiles=max(2, min(int(self.p["quantiles"]), k)),
                output_distribution="normal",
                subsample=k + 1,  # the subsample above is the whole sample; no second draw inside
                random_state=self.seed,
            ).fit(X[sub])
        self.nan_cols = np.flatnonzero(np.isnan(X).any(axis=0))
        self.levels: list[dict[str, int]] = []
        for c in self.cat_cols:
            vc = self._cat_str(frame, c).value_counts(name="n").sort(["n", c], descending=[True, False])
            keep = vc.filter(pl.col("n") >= int(self.p["cat_min_count"])).head(int(self.p["cat_max_levels"]))
            self.levels.append({v: i for i, v in enumerate(keep[c].to_list())})
        self.width = len(self.num_cols) + len(self.nan_cols) + sum(len(lv) + 1 for lv in self.levels)
        return self

    def _raw_num(self, frame: pl.DataFrame) -> np.ndarray:
        X = frame.select(self.num_cols).cast(pl.Float32).to_numpy()
        X[~np.isfinite(X)] = np.nan
        return X

    @staticmethod
    def _cat_str(frame: pl.DataFrame, c: str) -> pl.Series:
        return frame[c].cast(pl.Utf8).fill_null("__NA__").alias(c)

    def num(self, frame: pl.DataFrame, block: int = 200_000) -> np.ndarray:
        """The transformed numeric block, float32, NaN kept (filled in `design`)."""
        X = self._raw_num(frame)
        out = np.empty_like(X)
        for s in range(0, X.shape[0], block):
            out[s : s + block] = self.qt.transform(X[s : s + block]).astype(np.float32)
        return out

    def codes(self, frame: pl.DataFrame) -> np.ndarray:
        C = np.empty((frame.height, len(self.cat_cols)), dtype=np.int16)
        for j, (c, lv) in enumerate(zip(self.cat_cols, self.levels)):
            if not lv:
                C[:, j] = 0
                continue
            s = self._cat_str(frame, c)
            C[:, j] = s.replace_strict(lv, default=len(lv), return_dtype=pl.Int16).to_numpy()
        return C

    def design(self, Xn: np.ndarray, C: np.ndarray) -> np.ndarray:
        n = Xn.shape[0]
        out = np.zeros((n, self.width), dtype=np.float32)
        k = Xn.shape[1]
        nan = np.isnan(Xn)
        out[:, :k] = np.where(nan, np.float32(0.0), Xn)
        out[:, k : k + len(self.nan_cols)] = nan[:, self.nan_cols]
        off = k + len(self.nan_cols)
        rows = np.arange(n)
        for j, lv in enumerate(self.levels):
            out[rows, off + C[:, j].astype(np.int64)] = 1.0
            off += len(lv) + 1
        return out

    def summary(self) -> dict:
        return {
            "n_num": len(self.num_cols),
            "n_nan_indicators": int(len(self.nan_cols)),
            "cat_levels": {c: len(lv) for c, lv in zip(self.cat_cols, self.levels)},
            "input_width": int(self.width),
        }


# ---------- checkpointed epoch loop ----------


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str((p.dtype, p.shape)).encode())
            h.update(np.ascontiguousarray(p).view(np.uint8).ravel())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def _ckpt_dir() -> Path | None:
    d = os.environ.get("PRC_MLP_CKPT")
    return Path(d) if d else None


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _new_model(p: dict, seed: int):
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(
        hidden_layer_sizes=tuple(int(h) for h in p["hidden"]),
        activation=p["activation"],
        solver="adam",
        alpha=float(p["alpha"]),
        batch_size=int(p["batch_size"]),
        learning_rate_init=float(p["learning_rate"]),
        shuffle=False,  # the epoch's row order is drawn here, from (seed, epoch)
        random_state=seed,
    )


def _predict(model, pre: Preprocessor, Xn: np.ndarray, C: np.ndarray, block: int) -> np.ndarray:
    out = np.empty(Xn.shape[0], dtype=np.float64)
    for s in range(0, Xn.shape[0], block):
        out[s : s + block] = model.predict(pre.design(Xn[s : s + block], C[s : s + block]))
    return out


def _epoch(model, pre, Xn, C, y_std, rows, seed, epoch, lr, p):
    if hasattr(model, "_optimizer"):
        model._optimizer.learning_rate_init = lr
    else:
        model.set_params(learning_rate_init=lr)
    perm = rows[np.random.default_rng([seed, epoch]).permutation(rows.shape[0])]
    step = int(p["chunk_rows"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0, perm.shape[0], step):
            b = perm[s : s + step]
            model.partial_fit(pre.design(Xn[b], C[b]), y_std[b])


def _log(msg: str) -> None:
    print(msg, flush=True)


def _stage(key, p, seed, pre, Xn, C, y, rows, n_epochs, lrs=None, hold=None, log=_log):
    """Train one model on `rows`. With `hold` = (rows, y_seconds) and `lrs` None: early stopping with the
    halving rule, returning the history. With `lrs`: exactly len(lrs) epochs at those rates."""
    mu, sd = float(y[rows].mean()), float(y[rows].std() or 1.0)
    y_std = ((y - mu) / sd).astype(np.float32)
    ck = _ckpt_dir()
    path = None if ck is None else ck / f"{key}.pkl"
    st = _load(path) if path is not None else None
    if st is None:
        st = {"model": _new_model(p, seed), "epoch": 0, "hist": [], "lrs": [], "best": (np.inf, 0), "bad": 0}
        st["lr"] = float(p["learning_rate"])
    else:
        log(f"    resume {key} after epoch {st['epoch']}")
    model = st["model"]
    total = len(lrs) if lrs is not None else n_epochs
    while st["epoch"] < total:
        e = st["epoch"] + 1
        lr = lrs[e - 1] if lrs is not None else st["lr"]
        t = time.time()
        _epoch(model, pre, Xn, C, y_std, rows, seed, e, lr, p)
        st["lrs"].append(lr)
        msg = f"    epoch {e}/{total} lr {lr:.2e} {time.time() - t:.0f}s"
        if hold is not None:
            h_rows, h_y = hold
            r = float(np.sqrt(np.mean((_predict(model, pre, Xn[h_rows], C[h_rows], 200_000) * sd + mu - h_y) ** 2)))
            st["hist"].append(r)
            if r < st["best"][0]:
                st["best"], st["bad"] = (r, e), 0
            else:
                st["bad"] += 1
                if st["bad"] % int(p["lr_halve_every"]) == 0:
                    st["lr"] = st["lr"] * 0.5
            msg += f" holdout {r:.3f} best {st['best'][0]:.3f}@{st['best'][1]}"
        st["epoch"] = e
        log(msg)
        stop = hold is not None and st["bad"] >= int(p["patience"])
        if stop:
            st["stopped"] = True
        if path is not None:
            _save(path, st)
        if stop:
            break
    return model, mu, sd, st


def fit_mlp(
    train: pl.DataFrame,
    target_expr,
    cfg,
    feature_cols: list[str],
    cat_cols: list[str],
    predict_frames: list[pl.DataFrame],
    info: dict,
) -> tuple[list[np.ndarray], int]:
    from .train import _inner_holdout_mask

    t0 = time.time()
    p = resolve(cfg.mlp_params)
    pre = Preprocessor(p, cfg.seed).fit(train, feature_cols, cat_cols)
    Xn, C = pre.num(train), pre.codes(train)
    y = np.asarray(target_expr(train), dtype=np.float64)
    hold = _inner_holdout_mask(train, cfg.inner_holdout_months)
    n_hold = int(hold.sum())
    all_rows = np.arange(train.height)
    data_key = _digest(p, Xn, C, y, pre.summary())
    rec: dict = {"params": p, **pre.summary(), "train_rows": int(train.height), "holdout_rows": n_hold}

    if 0 < n_hold < train.height:
        fit_rows, h_rows = np.flatnonzero(~hold), np.flatnonzero(hold)
        _, _, _, st = _stage(
            _digest(data_key, "es", cfg.seed, hold),
            p,
            cfg.seed,
            pre,
            Xn,
            C,
            y,
            fit_rows,
            int(p["max_epochs"]),
            hold=(h_rows, y[h_rows]),
        )
        best_epoch = int(st["best"][1])
        lrs = st["lrs"][:best_epoch]
        rec.update(
            es_holdout_rmse=st["hist"],
            es_best_epoch=best_epoch,
            es_epochs_run=int(st["epoch"]),
            es_lrs=st["lrs"],
        )
    else:
        lrs = [float(p["learning_rate"])] * int(p["max_epochs"])
        rec["es_best_epoch"] = None
    pf_num = [(pre.num(f), pre.codes(f)) for f in predict_frames]
    acc = [np.zeros(f.height) for f in predict_frames]
    for k in range(max(1, cfg.n_seeds)):
        seed = cfg.seed + k
        key = _digest(data_key, "refit", seed, lrs)
        ck = _ckpt_dir()
        pred_path = None if ck is None else ck / f"{key}.pred.pkl"
        got = _load(pred_path) if pred_path is not None else None
        if got is None:
            model, mu, sd, _ = _stage(key, p, seed, pre, Xn, C, y, all_rows, len(lrs), lrs=lrs)
            got = [_predict(model, pre, a, b, 200_000) * sd + mu for a, b in pf_num]
            if pred_path is not None:
                _save(pred_path, got)
        for j, g in enumerate(got):
            acc[j] += g
    rec["epochs"] = len(lrs)
    rec["seconds"] = round(time.time() - t0, 1)
    info["mlp_matched"] = rec
    return [a / max(1, cfg.n_seeds) for a in acc], len(lrs)
