"""Probe: does a non-GBDT member (MLP with embeddings, ridge) add anything to the v12 clock mixture?

For the jan and jul folds (the scored months) this fits, on identical fold splits and features:
  * the v12 LightGBM taxi head and classifier (exactly prc.mixture's code path, no temperature)
  * an MLP taxi head (matched rows: target y - proxy, same as LightGBM model A) and ridge taxi head
  * an MLP clock-source classifier
and saves every head's predictions so blends can be scored offline. Nothing in the repo is modified.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from torch import nn

from prc.clocks import classes, with_candidates, with_clock_source
from prc.config import data_dir
from prc.cv import month_block_folds, year_month_key
from prc.features import CATEGORICALS, numeric_feature_columns
from prc.mixture import REAL_OFFBLOCK_SOURCES, _candidate_matrix, _class_probs
from prc.pipeline import build_features, load_raw
from prc.train import LABEL, PROXY, TrainConfig, _inner_holdout_mask, attach_encodings, fit_encodings, predict_fold

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(0)
np.random.seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- preprocessing shared by MLP and ridge (fitted on the training side only) ----------


class Prep:
    def __init__(self, tr: pl.DataFrame, num_cols: list[str], cat_cols: list[str], min_count: int = 20):
        self.num_cols, self.cat_cols = num_cols, cat_cols
        # statistics from a 300k-row sample, float32: the full float64 matrix and its sorted copies ran out of memory
        smp = tr.sample(n=min(300_000, tr.height), seed=0)
        X = smp.select([pl.col(c).cast(pl.Float32) for c in num_cols]).to_numpy()
        self.lo = np.nanpercentile(X, 0.1, axis=0).astype(np.float32)
        self.hi = np.nanpercentile(X, 99.9, axis=0).astype(np.float32)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu = np.nan_to_num(np.nanmean(Xc, axis=0)).astype(np.float32)
        sd = np.nanstd(Xc, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
        self.sd = sd.astype(np.float32)
        self.nan_cols = np.flatnonzero(np.isnan(X).mean(0) > 0.001)
        del X, Xc
        self.vocab = {}
        for c in cat_cols:
            vc = tr[c].cast(pl.Utf8).fill_null("__NA__").value_counts()
            keep = vc.filter(pl.col("count") >= min_count)[c].to_list()
            self.vocab[c] = {v: i + 1 for i, v in enumerate(sorted(keep))}  # 0 = rare / unseen

    def num(self, df: pl.DataFrame) -> np.ndarray:
        X = df.select([pl.col(c).cast(pl.Float32) for c in self.num_cols]).to_numpy()
        miss = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        np.clip(X, self.lo, self.hi, out=X)
        X -= self.mu
        X /= self.sd
        np.nan_to_num(X, copy=False, nan=0.0)
        return np.concatenate([X, miss], axis=1)

    def cat(self, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for c in self.cat_cols:
            m = self.vocab[c]
            s = df[c].cast(pl.Utf8).fill_null("__NA__").replace_strict(m, default=0, return_dtype=pl.Int64)
            cols.append(s.to_numpy())
        return np.stack(cols, axis=1).astype(np.int64)

    def card(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.cat_cols]


class MLP(nn.Module):
    def __init__(self, n_num: int, cards: list[int], out: int, width: int = 512, depth: int = 3, drop: float = 0.1):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(k, min(32, max(4, int(round(1.6 * k**0.56))))) for k in cards])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(drop)]
            d = width
        layers.append(nn.Linear(d, out))
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        e = [m(xc[:, i]) for i, m in enumerate(self.emb)]
        return self.net(torch.cat([xn, *e], dim=1))


def train_mlp(xn, xc, y, hold, out_dim, loss, epochs=6, bs=2048, lr=2e-3, ystd=1.0, tag=""):
    """Train on ~hold rows, keep the epoch with the best inner-holdout loss (never the evaluation fold)."""
    model = MLP(xn.shape[1], CARDS, out_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    idx_tr, idx_ho = np.flatnonzero(~hold), np.flatnonzero(hold)
    steps = epochs * int(np.ceil(len(idx_tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xn, Xc, Y = torch.from_numpy(xn), torch.from_numpy(xc), torch.from_numpy(y)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(idx_tr)
        t0 = time.time()
        for s in range(0, len(perm), bs):
            b = torch.from_numpy(perm[s : s + bs])
            opt.zero_grad()
            ls = loss(model(Xn[b], Xc[b]), Y[b])
            ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            hl = float(
                np.mean(
                    [
                        loss(model(Xn[idx_ho[s : s + 65536]], Xc[idx_ho[s : s + 65536]]), Y[idx_ho[s : s + 65536]]).item()
                        * len(idx_ho[s : s + 65536])
                        for s in range(0, len(idx_ho), 65536)
                    ]
                )
                * (len(range(0, len(idx_ho), 65536)) / len(idx_ho))
            )
        log(f"  {tag} epoch {ep} holdout loss {hl:.5f} ({time.time() - t0:.0f}s)")
        if hl < best:
            best, best_state = hl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_mlp(model, xn, xc):
    out = []
    with torch.no_grad():
        for s in range(0, len(xn), 65536):
            out.append(model(torch.from_numpy(xn[s : s + 65536]), torch.from_numpy(xc[s : s + 65536])).numpy())
    return np.concatenate(out)


def ridge(X, y, lam=10.0):
    X1 = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1).astype(np.float64)
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, X1.T @ y)


# ---------- main ----------

spec = yaml.safe_load(Path("configs/v12_weather_mechanism.yaml").read_text())
cfg = TrainConfig(**{k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__})
train, rank, _ = load_raw(data_dir() / "raw")
import os
feats, _ = build_features(train, rank, os.getenv("PROBE_WEATHER", "1") == "1", data_dir() / "work")
del train, rank
gc.collect()
log("features", feats.shape)
months = year_month_key(feats)
folds = [f for f in month_block_folds(months.unique().to_list(), "fast") if f.name in sys.argv[2:]]

for f in folds:
    log("fold", f.name)
    tr_f = feats.filter(months.is_in(list(f.train_months)))
    va_f = feats.filter(months.is_in(list(f.valid_months)))
    tr_f = tr_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    train_src = with_clock_source(with_candidates(tr_f), cfg.match_tol_s, with_rollover=False)
    target_c = with_candidates(va_f)
    target_src = with_clock_source(with_candidates(va_f), cfg.match_tol_s, with_rollover=True)["clock_src"]
    real = train_src.filter(pl.col("clock_src").is_in(list(REAL_OFFBLOCK_SOURCES))).drop("clock_src")

    # LightGBM taxi head (prc.train.predict_fold, unchanged)
    t0 = time.time()
    gp, ginfo = predict_fold(real, target_c, cfg)
    f_lgb = gp["pred"].to_numpy().astype(float)
    log("lgb taxi head", ginfo, f"{time.time() - t0:.0f}s")
    del gp, tr_f
    gc.collect()

    # MLP + ridge taxi head on matched rows, same encodings and feature columns as model A
    enc = fit_encodings(real, cfg)
    r_e, t_e = attach_encodings(real, enc), attach_encodings(target_c, enc)
    num_cols = [c for c in numeric_feature_columns(r_e) if c not in (LABEL, "month")]
    cat_cols = [c for c in CATEGORICALS if c in r_e.columns]
    m_tr = r_e.filter(pl.col(PROXY).is_not_null())
    matched_t = t_e[PROXY].is_not_null().to_numpy()
    prep = Prep(m_tr, num_cols, cat_cols)
    CARDS = prep.card()
    xn, xc = prep.num(m_tr), prep.cat(m_tr)
    g = (m_tr[LABEL].to_numpy() - m_tr[PROXY].to_numpy()).astype(np.float64)
    # the NN only has to correct the ordinary band; the tails stay with LightGBM and the routing (blend weights are
    # still judged on the unclipped label)
    g = np.clip(g, *np.percentile(g, [0.05, 99.95]))
    gstd = float(np.std(g))
    hold = _inner_holdout_mask(m_tr, cfg.inner_holdout_months)
    log("mlp taxi input", xn.shape, xc.shape, "hold", int(hold.sum()), "g std", gstd)
    model = train_mlp(
        xn, xc, (g / gstd).astype(np.float32)[:, None], hold, 1, nn.MSELoss(), epochs=int(os.getenv("PROBE_EPOCHS", "6")), tag="taxi"
    )
    t_m = t_e.filter(pl.Series(matched_t))
    xn_t, xc_t = prep.num(t_m), prep.cat(t_m)
    f_mlp = f_lgb.copy()
    f_mlp[matched_t] = np.clip(t_m[PROXY].to_numpy() + predict_mlp(model, xn_t, xc_t)[:, 0] * gstd, 0, None)
    # ridge on the same numeric block plus one-hot airport
    apt = sorted(m_tr["ADEP_mvt"].unique().to_list())

    def onehot(df):
        a = df["ADEP_mvt"].to_numpy()
        return np.stack([(a == x).astype(np.float32) for x in apt], 1)

    w = ridge(np.concatenate([xn[~hold], onehot(m_tr.filter(pl.Series(~hold)))], 1), g[~hold])
    Xt = np.concatenate([xn_t, onehot(t_m)], 1)
    f_ridge = f_lgb.copy()
    f_ridge[matched_t] = np.clip(t_m[PROXY].to_numpy() + Xt @ w[:-1] + w[-1], 0, None)
    del model, xn, xc, m_tr, r_e, t_e, real
    gc.collect()

    # classifier: LightGBM (prc.mixture._class_probs, unchanged) and MLP on the same encoded frame
    enc_c = fit_encodings(train_src, cfg)
    tr_e, tg_e = attach_encodings(train_src, enc_c), attach_encodings(target_c, enc_c)
    del train_src
    gc.collect()
    t0 = time.time()
    p_lgb, cinfo, _ = _class_probs(tr_e, tg_e, cfg)
    gc.collect()
    log("lgb classifier", cinfo["clf_rounds"], f"{time.time() - t0:.0f}s")
    names = classes()
    lookup = {c: i for i, c in enumerate(names)}
    num_c = [c for c in numeric_feature_columns(tr_e) if c not in (LABEL, "month")]
    prep_c = Prep(tr_e, num_c, cat_cols)
    CARDS = prep_c.card()
    xn, xc = prep_c.num(tr_e), prep_c.cat(tr_e)
    yc = tr_e["clock_src"].replace_strict(lookup, return_dtype=pl.Int64).to_numpy()
    hold_c = _inner_holdout_mask(tr_e, cfg.inner_holdout_months)
    ce = nn.CrossEntropyLoss()
    model_c = train_mlp(xn, xc, yc, hold_c, len(names), ce, epochs=max(1, int(os.getenv("PROBE_EPOCHS", "6")) - 2), tag="clf")
    p_mlp = torch.softmax(torch.from_numpy(predict_mlp(model_c, prep_c.num(tg_e), prep_c.cat(tg_e))), 1).numpy()
    cand = _candidate_matrix(target_c)
    np.savez_compressed(
        OUT / f"{f.name}.npz",
        y=target_c[LABEL].to_numpy().astype(float),
        ids=target_c["MVT_ID_mvt"].to_numpy(),
        airport=target_c["ADEP_mvt"].to_numpy().astype(str),
        src=target_src.to_numpy().astype(str),
        matched=matched_t,
        f_lgb=f_lgb,
        f_mlp=f_mlp,
        f_ridge=f_ridge,
        p_lgb=p_lgb,
        p_mlp=p_mlp,
        cand=cand,
    )
    log("saved", f.name)
    del tr_e, tg_e, xn, xc, model_c
    gc.collect()
