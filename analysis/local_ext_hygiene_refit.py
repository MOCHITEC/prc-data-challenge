"""External idea A (kind-mango MB3 / Phoenix): training-row hygiene for the matched-lane model (local, analysis only).

Runs the real `prc.train.predict_fold` with configs/main_flseq_osm_nolirf.yaml (n_seeds 1) on folds jan and jul
(train = the other 11 months of 2025, inner early-stopping holdout = the last training month), predicting the MATCHED
validation rows only (model B is not refitted; its rows are identical across variants by construction). Variants
differ only in which matched TRAINING rows are dropped before predict_fold (so the label encodings are refitted on the
same reduced set):

  base : nothing dropped (production)
  x1   : LIRF schedule copies |y - (MVT - SCHED)| <= 12 s, and y > 80,000
  x2   : x1 + |y - (MVT - AOBT_3)| > 7,200 (residual outliers)
  s2   : nothing dropped, LightGBM seed + 7 (refit-noise control)
  cap  : no rows dropped; label replaced by proxy + clip(y - proxy, +-7200) on training rows (Phoenix residual cap)

Writes $SCRATCH/agents/ext/refit_<fold>_<var>.parquet (MVT_ID_mvt, pred); skips a file that exists.

    .venv/bin/python analysis/local_ext_hygiene_refit.py run jan base   (one unit)
    .venv/bin/python analysis/local_ext_hygiene_refit.py eval            (compare all units present)
Env: POLARS_MAX_THREADS=3 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage SCRATCH=...
"""
import gc
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc.pipeline import drop_hidden_candidates, train_config_from_spec  # noqa: E402
from prc.features import CATEGORICALS  # noqa: E402
from prc.train import LABEL, PROXY, predict_fold, visible_numeric_columns  # noqa: E402

S = os.environ["SCRATCH"]
O = f"{S}/agents/ext"
os.makedirs(O, exist_ok=True)
FOLD_MONTH = {"jan": 1, "jul": 7}


def cfg():
    spec = yaml.safe_load(Path("configs/main_flseq_osm_nolirf.yaml").read_text())
    c = train_config_from_spec(spec)
    p = dict(c.params, num_threads=3)
    return replace(c, n_seeds=1, params=p)


def load(c):
    f = pl.read_parquet("data/work_flseq/feat_train.parquet")
    f = drop_hidden_candidates(f, c)
    # memory only: keep what predict_fold reads (visible numerics, categoricals, keys for encodings / OSM / folds)
    need = set(visible_numeric_columns(f, c)) | set(CATEGORICALS) | {
        "MVT_ID_mvt", "MVT_TIME_UTC_mvt", LABEL, PROXY, "hour_local", "mvt_minus_sched", "mvt_minus_lobt", "month"}
    f = f.select([n for n in f.columns if n in need])
    keep64 = {"MVT_ID_mvt", "FLIGHT_ID_mvt"}
    f = f.with_columns([pl.col(n).cast(pl.Float32) for n, t in f.schema.items() if t == pl.Float64 and n not in keep64])
    return f


def artefact_masks(f):
    y = pl.col(LABEL).cast(pl.Float64)
    copy = (pl.col("ADEP_mvt") == "LIRF") & ((y - pl.col("mvt_minus_sched")).abs() <= 12)
    x1 = (copy | (y > 80_000)).fill_null(False)
    big = ((y - pl.col(PROXY)).abs() > 7200).fill_null(False)
    return f.select(x1.alias("x1"), (x1 | big).alias("x2"), pl.col(PROXY).is_not_null().alias("m"))


def run(fold, var):
    out = f"{O}/refit_{fold}_{var}.parquet"
    if os.path.exists(out):
        print("exists", out)
        return
    t0 = time.time()
    c = cfg()
    f = load(c)
    mon = f["MVT_TIME_UTC_mvt"].dt.month()
    va = (mon == FOLD_MONTH[fold]) & f[PROXY].is_not_null()
    tr = mon != FOLD_MONTH[fold]
    mk = artefact_masks(f)
    if var in ("x1", "x2"):
        drop = mk[var] & mk["m"]
        print(f"{fold}/{var}: dropping {int((drop & tr).sum())} matched training rows")
        tr = tr & ~drop
    target = f.filter(va)
    train = f.filter(tr)
    del f
    gc.collect()
    if var == "cap":
        g = (pl.col(LABEL).cast(pl.Float64) - pl.col(PROXY)).clip(-7200, 7200)
        train = train.with_columns(
            pl.when(pl.col(PROXY).is_not_null()).then((pl.col(PROXY) + g).round(0)).otherwise(pl.col(LABEL))
            .cast(pl.Int32).alias(LABEL))
    print(f"{fold}/{var}: train {train.height:,} target {target.height:,} ({time.time() - t0:.0f}s)", flush=True)
    if var == "s2":  # seed control: production rows, different LightGBM seed
        c = replace(c, seed=c.seed + 7)
    p, info = predict_fold(train, target, c)
    p.write_parquet(out)
    info = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}
    Path(out.replace(".parquet", ".json")).write_text(json.dumps(info, default=str))
    print(f"{fold}/{var}: done rounds={info.get('rounds_matched')} ({time.time() - t0:.0f}s)", flush=True)


def _gain_line(tag, y, p_ref, p_new, day, n_all, rng):
    d = (p_ref - y) ** 2 - (p_new - y) ** 2  # >0 = new better
    o = np.sort(d)[::-1]
    ud, inv = np.unique(day, return_inverse=True)
    per = np.bincount(inv, weights=d) / n_all
    bs = [per[rng.integers(0, len(ud), len(ud))].sum() for _ in range(2000)]
    print(f"  {tag:28s} gain {d.sum() / n_all:+8.1f}  boot90 [{np.percentile(bs, 5):+7.1f},{np.percentile(bs, 95):+7.1f}]"
          f"  top10 {o[:10].sum() / n_all:+7.1f} top50 {o[:50].sum() / n_all:+7.1f}  days+ {int((per > 0).sum())}/{len(ud)}")
    return d


def evaluate():
    """Gains in MSE units over ALL validation rows of the fold month (unmatched rows: identical in every variant).
    q for the LIRF copy overlay = LOMO classifier from local_ext_overlays_bc.py (11 other months)."""
    from sklearn.isotonic import IsotonicRegression
    f = pl.read_parquet("data/work_flseq/feat_train.parquet",
                        columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", LABEL, PROXY, "mvt_minus_sched", "mvt_minus_lobt"])
    f = f.with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month(), day=pl.col("MVT_TIME_UTC_mvt").dt.day())
    Q = pl.read_parquet(f"{O}/lirf_q_lomo.parquet")
    rng = np.random.default_rng(0)
    for fold, m in FOLD_MONTH.items():
        n_all = int((f["month"] == m).sum())
        V = f.filter((pl.col("month") == m) & pl.col(PROXY).is_not_null()).join(Q.select("MVT_ID_mvt", "q"), on="MVT_ID_mvt", how="left")
        preds = {}
        for var in ("base", "x1", "x2", "s2", "cap"):
            fn = f"{O}/refit_{fold}_{var}.parquet"
            if os.path.exists(fn):
                V = V.join(pl.read_parquet(fn).rename({"pred": var}), on="MVT_ID_mvt", how="left")
                preds[var] = var
        if "base" not in preds:
            continue
        cal = Q.filter(pl.col("month") != m)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal["q"].to_numpy(), cal["copy"].to_numpy().astype(float))
        y = V[LABEL].to_numpy().astype(float)
        s_ = V["mvt_minus_sched"].to_numpy().astype(float)
        lo = V["mvt_minus_lobt"].to_numpy().astype(float)
        q = V["q"].fill_null(0).to_numpy()
        qi = np.where(V["q"].is_null().to_numpy(), 0, ir.predict(q))
        lirf = (V["ADEP_mvt"] == "LIRF").to_numpy() & V["q"].is_not_null().to_numpy()
        day = V["day"].to_numpy()

        def ov(p, qq, a):
            x = p.copy()
            x[lirf] = p[lirf] + a * qq[lirf] * (s_[lirf] - p[lirf])
            ok = ~np.isnan(lo)
            x[ok] = np.clip(x[ok], lo[ok] - 3606, lo[ok] + 3606)
            return np.maximum(x, 0)

        P0 = V["base"].to_numpy()
        ref = ov(P0, q, 0.4)
        print(f"\n=== fold {fold}: all rows {n_all:,}, matched {V.height:,}, LIRF matched {int(lirf.sum())}")
        print(f"  base raw RMSE(matched) {np.sqrt(np.mean((P0 - y) ** 2)):.2f}; base+copy0.4 {np.sqrt(np.mean((ref - y) ** 2)):.2f}")
        _gain_line("base: copy 0.4 vs none", y, P0, ref, day, n_all, rng)
        for var in preds:
            Pv = V[var].to_numpy()
            if var != "base":
                d = _gain_line(f"{var} raw vs base raw", y, P0, Pv, day, n_all, rng)
                A = V.select("ADEP_mvt").with_columns(d=pl.Series(d)).group_by("ADEP_mvt").agg(g=pl.col("d").sum() / n_all).sort("ADEP_mvt")
                print("     per airport:", [(a, round(g, 1)) for a, g in A.rows()])
                cp = lirf & (np.abs(y - s_) <= 12)
                print(f"     LIRF copy rows {int(cp.sum())}: {d[cp].sum() / n_all:+.1f}; LIRF other {d[lirf & ~cp].sum() / n_all:+.1f};"
                      f" non-LIRF {d[~lirf].sum() / n_all:+.1f}")
            for tag, qq, a in (("copy0.4", q, 0.4), ("copy0.65", q, 0.65), ("copy1.0", q, 1.0), ("iso1.0", qi, 1.0)):
                if var == "base" and tag == "copy0.4":
                    continue
                d = _gain_line(f"{var}+{tag} vs base+copy0.4", y, ref, ov(Pv, qq, a), day, n_all, rng)
                if var != "base" and tag in ("copy1.0", "iso1.0"):
                    for nm, msk in (("LIRF", lirf), ("non-LIRF", ~lirf)):
                        o = np.sort(d[msk])[::-1]
                        print(f"       {nm:8s} {d[msk].sum() / n_all:+7.1f}  top10 {o[:10].sum() / n_all:+6.1f}"
                              f"  top50 {o[:50].sum() / n_all:+6.1f}  w/o top50 {(d[msk].sum() - o[:50].sum()) / n_all:+6.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        evaluate()
