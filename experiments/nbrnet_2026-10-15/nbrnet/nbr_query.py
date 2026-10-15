"""Query features for NbrNet: a2o's member frame and model A's columns, per fold (PREREG.md §2).

`member_frame` replays `prc.pipeline.main` up to the frame `run_cv` receives (feature cache, the L2/L6 levers,
hidden candidates dropped). `prepare_fold` replays `prc.train.predict_fold` for this recipe up to model A's design:
fold-fitted encodings, the fold-level OSM distance, `visible_numeric_columns`, the categoricals and `cols_a`. The
`mlp.Preprocessor` (m4's recipe) is fitted on the fold's matched training rows only (R5). Nothing in src/ changes.

Output per fold (a directory of .npy files plus `valid_meta.parquet` and `pre.pkl`):
  tr_* / va_*: Xn (float32 transformed numerics, NaN kept), C (int16 category codes), y (label, s), proxy
  (MVT - AOBT_3, s), row (departure-table row of the query), id (MVT_ID_mvt); tr_hold (inner holdout mask).
  valid_meta.parquet: every labelled evaluation row (matched and unmatched) with y, ADEP, proxy, mvt_minus_lobt."""

from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from prc.cv import month_block_folds, year_month_key
from prc.features import CATEGORICALS
from prc.levers import L2_COLS
from prc.mlp import Preprocessor, resolve
from prc.pipeline import (
    build_features,
    drop_hidden_candidates,
    load_raw,
    train_config_from_spec,
    weather_flags,
)
from prc.train import (
    CAR_COLS,
    LABEL,
    LOBT_OFFSET,
    PROXY,
    STAMP_COLS,
    TrainConfig,
    _inner_holdout_mask,
    add_fold_features,
    attach_encodings,
    fit_encodings,
    visible_numeric_columns,
)

A2O_CONFIG = Path("configs/main_flseq_osm_nolirf_l2_opt_l6_cmix.yaml")


def load_cfg(path: Path = A2O_CONFIG) -> tuple[dict, TrainConfig]:
    spec = yaml.safe_load(path.read_text())
    return spec, train_config_from_spec(spec)


def member_frame(raw_dir: Path, work_dir: Path, config: Path = A2O_CONFIG) -> pl.DataFrame:
    """The labelled-months frame `run_cv` gets in `prc.pipeline.main` for this recipe."""
    spec, cfg = load_cfg(config)
    use_weather, wx_codes, wind_dir, gust_repair = weather_flags(spec)
    train, rank, _ = load_raw(raw_dir)
    f_tr, f_rk = build_features(train, rank, use_weather, work_dir, raw_dir, wx_codes, wind_dir, gust_repair)
    for flag in ("fe_arr_fill", "fe_month_scope", "fe_plan_nm_taxi", "fe_push_queue", "fe_tx_nm", "fe_fwd_nm_unmatched"):
        if getattr(cfg, flag):
            raise NotImplementedError(f"{flag} is on in the recipe; member_frame does not replay it")
    if cfg.fe_fwd_nm or cfg.fe_retention:
        from prc.levers import attach_levers

        f_tr, f_rk = attach_levers(f_tr, f_rk, train, rank, l1=False, l2=cfg.fe_fwd_nm, l3=False, l5=False, l6=cfg.fe_retention)
    del train, rank, f_rk
    gc.collect()
    return drop_hidden_candidates(f_tr, cfg)


def _check_replayed(cfg: TrainConfig) -> None:
    """Every predict_fold branch before model A that this replay does not implement must be off."""
    off = {
        "sim_unseen_stands": cfg.sim_unseen_stands == 0,
        "fe_carrier_echo": not cfg.fe_carrier_echo,
        "fe_stamp_hier": not cfg.fe_stamp_hier,
        "stand_dropout": cfg.stand_dropout == 0,
        "fe_day_demean": not cfg.fe_day_demean,
        "season_weight": cfg.season_weight == 0,
        "fe_meta_prior": not cfg.fe_meta_prior,
        "fe_arr_fill": not cfg.fe_arr_fill,
        "matched_target anchored": cfg.matched_target == "anchored",
        "use_categoricals": cfg.use_categoricals,
    }
    bad = [k for k, ok in off.items() if not ok]
    if bad:
        raise NotImplementedError(f"recipe uses {bad}, which prepare_fold does not replay")


def fold_by_name(feats: pl.DataFrame, cfg: TrainConfig, name: str):
    months = year_month_key(feats)
    folds = {f.name: f for f in month_block_folds(months.unique().to_list(), named=cfg.folds or ())}
    return folds[name], months


def prepare_fold(member: pl.DataFrame, fold_name: str, cfg: TrainConfig, dep_id: np.ndarray, out: Path) -> dict:
    """Model A's query design for one fold, written to `out` (see the module docstring)."""
    _check_replayed(cfg)
    t = time.time()
    fold, months = fold_by_name(member, cfg, fold_name)
    train_f = member.filter(months.is_in(list(fold.train_months)))
    target_f = member.filter(months.is_in(list(fold.valid_months)))
    train_f = train_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    target_f = target_f.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    enc = fit_encodings(train_f, cfg)  # fold-train rows only, exactly as predict_fold
    tr, tg = attach_encodings(train_f, enc), attach_encodings(target_f, enc)
    del train_f
    info: dict = {}
    tr, tg = add_fold_features(tr, tg, cfg, info)
    num_cols = visible_numeric_columns(tr, cfg)
    cat_cols = [c for c in CATEGORICALS if c in tr.columns and c not in set(cfg.exclude_features)]
    a_hidden = (set() if cfg.fe_fwd_nm else set(L2_COLS)) | set(CAR_COLS) | set(STAMP_COLS)
    cols_a = [c for c in num_cols if c != PROXY and c not in a_hidden] + [PROXY]
    meta = tg.select(
        "MVT_ID_mvt",
        pl.col(LABEL).cast(pl.Float64).alias("y"),
        "ADEP_mvt",
        pl.col(PROXY).cast(pl.Float64).alias("proxy"),
        pl.col(LOBT_OFFSET).cast(pl.Float64).alias("mvt_minus_lobt"),
        (pl.col("MVT_TIME_UTC_mvt").dt.year() * 100 + pl.col("MVT_TIME_UTC_mvt").dt.month()).alias("ym"),
    )
    m_tr, m_tg = tr.filter(pl.col(PROXY).is_not_null()), tg.filter(pl.col(PROXY).is_not_null())
    del tr, tg
    gc.collect()
    p = resolve({})
    pre = Preprocessor(p, cfg.seed).fit(m_tr, cols_a, cat_cols)  # m4 seeds its Preprocessor with cfg.seed
    out.mkdir(parents=True, exist_ok=True)
    lut = pl.DataFrame({"MVT_ID_mvt": dep_id, "_row": np.arange(len(dep_id), dtype=np.int32)})
    rec = {"fold": fold_name, "train_months": list(fold.train_months), "valid_months": list(fold.valid_months)}
    for tag, f in (("tr", m_tr), ("va", m_tg)):
        row = f.select("MVT_ID_mvt").join(lut, on="MVT_ID_mvt", how="left", maintain_order="left")["_row"]
        if row.null_count():
            raise AssertionError(f"{row.null_count()} {tag} query rows are not in the departure table")
        np.save(out / f"{tag}_Xn.npy", pre.num(f))
        np.save(out / f"{tag}_C.npy", pre.codes(f))
        np.save(out / f"{tag}_y.npy", f[LABEL].to_numpy().astype(np.float64))
        np.save(out / f"{tag}_proxy.npy", f[PROXY].to_numpy().astype(np.float64))
        np.save(out / f"{tag}_row.npy", row.to_numpy().astype(np.int32))
        np.save(out / f"{tag}_id.npy", f["MVT_ID_mvt"].to_numpy().astype(np.float64))
        rec[f"n_{tag}"] = f.height
    hold = _inner_holdout_mask(m_tr, cfg.inner_holdout_months)
    np.save(out / "tr_hold.npy", hold)
    meta.write_parquet(out / "valid_meta.parquet")
    with open(out / "pre.pkl", "wb") as fh:
        pickle.dump(pre, fh, protocol=pickle.HIGHEST_PROTOCOL)
    rec.update(
        n_hold=int(hold.sum()),
        hold_months=sorted(set(year_month_key(m_tr).to_numpy()[hold].tolist())),
        n_valid_all=meta.height,
        cols_a=cols_a,
        cat_cols=cat_cols,
        fold_feature_cols=info.get("fold_feature_cols"),
        preprocessor=pre.summary(),
        seconds=round(time.time() - t, 1),
    )
    (out / "fold.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["forward"])
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--tokens", type=Path, default=Path("data/nbrnet/tokens"))
    ap.add_argument("--member", type=Path, default=Path("data/nbrnet/member_train.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/nbrnet/folds"))
    a = ap.parse_args(argv)
    _, cfg = load_cfg()
    if a.member.exists():
        member = pl.read_parquet(a.member)
    else:
        member = member_frame(a.raw_dir, a.work_dir)
        a.member.parent.mkdir(parents=True, exist_ok=True)
        member.write_parquet(a.member)
    print(f"member frame {member.shape}", flush=True)
    dep_id = np.load(a.tokens / "dep_id.npy")
    for name in a.folds:
        rec = prepare_fold(member, name, cfg, dep_id, a.out / name)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("cols_a",)}, default=str), flush=True)
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
