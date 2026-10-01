"""Item 21 `refit-seed-semantics-changed`: does passing explicit bagging_seed / feature_fraction_seed at the refit
(commit d182fc1) change model B's booster at n_seeds = 1, and by how much on the 5,290 ranking unmatched rows?

Model B only. Reproduces predict_fold's data path for the unmatched branch (train.py:314-366):
  * training rows with a finite label; encodings fitted on ALL labelled training rows (matched + unmatched) with
    fit_encodings; attached with attach_encodings; unmatched = `mvt_minus_aobt3` null (routing rule);
  * cols_b = visible_numeric_columns(...) minus the NM-clock prefixes; categoricals = CATEGORICALS present;
  * design matrix through _matrix on the stacked [u_tr, u_tg] frame, exactly as _fit does.
Then, with a FIXED 658 rounds (the rounds_unmatched recorded for v2 and det_a/v8) and no early stopping:
  (a) params + {seed}                                              -- refit as the code before d182fc1 did
  (a2) the same again                                              -- bit-identity check
  (b) params + {seed, bagging_seed, feature_fraction_seed}         -- refit as _fit does at HEAD (n_seeds = 1)
Weather columns are hidden (det_a / v8 and v2 were built with use_weather: false) so that (b) is det_a's
model-B matrix and (a) is v2's matrix up to the code differences listed in results/v4-config-not-reproducible.md.

Run:  POLARS_MAX_THREADS=1 .venv/bin/python docs/investigation_2026-09-16/scripts/refit-seed-semantics-changed.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from prc.features import CATEGORICALS  # noqa: E402
from prc.train import (  # noqa: E402
    LABEL,
    PROXY,
    TrainConfig,
    _matrix,
    attach_encodings,
    fit_encodings,
    visible_numeric_columns,
)

FEAT_TRAIN = REPO / "data/work/feat_train.parquet"
FEAT_RANK = REPO / "data/work/feat_rank.parquet"
OUT_DIR = Path(os.environ.get("SCRATCH", "/tmp")) / "item21"
FIXED_ROUNDS = 658
NUM_THREADS = 2
ENC_COLS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour_local", "AIRCRAFT_OPERATOR_flt", LABEL, PROXY]
NM_PREFIXES = ("mvt_minus_aobt3", "aobt3", "lobt_minus", "mvt_minus_lobt", "mvt_minus_eobt1", "mvt_minus_iobt")


def load_cfg() -> TrainConfig:
    """configs/main.yaml resolved the way prc.pipeline.main does (pipeline.py:128-137)."""
    spec = yaml.safe_load((REPO / "configs/main.yaml").read_text())
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    return TrainConfig(**fields)


def weather_columns(schema: pl.Schema) -> list[str]:
    names = list(schema.names())
    return names[names.index("temp_c") :]


def load_model_b_frames(cfg: TrainConfig) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    lf = pl.scan_parquet(FEAT_TRAIN)
    labelled = lf.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())
    enc_frame = labelled.select(ENC_COLS).collect()
    enc = fit_encodings(enc_frame, cfg)
    info = {"labelled_train_rows": enc_frame.height, **{k: v.height for k, v in enc.items()}}
    del enc_frame
    u_tr = labelled.filter(pl.col(PROXY).is_null()).collect()
    u_tg = pl.scan_parquet(FEAT_RANK).filter(pl.col(PROXY).is_null()).collect()
    tr, tg = attach_encodings(u_tr, enc), attach_encodings(u_tg, enc)
    assert tr.height == u_tr.height and tg.height == u_tg.height
    assert tg["MVT_ID_mvt"].equals(u_tg["MVT_ID_mvt"])
    return tr, tg, info


def cols_b_for(tr: pl.DataFrame, cfg: TrainConfig) -> tuple[list[str], list[str]]:
    num_cols = visible_numeric_columns(tr, cfg)
    cat_cols = [c for c in CATEGORICALS if c in tr.columns and c not in set(cfg.exclude_features)]
    cols_b = [c for c in num_cols if not c.startswith(NM_PREFIXES)]
    return cols_b, cat_cols


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_cfg()
    train_schema = pl.scan_parquet(FEAT_TRAIN).collect_schema()
    wx = weather_columns(train_schema)
    # main.yaml resolved, then: model B copy, one seed, weather hidden (det_a / v2 frames had no weather), 2 threads
    cfg = replace(base, n_seeds=1, exclude_features=tuple(base.exclude_features) + tuple(wx)).for_unmatched()
    cfg = replace(cfg, params={**cfg.params, "num_threads": NUM_THREADS})
    params = cfg.lgb_params()
    print("resolved main.yaml:", json.dumps({k: v for k, v in base.__dict__.items() if k != "params"}, default=str))
    print("model-B params (early-stop fit / old-style refit):", json.dumps(params, sort_keys=True))
    print("hidden weather columns:", len(wx), wx)

    tr, tg, info = load_model_b_frames(cfg)
    print("frames:", info, "u_tr", tr.height, "u_tg", tg.height)
    cols_b, cat_cols = cols_b_for(tr, cfg)
    print(f"cols_b ({len(cols_b)} numeric):", cols_b)
    print(f"cat_cols ({len(cat_cols)}):", cat_cols)

    # ---- design matrix, exactly as _fit builds it (stacked train + predict frames, one Categorical cast) ----
    stacked = pl.concat([f.select(cols_b + cat_cols) for f in (tr, tg)], how="vertical_relaxed")
    Xall, cols = _matrix(stacked, cols_b, cat_cols)
    del stacked
    n_tr = tr.height
    Xtr, Xtg = Xall[:n_tr], Xall[n_tr:]
    cat_idx = [cols.index(c) for c in cat_cols]
    y = tr[LABEL].to_numpy().astype(float)
    print("matrix", Xall.shape, "y mean", y.mean().round(1))

    def refit(p: dict, tag: str) -> np.ndarray:
        t = time.time()
        d = lgb.Dataset(Xtr, label=y, categorical_feature=cat_idx, free_raw_data=False, params=p)
        m = lgb.train(p, d, num_boost_round=FIXED_ROUNDS, callbacks=[lgb.log_evaluation(0)])
        pred = m.predict(Xtg)
        print(f"  fit {tag}: {FIXED_ROUNDS} rounds, {m.num_trees()} trees, {time.time() - t:.1f}s")
        return pred

    p_old = dict(params)  # seed only (train.py @9f31cef: lgb.train(params, dfull, ...))
    p_new = dict(params, seed=cfg.seed, bagging_seed=cfg.seed, feature_fraction_seed=cfg.seed)  # train.py:302
    print("refit params (a) old:", json.dumps(p_old, sort_keys=True))
    print("refit params (b) HEAD:", json.dumps(p_new, sort_keys=True))
    pa1 = refit(p_old, "(a) seed-only, run 1")
    pa2 = refit(p_old, "(a2) seed-only, run 2")
    pb = refit(p_new, "(b) explicit sub-seeds")
    # what LightGBM itself reports as the effective seeds is not exposed through the Python API; the params
    # above are the exact dicts handed to lgb.train.

    # ---- reference submissions on the same rows ----
    ids = tg.select("MVT_ID_mvt")
    refs = {
        "v8_det_a": REPO / "outputs/det_a.parquet",
        "v4_probe": REPO / "outputs/probe_unmatched_from_v4.parquet",
        "v2_two_clock": REPO / "outputs/v2_two_clock.parquet",
        "v7_main": REPO / "outputs/main.parquet",
    }
    res = pl.DataFrame(
        {
            "MVT_ID_mvt": ids["MVT_ID_mvt"],
            "ADEP_mvt": tg["ADEP_mvt"],
            "month": tg["MVT_TIME_UTC_mvt"].dt.month(),
            "ms": tg["mvt_minus_sched"],
            "RUNWAY_mvt": tg["RUNWAY_mvt"],
            "a": pa1,
            "a2": pa2,
            "b": pb,
        }
    )
    for k, path in refs.items():
        r = pl.scan_parquet(path).select(["MVT_ID_mvt", pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias(k)])
        res = res.join(r.collect(), on="MVT_ID_mvt", how="left", maintain_order="left")
        assert res[k].null_count() == 0, k
    # the submission files carry rint'ed, clipped-at-0 Int32 values: compare like with like
    res = res.with_columns(
        [
            pl.col("a").clip(lower_bound=0.0).round(0).alias("a_int"),
            pl.col("b").clip(lower_bound=0.0).round(0).alias("b_int"),
        ]
    )
    res.write_parquet(OUT_DIR / "preds.parquet")

    A, A2, B = res["a"].to_numpy(), res["a2"].to_numpy(), res["b"].to_numpy()
    out: dict = {"fixed_rounds": FIXED_ROUNDS, "n_rows": res.height, "params_old": p_old, "params_new": p_new}
    out["a_vs_a2_bit_identical"] = bool(np.array_equal(A, A2))
    out["a_vs_a2_max_abs"] = float(np.max(np.abs(A - A2)))
    print("\n(a) run1 vs run2 bit-identical:", out["a_vs_a2_bit_identical"], "max|diff|", out["a_vs_a2_max_abs"])

    def block(mask: np.ndarray, name: str) -> dict:
        d = {
            "n": int(mask.sum()),
            "rms_a_b": rms(A[mask], B[mask]),
            "n_identical_a_b": int(np.sum(A[mask] == B[mask])),
            "rms_a_v8": rms(res["a_int"].to_numpy()[mask], res["v8_det_a"].to_numpy()[mask]),
            "rms_b_v8": rms(res["b_int"].to_numpy()[mask], res["v8_det_a"].to_numpy()[mask]),
            "n_identical_b_v8": int(np.sum(res["b_int"].to_numpy()[mask] == res["v8_det_a"].to_numpy()[mask])),
            "rms_a_v4": rms(res["a_int"].to_numpy()[mask], res["v4_probe"].to_numpy()[mask]),
            "rms_b_v4": rms(res["b_int"].to_numpy()[mask], res["v4_probe"].to_numpy()[mask]),
            "rms_a_v2": rms(res["a_int"].to_numpy()[mask], res["v2_two_clock"].to_numpy()[mask]),
            "n_identical_a_v2": int(np.sum(res["a_int"].to_numpy()[mask] == res["v2_two_clock"].to_numpy()[mask])),
            "rms_b_v2": rms(res["b_int"].to_numpy()[mask], res["v2_two_clock"].to_numpy()[mask]),
            "rms_v8_v4": rms(res["v8_det_a"].to_numpy()[mask], res["v4_probe"].to_numpy()[mask]),
            "rms_v8_v2": rms(res["v8_det_a"].to_numpy()[mask], res["v2_two_clock"].to_numpy()[mask]),
        }
        print(f"{name:>14} n={d['n']:5d} RMS(a-b)={d['rms_a_b']:8.1f} ident={d['n_identical_a_b']:5d} | "
              f"a~v8 {d['rms_a_v8']:8.1f} b~v8 {d['rms_b_v8']:8.1f} (ident {d['n_identical_b_v8']}) | "
              f"a~v4 {d['rms_a_v4']:8.1f} b~v4 {d['rms_b_v4']:8.1f} | a~v2 {d['rms_a_v2']:8.1f} (ident {d['n_identical_a_v2']}) "
              f"b~v2 {d['rms_b_v2']:8.1f} | v8~v4 {d['rms_v8_v4']:8.1f} v8~v2 {d['rms_v8_v2']:8.1f}")
        return d

    ms = res["ms"].to_numpy()
    all_mask = np.ones(res.height, dtype=bool)
    out["overall"] = block(all_mask, "overall")
    out["ms_ge_2h"] = block(ms >= 7200, "ms>=2h")
    out["ms_ge_12h"] = block(ms >= 43200, "ms>=12h")
    out["ms_lt_2h"] = block(ms < 7200, "ms<2h")
    out["by_airport"] = {}
    for apt in sorted(res["ADEP_mvt"].unique().to_list()):
        out["by_airport"][apt] = block((res["ADEP_mvt"] == apt).to_numpy(), apt)

    # tail counts of each prediction set
    def tail(v: np.ndarray) -> dict:
        return {
            "median": float(np.median(v)), "mean": float(np.mean(v)), "p99": float(np.quantile(v, 0.99)),
            "max": float(np.max(v)), "n_gt_7200": int((v > 7200).sum()), "n_gt_14400": int((v > 14400).sum()),
            "n_gt_43200": int((v > 43200).sum()),
        }
    out["tails"] = {k: tail(res[k].to_numpy()) for k in ("a", "b", "v8_det_a", "v4_probe", "v2_two_clock")}
    print("\ntails:")
    for k, v in out["tails"].items():
        print(f"  {k:>13}", {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()})

    # share of sum (a-b)^2 in the top-20 rows, LIRF share
    g2 = (A - B) ** 2
    order = np.argsort(-g2)
    out["gap_top20_share"] = float(g2[order[:20]].sum() / g2.sum())
    out["gap_lirf_share"] = float(g2[(res["ADEP_mvt"] == "LIRF").to_numpy()].sum() / g2.sum())
    out["gap_ms_ge_2h_share"] = float(g2[ms >= 7200].sum() / g2.sum())
    print(f"\nsum(a-b)^2: top-20 share {out['gap_top20_share']:.3f}, LIRF share {out['gap_lirf_share']:.3f}, "
          f"ms>=2h share {out['gap_ms_ge_2h_share']:.3f}")

    # the 11 LIRF ms >= 12 h rows and the top-20 (a-b) rows, side by side
    show = res.with_columns((pl.col("a") - pl.col("b")).pow(2).alias("gap2")).select(
        ["MVT_ID_mvt", "ADEP_mvt", "month", "ms", "RUNWAY_mvt", "a", "b", "v2_two_clock", "v8_det_a", "v4_probe", "gap2"]
    )
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=0):
        print("\nLIRF unmatched ranking rows with ms >= 12 h:")
        print(show.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= 43200)).sort("ms", descending=True))
        print("\ntop-20 rows by (a-b)^2:")
        print(show.sort("gap2", descending=True).head(20))
    out["lirf_ms_ge_12h"] = show.filter((pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") >= 43200)).sort("ms", descending=True).to_dicts()
    out["top20_a_b"] = show.sort("gap2", descending=True).head(20).to_dicts()
    out["cols_b"] = cols_b
    out["cat_cols"] = cat_cols
    out["frames"] = info
    out["seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\ndone in {out['seconds']}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
