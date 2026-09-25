"""One command from raw parquet to a validated submission file (docs/05 section 2).

    python -m prc.pipeline --config configs/v2_two_clock.yaml [--submit]

Steps: load raw -> build departure features (parity-masked context) -> optional weather join -> month-blocked CV
-> fit on all labelled months and predict the ranking set -> write and validate the submission -> optionally
upload it through the guard rails in prc.submit.

Everything is written to disk as it goes, so a session that dies mid-way can resume without recomputing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import polars as pl
import yaml

from .config import data_dir
from .features import build_departure_features
from .predict import log_submission, predict_ranking, write_submission
from .train import TrainConfig, run_cv


def load_raw(raw_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, Path]:
    train_files = sorted(raw_dir.rglob("training_*.parquet"))
    if not train_files:
        raise FileNotFoundError(f"no training_*.parquet under {raw_dir}; run `make download` first")
    train = pl.concat([pl.read_parquet(p) for p in train_files], how="diagonal")
    rank_path = next(iter(sorted(raw_dir.rglob("ranking.parquet"))), None)
    sub_path = next(iter(sorted(raw_dir.rglob("submitting.parquet"))), None)
    if rank_path is None or sub_path is None:
        raise FileNotFoundError(f"ranking.parquet / submitting.parquet missing under {raw_dir}")
    return train, pl.read_parquet(rank_path), sub_path


def maybe_join_weather(feats: pl.DataFrame, enabled: bool, cache: Path) -> pl.DataFrame:
    if not enabled:
        return feats
    from .weather import fetch_metar, hourly_weather_table, join_weather

    if cache.exists():
        hourly = pl.read_parquet(cache)
    else:
        hourly = hourly_weather_table(fetch_metar())
        cache.parent.mkdir(parents=True, exist_ok=True)
        hourly.write_parquet(cache)
    return join_weather(feats, hourly)


# Every module whose source decides which columns end up in the cached feature frames.
FEATURE_SOURCES = ("features.py", "features_extra.py", "weather.py")


def feature_fingerprint(use_weather: bool) -> str:
    """Identity of the feature builder, so a stale cache is rebuilt instead of silently used.

    Adding a column to prc.features used to leave the cached parquet in place: the new feature simply never
    reached the model, with no error anywhere.

    weather.py is in the hash because it decides columns too. Leaving it out made exactly the same mistake one
    level down: restricting the weather join to the mechanism columns changed nothing in features.py, so the
    fingerprint matched, the twenty-column cache was reused, and v12 would have been a bit-identical rerun of
    v11 while claiming to test a six-column subset."""
    src = b"".join((Path(__file__).parent / name).read_bytes() for name in FEATURE_SOURCES)
    return hashlib.sha1(src + str(use_weather).encode()).hexdigest()[:12]


def build_features(
    train: pl.DataFrame, rank: pl.DataFrame, use_weather: bool, work: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Feature frames for the labelled months and for the ranking months, cached on disk.

    Context features are computed per source frame: a ranking departure's neighbours are the other movements in
    ranking.parquet, exactly as at scoring time."""
    work.mkdir(parents=True, exist_ok=True)
    f_tr_path, f_rk_path = work / "feat_train.parquet", work / "feat_rank.parquet"
    stamp_path, stamp = work / "feat_fingerprint.txt", feature_fingerprint(use_weather)
    if f_tr_path.exists() and f_rk_path.exists():
        cached = stamp_path.read_text().strip() if stamp_path.exists() else "(none)"
        if cached == stamp:
            return pl.read_parquet(f_tr_path), pl.read_parquet(f_rk_path)
        print(f"feature cache is stale ({cached} != {stamp}); rebuilding")
    wcache = data_dir() / "ref" / "metar_hourly.parquet"
    f_tr = maybe_join_weather(build_departure_features(train), use_weather, wcache)
    f_rk = maybe_join_weather(build_departure_features(rank), use_weather, wcache)
    f_tr.write_parquet(f_tr_path)
    f_rk.write_parquet(f_rk_path)
    stamp_path.write_text(stamp)
    return f_tr, f_rk


def _float32(df: pl.DataFrame) -> pl.DataFrame:
    """Float64 feature columns to Float32; the label keeps its own dtype."""
    from .train import LABEL

    return df.with_columns(
        [pl.col(c).cast(pl.Float32) for c, dt in df.schema.items() if dt == pl.Float64 and c != LABEL]
    )


def resolve_predictor(name: str):
    """Which model the config asks for. The same function is used for CV and for the ranking prediction, so a
    config change can never validate one model and submit another."""
    if name == "two_clock":
        from .train import predict_fold

        return predict_fold
    if name == "clock_mixture":
        from .mixture import predict_fold_mixture

        return predict_fold_mixture
    raise SystemExit(f"unknown model {name!r}; expected 'two_clock' or 'clock_mixture'")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=data_dir() / "raw")
    ap.add_argument("--work-dir", type=Path, default=data_dir() / "work")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--skip-cv", action="store_true", help="predict only, reusing the last CV record")
    ap.add_argument("--submit", action="store_true", help="upload after the gate passes (uses the daily counter)")
    ap.add_argument("--version", type=int, help="submission version number, required with --submit")
    a = ap.parse_args(argv)

    spec = yaml.safe_load(a.config.read_text())
    cfg = TrainConfig(**{k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__})
    use_weather = bool(spec.get("use_weather", False))
    predictor = resolve_predictor(spec.get("model", "two_clock"))
    t0 = time.time()

    train, rank, sub_path = load_raw(a.raw_dir)
    f_tr, f_rk = build_features(train, rank, use_weather, a.work_dir)
    # The raw frames are dead after the feature build, and the feature frames are copied several times per fold
    # (candidates, clock-source label, encodings), so halving the float columns is what keeps v14 under the
    # cgroup limit: the v14 build OOM-killed at 14 GB with float64 features. LightGBM trains on float32 anyway.
    del train, rank
    gc.collect()
    f_tr, f_rk = _float32(f_tr), _float32(f_rk)
    print(f"features: train {f_tr.height:,} rows, ranking {f_rk.height:,} rows ({time.time() - t0:.0f}s)")
    # One cached feature build serves every ablation: a config names the column prefixes it leaves out.
    drop = tuple(spec.get("drop_feature_prefixes") or ())
    if drop:
        gone = [c for c in f_tr.columns if c.startswith(drop)]
        f_tr, f_rk = f_tr.drop(gone), f_rk.drop([c for c in gone if c in f_rk.columns])
        print(f"dropped {len(gone)} columns with prefixes {list(drop)}")

    cv_rec = None
    if not a.skip_cv:
        cv_rec = run_cv(f_tr, cfg, predictor=predictor)
        print(
            f"CV {cv_rec['exp_id']}: full={cv_rec['overall']['rmse_full']:.2f} "
            f"clean={cv_rec['overall']['rmse_clean']:.2f} ({cv_rec['train_seconds']}s)"
        )
        for name, m in cv_rec["folds"].items():
            print(
                f"  {name}: full={m['rmse_full']:.2f} clean={m['rmse_clean']:.2f} n={m['n']:,} "
                f"tail20={m['top20_share_of_sse']:.2f}"
            )

    pred, info = predict_ranking(f_tr, f_rk, cfg, predictor=predictor)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = spec.get("name", cfg.name)
    out_file = a.out_dir / f"{name}.parquet"
    write_submission(pred, sub_path, out_file)
    print(f"submission written: {out_file} ({info})")

    log_submission(
        Path("experiments/pipeline.jsonl"),
        {
            "config": str(a.config),
            "train_config": asdict(cfg),
            "use_weather": use_weather,
            "drop_feature_prefixes": list(drop),
            "model": spec.get("model", "two_clock"),
            "cv": cv_rec["overall"] if cv_rec else None,
            "predict_info": info,
            "file": str(out_file),
            "seconds": round(time.time() - t0, 1),
        },
    )

    if a.submit:
        if a.version is None:
            raise SystemExit("--submit requires --version")
        from .submit import submit

        entry = submit(
            out_file,
            sub_path,
            a.version,
            cv_rec["exp_id"] if cv_rec else name,
            cv_rec["overall"] if cv_rec else {},
            spec.get("note", ""),
        )
        print("uploaded:", json.dumps(entry, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
