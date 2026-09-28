"""One command from raw parquet to a validated submission file (docs/README).

    python -m prc.pipeline --config configs/main.yaml [--submit]

Steps: load raw -> build departure features (parity-masked context) -> optional weather join -> month-blocked CV
-> fit on all labelled months and predict the ranking set -> write and validate the submission -> optionally
upload it through the guard rails in prc.submit.

Everything is written to disk as it goes, so a session that dies mid-way can resume without recomputing."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import polars as pl
import yaml

from .config import data_dir
from .features import build_feature_frames
from .predict import log_submission, predict_ranking, write_submission
from .train import LABEL, TrainConfig, run_cv


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


def weather_cache_path(wx_codes: bool, wind_dir: bool = False, gust_repair: bool = False) -> Path:
    """One cache file per weather recipe. Two recipes sharing one path is how a `weather.py` edit gets
    silently reverted at read time - so every flag that changes the hourly table's columns is in the name,
    and the two names that already exist keep their spelling exactly."""
    name = "metar_hourly" + ("_wxcodes" if wx_codes else "") + ("_winddir" if wind_dir else "")
    name += "_gust" if gust_repair else ""
    return data_dir() / "ref" / f"{name}.parquet"


def maybe_join_weather(
    feats: pl.DataFrame,
    enabled: bool,
    cache: Path,
    wx_codes: bool = False,
    wind_dir: bool = False,
    gust_repair: bool = False,
) -> pl.DataFrame:
    if not enabled:
        return feats
    from .weather import fetch_metar, hourly_weather_table, join_weather

    if cache.exists():
        hourly = pl.read_parquet(cache)
    else:
        hourly = hourly_weather_table(fetch_metar(), wx_codes=wx_codes, wind_dir=wind_dir, gust_repair=gust_repair)
        cache.parent.mkdir(parents=True, exist_ok=True)
        hourly.write_parquet(cache)
    return join_weather(feats, hourly)


def weather_flags(spec: dict) -> tuple[bool, bool, bool, bool]:
    """(use_weather, wx_codes, wind_dir, gust_repair) as a recipe asks for them.

    Every one of them defaults to False, which is what makes an archived recipe rebuild the frame it built
    before. It is a function rather than four `spec.get` calls in `main` so that the defaults are something a
    test can hold on to: a flag that silently defaults to on is a change to every recipe in the repository,
    and `main` is the one place nothing else covers."""
    use_weather = bool(spec.get("use_weather", False))
    return (
        use_weather,
        bool(spec.get("wx_codes", False)),
        bool(spec.get("wind_dir", False)),
        bool(spec.get("gust_repair", False)),
    )


def cache_fingerprint(
    train: pl.DataFrame,
    rank: pl.DataFrame,
    raw_dir: Path,
    use_weather: bool,
    wx_codes: bool = False,
    wind_dir: bool = False,
    gust_repair: bool = False,
) -> dict:
    """What the cached feature frames were built from. A cache hit requires every field to match.

    `weather_code` is here because it was not, and that was a trap: the fingerprint hashed `features.py`
    only, so an edit to `weather.py` left every cached frame valid and the next run reported the old numbers
    with the new code on disk. The plan calls that the highest-probability way for this theme to waste a day.
    It is hashed only when the weather join is on, so a `use_weather: false` recipe is unaffected by weather
    edits and its cache is not invalidated by them."""
    return {
        "raw_dir": str(raw_dir.resolve()),
        "files": sorted((p.name, p.stat().st_size) for p in raw_dir.rglob("*.parquet")),
        "train_rows": train.height,
        "rank_rows": rank.height,
        "use_weather": use_weather,
        "wx_codes": bool(use_weather and wx_codes),
        # Same reason as `wx_codes`: these change which columns the hourly table has, so a frame built
        # without them must not be handed to a recipe that asks for them. Hashed only when the weather join
        # is on, so a `use_weather: false` recipe's cache is untouched by them.
        "wind_dir": bool(use_weather and wind_dir),
        "gust_repair": bool(use_weather and gust_repair),
        "feature_code": _features_version(),
        "weather_code": _weather_version() if use_weather else None,
    }


def _features_version() -> str:
    """Hash of the feature builder source, so editing it invalidates the cache."""
    from . import features

    return hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest()[:16]


def _weather_version() -> str:
    """Hash of the weather module source, for the same reason and with the same effect."""
    from . import weather

    return hashlib.sha256(Path(weather.__file__).read_bytes()).hexdigest()[:16]


def build_features(
    train: pl.DataFrame,
    rank: pl.DataFrame,
    use_weather: bool,
    work: Path,
    raw_dir: Path,
    wx_codes: bool = False,
    wind_dir: bool = False,
    gust_repair: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Feature frames for the labelled months and for the ranking months, cached on disk.

    Context features are computed per source frame: a ranking departure's neighbours are the other movements in
    ranking.parquet, exactly as at scoring time.

    The cache is keyed by what produced it, not by path alone. A synthetic run and a real run share the default
    work directory, and reusing one for the other trains on the wrong data while printing plausible numbers -
    that happened once (docs/README) and the dtype error that caught it was luck."""
    work.mkdir(parents=True, exist_ok=True)
    f_tr_path, f_rk_path = work / "feat_train.parquet", work / "feat_rank.parquet"
    meta_path = work / "feat_meta.json"
    want = cache_fingerprint(train, rank, raw_dir, use_weather, wx_codes, wind_dir, gust_repair)
    if f_tr_path.exists() and f_rk_path.exists() and meta_path.exists():
        have = json.loads(meta_path.read_text())
        # json round-trips the file tuples as lists
        if have == json.loads(json.dumps(want)):
            print(f"features: reusing the cache in {work}")
            return pl.read_parquet(f_tr_path), pl.read_parquet(f_rk_path)
        differing = sorted(k for k in want if have.get(k) != json.loads(json.dumps(want))[k])
        print(f"features: cache in {work} was built from different inputs ({', '.join(differing)}); rebuilding")
    wcache = weather_cache_path(wx_codes, wind_dir, gust_repair)
    # One call, because C1's modal-offset table is fitted on the two frames together; see
    # `features.build_feature_frames`.
    raw_tr, raw_rk = build_feature_frames(train, rank)
    f_tr = maybe_join_weather(raw_tr, use_weather, wcache, wx_codes, wind_dir, gust_repair)
    f_rk = maybe_join_weather(raw_rk, use_weather, wcache, wx_codes, wind_dir, gust_repair)
    f_tr.write_parquet(f_tr_path)
    f_rk.write_parquet(f_rk_path)
    meta_path.write_text(json.dumps(want, indent=2, default=str))
    return f_tr, f_rk


# Recipe keys that are not TrainConfig fields and are read by `main` itself (the weather flags, the coverage
# gate, the submission note). Every recipe in configs/ uses only these and TrainConfig's fields.
PIPELINE_KEYS = frozenset(
    {"use_weather", "wx_codes", "wind_dir", "gust_repair", "coverage_gate", "coverage_allow", "note"}
)


def train_config_from_spec(spec: dict) -> TrainConfig:
    """The TrainConfig a recipe asks for.

    Unknown keys raise. They used to be dropped silently, so `model: catboost` - or a typo of any field - ran
    the default LightGBM recipe and logged a record that said nothing about it (P10 §1.2); every negative result
    would have been uninterpretable. The keys `main` reads itself are listed in PIPELINE_KEYS."""
    unknown = sorted(k for k in spec if k not in TrainConfig.__dataclass_fields__ and k not in PIPELINE_KEYS)
    if unknown:
        raise ValueError(f"unknown recipe keys {unknown}: not a TrainConfig field and not in PIPELINE_KEYS")
    fields = {k: v for k, v in spec.items() if k in TrainConfig.__dataclass_fields__}
    # A config that sets `params` overrides only the keys it names: replacing the whole dict would silently
    # drop objective/verbose/num_threads and make one experiment differ from the others in ways nobody chose.
    # `params` is LightGBM's; CatBoost's settings live in `cb_params`, merged over CB_DEFAULTS by the config
    # itself, so the two families never share one merged dict.
    if "params" in fields:
        fields["params"] = {**TrainConfig().params, **fields["params"]}
    for k in ("folds", "exclude_features", "include_candidates", "unmatched_sched_reparam_airports"):
        if isinstance(fields.get(k), list):
            fields[k] = tuple(fields[k])
    return TrainConfig(**fields)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=data_dir() / "raw")
    ap.add_argument("--work-dir", type=Path, default=data_dir() / "work")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--skip-cv", action="store_true", help="predict only, reusing the last CV record")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="log to experiments/synthetic_smoke.jsonl instead of the real-data logs (see experiments/README)",
    )
    ap.add_argument(
        "--skip-predict",
        action="store_true",
        help="stop after cross-validation: no full-data refit, no submission file. About a quarter of the "
        "wall clock of a run, and pure waste for an experiment whose only output is a CV number.",
    )
    ap.add_argument("--submit", action="store_true", help="upload after the gate passes (uses the daily counter)")
    ap.add_argument("--version", type=int, help="submission version number, required with --submit")
    a = ap.parse_args(argv)

    spec = yaml.safe_load(a.config.read_text())
    cfg = train_config_from_spec(spec)
    # P8-B (`wx_codes`), C3 (`wind_dir`) and C3's hygiene rider (`gust_repair`) are all off by default:
    # with them off `hourly_weather_table` emits exactly the columns it always has, so every archived recipe
    # rebuilds to the same frame it did before.
    use_weather, wx_codes, wind_dir, gust_repair = weather_flags(spec)
    t0 = time.time()

    train, rank, sub_path = load_raw(a.raw_dir)
    f_tr, f_rk = build_features(train, rank, use_weather, a.work_dir, a.raw_dir, wx_codes, wind_dir, gust_repair)
    print(f"features: train {f_tr.height:,} rows, ranking {f_rk.height:,} rows ({time.time() - t0:.0f}s)")

    # Coverage parity, before CV, because CV cannot see this class of defect: it validates on the training
    # side, which is intact by construction. Off by default so every existing recipe runs exactly as it did.
    # It cannot be turned on for production today - the five identically-zero columns P8-D found make it
    # fail - which is the finding, not a reason to weaken the rule (P8-D results note).
    if bool(spec.get("coverage_gate", False)):
        from .coverage import gate

        gate(f_tr, f_rk, cfg, allow=spec.get("coverage_allow") or {}, strict=True)
    elif a.submit:
        print(
            "WARNING: submitting without the coverage gate (coverage_gate is not set in this recipe). "
            "Nothing checks that the columns the model trained on still look the same on the scored months."
        )

    log_name = "synthetic_smoke.jsonl" if a.synthetic else "experiments.jsonl"
    cv_rec = None
    if not a.skip_cv:
        cv_rec = run_cv(f_tr, cfg, log_name=log_name)
        print(
            f"CV {cv_rec['exp_id']}: full={cv_rec['overall']['rmse_full']:.2f} "
            f"clean={cv_rec['overall']['rmse_clean']:.2f} ({cv_rec['train_seconds']}s)"
        )
        for name, m in cv_rec["folds"].items():
            print(
                f"  {name}: full={m['rmse_full']:.2f} clean={m['rmse_clean']:.2f} n={m['n']:,} "
                f"tail20={m['top20_share_of_sse']:.2f}"
            )

    if a.skip_predict:
        print("--skip-predict: stopping after CV")
        return 0

    pred, info = predict_ranking(f_tr, f_rk, cfg)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = spec.get("name", cfg.name)
    out_file = a.out_dir / f"{name}.parquet"
    ref_median = f_tr.filter(pl.col(LABEL).is_not_null() & pl.col(LABEL).is_finite())[LABEL].median()
    write_submission(pred, sub_path, out_file, reference_median=ref_median)
    print(f"submission written: {out_file} ({info})")

    log_submission(
        Path("experiments") / ("synthetic_smoke.jsonl" if a.synthetic else "pipeline.jsonl"),
        {
            "config": str(a.config),
            "train_config": asdict(cfg),
            "use_weather": use_weather,
            "wx_codes": wx_codes,
            "wind_dir": wind_dir,
            "gust_repair": gust_repair,
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
