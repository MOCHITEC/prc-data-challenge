"""Item 25 `n-features-72-mislabel`: re-derive, per submission v1..v9, the feature matrix model A and model B saw.

No training. For each submission's commit the script
  1. `git archive`s src/prc and configs into a scratch directory,
  2. runs a driver under that commit's code (PYTHONPATH, POLARS_MAX_THREADS=1) which
     - loads the submission's YAML the way that commit's pipeline.main did,
     - builds features on a slice of one raw training month with that commit's features.py,
     - joins weather with that commit's weather.join_weather and the cached data/ref/metar_hourly.parquet
       when the YAML says use_weather (falls back to a naive-timestamp workaround if the join crashes and says so),
     - replaces train._fit with a stub that records (feature_cols, cat_cols) and returns zeros, then calls
       predict_fold so the exact column lists for model A (matched) and model B (unmatched) are captured,
     - evaluates the `n_features` expression that run_cv logged at that commit (frame numerics vs visible columns),
  3. prints one markdown table.

Usage: .venv/bin/python docs/investigation_2026-09-16/scripts/n-features-72-mislabel.py [--rows N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# (submission, commit whose code is taken, why that commit, config path at that commit)
SUBMISSIONS = [
    ("v1", "e0099d7", "git_sha of CV record v1_baseline-cf5c3c4bb9 (recovered from 4e807b5^)", "configs/v1_baseline.yaml"),
    ("v2", "07ae6a3", "git_sha of CV record v2_two_clock-ce1c9c23ce (recovered from 4e807b5^)", "configs/v2_two_clock.yaml"),
    ("v3", "9f31cef", "no CV/pipeline record exists; last commit before upload 09-10 22:21Z", "configs/v3_two_clock_weather.yaml"),
    ("v4", "9f31cef", "no CV/pipeline record exists; last commit before upload 09-11 00:42Z", "configs/v4_linear_tree.yaml"),
    ("v5", "949d33b", "git_sha of CV record main-f9d57a661f", "configs/main.yaml"),
    ("v6", "062a6f0", "git_sha of CV record main-63086f0648", "configs/main.yaml"),
    ("v7", "fd848bf", "submit commit; src/prc identical to 97e67c6 (git_sha of main-48c19bc830)", "configs/main.yaml"),
    ("v8", "70ef994", "git_sha of CV record det_a-cfc9417c52", "configs/det_a.yaml"),
    ("today", "HEAD", "current code", "configs/main.yaml"),
]

DRIVER = r'''
import inspect, json, sys, traceback
from pathlib import Path
import numpy as np
import polars as pl
import yaml

sha, cfg_path, n_rows, repo = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
out = {"sha": sha, "config": cfg_path}
import prc.features as F
import prc.train as T
out["prc_file"] = F.__file__

spec = yaml.safe_load(Path(cfg_path).read_text())
fields = {k: v for k, v in spec.items() if k in T.TrainConfig.__dataclass_fields__}
if "params" in fields:
    fields["params"] = {**T.TrainConfig().params, **fields["params"]}
for k in ("folds", "exclude_features", "include_candidates"):
    if isinstance(fields.get(k), list):
        fields[k] = tuple(fields[k])
cfg = T.TrainConfig(**fields)
use_weather = bool(spec.get("use_weather", False))
out["use_weather"] = use_weather
out["cfg"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items() if k != "params"}

raw = pl.read_parquet(repo / "data/raw/training_2025-01-01_2025-02-01.parquet")
if n_rows > 0:
    raw = raw.head(n_rows)
feats = F.build_departure_features(raw)
out["weather_join"] = "off"
if use_weather:
    import prc.weather as W
    hourly = pl.read_parquet(repo / "data/ref/metar_hourly.parquet")
    try:
        feats = W.join_weather(feats, hourly)
        out["weather_join"] = "ok (commit code, as-is)"
    except Exception as e:
        out["weather_join_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        feats2 = feats.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None))
        feats = W.join_weather(feats2, hourly).with_columns(
            pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone("UTC"))
        out["weather_join"] = "WORKAROUND: departures cast to naive before the commit's join_weather"

out["frame_width"] = feats.width
num_frame = F.numeric_feature_columns(feats)
out["numeric_feature_columns"] = num_frame
out["hour_utc_in_frame"] = "hour_utc" in feats.columns
out["hour_utc_dtype_in_frame"] = str(feats.schema["hour_utc"]) if "hour_utc" in feats.columns else None

src_run_cv = inspect.getsource(T.run_cv)
if "visible_numeric_columns(feats" in src_run_cv:
    out["n_features_rule"] = "len(visible_numeric_columns(feats, cfg))"
    out["n_features_logged"] = len(T.visible_numeric_columns(feats, cfg))
    out["n_frame_columns_logged"] = len(num_frame) if "n_frame_columns" in src_run_cv else None
else:
    out["n_features_rule"] = "len(numeric_feature_columns(feats))"
    out["n_features_logged"] = len(num_frame)

calls = []
def fake_fit(train, target_expr, cfg_, feature_cols, cat_cols, predict_frames, *a, **k):
    calls.append({"n_train": train.height, "feature_cols": list(feature_cols), "cat_cols": list(cat_cols)})
    return [np.zeros(f.height) for f in predict_frames], 0
T._fit = fake_fit

lab = feats.filter(pl.col(T.LABEL).is_not_null())
out["n_labelled"] = lab.height
out["n_unmatched_labelled"] = lab.filter(pl.col(T.PROXY).is_null()).height
try:
    pred, info = T.predict_fold(lab, lab, cfg)
    out["predict_info"] = {k: v for k, v in info.items() if k != "by_airport"}
except Exception:
    out["predict_fold_error"] = traceback.format_exc()[-1500:]
out["fit_calls"] = calls
print(json.dumps(out, default=str))
'''


def run_one(sub: str, sha: str, cfg_path: str, n_rows: int, work: Path) -> dict:
    dest = work / f"{sub}_{sha}"
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / "src.tar"
    with open(tar, "wb") as fh:
        subprocess.run(["git", "archive", sha, "src/prc", "configs"], cwd=REPO, stdout=fh, check=True)
    with tarfile.open(tar) as t:
        t.extractall(dest)
    env = dict(os.environ, PYTHONPATH=str(dest / "src"), POLARS_MAX_THREADS="1", PYTHONWARNINGS="ignore")
    py = str(REPO / ".venv/bin/python")
    proc = subprocess.run(
        [py, "-c", DRIVER, sha, cfg_path, str(n_rows), str(REPO)],
        cwd=dest, env=env, capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        return {"sha": sha, "config": cfg_path, "error": proc.stderr[-2000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400_000, help="rows of the January raw file to use (0 = all)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--json", type=Path, default=None, help="write raw per-commit results here")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="nfeat72_"))
    results = []
    for sub, sha, why, cfg_path in SUBMISSIONS:
        if a.only and sub not in a.only:
            continue
        r = run_one(sub, sha, cfg_path, a.rows, work)
        r.update({"submission": sub, "why": why})
        results.append(r)
        print(f"# {sub} {sha} done: {'ERROR' if 'error' in r else 'ok'}", file=sys.stderr)
    if a.json:
        a.json.write_text(json.dumps(results, indent=1, default=str))

    hdr = ["sub", "commit", "config", "use_weather", "weather join", "frame numerics", "n_features (logged rule)",
           "A numeric", "A cat", "B numeric", "B cat", "hour_utc A/B", "weather cols A/B", "encodings A/B", "candidates A/B"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"| {r['submission']} | {r['sha']} | {r['config']} | ERROR: {r['error'][-300:]!r} |")
            continue
        calls = r.get("fit_calls", [])
        A = calls[0] if len(calls) > 0 else {"feature_cols": [], "cat_cols": []}
        B = calls[1] if len(calls) > 1 else {"feature_cols": [], "cat_cols": []}
        wx = lambda cols: sum(1 for c in cols if c.split("_prev1h")[0] in
                              ("temp_c", "dewp_spread_c", "wind_ms", "gust_ms", "precip_mm", "vis_km",
                               "deicing_gate", "low_vis", "temp_x_precip", "frozen_wx"))
        enc_names = ("ref_p10", "ref_p20", "ref_med", "ref_n", "arh_med", "arh_n", "apt_med", "ref_p10_fb",
                     "ref_missing", "op_offset_med", "op_offset_n")
        cand_names = ("queue_ahead", "bank_position", "stand_rwy_share", "noise_control", "active_runways_60m",
                      "dep_arr_ratio_30m", "congestion_vs_usual_30m", "dest_distance_km", "planned_flight_s")
        enc = lambda cols: sum(1 for c in cols if c in enc_names)
        cand = lambda cols: sum(1 for c in cols if c in cand_names)
        row = [
            r["submission"], r["sha"], r["config"], str(r["use_weather"]), r["weather_join"],
            str(len(r["numeric_feature_columns"])),
            f"{r['n_features_logged']} ({r['n_features_rule']})",
            str(len(A["feature_cols"])), str(len(A["cat_cols"])), str(len(B["feature_cols"])), str(len(B["cat_cols"])),
            f"{'hour_utc' in A['feature_cols']}/{'hour_utc' in B['feature_cols']}",
            f"{wx(A['feature_cols'])}/{wx(B['feature_cols'])}",
            f"{enc(A['feature_cols'])}/{enc(B['feature_cols'])}",
            f"{cand(A['feature_cols'])}/{cand(B['feature_cols'])}",
        ]
        print("| " + " | ".join(row) + " |")
    print()
    for r in results:
        if "error" in r:
            continue
        print(f"## {r['submission']} @ {r['sha']} ({r['why']})")
        print(f"weather_join: {r['weather_join']}; error: {r.get('weather_join_error')}")
        print(f"n_labelled {r['n_labelled']} / unmatched {r['n_unmatched_labelled']}; predict_info {r.get('predict_info')}; "
              f"predict_fold_error {r.get('predict_fold_error')}")
        print(f"hour_utc in frame: {r['hour_utc_in_frame']} dtype {r['hour_utc_dtype_in_frame']}")
        print("frame numerics:", r["numeric_feature_columns"])
        for i, c in enumerate(r.get("fit_calls", [])):
            print(f"model {'AB'[i] if i < 2 else i}: n_train {c['n_train']}, {len(c['feature_cols'])} numeric + {len(c['cat_cols'])} cat")
            print("  numeric:", c["feature_cols"])
            print("  cat:", c["cat_cols"])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
