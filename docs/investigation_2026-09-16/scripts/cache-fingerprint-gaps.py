"""Item 28 `cache-fingerprint-gaps`: what the feature-cache fingerprint covers, and whether any recorded
experiment could have run on a cache built from feature code other than the code its `git_sha` points at.

Reads only: git objects, experiments/*.jsonl (current and historical via `git show`), file mtimes, and
data/work/feat_meta.json. No parquet is opened, nothing is trained, nothing is rebuilt.

Run from the repository root:  python docs/investigation_2026-09-16/scripts/cache-fingerprint-gaps.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UTC = timezone.utc
WATCHED = ["src/prc/features.py", "src/prc/weather.py", "src/prc/airports.py", "src/prc/train.py",
           "src/prc/pipeline.py", "src/prc/schema.py"]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def blob_hash(sha: str, path: str, n: int = 16) -> str:
    try:
        b = subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return "-"
    return hashlib.sha256(b).hexdigest()[:n]


def commit_time(sha: str) -> datetime:
    return datetime.fromisoformat(git("show", "-s", "--format=%cI", sha))


def mtime(p: Path) -> datetime | None:
    return datetime.fromtimestamp(p.stat().st_mtime, UTC) if p.exists() else None


def fmt(t: datetime | None) -> str:
    return t.astimezone(UTC).strftime("%m-%d %H:%M:%S") if t else "-"


# ---------------------------------------------------------------- 1. fingerprint coverage (static read)
src = (ROOT / "src/prc/pipeline.py").read_text()
fp_body = src[src.index("def cache_fingerprint"):src.index("def build_features")]
keys = re.findall(r'^\s+"(\w+)":', fp_body, flags=re.M)
print("## fingerprint keys (src/prc/pipeline.py cache_fingerprint):", keys)
hashed = re.findall(r"Path\((\w+)\.__file__\)", fp_body)
print("## files hashed:", [f"{m}.py" for m in hashed])
imports = re.findall(r"^from \.(\w+) import", (ROOT / "src/prc/features.py").read_text(), flags=re.M)
print("## features.py imports (not hashed):", imports)

# ---------------------------------------------------------------- 2. commits touching the watched files
print("\n## commits since 2026-09-10 touching watched files (commit time UTC)")
log = git("log", "--since=2026-09-09", "--format=%h|%cI", "--name-only", "--", *WATCHED)
commits: list[tuple[str, datetime, list[str]]] = []
cur = None
for line in log.splitlines():
    if "|" in line and re.match(r"^[0-9a-f]{7,}\|", line):
        h, t = line.split("|")
        cur = (h, datetime.fromisoformat(t), [])
        commits.append(cur)
    elif line.strip() and cur is not None:
        cur[2].append(line.strip().replace("src/prc/", ""))
commits.sort(key=lambda c: c[1])
for h, t, files in commits:
    print(f"{h} {fmt(t)} feat={blob_hash(h, 'src/prc/features.py')} wx={blob_hash(h, 'src/prc/weather.py', 8)} "
          f"ap={blob_hash(h, 'src/prc/airports.py', 8)} :: {', '.join(files)}")

# ---------------------------------------------------------------- 3. experiment records, current + dropped
recs = []
for line in (ROOT / "experiments/experiments.jsonl").read_text().splitlines():
    r = json.loads(line)
    r["_end"] = mtime(ROOT / "experiments" / f"oof_{r['exp_id']}.parquet")
    r["_src"] = "experiments.jsonl"
    recs.append(r)
dropped = subprocess.check_output(["git", "show", "4e807b5^:experiments/experiments.jsonl"], cwd=ROOT, text=True)
for line in dropped.splitlines():
    r = json.loads(line)
    if any(x["exp_id"] == r["exp_id"] for x in recs):
        continue
    name = r["config"]["name"]
    r["_end"] = mtime(ROOT / "outputs" / f"{name}.parquet")  # OOF files for these were deleted; output mtime instead
    r["_src"] = "dropped@4e807b5"
    recs.append(r)
recs.sort(key=lambda r: r["_end"] or datetime.min.replace(tzinfo=UTC))


def head_at(t: datetime) -> str:
    """Last commit on the first-parent line whose commit time is <= t."""
    out = git("log", "--first-parent", "--format=%h|%cI", "HEAD")
    for line in out.splitlines():
        h, ti = line.split("|")
        if datetime.fromisoformat(ti) <= t:
            return h
    return "-"


def use_weather_at(sha: str, name: str, nf: int) -> str:
    try:
        y = subprocess.check_output(["git", "show", f"{sha}:configs/{name}.yaml"], cwd=ROOT, text=True,
                                    stderr=subprocess.DEVNULL)
        m = re.search(r"^use_weather:\s*(\w+)", y, flags=re.M)
        if m:
            return m.group(1).lower()
    except subprocess.CalledProcessError:
        pass
    return f"inferred:{'true' if nf and nf >= 90 else 'false'}"


print("\n## experiment timeline (end = OOF/output mtime UTC; start_est = end - train_seconds - 60 s load)")
hdr = ("| # | exp_id | git_sha (commit time) | feat@sha | wx@sha | ap@sha | use_weather | n_feat | end | train_s "
       "| gap_prev-train (s) | HEAD@start_est | feat@HEAD@start | rebuild expected | note |")
print(hdr)
print("|" + "---|" * (hdr.count("|") - 1))
prev_end, prev_feat, prev_wx = None, None, None
for i, r in enumerate(recs):
    sha, end, ts = r["git_sha"], r["_end"], float(r.get("train_seconds") or 0)
    ct = commit_time(sha)
    feat = blob_hash(sha, "src/prc/features.py")
    wx = blob_hash(sha, "src/prc/weather.py", 8)
    ap = blob_hash(sha, "src/prc/airports.py", 8)
    name = r["config"].get("name", r["exp_id"].rsplit("-", 1)[0])
    uw = use_weather_at(sha, name, r.get("n_features") or 0)
    uw_bool = uw.endswith("true")
    start_est = end - timedelta(seconds=ts + 60) if end else None
    h0 = head_at(start_est) if start_est else "-"
    feat0 = blob_hash(h0, "src/prc/features.py") if h0 != "-" else "-"
    gap = (end - prev_end).total_seconds() - ts if (end and prev_end) else None
    expect = "yes" if (prev_feat is None or feat != prev_feat or uw_bool != prev_wx) else "no"
    notes = []
    if ct > (end or ct):
        notes.append("COMMIT AFTER RUN END")
    if start_est and ct > start_est:
        notes.append(f"sha committed {int((ct - start_est).total_seconds())} s after start_est")
    if feat0 not in ("-", feat):
        notes.append("features.py@HEAD@start != features.py@git_sha")
    if r["_src"] != "experiments.jsonl":
        notes.append("record dropped at 4e807b5; end = outputs mtime")
    print(f"| {i} | {r['exp_id']} | {sha} ({fmt(ct)}) | {feat} | {wx} | {ap} | {uw} | {r.get('n_features')} | "
          f"{fmt(end)} | {ts:.0f} | {'-' if gap is None else f'{gap:.0f}'} | {h0} | {feat0} | {expect} | "
          f"{'; '.join(notes)} |")
    prev_end, prev_feat, prev_wx = end, feat, uw_bool

# ---------------------------------------------------------------- 4. current cache vs current code
meta = json.loads((ROOT / "data/work/feat_meta.json").read_text())
now_feat = hashlib.sha256((ROOT / "src/prc/features.py").read_bytes()).hexdigest()[:16]
print("\n## current cache")
print("feat_meta.json mtime:", fmt(mtime(ROOT / "data/work/feat_meta.json")),
      "| feature_code:", meta["feature_code"], "| sha256(features.py) now:", now_feat,
      "| use_weather:", meta["use_weather"], "| train_rows:", meta["train_rows"], "| rank_rows:", meta["rank_rows"])
for p in ["data/ref/metar_hourly.parquet", "data/external/airports.parquet", "data/work/feat_train.parquet",
          "data/work/feat_rank.parquet"]:
    q = ROOT / p
    print(f"{p}: mtime {fmt(mtime(q))} size {q.stat().st_size if q.exists() else '-'}")
for p in sorted((ROOT / "data/ref/metar").glob("*.csv")):
    print(f"{p.relative_to(ROOT)}: mtime {fmt(mtime(p))}")
print("HEAD now:", git("rev-parse", "--short", "HEAD"), "| git status --short:", git("status", "--short") or "(clean)")
