"""Upload a validated submission with guard rails: daily counter, version uniqueness, mandatory log entry.
Never uploads probes: the caller must pass the experiment id and local CV numbers (docs/04 §8)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pyarrow.parquet as pq

from .config import OsnConfig
from .io_minio import upload_submission
from .validate_submission import validate_files

DAILY_LIMIT = 5
LOG = Path("experiments/submissions.jsonl")


def _today_count(log: Path) -> tuple[int, set[int]]:
    n, versions = 0, set()
    if not log.exists():
        return 0, versions
    today_utc = dt.datetime.now(dt.UTC).date()
    today_cet = (dt.datetime.now(dt.UTC) + dt.timedelta(hours=2)).date()  # CEST during the competition
    for line in log.read_text().splitlines():
        rec = json.loads(line)
        versions.add(int(rec["version"]))
        ts = dt.datetime.fromisoformat(rec["uploaded_at"])
        if ts.date() in (today_utc, today_cet):
            n += 1
    return n, versions


def submit(
    file_path: Path,
    template_path: Path,
    version: int,
    exp_id: str,
    cv_summary: dict,
    note: str,
    log: Path = LOG,
    cfg: OsnConfig | None = None,
    dry_run: bool = False,
) -> dict:
    res = validate_files(file_path, template_path)
    res.raise_if_failed()
    n_today, versions = _today_count(log)
    if version in versions:
        raise ValueError(f"version v{version} already used")
    if n_today >= DAILY_LIMIT:
        raise RuntimeError(f"daily limit reached ({n_today}/{DAILY_LIMIT}); wait for the next day")
    n_rows = pq.read_metadata(file_path).num_rows
    entry = {
        "version": version,
        "exp_id": exp_id,
        "cv": cv_summary,
        "note": note,
        "n_rows": n_rows,
        "file": str(file_path),
        "uploaded_at": dt.datetime.now(dt.UTC).isoformat(),
        "dry_run": dry_run,
    }
    if not dry_run:
        entry["object_name"] = upload_submission(file_path, version, cfg)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry
