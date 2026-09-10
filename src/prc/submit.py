"""Upload a validated submission with guard rails: daily counter, version uniqueness, mandatory log entry.
Never uploads probes: the caller must pass the experiment id and local CV numbers (docs/04 §8)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from .config import OsnConfig
from .io_minio import upload_submission
from .validate_submission import validate_files

DAILY_LIMIT = 5
LOG = Path("experiments/submissions.jsonl")
BRUSSELS = ZoneInfo("Europe/Brussels")  # the organisers state the deadline in CET/CEST


def _today_count(log: Path, now: dt.datetime | None = None) -> tuple[int, set[int]]:
    """Real uploads counted today, plus the versions already used.

    The daily quota's time zone is undocumented (docs/01 U9), so count on the UTC day and on the Brussels day
    and take the larger - the safe side. Dry runs consume neither the quota nor a version number."""
    now = now or dt.datetime.now(dt.UTC)
    versions: set[int] = set()
    if not log.exists():
        return 0, versions
    n_utc = n_cet = 0
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("dry_run"):
            continue
        versions.add(int(rec["version"]))
        ts = dt.datetime.fromisoformat(rec["uploaded_at"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.UTC)
        if ts.astimezone(dt.UTC).date() == now.astimezone(dt.UTC).date():
            n_utc += 1
        if ts.astimezone(BRUSSELS).date() == now.astimezone(BRUSSELS).date():
            n_cet += 1
    return max(n_utc, n_cet), versions


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
    for w in res.warnings:
        print(f"submission gate warning: {w}")
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
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        _append(log, entry)
        return entry
    # write-ahead: an attempted upload must leave a trace even if the transfer dies, because whether a failed
    # upload consumes the daily quota is undocumented (docs/01 U10) and we count it on the safe side.
    _append(log, {**entry, "status": "attempted"})
    try:
        entry["object_name"] = upload_submission(file_path, version, cfg)
    except Exception as exc:
        _append(log, {**entry, "status": "failed", "error": repr(exc)})
        raise
    entry["status"] = "uploaded"
    _append(log, entry)
    return entry


def _append(log: Path, entry: dict) -> None:
    with open(log, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
