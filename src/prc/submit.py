"""Upload a validated submission with guard rails: daily counter, version uniqueness, mandatory log entry.
Never uploads probes: the caller must pass the experiment id and local CV numbers (docs/README).

Version uniqueness is checked locally (experiments/submissions.jsonl) and, before any real upload, remotely:
the team bucket and the official leaderboard, because the local ledger cannot see other sessions' uploads
(two sessions both uploaded resilient-hedgehog_v42.parquet and the second overwrote the first). If either
remote source cannot be read, the upload is refused (fail closed).

Dry check, no upload:  python -m prc.submit --check-version N"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from .config import OsnConfig
from .io_minio import list_team_submissions, stat_submission, submission_object_name, upload_submission
from .leaderboard import team_versions, version_of
from .validate_submission import validate_files

DAILY_LIMIT = 5
LOG = Path("experiments/submissions.jsonl")
BRUSSELS = ZoneInfo("Europe/Brussels")  # the organisers state the deadline in CET/CEST


def _today_count(log: Path, now: dt.datetime | None = None) -> tuple[int, set[int]]:
    """Real uploads counted today, plus the versions already used.

    The daily quota's time zone is undocumented (docs/README), so count on the UTC day and on the Brussels day
    and take the larger - the safe side. Dry runs consume neither the quota nor a version number."""
    now = now or dt.datetime.now(dt.UTC)
    versions: set[int] = set()
    if not log.exists():
        return 0, versions
    # Count versions, not lines. Every real upload writes twice - a write-ahead "attempted" entry and the
    # final one - so counting lines doubled the day's total and refused the fourth submission of the day as
    # the seventh. The write-ahead entry is the point of the design, so the counter is what had to change.
    utc_versions: set[int] = set()
    cet_versions: set[int] = set()
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("dry_run"):
            continue
        v = int(rec["version"])
        versions.add(v)
        ts = dt.datetime.fromisoformat(rec["uploaded_at"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.UTC)
        if ts.astimezone(dt.UTC).date() == now.astimezone(dt.UTC).date():
            utc_versions.add(v)
        if ts.astimezone(BRUSSELS).date() == now.astimezone(BRUSSELS).date():
            cet_versions.add(v)
    return max(len(utc_versions), len(cet_versions)), versions


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
    # Remote checks before the write-ahead entry: a refused version must not be recorded as attempted.
    remote = remote_version_check(version, cfg or OsnConfig.from_env(), local_versions=versions)
    if not remote["allowed"]:
        raise RemoteVersionError("refusing to upload: " + "; ".join(remote["reasons"]))
    # write-ahead: an attempted upload must leave a trace even if the transfer dies, because whether a failed
    # upload consumes the daily quota is undocumented (docs/README) and we count it on the safe side.
    _append(log, {**entry, "status": "attempted"})
    try:
        entry["object_name"] = upload_submission(file_path, version, cfg)
    except Exception as exc:
        _append(log, {**entry, "status": "failed", "error": repr(exc)})
        raise
    entry["status"] = "uploaded"
    _append(log, entry)
    return entry


class RemoteVersionError(RuntimeError):
    """The team bucket or the leaderboard already holds this version, or could not be read."""


def remote_version_check(version: int, cfg: OsnConfig, local_versions: set[int] | None = None) -> dict:
    """What the team bucket and the leaderboard say about `version`. Never uploads; never raises on a conflict.

    Returns bucket_objects, bucket_versions, leaderboard_versions, next_free, allowed and reasons. Refuses when
    (a) <team>_v<version>.parquet exists in the team bucket, (b) the leaderboard lists that version for our
    team, (c) the version is not above the max of the bucket and leaderboard versions, or either source cannot
    be read (fail closed)."""
    team = cfg.team_name
    reasons: list[str] = []
    out: dict = {"version": version, "team_bucket": cfg.team_bucket, "team_name": team}
    object_name = submission_object_name(version, cfg) if team else f"<team>_v{version}.parquet"
    out["object_name"] = object_name

    bucket_ok = lb_ok = True
    try:
        objects = list_team_submissions(cfg)
        existing = stat_submission(object_name, cfg)
    except Exception as exc:  # noqa: BLE001 - any failure to read the bucket must refuse the upload
        bucket_ok, objects, existing = False, [], None
        reasons.append(f"team bucket unreachable ({type(exc).__name__}: {exc})")
    out["bucket_objects"] = objects
    out["bucket_versions"] = sorted({v for n, _ in objects if (v := version_of(n, team)) is not None})
    if existing is not None:
        reasons.append(f"{cfg.team_bucket}/{object_name} already exists ({existing.size:,} bytes)")

    try:
        lb = team_versions(team)
    except Exception as exc:  # noqa: BLE001
        lb_ok, lb = False, set()
        reasons.append(f"leaderboard API unreachable ({type(exc).__name__}: {exc})")
    out["leaderboard_versions"] = sorted(lb)
    if version in lb:
        reasons.append(f"v{version} is already on the leaderboard for {team}")

    remote_max = max([*out["bucket_versions"], *lb], default=0)
    out["remote_max"] = remote_max
    if bucket_ok and lb_ok and version <= remote_max:
        reasons.append(f"v{version} is not above the highest remote version v{remote_max}")
    local = local_versions or set()
    out["local_versions"] = sorted(local)
    if version in local:
        reasons.append(f"v{version} is already in the local ledger")
    out["next_free"] = max([remote_max, *local], default=0) + 1 if bucket_ok and lb_ok else None
    out["reasons"] = reasons
    out["allowed"] = not reasons
    return out


def _append(log: Path, entry: dict) -> None:
    with open(log, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _print_check(res: dict) -> None:
    print(f"team bucket: {res['team_bucket']}  team: {res['team_name']}")
    print(f"bucket objects ({len(res['bucket_objects'])}):")
    for name, size in sorted(res["bucket_objects"], key=lambda o: (version_of(o[0], res["team_name"]) or 0, o[0])):
        print(f"  {name}  {size:,} bytes")
    print(f"bucket versions: {res['bucket_versions']}")
    print(f"leaderboard versions: {res['leaderboard_versions']}")
    print(f"local ledger versions: {res['local_versions']}")
    print(f"remote max: v{res['remote_max']}  next free version: {res['next_free']}")
    verdict = "ALLOWED" if res["allowed"] else "REFUSED"
    print(f"v{res['version']} ({res['object_name']}): {verdict}")
    for r in res["reasons"]:
        print(f"  - {r}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m prc.submit", description="Version checks; never uploads.")
    ap.add_argument("--check-version", type=int, required=True, metavar="N", help="version to check")
    ap.add_argument("--log", type=Path, default=LOG)
    args = ap.parse_args(argv)
    try:
        cfg = OsnConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: cannot build the OSN config ({exc})")
        return 2
    _, local = _today_count(args.log)
    res = remote_version_check(args.check_version, cfg, local_versions=local)
    _print_check(res)
    return 0 if res["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
