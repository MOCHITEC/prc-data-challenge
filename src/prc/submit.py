"""Upload a validated submission with guard rails: daily counter, version uniqueness, mandatory log entry.
Never uploads probes: the caller must pass the experiment id and local CV numbers (docs/README).

Version uniqueness is checked locally (experiments/submissions.jsonl) and, before any real upload, remotely:
the team bucket and the official leaderboard, because the local ledger cannot see other sessions' uploads
(two sessions both uploaded resilient-hedgehog_v42.parquet and the second overwrote the first). If either
remote source cannot be read, the upload is refused (fail closed).

Dry check, no upload:  python -m prc.submit --check-version N

Final phase (one object, <team>_final.parquet, no _vNN checks; see docs/final_phase_2026-10-11/RUNBOOK_final_submit.md):
    python -m prc.submit --final --file F --template final_submitting.parquet --ranking final_ranking.parquet \
        --source-commit C --source-path P --expect-md5 M [--dry-run] [--confirm-final-overwrite]"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from .config import OsnConfig
from .io_minio import (
    download_team_object,
    final_object_name,
    list_team_submissions,
    stat_submission,
    submission_object_name,
    upload_final,
    upload_submission,
)
from .leaderboard import team_versions, version_of
from .validate_final import file_md5, format_report, validate_final_files
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
        if rec.get("dry_run") or rec.get("kind") == "final":  # the final is one object outside the _vNN series
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


class FinalRefused(RuntimeError):
    """The final upload was refused before anything was written to the bucket."""


class FinalVerifyError(RuntimeError):
    """The final object was uploaded, but the bytes read back do not match the local file."""


def source_blob_md5(commit: str, path: str) -> str:
    """md5 of `git show <commit>:<path>`, so the ledger's source commit and path are checked, not just copied."""
    import hashlib

    blob = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    return hashlib.md5(blob).hexdigest()  # integrity fingerprint, not a security hash


def _stat_dict(st) -> dict | None:
    if st is None:
        return None
    return {"size": st.size, "last_modified": str(st.last_modified), "etag": getattr(st, "etag", None)}


def submit_final(
    file_path: Path,
    template_path: Path,
    ranking_path: Path,
    source_commit: str,
    source_path: str,
    expect_md5: str | None = None,
    note: str = "",
    log: Path = LOG,
    cfg: OsnConfig | None = None,
    dry_run: bool = False,
    confirm_overwrite: bool = False,
) -> dict:
    """Validate, then upload <team>_final.parquet once, then read it back and compare md5.

    Order: (1) the final gate (prc.validate_final), (2) the local file must equal the blob at
    source_commit:source_path, (3) HEAD the final object and print its size and mtime, (4) refuse if it exists and
    `confirm_overwrite` is False, (5) dry run: one ledger entry and stop, (6) write-ahead "attempted" entry, PUT
    (io_minio.upload_final re-HEADs and refuses if the object changed since step 3), (7) GET it back and compare
    md5, (8) final ledger entry. The _vNN checks (daily counter, version numbers, leaderboard) do not apply.
    A refusal in steps 1 to 4 raises FinalRefused and writes nothing to the ledger."""
    file_path = Path(file_path)
    rep = validate_final_files(file_path, Path(template_path), Path(ranking_path), expect_md5=expect_md5)
    print(format_report(rep))
    if not rep.ok:
        raise FinalRefused("final gate failed, nothing uploaded:\n- " + "\n- ".join(rep.errors))
    md5 = rep.md5

    try:
        src_md5 = source_blob_md5(source_commit, source_path)
    except Exception as exc:  # an unreadable source must refuse, not skip the check
        raise FinalRefused(f"cannot read {source_commit}:{source_path} ({type(exc).__name__}: {exc})") from exc
    if src_md5 != md5:
        raise FinalRefused(f"{file_path} (md5 {md5}) is not {source_commit}:{source_path} (md5 {src_md5})")
    print(f"source: {source_commit}:{source_path} md5 {src_md5} == local file")

    cfg = cfg or OsnConfig.from_env()
    if not cfg.team_bucket or not cfg.team_name:
        raise FinalRefused("OSN_TEAM_BUCKET and PRC_TEAM_NAME must be set")
    object_name = final_object_name(cfg)
    try:
        existing = stat_submission(object_name, cfg)
    except Exception as exc:  # unreachable or denied is not "absent": fail closed
        raise FinalRefused(f"cannot HEAD {cfg.team_bucket}/{object_name} ({type(exc).__name__}: {exc})") from exc
    if existing is None:
        print(f"target: {cfg.team_bucket}/{object_name} does not exist yet")
    else:
        print(
            f"target: {cfg.team_bucket}/{object_name} EXISTS: {existing.size:,} bytes, "
            f"last modified {existing.last_modified}, etag {getattr(existing, 'etag', None)}"
        )
        if not confirm_overwrite:
            raise FinalRefused(
                f"{cfg.team_bucket}/{object_name} already exists ({existing.size:,} bytes, last modified "
                f"{existing.last_modified}); pass --confirm-final-overwrite to replace it"
            )
        print("overwrite confirmed (--confirm-final-overwrite)")

    entry = {
        "version": "final",
        "kind": "final",
        "object_name": object_name,
        "team_bucket": cfg.team_bucket,
        "file": str(file_path),
        "md5": md5,
        "source_commit": source_commit,
        "source_path": source_path,
        "n_rows": rep.n_rows,
        "validation": rep.summary(),
        "existing_object": _stat_dict(existing),
        "overwrite_confirmed": bool(confirm_overwrite),
        "note": note,
        "uploaded_at": dt.datetime.now(dt.UTC).isoformat(),
        "dry_run": dry_run,
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        entry["status"] = "dry_run"
        _append(log, entry)
        print(f"DRY RUN: would upload {file_path} -> {cfg.team_bucket}/{object_name}; nothing uploaded")
        return entry

    _append(log, {**entry, "status": "attempted"})  # write-ahead: a transfer that dies still leaves a trace
    try:
        upload_final(file_path, cfg, expected_existing=existing)
    except Exception as exc:
        _append(log, {**entry, "status": "failed", "error": repr(exc)})
        raise

    try:
        remote_stat = stat_submission(object_name, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            back = download_team_object(object_name, Path(tmp) / object_name, cfg)
            remote_md5 = file_md5(back)
    except Exception as exc:
        _append(log, {**entry, "status": "uploaded_unverified", "error": repr(exc)})
        raise FinalVerifyError(f"uploaded, but the read-back failed ({type(exc).__name__}: {exc})") from exc
    entry["remote_object"] = _stat_dict(remote_stat)
    entry["remote_md5"] = remote_md5
    entry["verified_at"] = dt.datetime.now(dt.UTC).isoformat()
    if remote_md5 != md5:
        entry["status"] = "verify_failed"
        _append(log, entry)
        raise FinalVerifyError(f"{cfg.team_bucket}/{object_name} reads back md5 {remote_md5}, local {md5}")
    entry["status"] = "uploaded"
    entry["verified"] = True
    _append(log, entry)
    print(f"UPLOADED and verified: {cfg.team_bucket}/{object_name} md5 {remote_md5} ({_stat_dict(remote_stat)})")
    return entry


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

    ap = argparse.ArgumentParser(
        prog="python -m prc.submit",
        description="--check-version N: version checks, never uploads. --final: the one final upload.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-version", type=int, metavar="N", help="version to check")
    mode.add_argument("--final", action="store_true", help="upload <team>_final.parquet (no _vNN checks)")
    ap.add_argument("--log", type=Path, default=LOG)
    fin = ap.add_argument_group("--final")
    fin.add_argument("--file", type=Path, help="the final parquet")
    fin.add_argument("--template", type=Path, help="final_submitting.parquet")
    fin.add_argument("--ranking", type=Path, help="final_ranking.parquet (per-month checks)")
    fin.add_argument("--source-commit", help="commit the file comes from (checked with git show)")
    fin.add_argument("--source-path", help="path of the file in that commit")
    fin.add_argument("--expect-md5", help="md5 the hand-off names")
    fin.add_argument("--note", default="")
    fin.add_argument("--dry-run", action="store_true", help="everything except the upload")
    fin.add_argument("--confirm-final-overwrite", action="store_true", help="replace an existing final object")
    args = ap.parse_args(argv)
    try:
        cfg = OsnConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: cannot build the OSN config ({exc})")
        return 2
    if args.final:
        need = ("file", "template", "ranking", "source_commit", "source_path")
        missing = [f"--{n.replace('_', '-')}" for n in need if getattr(args, n) is None]
        if missing:
            ap.error(f"--final needs {' '.join(missing)}")
        try:
            entry = submit_final(
                args.file,
                args.template,
                args.ranking,
                args.source_commit,
                args.source_path,
                expect_md5=args.expect_md5,
                note=args.note,
                log=args.log,
                cfg=cfg,
                dry_run=args.dry_run,
                confirm_overwrite=args.confirm_final_overwrite,
            )
        except FinalRefused as exc:
            print(f"REFUSED: {exc}")
            return 1
        except FinalVerifyError as exc:
            print(f"VERIFY FAILED: {exc}")
            return 3
        print(f"status: {entry['status']}  ledger: {args.log}")
        return 0
    _, local = _today_count(args.log)
    res = remote_version_check(args.check_version, cfg, local_versions=local)
    _print_check(res)
    return 0 if res["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
