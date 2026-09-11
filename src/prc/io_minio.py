"""Download competition files from, and upload submissions to, the OpenSky Network MinIO buckets.

Credentials come from environment variables only (never from arguments or chat)."""

from __future__ import annotations

import logging
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from .config import OsnConfig, data_dir

log = logging.getLogger(__name__)


def client(cfg: OsnConfig | None = None) -> Minio:
    cfg = cfg or OsnConfig.from_env()
    return Minio(cfg.endpoint, access_key=cfg.access_key, secret_key=cfg.secret_key, secure=True, region=cfg.region)


def list_buckets(cfg: OsnConfig | None = None) -> list[str]:
    return [b.name for b in client(cfg).list_buckets()]


def list_objects(bucket: str, prefix: str = "", cfg: OsnConfig | None = None) -> list[tuple[str, int]]:
    c = client(cfg)
    return [(o.object_name, o.size or 0) for o in c.list_objects(bucket, prefix=prefix, recursive=True)]


def download_all(bucket: str | None = None, dest: Path | None = None, cfg: OsnConfig | None = None) -> list[Path]:
    """Download every object of the competition bucket into dest (skips files already present with same size)."""
    cfg = cfg or OsnConfig.from_env()
    bucket = bucket or cfg.competition_bucket
    dest = dest or data_dir() / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    c = client(cfg)
    out: list[Path] = []
    for obj in c.list_objects(bucket, recursive=True):
        target = dest / obj.object_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == obj.size:
            log.info("skip %s (exists, %d bytes)", obj.object_name, obj.size)
        else:
            log.info("download %s (%d bytes)", obj.object_name, obj.size)
            c.fget_object(bucket, obj.object_name, str(target))
        out.append(target)
    return out


def upload_submission(local_file: Path, version: int, cfg: OsnConfig | None = None, overwrite: bool = False) -> str:
    """Upload <team-name>_v<version>.parquet to the team bucket. Caller must have run the validation gate.

    Refuses to replace an object that already exists, unless `overwrite=True` is passed deliberately.

    This guard is not about tidiness. The organisers' scorer writes its verdict back into the team bucket
    (<file>_result.json, <file>_persist.json) and the public leaderboard holds one record per submission - and
    those records are observably mutable: across seven leaderboard snapshots, three submissionIds changed score
    in place while keeping their filename. So re-PUTting a name that has already been scored is the one
    observed way a score can move, and it can move DOWN. Overwriting resilient-hedgehog_v4.parquet with a worse
    model could replace a 305.027 with something worse and there is no delete endpoint to undo it.

    The only thing that previously prevented this was prc.submit's local ledger check against
    experiments/submissions.jsonl, which lives on a disposable sandbox disk and is bypassed entirely by calling
    this function directly."""
    cfg = cfg or OsnConfig.from_env()
    if not cfg.team_bucket or not cfg.team_name:
        raise RuntimeError("OSN_TEAM_BUCKET and PRC_TEAM_NAME must be set to upload submissions")
    object_name = f"{cfg.team_name}_v{version}.parquet"
    c = client(cfg)
    if not overwrite:
        try:
            existing = c.stat_object(cfg.team_bucket, object_name)  # HEAD, non-mutating
        except S3Error as exc:
            if exc.code not in ("NoSuchKey", "NoSuchObject", "NotFound"):
                raise
        else:
            raise FileExistsError(
                f"{cfg.team_bucket}/{object_name} already exists "
                f"({existing.size:,} bytes, uploaded {existing.last_modified}). "
                "Re-uploading a scored name can change its leaderboard score in place, downwards, with no way "
                "to undo it. Use the next version number, or pass overwrite=True if replacing it is intended."
            )
    c.fput_object(cfg.team_bucket, object_name, str(local_file))
    log.info("uploaded %s -> %s/%s", local_file, cfg.team_bucket, object_name)
    return object_name
