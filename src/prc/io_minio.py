"""Download competition files from, and upload submissions to, the OpenSky Network MinIO buckets.

Credentials come from environment variables only (never from arguments or chat)."""

from __future__ import annotations

import logging
from pathlib import Path

from minio import Minio

from .config import OsnConfig, data_dir

log = logging.getLogger(__name__)


def client(cfg: OsnConfig | None = None) -> Minio:
    cfg = cfg or OsnConfig.from_env()
    return Minio(cfg.endpoint, access_key=cfg.access_key, secret_key=cfg.secret_key, secure=True)


def list_objects(bucket: str, prefix: str = "", cfg: OsnConfig | None = None) -> list[str]:
    c = client(cfg)
    return [o.object_name for o in c.list_objects(bucket, prefix=prefix, recursive=True)]


def download_all(
    bucket: str | None = None, dest: Path | None = None, cfg: OsnConfig | None = None
) -> list[Path]:
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


def upload_submission(local_file: Path, version: int, cfg: OsnConfig | None = None) -> str:
    """Upload <team-name>_v<version>.parquet to the team bucket. Caller must have run the validation gate."""
    cfg = cfg or OsnConfig.from_env()
    if not cfg.team_bucket or not cfg.team_name:
        raise RuntimeError("OSN_TEAM_BUCKET and PRC_TEAM_NAME must be set to upload submissions")
    object_name = f"{cfg.team_name}_v{version}.parquet"
    client(cfg).fput_object(cfg.team_bucket, object_name, str(local_file))
    log.info("uploaded %s -> %s/%s", local_file, cfg.team_bucket, object_name)
    return object_name
