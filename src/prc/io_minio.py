"""Download competition files from, and upload submissions to, the OpenSky Network MinIO buckets.

Credentials come from environment variables only (never from arguments or chat)."""

from __future__ import annotations

import logging
from pathlib import Path

import certifi
import urllib3
from minio import Minio
from minio.error import S3Error

from .config import OsnConfig, _first_env, data_dir

log = logging.getLogger(__name__)

# The sandbox has no direct egress: everything leaves through the agent proxy named by HTTPS_PROXY, whose
# TLS interception is trusted via the CA bundle in SSL_CERT_FILE / REQUESTS_CA_BUNDLE. minio builds a plain
# urllib3.PoolManager that ignores both, so a direct connection dies with "too many 503 error responses".
PROXY_VARS = ("HTTPS_PROXY", "https_proxy")
CA_BUNDLE_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def http_client(timeout: int = 300) -> urllib3.PoolManager:
    """A urllib3 pool that honours the proxy and CA bundle in the environment (both unset: a direct pool)."""
    ca = _first_env(CA_BUNDLE_VARS) or certifi.where()
    kwargs = {
        "cert_reqs": "CERT_REQUIRED",
        "ca_certs": ca,
        "timeout": urllib3.Timeout(connect=20, read=timeout),
        "retries": urllib3.Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]),
    }
    proxy = _first_env(PROXY_VARS)
    return urllib3.ProxyManager(proxy, **kwargs) if proxy else urllib3.PoolManager(**kwargs)


def client(cfg: OsnConfig | None = None) -> Minio:
    cfg = cfg or OsnConfig.from_env()
    return Minio(
        cfg.endpoint,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=True,
        http_client=http_client(),
    )


def list_buckets(cfg: OsnConfig | None = None) -> list[str]:
    return [b.name for b in client(cfg).list_buckets()]


def list_objects(bucket: str, prefix: str = "", cfg: OsnConfig | None = None) -> list[tuple[str, int]]:
    c = client(cfg)
    return [(o.object_name, o.size or 0) for o in c.list_objects(bucket, prefix=prefix, recursive=True)]


def download_all(bucket: str | None = None, dest: Path | None = None, cfg: OsnConfig | None = None) -> list[Path]:
    """Download every object of the competition bucket into dest (skips files already present with same size)."""
    cfg = cfg or OsnConfig.from_env()
    bucket = bucket or cfg.competition_bucket
    if not bucket:
        raise RuntimeError(
            "OSN_COMPETITION_BUCKET is not set. It names the bucket holding training_*.parquet, "
            "ranking.parquet and submitting.parquet; the 2024 edition still occupies 'competition-data', "
            "so the 2026 files are somewhere else (docs/README)."
        )
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


FINAL_COMPETITION_FILES = ("final_submitting.parquet", "final_ranking.parquet", "ranking.parquet")


def download_files(
    names: tuple[str, ...] | list[str] = FINAL_COMPETITION_FILES,
    bucket: str | None = None,
    dest: Path | None = None,
    cfg: OsnConfig | None = None,
) -> list[Path]:
    """Download the named objects of the competition bucket into dest. Read-only: GET only, no PUT.

    Re-downloads a file whose local size differs from the bucket's; skips one whose size matches."""
    cfg = cfg or OsnConfig.from_env()
    bucket = bucket or cfg.competition_bucket
    if not bucket:
        raise RuntimeError("no competition bucket: pass bucket= or set OSN_COMPETITION_BUCKET (prc-2026-datasets)")
    dest = dest or data_dir() / "raw" / "final"
    dest.mkdir(parents=True, exist_ok=True)
    c = client(cfg)
    out: list[Path] = []
    for name in names:
        st = c.stat_object(bucket, name)
        target = dest / name
        if target.exists() and target.stat().st_size == st.size:
            log.info("skip %s (exists, %d bytes)", name, st.size)
        else:
            log.info("download %s/%s (%d bytes)", bucket, name, st.size)
            c.fget_object(bucket, name, str(target))
        out.append(target)
    return out


# stat_object answers these codes for "no such object"; every other error (403, 503, network) is not "absent".
_ABSENT_CODES = ("NoSuchKey", "NoSuchObject", "NotFound")


def submission_object_name(version: int, cfg: OsnConfig) -> str:
    return f"{cfg.team_name}_v{version}.parquet"


def stat_submission(object_name: str, cfg: OsnConfig, c: Minio | None = None):
    """HEAD the object in the team bucket: its stat if it exists, None if the bucket says it does not.

    Any other failure (unreachable, denied) raises, so callers fail closed instead of reading it as "absent"."""
    c = c or client(cfg)
    try:
        return c.stat_object(cfg.team_bucket, object_name)
    except S3Error as exc:
        if exc.code in _ABSENT_CODES:
            return None
        raise


def list_team_submissions(cfg: OsnConfig | None = None) -> list[tuple[str, int]]:
    """Every object in the team bucket that starts with <team-name>_v (submissions and the scorer's sidecars)."""
    cfg = cfg or OsnConfig.from_env()
    if not cfg.team_bucket or not cfg.team_name:
        raise RuntimeError("OSN_TEAM_BUCKET and PRC_TEAM_NAME must be set to list submissions")
    return list_objects(cfg.team_bucket, prefix=f"{cfg.team_name}_v", cfg=cfg)


def upload_submission(local_file: Path, version: int, cfg: OsnConfig | None = None, overwrite: bool = False) -> str:
    """Upload <team-name>_v<version>.parquet to the team bucket. Caller must have run the validation gate.

    Refuses to replace an object that already exists, unless `overwrite=True` is passed deliberately (guard
    ported from claude/quirky-hopper-b7q73j). Two sessions both uploaded resilient-hedgehog_v42.parquet and the
    second PUT silently replaced the first: the bucket has no versioning and the scorer re-reads a re-PUT name,
    so an overwrite can move an already-scored submission's leaderboard score in place. prc.submit adds the
    leaderboard and next-version checks on top; this HEAD is the last line, right before the PUT."""
    cfg = cfg or OsnConfig.from_env()
    if not cfg.team_bucket or not cfg.team_name:
        raise RuntimeError("OSN_TEAM_BUCKET and PRC_TEAM_NAME must be set to upload submissions")
    object_name = submission_object_name(version, cfg)
    c = client(cfg)
    if not overwrite:
        existing = stat_submission(object_name, cfg, c)
        if existing is not None:
            raise FileExistsError(
                f"{cfg.team_bucket}/{object_name} already exists "
                f"({existing.size:,} bytes, uploaded {existing.last_modified}). "
                "Re-uploading a scored name can change its leaderboard score in place, with no way to undo it. "
                "Use the next free version (python -m prc.submit --check-version N)."
            )
    c.fput_object(cfg.team_bucket, object_name, str(local_file))
    log.info("uploaded %s -> %s/%s", local_file, cfg.team_bucket, object_name)
    return object_name
