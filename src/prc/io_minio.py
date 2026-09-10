"""Download competition files from, and upload submissions to, the OpenSky Network MinIO buckets.

Credentials come from environment variables only (never from arguments or chat)."""

from __future__ import annotations

import logging
from pathlib import Path

import certifi
import urllib3
from minio import Minio

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
            "OSN_COMPETITION_BUCKET is not set. Run `make doctor`: it lists the buckets the credentials can "
            "see and flags the 2024 dataset, so you can set the variable to the 2026 bucket (docs/05 §1)."
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


def upload_submission(local_file: Path, version: int, cfg: OsnConfig | None = None) -> str:
    """Upload <team-name>_v<version>.parquet to the team bucket. Caller must have run the validation gate."""
    cfg = cfg or OsnConfig.from_env()
    if not cfg.team_bucket or not cfg.team_name:
        raise RuntimeError("OSN_TEAM_BUCKET and PRC_TEAM_NAME must be set to upload submissions")
    object_name = f"{cfg.team_name}_v{version}.parquet"
    client(cfg).fput_object(cfg.team_bucket, object_name, str(local_file))
    log.info("uploaded %s -> %s/%s", local_file, cfg.team_bucket, object_name)
    return object_name
