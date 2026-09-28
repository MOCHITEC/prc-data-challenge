"""Configuration loaded from environment variables (.env is git-ignored; see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Official submission template contract (docs/01 §2.1, §3)
SUBMISSION_ID_COL = "MVT_ID_mvt"
SUBMISSION_TARGET_COL = "TAXITIME_SEC_mvt"
EXPECTED_SUBMISSION_ROWS = 344_841  # usedPairs on the leaderboard API (2026-09-09)
TARGET_AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]


# The organisers' key pair is labelled differently depending on where you read it (the MinIO console says
# "access key", the OSN account page says "ID"), so accept every spelling we have seen.
ACCESS_KEY_VARS = ("OSN_ACCESS_KEY", "OSN_ID", "OSN_ACCESS_KEY_ID", "OSN_KEY_ID")
SECRET_KEY_VARS = ("OSN_SECRET_KEY", "OSN_SECRET", "OSN_SECRET_ACCESS_KEY")


def _first_env(names: tuple[str, ...]) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


@dataclass(frozen=True)
class OsnConfig:
    endpoint: str
    access_key: str
    secret_key: str
    competition_bucket: str
    team_bucket: str
    team_name: str
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> OsnConfig:
        access, secret = _first_env(ACCESS_KEY_VARS), _first_env(SECRET_KEY_VARS)
        missing = []
        if not access:
            missing.append(" / ".join(ACCESS_KEY_VARS))
        if not secret:
            missing.append(" / ".join(SECRET_KEY_VARS))
        if missing:
            raise RuntimeError(f"Missing environment variables, set one of each: {missing} (see .env.example)")
        return cls(
            # minio wants a bare host, so tolerate a pasted URL
            endpoint=os.getenv("OSN_S3_ENDPOINT", "s3.opensky-network.org").replace("https://", "").rstrip("/"),
            access_key=access,
            secret_key=secret,
            # both of these are discoverable from the buckets themselves - see prc.doctor
            competition_bucket=os.getenv("OSN_COMPETITION_BUCKET", "prc-2026-datasets"),
            team_bucket=os.getenv("OSN_TEAM_BUCKET", ""),
            team_name=os.getenv("PRC_TEAM_NAME", ""),
            # Pinning the region stops minio from calling GET /<bucket>?location=, which the OpenSky endpoint
            # answers with 503 and which made the first upload attempt fail before any data was sent.
            region=os.getenv("OSN_S3_REGION", "us-east-1"),
        )


def data_dir() -> Path:
    p = Path(os.getenv("PRC_DATA_DIR", "./data"))
    p.mkdir(parents=True, exist_ok=True)
    return p
