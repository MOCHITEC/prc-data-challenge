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


@dataclass(frozen=True)
class OsnConfig:
    endpoint: str
    access_key: str
    secret_key: str
    competition_bucket: str
    team_bucket: str
    team_name: str

    @classmethod
    def from_env(cls) -> OsnConfig:
        missing = [k for k in ("OSN_ACCESS_KEY", "OSN_SECRET_KEY") if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing environment variables: {missing} (see .env.example)")
        return cls(
            endpoint=os.getenv("OSN_S3_ENDPOINT", "s3.opensky-network.org"),
            access_key=os.environ["OSN_ACCESS_KEY"],
            secret_key=os.environ["OSN_SECRET_KEY"],
            competition_bucket=os.getenv("OSN_COMPETITION_BUCKET", "competition-data"),
            team_bucket=os.getenv("OSN_TEAM_BUCKET", ""),
            team_name=os.getenv("PRC_TEAM_NAME", ""),
        )


def data_dir() -> Path:
    p = Path(os.getenv("PRC_DATA_DIR", "./data"))
    p.mkdir(parents=True, exist_ok=True)
    return p
