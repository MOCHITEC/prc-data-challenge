"""Credential and bucket check: run this first in a new session (docs/05 section 0).

    python -m prc.doctor

Verifies that the OpenSky credentials in the environment work, discovers which buckets they can see, reports
what the competition bucket holds, and tells you the team name and submission bucket to put in the
environment if they are not set yet. Read-only: it never writes to a bucket."""

from __future__ import annotations

import os
import re

import requests

from .config import ACCESS_KEY_VARS, SECRET_KEY_VARS, OsnConfig, _first_env
from .io_minio import list_buckets, list_objects

EXPECTED_OFFICIAL = re.compile(
    r"^(training_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet|ranking\.parquet|submitting\.parquet)$"
)
# The 2024 edition (ATOW) still occupies the bucket literally named "competition-data": one parquet per day
# plus the three csv split files. Seeing those means the 2026 files live somewhere else, not that we are early.
LEGACY_2024 = re.compile(r"^(\d{4}-\d{2}-\d{2}\.parquet|challenge_set\.csv|(final_)?submission_set\.csv)$")

OAUTH_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def looks_like_2024_dataset(object_names: list[str]) -> bool:
    """True when the listing is the 2024 ATOW dataset rather than the 2026 taxi-time files."""
    leaves = [n.split("/")[-1] for n in object_names]
    if not leaves or any(EXPECTED_OFFICIAL.match(n) for n in leaves):
        return False
    return sum(bool(LEGACY_2024.match(n)) for n in leaves) > len(leaves) / 2


def is_oauth_api_client(access_key: str, secret_key: str, timeout: int = 20) -> bool:
    """True when the pair is an OpenSky *API client* (OAuth2), which the S3 endpoint does not accept.

    The account page hands out two unrelated credentials under similar names; mistaking one for the other is
    the failure this whole function exists to name. Read-only: it only asks the issuer for a token."""
    try:
        r = requests.post(
            OAUTH_TOKEN_URL,
            data={"grant_type": "client_credentials", "client_id": access_key, "client_secret": secret_key},
            timeout=timeout,
        )
    except requests.RequestException:
        return False
    return r.status_code == 200 and "access_token" in r.text


def main() -> int:
    print("== environment")
    for names in (ACCESS_KEY_VARS, SECRET_KEY_VARS):
        found = [n for n in names if os.getenv(n)]
        print(f"  {' / '.join(names)}: {'set via ' + found[0] if found else 'MISSING'}")
    for n in ("OSN_TEAM_BUCKET", "PRC_TEAM_NAME", "OSN_S3_ENDPOINT", "OSN_COMPETITION_BUCKET"):
        print(f"  {n}: {os.getenv(n) or '(unset)'}")
    if not (_first_env(ACCESS_KEY_VARS) and _first_env(SECRET_KEY_VARS)):
        print(
            "\nAdd the access key and secret to the cloud environment, then start a NEW session: the values are\n"
            "copied once at session start, so a running session cannot pick them up."
        )
        return 1

    cfg = OsnConfig.from_env()
    print(f"\n== connection to {cfg.endpoint}")
    try:
        buckets = list_buckets(cfg)
    except Exception as exc:  # noqa: BLE001 - the point is to report any failure legibly
        print(f"  FAILED: {exc!r}")
        if "InvalidAccessKeyId" in repr(exc) and is_oauth_api_client(cfg.access_key, cfg.secret_key):
            print(
                "\n  DIAGNOSIS: this pair is an OpenSky *API client* (OAuth2 client id + secret), not an S3\n"
                "  access key. It authenticates fine against the OpenSky token endpoint, and the S3 endpoint\n"
                "  rejects it. The two are issued in different places under confusingly similar names.\n"
                "  Fix: sign in to https://s3.opensky-network.org/ with the OpenSky account, open\n"
                "  'Access Keys' -> 'Create access key', and put THOSE two values in the cloud environment as\n"
                "  OSN_ACCESS_KEY and OSN_SECRET_KEY. Then start a NEW session (variables are copied once).\n"
                "  (The MinIO console's own OIDC client is the only one its STS accepts, so the API client\n"
                "  cannot be exchanged for temporary S3 credentials either - this was tried.)"
            )
        else:
            print("  Check the key/secret pair and that the team request was approved.")
        return 1
    print(f"  ok, {len(buckets)} bucket(s) visible: {buckets}")

    comp = (
        cfg.competition_bucket
        if cfg.competition_bucket and cfg.competition_bucket in buckets
        else next((b for b in buckets if "competition" in b or "data" in b), None)
    )
    team = cfg.team_bucket if cfg.team_bucket in buckets else next((b for b in buckets if b != comp), None)

    if comp:
        objs = list_objects(comp, cfg=cfg)
        official = [o for o in objs if EXPECTED_OFFICIAL.match(o[0].split("/")[-1])]
        total_mb = sum(s for _, s in objs) / 1e6
        print(f"\n== competition bucket '{comp}': {len(objs)} object(s), {total_mb:.0f} MB")
        for name, size in sorted(objs)[:20]:
            print(f"  {name}  {size / 1e6:.1f} MB")
        if len(objs) > 20:
            print(f"  ... and {len(objs) - 20} more")
        missing_kinds = []
        if not any(o[0].endswith("ranking.parquet") for o in objs):
            missing_kinds.append("ranking.parquet")
        if not any(o[0].endswith("submitting.parquet") for o in objs):
            missing_kinds.append("submitting.parquet")
        n_train = sum(1 for o in objs if "training_" in o[0])
        print(f"  official-looking files: {len(official)} (training months: {n_train})")
        if looks_like_2024_dataset([o[0] for o in objs]):
            print(
                "  NOTE: this is the 2024 ATOW dataset (daily parquet + challenge_set.csv), not the 2026\n"
                "  taxi-time files. The 2026 data is in a different bucket - set OSN_COMPETITION_BUCKET to it."
            )
        if missing_kinds:
            print(f"  NOTE: not found in this bucket: {missing_kinds} - check the other buckets or ask on Discord")
    else:
        print("\n== no competition-data bucket found among the visible buckets")

    print("\n== submission target")
    if team:
        print(f"  candidate team bucket: '{team}'")
        guess = re.sub(r"^(submissions?[-_]?|prc[-_]?)", "", team)
        print(
            f"  suggested PRC_TEAM_NAME: '{cfg.team_name or guess}'  "
            f"(submission files are named <team-name>_v<N>.parquet)"
        )
        if not cfg.team_bucket or not cfg.team_name:
            print("  add OSN_TEAM_BUCKET and PRC_TEAM_NAME to the environment to enable uploads")
    else:
        print("  no separate bucket visible; the submission bucket may be write-only and not listable.")
        print("  set OSN_TEAM_BUCKET and PRC_TEAM_NAME from the approval e-mail.")
    print("\nNext: make download  ->  make profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
