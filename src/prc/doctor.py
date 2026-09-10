"""Credential and bucket check: run this first in a new session (docs/05 section 0).

    python -m prc.doctor

Verifies that the OpenSky credentials in the environment work, discovers which buckets they can see, reports
what the competition bucket holds, and tells you the team name and submission bucket to put in the
environment if they are not set yet. Read-only: it never writes to a bucket."""

from __future__ import annotations

import os
import re

from .config import ACCESS_KEY_VARS, SECRET_KEY_VARS, OsnConfig, _first_env
from .io_minio import list_buckets, list_objects

EXPECTED_OFFICIAL = re.compile(
    r"^(training_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet|ranking\.parquet|submitting\.parquet)$"
)


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
        print("  Check the key/secret pair and that the team request was approved.")
        return 1
    print(f"  ok, {len(buckets)} bucket(s) visible: {buckets}")

    comp = (
        cfg.competition_bucket
        if cfg.competition_bucket in buckets
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
        if missing_kinds:
            print(f"  NOTE: not found in this bucket: {missing_kinds} - check the other buckets or ask on Discord")
    else:
        print("\n== no competition-data bucket found among the visible buckets")

    print("\n== submission target")
    if team:
        print(f"  candidate team bucket: '{team}'")
        # buckets are named prc-<year>-<team-name>, and the team name is what goes in the submission filename
        guess = re.sub(r"^prc[-_]\d{4}[-_]|^(submissions?[-_]?|prc[-_])", "", team)
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
