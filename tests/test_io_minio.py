import urllib.parse

import pytest

from prc.config import OsnConfig
from prc.io_minio import OFFICIAL_FILES, presigned_urls, snowflake_fetch_block

# Dummy keys: signing is local because the region is pinned, so none of this touches the network.
CFG = OsnConfig("s3.opensky-network.org", "AKDUMMY", "not-a-real-secret", "prc-2026-datasets", "", "")


def test_official_files_match_the_profiled_names():
    assert len(OFFICIAL_FILES) == len(set(OFFICIAL_FILES)) == 14
    assert "training_2025-12-01_2026-01-01.parquet" in OFFICIAL_FILES


def test_presigned_links_are_short_lived_and_never_carry_the_secret():
    ((name, url),) = presigned_urls(["ranking.parquet"], minutes=15, cfg=CFG)
    q = urllib.parse.urlparse(url)
    assert name == "ranking.parquet" and q.hostname == "s3.opensky-network.org"
    assert q.path == "/prc-2026-datasets/ranking.parquet"
    assert urllib.parse.parse_qs(q.query)["X-Amz-Expires"] == ["900"]
    assert "not-a-real-secret" not in url
    with pytest.raises(ValueError):
        presigned_urls(["ranking.parquet"], minutes=24 * 60, cfg=CFG)


def test_fetch_block_is_one_statement_with_one_call_per_file():
    sql = snowflake_fetch_block(cfg=CFG)
    assert sql.startswith("EXECUTE IMMEDIATE $$") and sql.endswith("$$") and sql.count("$$") == 2
    assert sql.count("CALL PRC_DC2026.RAW.FETCH_URL_TO_STAGE(") == 14
    assert "not-a-real-secret" not in sql
