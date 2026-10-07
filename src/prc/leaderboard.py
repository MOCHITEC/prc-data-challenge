"""Read the official leaderboard API: which of our filenames the organisers have already processed.

The API is public (no credentials) and answers {"items": [...], "nextCursor": ...}; each item carries
teamName, filename, score, usedPairs and processedAt. prc.submit uses it to refuse a version number that
another session has already uploaded, which the local experiments/submissions.jsonl cannot see."""

from __future__ import annotations

import json
import re

from .io_minio import http_client

COMPETITION_ID = "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
LEADERBOARD_URL = f"https://datacomp.opensky-network.org/api/competitions/{COMPETITION_ID}/leaderboard"
PAGE_LIMIT = 200
MAX_PAGES = 50


def version_of(name: str, team_name: str) -> int | None:
    """The N in <team>_vN.parquet (also matches the scorer's <team>_vN.parquet_result.json sidecars)."""
    m = re.match(rf"^{re.escape(team_name)}_v(\d+)\.parquet", name)
    return int(m.group(1)) if m else None


def _get_json(url: str, fields: dict, timeout: int = 30) -> dict:
    resp = http_client(timeout=timeout).request("GET", url, fields=fields)
    if resp.status != 200:
        raise RuntimeError(f"leaderboard API answered HTTP {resp.status}")
    return json.loads(resp.data)


def team_entries(team_name: str, get_json=_get_json) -> list[dict]:
    """Every leaderboard item for `team_name`, following nextCursor. Raises on anything unexpected."""
    if not team_name:
        raise RuntimeError("PRC_TEAM_NAME must be set to read the leaderboard")
    items: list[dict] = []
    seen: set[str] = set()
    cursor = None
    for _ in range(MAX_PAGES):
        fields = {"teamName": team_name, "limit": str(PAGE_LIMIT)}
        if cursor:
            fields["cursor"] = cursor
        page = get_json(LEADERBOARD_URL, fields)
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise TypeError(f"unexpected leaderboard response shape: {str(page)[:200]}")
        items.extend(i for i in page["items"] if i.get("teamName", team_name) == team_name)
        cursor = page.get("nextCursor")
        if not cursor:
            return items
        if cursor in seen:  # the cursor parameter is not documented; a repeat means it was ignored
            raise RuntimeError("leaderboard pagination did not advance; cannot read the full list")
        seen.add(cursor)
    raise RuntimeError(f"leaderboard pagination exceeded {MAX_PAGES} pages")


def team_versions(team_name: str, get_json=_get_json) -> set[int]:
    out = set()
    for item in team_entries(team_name, get_json):
        v = version_of(str(item.get("filename", "")), team_name)
        if v is not None:
            out.add(v)
    return out
