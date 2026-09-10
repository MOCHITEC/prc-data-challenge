"""Leaderboard tracking (docs/04 section 0: targets are relative to the current leader).

Pulls the official REST API, stores a dated snapshot under refs/leaderboard_snapshots/ and prints the standings
plus the day-over-day movement. Read-only; nothing here submits anything."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import requests

COMPETITION_ID = "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
API = f"https://datacomp.opensky-network.org/api/competitions/{COMPETITION_ID}/leaderboard"
SNAPSHOT_DIR = Path("refs/leaderboard_snapshots")


def fetch_all(limit: int = 200, timeout: int = 60) -> list[dict]:
    """Every SUCCEEDED submission, following the cursor pagination."""
    items: list[dict] = []
    cursor: str | None = None
    for _ in range(200):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(API, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        items.extend(payload.get("items", []))
        cursor = payload.get("nextCursor")
        if not cursor:
            break
    return items


def best_per_team(items: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for it in items:
        score = it.get("score")
        if score is None:
            continue
        cur = best.get(it["teamName"])
        if cur is None or score < cur["score"]:
            best[it["teamName"]] = it
    return sorted(best.values(), key=lambda x: x["score"])


def save_snapshot(items: list[dict], now: dt.datetime | None = None, out_dir: Path = SNAPSHOT_DIR) -> Path:
    now = now or dt.datetime.now(dt.UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{now:%Y-%m-%dT%HZ}_leaderboard.json"
    path.write_text(json.dumps(items))
    return path


def load_snapshot(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def previous_snapshot(out_dir: Path = SNAPSHOT_DIR, exclude: Path | None = None) -> Path | None:
    files = sorted(p for p in out_dir.glob("*_leaderboard.json") if p != exclude)
    return files[-1] if files else None


def summarise(items: list[dict], our_team: str | None = None, prev: list[dict] | None = None, top: int = 10) -> str:
    board = best_per_team(items)
    prev_board = {b["teamName"]: b["score"] for b in best_per_team(prev)} if prev else {}
    lines = [f"submissions={len(items)} teams={len(board)} leader={board[0]['score']:.2f} ({board[0]['teamName']})"]
    for i, b in enumerate(board[:top], 1):
        delta = b["score"] - prev_board[b["teamName"]] if b["teamName"] in prev_board else None
        d = f" ({delta:+.2f})" if delta is not None else " (new)"
        lines.append(f"{i:2d} {b['score']:8.2f}{d}  {b['teamName']}  {b['filename']}")
    if our_team:
        ours = next((i for i, b in enumerate(board, 1) if b["teamName"] == our_team), None)
        if ours:
            gap = board[ours - 1]["score"] - board[0]["score"]
            lines.append(f"us: rank {ours}/{len(board)} score {board[ours - 1]['score']:.2f} gap_to_leader {gap:+.2f}")
        else:
            lines.append(f"us: {our_team} has no scored submission yet")
    return "\n".join(lines)


def main() -> int:
    import os

    items = fetch_all()
    path = save_snapshot(items)
    prev_path = previous_snapshot(exclude=path)
    prev = load_snapshot(prev_path) if prev_path else None
    print(summarise(items, our_team=os.getenv("PRC_TEAM_NAME") or None, prev=prev))
    print(f"snapshot: {path}" + (f"  (compared with {prev_path.name})" if prev_path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
