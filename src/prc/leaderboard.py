"""Leaderboard tracking (docs/04 section 0: targets are relative to the current leader).

Pulls the official REST API, stores a dated snapshot under refs/leaderboard_snapshots/ and prints the standings
plus the day-over-day movement. Read-only; nothing here submits anything."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # PRC_TEAM_NAME lives in .env, which is git-ignored

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


def reconcile(log_path: Path, items: list[dict], our_team: str | None) -> list[dict]:
    """Join our upload log to the scores the board reports, keyed by the object name we uploaded.

    This is the CV-to-leaderboard calibration table. Doing it from the API rather than by hand is the point:
    a remembered score is a score that drifts. Uploads with no score yet are kept with score=None."""
    scored = {it.get("filename"): it.get("score") for it in items if our_team is None or it.get("teamName") == our_team}
    rows: list[dict] = []
    seen: set[str] = set()
    if not log_path.exists():
        return rows
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        obj = rec.get("object_name")
        if rec.get("dry_run") or rec.get("status") != "uploaded" or not obj or obj in seen:
            continue
        seen.add(obj)
        cv = rec.get("cv") or {}
        rows.append(
            {
                "version": rec.get("version"),
                "object": obj,
                "exp_id": rec.get("exp_id"),
                "cv_full": cv.get("rmse_full"),
                "lb": scored.get(obj),
                "gap": (scored[obj] - cv["rmse_full"]) if scored.get(obj) and cv.get("rmse_full") else None,
                "note": rec.get("note", "")[:60],
            }
        )
    return sorted(rows, key=lambda r: (r["version"] is None, r["version"]))


def format_reconciliation(rows: list[dict]) -> str:
    if not rows:
        return "no scored uploads in the submission log yet"
    out = ["", "our submissions (CV vs leaderboard):", f"{'v':>3} {'CV':>8} {'LB':>8} {'LB-CV':>7}  exp_id"]
    for r in rows:
        cv = f"{r['cv_full']:8.2f}" if r["cv_full"] else "       -"
        lb = f"{r['lb']:8.2f}" if r["lb"] else "  unscored"
        gap = f"{r['gap']:+7.2f}" if r["gap"] is not None else "      -"
        out.append(f"{r['version']:>3} {cv} {lb} {gap}  {r['exp_id']}")
    return "\n".join(out)


def main() -> int:
    import os

    team = os.getenv("PRC_TEAM_NAME") or None
    if not team:
        print("PRC_TEAM_NAME is not set: matching submissions by object name across every team")
    items = fetch_all()
    path = save_snapshot(items)
    prev_path = previous_snapshot(exclude=path)
    prev = load_snapshot(prev_path) if prev_path else None
    print(summarise(items, our_team=team, prev=prev))
    print(format_reconciliation(reconcile(Path("experiments/submissions.jsonl"), items, team)))
    print(f"snapshot: {path}" + (f"  (compared with {prev_path.name})" if prev_path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
