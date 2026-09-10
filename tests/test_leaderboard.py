import json

import pytest

from prc.leaderboard import best_per_team, summarise

ITEMS = [
    {"teamName": "alpha", "score": 300.0, "filename": "alpha_v1.parquet"},
    {"teamName": "alpha", "score": 280.5, "filename": "alpha_v2.parquet"},
    {"teamName": "beta", "score": 250.25, "filename": "beta_v9.parquet"},
    {"teamName": "gamma", "score": None, "filename": "gamma_v1.parquet"},
]


def test_best_per_team_takes_minimum_and_sorts():
    board = best_per_team(ITEMS)
    assert [b["teamName"] for b in board] == ["beta", "alpha"]
    assert board[1]["score"] == 280.5  # best of alpha, not the latest
    assert all(b["teamName"] != "gamma" for b in board)  # unscored rows dropped


def test_summarise_reports_gap_and_delta():
    prev = [
        {"teamName": "alpha", "score": 290.0, "filename": "alpha_v1.parquet"},
        {"teamName": "beta", "score": 249.0, "filename": "beta_v8.parquet"},
    ]
    out = summarise(ITEMS, our_team="alpha", prev=prev)
    assert "leader=250.25 (beta)" in out
    assert "-9.50" in out  # alpha improved from 290.0 to 280.5
    assert "gap_to_leader +30.25" in out
    assert "rank 2/2" in out
    assert "no scored submission yet" in summarise(ITEMS, our_team="delta")


def test_reconcile_joins_uploads_to_scores_and_skips_dry_runs_and_failures(tmp_path):
    from prc.leaderboard import format_reconciliation, reconcile

    log = tmp_path / "submissions.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"version": 1, "dry_run": True, "status": "uploaded", "object_name": "t_v1.parquet"},
                {"version": 1, "dry_run": False, "status": "attempted", "object_name": None},
                {
                    "version": 1,
                    "dry_run": False,
                    "status": "uploaded",
                    "object_name": "t_v1.parquet",
                    "exp_id": "a",
                    "cv": {"rmse_full": 337.02},
                    "note": "baseline",
                },
                {"version": 2, "dry_run": False, "status": "failed", "object_name": "t_v2.parquet"},
                {
                    "version": 3,
                    "dry_run": False,
                    "status": "uploaded",
                    "object_name": "t_v3.parquet",
                    "exp_id": "b",
                    "cv": {"rmse_full": 320.0},
                },
            ]
        )
    )
    items = [
        {"teamName": "t", "filename": "t_v1.parquet", "score": 345.35},
        {"teamName": "other", "filename": "t_v3.parquet", "score": 1.0},  # another team's row must not leak in
    ]
    rows = reconcile(log, items, our_team="t")
    assert [r["version"] for r in rows] == [1, 3]
    assert rows[0]["lb"] == 345.35
    assert rows[0]["gap"] == pytest.approx(345.35 - 337.02)
    assert rows[1]["lb"] is None and rows[1]["gap"] is None  # uploaded but not scored yet
    text = format_reconciliation(rows)
    assert "unscored" in text and "345.35" in text
