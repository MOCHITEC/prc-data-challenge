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
