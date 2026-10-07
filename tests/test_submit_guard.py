"""Remote version guard: two sessions both uploaded resilient-hedgehog_v42.parquet and the second overwrote the
first, because prc.submit only checked the local ledger. Every bucket and API call here is mocked."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from minio.error import S3Error

import prc.submit as submit_mod
from prc import io_minio, leaderboard
from prc.config import SUBMISSION_ID_COL, SUBMISSION_TARGET_COL, OsnConfig
from prc.submit import RemoteVersionError, main, remote_version_check, submit

CFG = OsnConfig(
    endpoint="s3.example", access_key="AK", secret_key="SK", competition_bucket="", team_bucket="tb", team_name="team"
)


def _s3_error(code):
    return S3Error(None, code, "msg", "/tb/x", "rid", "hid")


class FakeBucket:
    def __init__(self, versions=(), sidecars=True):
        self.objects = {}
        for v in versions:
            self.objects[f"team_v{v}.parquet"] = 1000 + v
            if sidecars:
                self.objects[f"team_v{v}.parquet_result.json"] = 280
        self.put = []

    def list(self, cfg=None):
        return sorted(self.objects.items())

    def stat(self, name, cfg=None, c=None):
        return SimpleNamespace(size=self.objects[name], last_modified="t") if name in self.objects else None


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Mocked bucket + leaderboard, a no-op validation gate, and a tiny real parquet file."""
    state = SimpleNamespace(bucket=FakeBucket(), lb=set(), uploads=[])
    monkeypatch.setattr(submit_mod, "list_team_submissions", lambda cfg: state.bucket.list(cfg))
    monkeypatch.setattr(submit_mod, "stat_submission", lambda name, cfg: state.bucket.stat(name, cfg))
    monkeypatch.setattr(submit_mod, "team_versions", lambda team: set(state.lb))

    def fake_upload(path, version, cfg):
        state.uploads.append(version)
        return f"team_v{version}.parquet"

    monkeypatch.setattr(submit_mod, "upload_submission", fake_upload)
    monkeypatch.setattr(
        submit_mod, "validate_files", lambda f, t: SimpleNamespace(raise_if_failed=lambda: None, warnings=[])
    )
    f = tmp_path / "sub.parquet"
    pq.write_table(pa.table({SUBMISSION_ID_COL: [1, 2], SUBMISSION_TARGET_COL: [600.0, 700.0]}), f)
    state.file, state.log = f, tmp_path / "submissions.jsonl"
    return state


def _submit(state, version):
    return submit(state.file, state.file, version, "exp", {}, "n", log=state.log, cfg=CFG)


def _log_lines(state):
    return [json.loads(x) for x in state.log.read_text().splitlines()] if state.log.exists() else []


def test_collision_with_an_object_in_the_bucket_is_refused(env):
    env.bucket = FakeBucket(versions=[41, 42])
    env.lb = {41}  # v42 uploaded by another session but not yet scored
    with pytest.raises(RemoteVersionError, match="already exists"):
        _submit(env, 42)
    assert env.uploads == []
    assert _log_lines(env) == [], "a refused version must not leave a write-ahead 'attempted' entry"


def test_a_version_already_on_the_leaderboard_is_refused(env):
    env.bucket = FakeBucket(versions=[])  # e.g. the object was moved, the score remains
    env.lb = {12}
    with pytest.raises(RemoteVersionError, match="already on the leaderboard"):
        _submit(env, 12)
    assert env.uploads == []


def test_a_version_below_the_remote_max_is_refused(env):
    env.bucket = FakeBucket(versions=[8, 10])
    env.lb = {8, 10}
    with pytest.raises(RemoteVersionError, match="not above the highest remote version v10"):
        _submit(env, 9)  # free everywhere, but lower than v10
    assert env.uploads == []


def test_bucket_max_counts_even_when_the_leaderboard_lags(env):
    env.bucket = FakeBucket(versions=[44, 45])
    env.lb = {44}
    res = remote_version_check(45, CFG)
    assert not res["allowed"] and res["next_free"] == 46


def test_unreachable_bucket_is_refused(env, monkeypatch):
    def boom(cfg):
        raise ConnectionError("503 from the endpoint")

    monkeypatch.setattr(submit_mod, "list_team_submissions", boom)
    env.lb = {1}
    with pytest.raises(RemoteVersionError, match="team bucket unreachable"):
        _submit(env, 2)
    assert env.uploads == [] and _log_lines(env) == []


def test_unreachable_leaderboard_is_refused(env, monkeypatch):
    def boom(team):
        raise TimeoutError("api down")

    monkeypatch.setattr(submit_mod, "team_versions", boom)
    env.bucket = FakeBucket(versions=[1])
    with pytest.raises(RemoteVersionError, match="leaderboard API unreachable"):
        _submit(env, 2)
    assert env.uploads == []
    assert remote_version_check(2, CFG)["next_free"] is None


def test_the_normal_path_uploads_the_next_free_version(env):
    env.bucket = FakeBucket(versions=[1, 2, 3])
    env.lb = {1, 2, 3}
    entry = _submit(env, 4)
    assert env.uploads == [4] and entry["status"] == "uploaded"
    assert [r["status"] for r in _log_lines(env)] == ["attempted", "uploaded"]


def test_local_ledger_check_is_kept(env):
    env.bucket = FakeBucket(versions=[1])
    env.lb = {1}
    env.log.write_text(json.dumps({"version": 2, "uploaded_at": "2020-01-01T00:00:00+00:00"}) + "\n")
    with pytest.raises(ValueError, match="already used"):
        _submit(env, 2)
    assert env.uploads == []


def test_dry_run_stays_offline(env, monkeypatch):
    monkeypatch.setattr(submit_mod, "team_versions", lambda team: pytest.fail("dry run must not call the API"))
    submit(env.file, env.file, 1, "exp", {}, "n", log=env.log, cfg=CFG, dry_run=True)
    assert env.uploads == []


# ---- the HEAD guard inside upload_submission itself (ported from claude/quirky-hopper-b7q73j) ----


class FakeMinio:
    def __init__(self, stat):
        self._stat, self.put = stat, []

    def stat_object(self, bucket, name):
        if isinstance(self._stat, Exception):
            raise self._stat
        return self._stat

    def fput_object(self, bucket, name, path):
        self.put.append((bucket, name))


@pytest.mark.parametrize(
    ("stat", "outcome"),
    [
        (SimpleNamespace(size=5, last_modified="t"), FileExistsError),
        (_s3_error("NoSuchKey"), None),
        (_s3_error("AccessDenied"), S3Error),
        (ConnectionError("unreachable"), ConnectionError),
    ],
)
def test_upload_submission_refuses_existing_or_unknown(monkeypatch, tmp_path, stat, outcome):
    fake = FakeMinio(stat)
    monkeypatch.setattr(io_minio, "client", lambda cfg=None: fake)
    if outcome is None:
        assert io_minio.upload_submission(tmp_path / "f", 7, CFG) == "team_v7.parquet"
        assert fake.put == [("tb", "team_v7.parquet")]
    else:
        with pytest.raises(outcome):
            io_minio.upload_submission(tmp_path / "f", 7, CFG)
        assert fake.put == []


# ---- leaderboard reader ----


def test_leaderboard_follows_cursor_and_filters_other_teams():
    pages = {
        None: {"items": [{"teamName": "team", "filename": "team_v1.parquet"}], "nextCursor": "c1"},
        "c1": {
            "items": [
                {"teamName": "team", "filename": "team_v3.parquet"},
                {"teamName": "other", "filename": "other_v9.parquet"},
            ],
            "nextCursor": None,
        },
    }
    assert leaderboard.team_versions("team", get_json=lambda url, f: pages[f.get("cursor")]) == {1, 3}


def test_leaderboard_bad_shape_or_stuck_cursor_raises():
    with pytest.raises(TypeError):
        leaderboard.team_versions("team", get_json=lambda url, f: {"detail": "error"})
    with pytest.raises(RuntimeError, match="did not advance"):
        leaderboard.team_versions("team", get_json=lambda url, f: {"items": [], "nextCursor": "same"})


def test_leaderboard_http_error_raises(monkeypatch):
    class Pool:
        def request(self, *a, **k):
            return SimpleNamespace(status=503, data=b"")

    monkeypatch.setattr(leaderboard, "http_client", lambda timeout=30: Pool())
    with pytest.raises(RuntimeError, match="HTTP 503"):
        leaderboard.team_versions("team")


# ---- dry-run CLI ----


def test_check_version_cli_prints_next_free(env, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(submit_mod.OsnConfig, "from_env", classmethod(lambda cls: CFG))
    env.bucket = FakeBucket(versions=[1, 2])
    env.lb = {1, 2}
    assert main(["--check-version", "2", "--log", str(tmp_path / "none.jsonl")]) == 1
    out = capsys.readouterr().out
    assert "next free version: 3" in out and "REFUSED" in out and "team_v2.parquet_result.json" in out
    assert main(["--check-version", "3", "--log", str(tmp_path / "none.jsonl")]) == 0
    assert env.uploads == []
