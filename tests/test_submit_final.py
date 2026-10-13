"""`python -m prc.submit --final`: every bucket call is mocked; nothing here can reach a real bucket."""

import json
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from minio.error import S3Error
from test_validate_final import make_final_set

import prc.submit as submit_mod
from prc import io_minio
from prc.config import SUBMISSION_TARGET_COL, OsnConfig
from prc.submit import FinalRefused, FinalVerifyError, _today_count, main, submit_final
from prc.validate_final import file_md5

CFG = OsnConfig(
    endpoint="s3.example", access_key="AK", secret_key="SK", competition_bucket="", team_bucket="tb", team_name="team"
)
EXISTING = SimpleNamespace(size=2_000_000, last_modified="2026-10-09 16:02:00+00:00", etag="abc")


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A valid synthetic final set, a mocked team bucket and a mocked git source."""
    paths, *_ = make_final_set(tmp_path)
    state = SimpleNamespace(paths=paths, existing=None, uploads=[], downloads=[], stats=0, log=tmp_path / "s.jsonl")
    state.read_back = lambda local, dest: shutil.copyfile(local, dest)  # a faithful bucket

    def fake_stat(name, cfg, c=None):
        assert name == "team_final.parquet"
        state.stats += 1
        if isinstance(state.existing, Exception):
            raise state.existing
        if state.uploads:  # after the PUT the bucket holds our file
            return SimpleNamespace(size=paths["sub"].stat().st_size, last_modified="now", etag="new")
        return state.existing

    def fake_upload(local, cfg, expected_existing=None, c=None):
        state.uploads.append((str(local), expected_existing))
        return "team_final.parquet"

    def fake_download(name, dest, cfg, c=None):
        state.downloads.append(name)
        state.read_back(paths["sub"], dest)
        return dest

    monkeypatch.setattr(submit_mod, "stat_submission", fake_stat)
    monkeypatch.setattr(submit_mod, "upload_final", fake_upload)
    monkeypatch.setattr(submit_mod, "download_team_object", fake_download)
    monkeypatch.setattr(submit_mod, "source_blob_md5", lambda commit, path: file_md5(paths["sub"]))
    # the _vNN machinery must not be touched in final mode
    monkeypatch.setattr(submit_mod, "team_versions", lambda team: pytest.fail("final mode read the leaderboard"))
    monkeypatch.setattr(submit_mod, "list_team_submissions", lambda cfg: pytest.fail("final mode listed _vNN"))
    monkeypatch.setattr(submit_mod, "upload_submission", lambda *a, **k: pytest.fail("final mode used the _vNN PUT"))
    monkeypatch.setattr("prc.validate_final.EXPECTED_FINAL_ROWS", None)
    return state


def _final(state, file=None, **kw):
    kw.setdefault("cfg", CFG)
    return submit_final(
        file or state.paths["sub"],
        state.paths["template"],
        state.paths["ranking"],
        "abc1234",
        "handoff/final.parquet",
        log=state.log,
        **kw,
    )


def _ledger(state):
    return [json.loads(x) for x in state.log.read_text().splitlines()] if state.log.exists() else []


@pytest.mark.parametrize("dry_run", [False, True])
def test_existing_object_is_refused_without_the_confirm_flag(env, capsys, dry_run):
    env.existing = EXISTING
    with pytest.raises(FinalRefused, match="--confirm-final-overwrite"):
        _final(env, dry_run=dry_run)
    out = capsys.readouterr().out
    assert "2,000,000 bytes" in out and "2026-10-09 16:02:00" in out, "size and mtime are printed before refusing"
    assert env.uploads == [] and env.downloads == [] and _ledger(env) == []


def test_failed_validation_is_refused_before_any_bucket_call(env, tmp_path):
    bad = tmp_path / "bad.parquet"
    t = pq.read_table(env.paths["sub"])
    vals = t[SUBMISSION_TARGET_COL].to_pylist()
    vals[0] = -1
    pq.write_table(t.set_column(1, SUBMISSION_TARGET_COL, pa.array(vals, pa.int32())), bad)
    with pytest.raises(FinalRefused, match="final gate failed"):
        _final(env, file=bad, confirm_overwrite=True)
    assert env.stats == 0 and env.uploads == [] and _ledger(env) == []


def test_a_file_that_is_not_the_source_blob_is_refused(env, monkeypatch):
    monkeypatch.setattr(submit_mod, "source_blob_md5", lambda commit, path: "0" * 32)
    with pytest.raises(FinalRefused, match="is not abc1234:handoff/final.parquet"):
        _final(env, confirm_overwrite=True)
    assert env.stats == 0 and env.uploads == []


def test_unreadable_bucket_is_refused(env):
    env.existing = S3Error(None, "AccessDenied", "msg", "/tb/x", "rid", "hid")
    with pytest.raises(FinalRefused, match="cannot HEAD"):
        _final(env, confirm_overwrite=True)
    assert env.uploads == [] and _ledger(env) == []


def test_dry_run_does_everything_but_the_upload(env, capsys):
    env.existing = EXISTING
    entry = _final(env, dry_run=True, confirm_overwrite=True, note="dry")
    assert env.uploads == [] and env.downloads == []
    assert env.stats == 1, "the dry run HEADs the target like the real run"
    assert "DRY RUN" in capsys.readouterr().out
    (rec,) = _ledger(env)
    assert rec == json.loads(json.dumps(entry, default=str))
    assert rec["status"] == "dry_run" and rec["dry_run"] is True and rec["kind"] == "final"
    assert rec["md5"] == file_md5(env.paths["sub"])
    assert (rec["source_commit"], rec["source_path"]) == ("abc1234", "handoff/final.parquet")
    assert rec["existing_object"]["size"] == 2_000_000 and rec["validation"]["ok"] is True


def test_upload_reads_back_and_verifies_md5(env):
    env.existing = EXISTING
    entry = _final(env, confirm_overwrite=True)
    assert len(env.uploads) == 1 and env.uploads[0][1] is EXISTING, "the PUT is pinned to the object shown"
    assert env.downloads == ["team_final.parquet"]
    assert entry["status"] == "uploaded" and entry["verified"] and entry["remote_md5"] == entry["md5"]
    assert [r["status"] for r in _ledger(env)] == ["attempted", "uploaded"]


def test_md5_mismatch_after_upload_raises_and_is_logged(env):
    env.existing = None  # an absent object needs no confirmation

    def corrupt(local, dest):
        dest.write_bytes(local.read_bytes()[:-1] + b"X")

    env.read_back = corrupt
    with pytest.raises(FinalVerifyError, match="reads back md5"):
        _final(env)
    log = _ledger(env)
    assert [r["status"] for r in log] == ["attempted", "verify_failed"]
    assert log[1]["remote_md5"] != log[1]["md5"]


def test_failed_put_is_logged_after_the_write_ahead_entry(env, monkeypatch):
    def boom(local, cfg, expected_existing=None, c=None):
        raise ConnectionError("reset")

    monkeypatch.setattr(submit_mod, "upload_final", boom)
    with pytest.raises(ConnectionError):
        _final(env)
    assert [r["status"] for r in _ledger(env)] == ["attempted", "failed"]


def test_final_entries_do_not_break_the_daily_counter(env):
    env.existing = None
    _final(env)
    _final(env, dry_run=True, confirm_overwrite=True)  # the bucket now holds our own upload
    assert len(_ledger(env)) == 3
    assert _today_count(env.log) == (0, set())


def test_cli_final_dry_run_and_refusal(env, monkeypatch, capsys):
    monkeypatch.setattr(submit_mod.OsnConfig, "from_env", classmethod(lambda cls: CFG))
    env.existing = EXISTING
    p = env.paths
    base = ["--final", "--file", str(p["sub"]), "--template", str(p["template"]), "--ranking", str(p["ranking"])]
    base += ["--source-commit", "abc1234", "--source-path", "handoff/final.parquet", "--log", str(env.log)]
    assert main(base + ["--dry-run"]) == 1  # exists, not confirmed
    assert "REFUSED" in capsys.readouterr().out
    assert main(base + ["--dry-run", "--confirm-final-overwrite"]) == 0
    assert "status: dry_run" in capsys.readouterr().out
    assert env.uploads == []
    with pytest.raises(SystemExit):
        main(["--final", "--file", str(p["sub"]), "--dry-run"])  # template, ranking and source missing


# ---- the re-HEAD guard inside io_minio.upload_final ----


class FakeMinio:
    def __init__(self, stat):
        self._stat, self.put = stat, []

    def stat_object(self, bucket, name):
        if isinstance(self._stat, Exception):
            raise self._stat
        return self._stat

    def fput_object(self, bucket, name, path):
        self.put.append((bucket, name))


def _absent():
    return S3Error(None, "NoSuchKey", "msg", "/tb/x", "rid", "hid")


@pytest.mark.parametrize(
    ("now", "shown", "ok"),
    [
        (EXISTING, EXISTING, True),  # the object the user confirmed
        (_absent(), None, True),  # absent, and shown as absent
        (SimpleNamespace(size=5, last_modified="later", etag="x"), EXISTING, False),  # replaced in between
        (EXISTING, None, False),  # appeared in between
        (_absent(), EXISTING, False),  # vanished in between
        (S3Error(None, "AccessDenied", "m", "/", "r", "h"), EXISTING, None),  # unreadable: raise
    ],
)
def test_upload_final_puts_only_over_the_object_that_was_shown(tmp_path, now, shown, ok):
    fake = FakeMinio(now)
    if ok:
        assert io_minio.upload_final(tmp_path / "f", CFG, expected_existing=shown, c=fake) == "team_final.parquet"
        assert fake.put == [("tb", "team_final.parquet")]
    else:
        with pytest.raises(FileExistsError if ok is False else S3Error):
            io_minio.upload_final(tmp_path / "f", CFG, expected_existing=shown, c=fake)
        assert fake.put == []
