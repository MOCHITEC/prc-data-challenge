import pytest

from prc.config import OsnConfig


def test_access_key_aliases(monkeypatch):
    for name in ("OSN_ACCESS_KEY", "OSN_ID", "OSN_ACCESS_KEY_ID", "OSN_KEY_ID"):
        for v in ("OSN_ACCESS_KEY", "OSN_ID", "OSN_ACCESS_KEY_ID", "OSN_KEY_ID", "OSN_SECRET_KEY"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv(name, "  AKID  ")
        monkeypatch.setenv("OSN_SECRET_KEY", "SECRET")
        cfg = OsnConfig.from_env()
        assert cfg.access_key == "AKID"  # whitespace from a pasted value is stripped
        assert cfg.secret_key == "SECRET"


def test_secret_aliases_and_endpoint_normalisation(monkeypatch):
    monkeypatch.delenv("OSN_SECRET_KEY", raising=False)
    monkeypatch.setenv("OSN_ID", "AKID")
    monkeypatch.setenv("OSN_SECRET", "S3CRET")
    monkeypatch.setenv("OSN_S3_ENDPOINT", "https://s3.opensky-network.org/")
    cfg = OsnConfig.from_env()
    assert cfg.secret_key == "S3CRET"
    assert cfg.endpoint == "s3.opensky-network.org"  # minio wants a bare host


def test_missing_credentials_name_every_accepted_variable(monkeypatch):
    for v in (
        "OSN_ACCESS_KEY",
        "OSN_ID",
        "OSN_ACCESS_KEY_ID",
        "OSN_KEY_ID",
        "OSN_SECRET_KEY",
        "OSN_SECRET",
        "OSN_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError) as e:
        OsnConfig.from_env()
    assert "OSN_ID" in str(e.value) and "OSN_SECRET" in str(e.value)
