"""The two failure modes that cost a session's first hour: minio bypassing the sandbox proxy, and an OpenSky
API client pasted where an S3 access key belongs."""

import urllib3

from prc.doctor import is_oauth_api_client, looks_like_2024_dataset
from prc.io_minio import http_client

LEGACY = ["2022-01-01.parquet", "2022-01-02.parquet", "challenge_set.csv", "final_submission_set.csv"]
OFFICIAL_2026 = [
    "training_2025-01-01_2025-02-01.parquet",
    "training_2025-02-01_2025-03-01.parquet",
    "ranking.parquet",
    "submitting.parquet",
]


def _clear_proxy(monkeypatch):
    for n in ("HTTPS_PROXY", "https_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(n, raising=False)


def test_http_client_uses_the_proxy_when_one_is_configured(monkeypatch):
    _clear_proxy(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    assert isinstance(http_client(), urllib3.ProxyManager)


def test_http_client_is_direct_without_a_proxy(monkeypatch):
    _clear_proxy(monkeypatch)
    pool = http_client()
    assert isinstance(pool, urllib3.PoolManager) and not isinstance(pool, urllib3.ProxyManager)


def test_http_client_prefers_the_environment_ca_bundle(monkeypatch, tmp_path):
    _clear_proxy(monkeypatch)
    ca = tmp_path / "ca.crt"
    ca.write_text("")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    assert http_client().connection_pool_kw["ca_certs"] == str(ca)


def test_legacy_dataset_is_recognised():
    assert looks_like_2024_dataset(LEGACY)
    assert looks_like_2024_dataset([f"sub/{n}" for n in LEGACY])  # a prefix must not hide it


def test_the_2026_files_are_not_mistaken_for_the_legacy_dataset():
    assert not looks_like_2024_dataset(OFFICIAL_2026)
    assert not looks_like_2024_dataset(OFFICIAL_2026 + LEGACY)  # any official file settles it
    assert not looks_like_2024_dataset([])


def test_oauth_probe_reads_the_token_endpoint_response(monkeypatch):
    import requests

    class Resp:
        def __init__(self, code, text):
            self.status_code, self.text = code, text

    def ok(url, data, timeout):
        assert data["grant_type"] == "client_credentials"
        return Resp(200, '{"access_token": "x"}')

    monkeypatch.setattr(requests, "post", ok)
    assert is_oauth_api_client("id", "secret")

    monkeypatch.setattr(requests, "post", lambda url, data, timeout: Resp(401, "unauthorized"))
    assert not is_oauth_api_client("id", "secret")


def test_oauth_probe_stays_quiet_when_the_network_is_down(monkeypatch):
    import requests

    def boom(url, data, timeout):
        raise requests.RequestException("no route")

    monkeypatch.setattr(requests, "post", boom)
    assert not is_oauth_api_client("id", "secret")  # a diagnosis must never mask the original error


def test_the_competition_bucket_has_no_guessed_default(monkeypatch):
    """ "competition-data" is the 2024 edition (306 GB), so download_all must ask rather than guess."""
    import pytest

    from prc.config import OsnConfig
    from prc.io_minio import download_all

    monkeypatch.delenv("OSN_COMPETITION_BUCKET", raising=False)
    monkeypatch.setenv("OSN_ACCESS_KEY", "AKID")
    monkeypatch.setenv("OSN_SECRET_KEY", "SECRET")
    assert OsnConfig.from_env().competition_bucket == ""
    with pytest.raises(RuntimeError, match="OSN_COMPETITION_BUCKET"):
        download_all()
