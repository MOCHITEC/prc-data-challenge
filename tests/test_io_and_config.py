"""The failure mode that cost a session's first hour: minio bypassing the sandbox proxy."""

import urllib3

from prc.io_minio import http_client


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
