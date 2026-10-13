from src.scraper.client import NetkeibaClient

TABLE = "<table class='SpeedIndex_Table'><tbody>{rows}</tbody></table>"


def _client(monkeypatch, html=None, error=None):
    c = NetkeibaClient(auto_login=False, cache_dir=None)

    def fake_fetch(url, use_cache=True, encoding=None):
        if error:
            raise error
        return html

    monkeypatch.setattr(c, "fetch", fake_fetch)
    return c


def test_full_table_means_login_is_effective(monkeypatch):
    c = _client(monkeypatch, TABLE.format(rows="<tr><td>x</td></tr>" * 11))
    assert c._verify_premium() is True


def test_partial_table_means_login_is_not_effective(monkeypatch):
    c = _client(monkeypatch, TABLE.format(rows="<tr><td>x</td></tr>" * 3))
    assert c._verify_premium() is False


def test_probe_failure_is_unknown_not_blocking(monkeypatch):
    c = _client(monkeypatch, error=RuntimeError("timeout"))
    assert c._verify_premium() is None
    assert c.premium_ok is None
