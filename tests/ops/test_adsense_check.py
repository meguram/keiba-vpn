"""AdSense の確かめ（src/ops/adsense_check.py）。外への通信はしない。"""
from __future__ import annotations

from src.ops import adsense_check as ac


def _fake(routes):
    def get(url, follow=True):
        return routes.get(url, (404, url, "", []))
    return get


def test_all_ok(monkeypatch, capsys):
    meta = f'<meta name="google-adsense-account" content="ca-{ac.PUB_ID}"/>' + "<p>" + "本文" * 200 + "</p>"
    routes = {
        f"{ac.ROOT}/ads.txt": (200, f"{ac.APP}/ads.txt", ac.ADS_LINE + "\n", [f"301 → {ac.APP}/ads.txt"]),
        f"{ac.APP}/ads.txt": (200, f"{ac.APP}/ads.txt", ac.ADS_LINE + "\n", []),
        f"{ac.APP}/robots.txt": (200, "", "User-agent: Mediapartners-Google\nAllow: /\nUser-agent: Google-Display-Ads-Bot\nAllow: /\nUser-agent: *\nDisallow: /\n", []),
        **{f"{ac.APP}{p}": (200, "", meta, []) for p in ac.PAGES},
    }
    monkeypatch.setattr(ac, "_get", _fake(routes))
    assert ac.main(["--no-api"]) == 0  # stg が届かなくても 0
    out = capsys.readouterr().out
    assert "-- ads.txt stg" in out and "NG" not in out


def test_missing_root_and_meta(monkeypatch):
    routes = {
        f"{ac.ROOT}/ads.txt": (0, "", "URLError", []),
        f"{ac.APP}/ads.txt": (200, "", ac.ADS_LINE, []),
        f"{ac.APP}/robots.txt": (200, "", "User-agent: *\nDisallow: /\n", []),
        f"{ac.APP}/": (200, "", "<html></html>", []),
        f"{ac.APP}/races": (200, "", "<html></html>", []),
    }
    monkeypatch.setattr(ac, "_get", _fake(routes))
    assert ac.main(["--no-api"]) == 1
    assert ac.check_page(ac.APP, "/")[0] is False
    assert ac.check_robots(ac.APP)[0] is False


def test_review_state_unset(monkeypatch):
    for k in ("ADSENSE_OAUTH_FILE", "ADSENSE_OAUTH_CLIENT_ID", "ADSENSE_OAUTH_CLIENT_SECRET", "ADSENSE_OAUTH_REFRESH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert ac.review_state()[0].startswith("審査の状態: 未設定")


def test_review_state_never_prints_secrets(monkeypatch):
    monkeypatch.setenv("ADSENSE_OAUTH_CLIENT_ID", "cid-SECRET1")
    monkeypatch.setenv("ADSENSE_OAUTH_CLIENT_SECRET", "cs-SECRET2")
    monkeypatch.setenv("ADSENSE_OAUTH_REFRESH_TOKEN", "rt-SECRET3")

    class R:
        def __init__(self, body):
            self.body = body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return self.body

    monkeypatch.setattr(ac.urllib.request, "urlopen", lambda req, timeout=0: R(b'{"access_token": "at-SECRET4"}'))
    calls = []

    def api(url, token):
        calls.append(url)
        if url.endswith("/accounts"):
            return {"accounts": [{"name": "accounts/pub-1", "state": "READY"}]}
        return {"sites": [{"domain": "megu-keiba.com", "state": "GETTING_READY"}]}

    monkeypatch.setattr(ac, "_api_json", api)
    lines = ac.review_state()
    text = "\n".join(lines)
    assert "GETTING_READY" in text and "megu-keiba.com" in text
    assert "SECRET" not in text


def test_review_state_bad_file_is_not_unset(monkeypatch, tmp_path):
    for k in ("ADSENSE_OAUTH_CLIENT_ID", "ADSENSE_OAUTH_CLIENT_SECRET", "ADSENSE_OAUTH_REFRESH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ADSENSE_OAUTH_FILE", str(tmp_path / "missing.json"))
    assert "設定の誤り" in ac.review_state()[0]
    bad = tmp_path / "bad.json"
    bad.write_text('{"client_id": "x"}', encoding="utf-8")
    monkeypatch.setenv("ADSENSE_OAUTH_FILE", str(bad))
    assert "設定の誤り" in ac.review_state()[0]


def test_text_chars_ignores_scripts_and_tags():
    html = "<html><head><script>var a='長い文字列';</script><style>p{}</style></head><body><p>本文 です</p></body></html>"
    assert ac.text_chars(html) == 4
