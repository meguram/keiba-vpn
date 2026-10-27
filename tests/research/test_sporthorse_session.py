"""sporthorse_session.py のテスト（実アクセスなし。probe/fetch_html は他テストで monkeypatch 済み）。"""
import pytest

from src.research.pedigree import sporthorse_session as ss


def test_cookie_header_joins_name_value_pairs():
    st = ss.SessionState(user_agent="ua", cookies={"a": "1", "b": "2"})
    assert st.cookie_header() == "a=1; b=2"


def test_cookie_header_skips_empty_values():
    st = ss.SessionState(user_agent="ua", cookies={"a": "1", "b": ""})
    assert st.cookie_header() == "a=1"


def test_save_and_load_session_roundtrip(tmp_path):
    p = tmp_path / "session.json"
    st = ss.SessionState(user_agent="ua-x", cookies={"cf_clearance": "abc"})
    ss.save_session(st, p)
    loaded = ss.load_session(p)
    assert loaded.user_agent == "ua-x"
    assert loaded.cookies == {"cf_clearance": "abc"}
    assert loaded.saved_at  # 自動付与される


def test_load_session_missing_file_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="ブラウザで"):
        ss.load_session(tmp_path / "missing.json")


def test_normalize_cookie_import_from_dict_cookies():
    data = {"cookies": {"a": "1", "b": 2}}
    assert ss.normalize_cookie_import(data) == {"a": "1", "b": "2"}


def test_normalize_cookie_import_from_cookie_object_list_filters_other_domains():
    data = [
        {"domain": ".sporthorse-data.com", "name": "cf_clearance", "value": "xyz"},
        {"domain": ".example.com", "name": "unrelated", "value": "nope"},
        {"domain": "", "name": "no_domain", "value": "keep"},
    ]
    assert ss.normalize_cookie_import(data) == {"cf_clearance": "xyz", "no_domain": "keep"}


def test_normalize_cookie_import_rejects_unreadable_payload():
    with pytest.raises(ValueError):
        ss.normalize_cookie_import("not a dict or list")


def test_import_session_rejects_empty_cookies(tmp_path):
    with pytest.raises(ValueError, match="クッキーが 1 件もありません"):
        ss.import_session([{"domain": ".example.com", "name": "x", "value": "y"}], path=tmp_path / "s.json")


def test_import_session_saves_and_returns_path(tmp_path):
    p = tmp_path / "s.json"
    payload = [{"domain": ".sporthorse-data.com", "name": "cf_clearance", "value": "zzz"}]
    out = ss.import_session(payload, user_agent="custom-ua", path=p)
    assert out == p
    st = ss.load_session(p)
    assert st.cookies == {"cf_clearance": "zzz"}
    assert st.user_agent == "custom-ua"


def test_is_challenge_detects_cf_mitigated_header():
    assert ss.is_challenge("<html>普通のページ</html>", {"cf-mitigated": "challenge"})


def test_is_challenge_detects_challenge_markup():
    assert ss.is_challenge("Just a moment... checking your browser", {})


def test_is_challenge_false_for_normal_page():
    assert not ss.is_challenge("<title>Sunday Silence</title>Date of Birth: 25-Mar-1986", {})


def test_session_path_prefers_explicit_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPORTHORSE_SESSION", str(tmp_path / "from_env.json"))
    explicit = tmp_path / "explicit.json"
    assert ss.session_path(explicit) == explicit


def test_session_path_uses_env_when_no_explicit(monkeypatch, tmp_path):
    env_path = tmp_path / "from_env.json"
    monkeypatch.setenv("SPORTHORSE_SESSION", str(env_path))
    assert ss.session_path(None) == env_path


def test_session_path_defaults_when_no_env_or_explicit(monkeypatch):
    monkeypatch.delenv("SPORTHORSE_SESSION", raising=False)
    assert ss.session_path(None) == ss.DEFAULT_SESSION_PATH
