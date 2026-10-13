"""NetkeibaClient の文字コード判定: 宣言（Content-Type / meta）をホスト別の固定値より優先する。"""

from __future__ import annotations

from types import SimpleNamespace

from src.scraper.client import NetkeibaClient

decl = NetkeibaClient._declared_charset


def _resp(content_type: str, body: bytes):
    return SimpleNamespace(headers={"Content-Type": content_type}, content=body)


def test_header_charset_wins():
    assert decl(_resp("text/html; charset=UTF-8", b"<html>")) == "UTF-8"
    assert decl(_resp("text/html; charset=EUC-JP", b"<html>")) == "EUC-JP"


def test_meta_charset_used_when_header_has_none():
    assert decl(_resp("text/html", b'<html><head><meta charset="utf-8">')).lower() == "utf-8"
    assert decl(_resp("text/html", b'<meta http-equiv="Content-Type" content="text/html; charset=euc-jp">')).lower() == "euc-jp"


def test_unknown_or_missing_charset_falls_back_to_none():
    assert decl(_resp("text/html; charset=x-not-a-codec", b"<html>")) is None
    assert decl(_resp("text/html", b"<html>")) is None


def test_utf8_japanese_decodes_correctly_via_declared_charset():
    body = "<html><head><meta charset=\"utf-8\"></head>3歳未勝利 ワンダーアスク</html>".encode("utf-8")
    r = _resp("text/html; charset=UTF-8", body)
    assert body.decode(decl(r)) .endswith("ワンダーアスク</html>")
