"""JRA 事実データ（テキスト）の部品: 既定で取得しない・テキスト以外は拒否・表記の揺れに強い解析。合成 HTML のみ（実ページ・実アクセスなし）。"""

import pytest

from src.scraper import jra_facts as J

SHUTUBA = """<table><tr><th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th><th>騎手</th><th>調教師</th><th>馬体重</th></tr>
<tr><td>1</td><td>１</td><td>テストA</td><td>牡4</td><td>56.0</td><td>騎手甲</td><td>調教師乙</td><td>480(+2)</td></tr>
<tr><td>2</td><td>2</td><td>テストB</td><td>牝5</td><td>54.0</td><td>騎手丙</td><td>調教師丁</td><td>452(-4)</td></tr></table>"""
RESULT = """<table><tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>タイム</th><th>着差</th><th>上り3F</th><th>人気</th></tr>
<tr><td>1</td><td>1</td><td>1</td><td>テストA</td><td>2:00.1</td><td></td><td>34.5</td><td>2</td></tr></table>"""


def test_fetch_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(J.ENV_FLAG, raising=False)
    with pytest.raises(J.JraFetchDisabled):
        J.fetch_text("https://www.jra.go.jp/x")


def test_fetch_guards(monkeypatch):
    monkeypatch.setenv(J.ENV_FLAG, "1")
    with pytest.raises(J.JraFetchRefused):
        J.fetch_text("https://example.com/x")                      # JRA 公式以外

    class R:
        def __init__(self, ctype, body=b"<html></html>"):
            self.headers = {"Content-Type": ctype}
            self._b = body

        def read(self, n=-1):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with pytest.raises(J.JraFetchRefused):
        J.fetch_text("https://www.jra.go.jp/a.jpg", opener=lambda req, timeout: R("image/jpeg"), sleep=lambda s: None)   # 画像は取らない
    out = J.fetch_text("https://www.jra.go.jp/a.html", opener=lambda req, timeout: R("text/html; charset=utf-8", "テスト".encode()), sleep=lambda s: None)
    assert out == "テスト"


def test_parse_shutuba_tolerant():
    rows = J.parse_shutuba(SHUTUBA)
    assert [r["horse_number"] for r in rows] == [1, 2] and rows[0]["horse_name"] == "テストA"
    assert rows[0]["weight"] == 480 and rows[0]["weight_change"] == 2 and rows[1]["weight_change"] == -4
    assert rows[0]["jockey_weight"] == 56.0 and rows[0]["trainer_name"] == "調教師乙"


def test_parse_result_and_unreadable_returns_none():
    rows = J.parse_result(RESULT)
    assert rows[0]["finish_position"] == 1 and rows[0]["last_3f"] == 34.5 and rows[0]["popularity"] == 2
    assert J.parse_shutuba("<p>表なし</p>") is None and J.parse_shutuba("<table><tr><th>列A</th></tr><tr><td>1</td></tr></table>") is None


def test_to_storage_doc_is_stamped_jra():
    doc = J.to_storage_doc("202608020311", "shutuba", J.parse_shutuba(SHUTUBA))
    assert doc["_meta"]["source"] == "jra" and doc["race_id"] == "202608020311" and len(doc["entries"]) == 2
