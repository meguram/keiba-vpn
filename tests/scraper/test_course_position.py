"""芝のコース区分（A〜D）・開催回・日目の取得（course_position）: 解析、応答の検証、group の使い回し、再開、停止、保存。"""

import json
import re
from pathlib import Path

import pytest

from src.scraper import course_position as cp

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "course_position" / "race_list_sub_20261004.html"


def _sub(ds, blocks=(("東京", 4, 2, "A", "良", "良"), ("京都", 4, 2, "A", "稍", "稍"))):
    """日付 ds の race_list_sub の最小の断片（払戻一覧のリンクつき）。"""
    out = ['<div id="RaceTopRace">']
    for v, kai, day, letter, tc, dc in blocks:
        lt = f"({letter})" if letter else ""
        out.append(f'<dl class="RaceList_DataList"><dt class="RaceList_DataHeader"><div><p class="RaceList_DataTitle"><small>{kai}回</small> {v} <small>{day}日目</small></p>'
                   f'<a href="../top/payback_list.html?kaisai_id={ds[:4]}{cp.JRA_VENUES.get(v, "99")}{kai:02d}{day:02d}&kaisai_date={ds}&rf=race_list">払戻一覧</a></div>'
                   f'<div class="RaceList_DataDesc"><span class="Shiba">芝{lt}：{tc}</span><span class="Da">ダ：{dc}</span></div></dt></dl>')
    return "".join(out) + "</div>"


def test_parse_sub_reads_the_fixture():
    recs, info = cp.parse_sub(FIX.read_text(encoding="utf-8"))
    assert recs == [
        {"venue": "東京", "kai": 4, "day": 2, "turf_course": "A", "turf_condition": "良", "dirt_condition": "良"},
        {"venue": "京都", "kai": 4, "day": 2, "turf_course": "A", "turf_condition": "稍", "dirt_condition": "稍"}]
    assert info["blocks"] == 2 and info["unparsed"] == 0 and cp.verify("20261004", recs, info) is None


def test_parse_sub_handles_fullwidth_parentheses_no_letter_and_foreign_venues():
    html = ('<dl class="RaceList_DataList"><dt><p class="RaceList_DataTitle"><small>1回</small> 札幌 <small>10日目</small></p><span class="Shiba">芝（B）：重</span><span class="Da">ダ：不</span></dt></dl>'
            '<dl class="RaceList_DataList"><dt><p class="RaceList_DataTitle"><small>2回</small> 函館 <small>3日目</small></p><span class="Shiba">芝：良</span></dt></dl>'
            '<dl class="RaceList_DataList"><dt><p class="RaceList_DataTitle">ロンシャン <small>1日目</small></p><span class="Shiba">芝：良</span></dt></dl>')
    recs, info = cp.parse_sub(html)
    assert [(r["venue"], r["turf_course"], r["turf_condition"]) for r in recs] == [("札幌", "B", "重"), ("函館", None, "良")]
    assert info["unparsed"] == 0                                                   # 海外の会場は読み飛ばす（読み切れないのではない）
    broken = '<dl class="RaceList_DataList"><dt><p class="RaceList_DataTitle">東京 開催中</p></dt></dl>'
    assert cp.parse_sub(broken)[1]["unparsed"] == 1                                # JRA の会場なのに回・日目が読めない
    assert cp.parse_sub("<div>開催なし</div>") == ([], {"blocks": 0, "unparsed": 0, "links": []})


def test_verify_rejects_empty_unparsed_wrong_day_and_mismatched_kai_day():
    recs, info = cp.parse_sub(_sub("20190309"))
    assert cp.verify("20190309", recs, info) is None
    assert "別の日" in cp.verify("20190310", recs, info)                            # 応答のリンクの日付が違う
    assert cp.verify("20190309", [], info) == "解析結果が空"
    recs2, info2 = cp.parse_sub(_sub("20190309").replace("2日目", "3日目", 1))
    assert "合わない" in cp.verify("20190309", recs2, info2)                         # リンクの回・日目と表示が食い違う
    assert "リンクが無く" in cp.verify("20190309", recs, {**info, "links": []})


class FakeClient:
    def __init__(self, window=None, fail_on=None):
        self.urls, self.window, self.fail_on = [], window or {}, fail_on

    def fetch(self, url, use_cache=True, encoding=None):
        self.urls.append(url)
        ds = re.search(r"kaisai_date=(\d{8})", url).group(1)
        if self.fail_on and self.fail_on(url):
            raise BaseException("アクセス制限")
        if "race_list_get_date_list" in url:
            ds_list = self.window.get(ds, [ds])
            return "<ul>" + "".join(f'<li date="{d}" group="G{ds_list[0]}"></li>' for d in ds_list) + "</ul>"
        return _sub(ds)


def test_collect_reuses_group_skips_done_dates_and_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "STORE", tmp_path)
    days = ["20190302", "20190303", "20190309", "20190310"]
    c = FakeClient(window={d: days for d in days})
    stats = cp.collect(c, days)
    assert stats["fetched"] == 4 and stats["skipped"] == 0 and stats["aborted"] is None
    assert sum("race_list_get_date_list" in u for u in c.urls) == 1                 # 日付一覧は 1 回だけ（4 日分の group を使い回す）
    assert sum("race_list_sub" in u for u in c.urls) == 4
    st = json.loads((tmp_path / "course_position_2019.json").read_text(encoding="utf-8"))
    assert st["dates"]["2019-03-09"][0]["turf_course"] == "A" and len(st["dates"]) == 4
    c2 = FakeClient()
    assert cp.collect(c2, days)["skipped"] == 4 and c2.urls == []                   # 取得済みの日は取り直さない（再開できる）


def test_collect_does_not_store_bad_days_and_stops_after_a_streak(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "STORE", tmp_path)
    class Wrong(FakeClient):
        def fetch(self, url, use_cache=True, encoding=None):
            if "race_list_sub" in url:
                return _sub("20200101")                                               # いつも別の日の応答を返す
            return super().fetch(url, use_cache, encoding)
    days = [f"2019040{i}" for i in range(1, 9)]
    stats = cp.collect(Wrong(window={d: days for d in days}), days)
    assert stats["fetched"] == 0 and stats["bad"] == cp.MAX_BAD_STREAK and "続いた" in stats["aborted"]
    assert cp.load_store(2019)["dates"] == {}                                        # 別の日のデータを、その日として保存しない
    stats2 = cp.collect(FakeClient(window={"20190302": ["20190301"]}), ["20190301", "20190302"])   # group の無い日は保存せず、次へ進む
    assert stats2["fetched"] == 1 and stats2["bad"] == 1


def test_collect_checks_before_each_date_and_respects_max(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "STORE", tmp_path)
    days = ["20190401", "20190402", "20190403"]
    c = FakeClient(window={d: days for d in days})
    n = {"k": 0}

    def check():
        n["k"] += 1
        return "アクセス保留" if n["k"] == 3 else None                                # 3 日目の前で停止の印
    stats = cp.collect(c, days, check=check)
    assert stats["fetched"] == 2 and stats["aborted"] == "アクセス保留"
    assert set(cp.load_store(2019)["dates"]) == {"2019-04-01", "2019-04-02"}         # 止めても取れた分は残る
    monkeypatch.setattr(cp, "STORE", tmp_path / "b")
    assert cp.collect(FakeClient(window={d: days for d in days}), days, max_dates=2)["fetched"] == 2


def test_collect_saves_what_it_has_when_the_guard_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "STORE", tmp_path)
    days = ["20190501", "20190502"]
    c = FakeClient(window={d: days for d in days}, fail_on=lambda u: "20190502" in u and "race_list_sub" in u)
    with pytest.raises(BaseException):
        cp.collect(c, days)                                                           # AccessRestrictionDetected は BaseException。例外が突き抜けても保存する
    assert "2019-05-01" in cp.load_store(2019)["dates"]


def test_race_dates_reads_days_with_races_and_ignores_broken_files(tmp_path):
    (tmp_path / "20190309.json").write_text(json.dumps({"date": "20190309", "races": [{"race_id": "x"}]}), encoding="utf-8")
    (tmp_path / "20190310.json").write_text(json.dumps({"date": "20190310", "races": []}), encoding="utf-8")
    (tmp_path / "20200105.json").write_text(json.dumps({"date": "20200105", "races": [{"race_id": "y"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{壊れた", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1,2]", encoding="utf-8")
    (tmp_path / "noname.json").write_text(json.dumps({"races": [{"race_id": "z"}]}), encoding="utf-8")
    assert cp.race_dates(2019, 2020, tmp_path) == ["20190309", "20200105"]
    assert cp.race_dates(2019, 2020, tmp_path, before="20200105") == ["20190309"]     # 今日より前の日だけ
