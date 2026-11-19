"""めぐ分析の「好走ルール（通常）」（src/research/race_analysis/course_rules.py）と、重賞の比べる相手（src/research/graded/pop_base.py）。"""

import json

from src.research.graded import pop_base as PB
from src.research.race_analysis import course_rules as CR
from src.research.race_analysis.base import RaceContext


def _ctx(**kw):
    base = dict(race_id="202605050601", date="2026-05-10", venue="東京", surface="芝", distance=1600, grade="1勝", race_name="3歳以上1勝クラス")
    return RaceContext(**(base | kw))


def test_unit_of_and_grade_from():
    assert CR.unit_of(_ctx(), "1勝") == "東京芝1600|1勝・2勝"
    assert CR.unit_of(_ctx(grade="新馬"), "新馬") == "東京芝1600|新馬"
    assert CR.unit_of(_ctx(venue="京都", distance=1800, surface="芝"), "未勝利") == "京都芝1800外|新馬・未勝利"
    assert CR.unit_of(_ctx(venue="京都", distance=1600), "2勝") == "京都芝1600内外不明|1勝・2勝"         # 内外が決まらないコース（frame と同じ決まり）
    assert CR.unit_of(_ctx(venue="京都", distance=1600), "OP", "") == "京都芝1600外|3勝・OP"           # 出馬表の空文字は「外」（学習の表と同じ）
    assert CR.unit_of(_ctx(venue="京都", distance=1600), "OP", "内") == "京都芝1600内|3勝・OP"
    assert CR.unit_of(_ctx(venue="中山", surface="ダート", distance=1200), "3勝") == "中山ダ1200|3勝・OP"
    assert CR.unit_of(_ctx(), "G3") is None                                                          # 重賞は単位の外
    assert CR.grade_from("", "サラ系３歳以上 ２勝クラス") == "2勝" and CR.grade_from("", "サラ系３歳以上 オープン") == "OP"
    assert CR.grade_from("未勝利", "") == "未勝利" and CR.grade_from("", "") is None


def _stats(sha):
    st = {"n": 10, "rec": [3, 2, 1, 4], "place": 0.6, "roi_t": 1.1, "roi_f": 0.9, "base_same_pop": {"place": 0.5, "roi_t": 0.8, "roi_f": 0.8}}
    unit = "東京芝1600|1勝・2勝"
    vals = {p: {unit: {"unit": st, "buy": [st], "kill": [st]}} for p in ("train", "2025")}
    return {"sha256_cells": sha, "data_until": "2026-11-15", "cycle": 17,
            "rules": {unit: {"layer": "主", "n_buy": 3, "n_kill": 2, "buy": [{"text": "「前走の着順」が 1着", "hyp": "能力"}],
                             "kill": [{"text": "「前走の着順」が 10着以下", "hyp": "能力"}]}},
            "values": vals | {"2026": {}}}


def test_module_reads_frozen_stats(tmp_path, monkeypatch):
    sha = "abc"
    (tmp_path / "s.json").write_text(json.dumps(_stats(sha)), encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps({"sha256_cells": sha}), encoding="utf-8")
    monkeypatch.setattr(CR, "STATS", tmp_path / "s.json")
    monkeypatch.setattr(CR, "PREREG3", tmp_path / "p.json")
    CR._load.cache_clear()
    m = CR.CourseRulesModule()
    assert m.applicable(_ctx())
    assert not m.applicable(_ctx(distance=2000))                                                     # 条件の無い単位
    assert not m.applicable(_ctx(grade="G3"))                                                        # 重賞は除く
    assert not m.applicable(_ctx(race_class="障害3歳以上未勝利"))                                     # 障害は除く
    out = m.compute(_ctx(track_condition="重"))
    f = out.figure
    assert f["unit"] == "東京芝1600|1勝・2勝" and not f["track_ok"]                                   # 良でなければ「当てはまらない」と出す
    assert f["rules"]["buy"][0]["stats"]["train"]["rec"] == [3, 2, 1, 4] and f["rules"]["buy"][0]["stats"]["2026"] is None
    assert any("保証しません" in n for n in out.notes)
    assert m.compute(_ctx(date="2026-05-10")).figure["includes_after"]                               # 結果の最終日より前のレースは、後の結果が入ると印を付ける
    assert not m.compute(_ctx(date="2026-11-22")).figure["includes_after"]
    (tmp_path / "p.json").write_text(json.dumps({"sha256_cells": "zzz"}), encoding="utf-8")         # 凍結と違う成績は使わない
    CR._load.cache_clear()
    assert not m.applicable(_ctx())


def test_same_pop_weights_by_picks():
    table = {"1": {"n": 10, "rec": [5, 2, 1, 2], "npay": 10, "tan": 15.0, "fuku": 9.0},
             "5": {"n": 10, "rec": [1, 1, 1, 7], "npay": 10, "tan": 8.0, "fuku": 6.0}}
    r = PB.same_pop(table, [1, 1, 5, None, 0])
    assert abs(r["place"] - (0.8 * 2 + 0.3) / 3) < 1e-9 and abs(r["roi_t"] - (1.5 * 2 + 0.8) / 3) < 1e-9 and r["n_picks"] == 3
    assert PB.same_pop(table, []) is None
