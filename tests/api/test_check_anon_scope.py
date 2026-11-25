"""漏れの検査（src/scripts/ops/check_anon_scope.py）が、出してはいけない欄を拾い、確定の結果のオッズは許すこと。外部接続なし。"""

from __future__ import annotations

from src.scripts.ops import check_anon_scope as m


_MEMBER_ONLY = ("/api/horse/", "/api/data/", "/api/betting/pair-odds/", "/api/odds/")


def _fake(responses, open_paths=()):
    def get(_base, path, _cookie):
        if path.startswith(_MEMBER_ONLY) and path not in open_paths:
            return 401, None
        return (200, responses[path]) if path in responses else (404, None)
    return get


_CLEAN = {
    "/api/prediction-results": {"races": [{"top1": {"mark": "◎"}}]},
    "/api/prediction-results/news": {"items": []},
    "/api/prediction-results/mark-card": {"periods": []},
    "/api/race-list/20261123": {"races": [{"race_id": "r1"}, {"race_id": "r2"}]},
    "/api/race/r1": {"access": {"result_confirmed": True}, "race_shutuba": {"entries": [{"horse_id": "h"}]},
                     "race_result": {"entries": [{"odds": 3.0, "popularity": 1}], "payoff": {}}},
    "/api/race/r1/predictions": {"predictions": [{"horse_number": 1, "mark": "◎"}]},
    "/api/race/r1/entry/h/card": {"horse": {"recent": [{"finish": 1}]}},
    "/api/race/r2": {"access": {"result_confirmed": False}, "race_shutuba": {"entries": []}, "race_result": {"entries": [{"finish_position": 1}]}},
    "/api/race/r2/predictions": {"predictions": []},
}


def test_check_passes_clean_response(monkeypatch):
    monkeypatch.setattr(m, "_get", _fake(_CLEAN))
    assert m.check("http://x", ["20261123"], 10) == []


def test_check_flags_leaks(monkeypatch):
    dirty = {**_CLEAN,
             "/api/race/r2": {"access": {"result_confirmed": False}, "race_result": {"entries": [{"odds": 3.0}]}, "horses": {"h": {"race_history": []}}},
             "/api/race/r1/predictions": {"predictions": [{"win_prob": 0.3}]},
             "/api/race/r1/entry/h/card": {"horse": {"training": {}}}}
    monkeypatch.setattr(m, "_get", _fake(dirty))
    got = m.check("http://x", ["20261123"], 10)
    assert "/api/race/r2: .race_result.entries[].odds" in got
    assert "/api/race/r2: .horses.h.race_history" in got
    assert "/api/race/r1/predictions: .predictions[].win_prob" in got
    assert "/api/race/r1/entry/h/card: .horse.training" in got


def test_check_flags_open_member_only_path_and_text(monkeypatch):
    dirty = {**_CLEAN, "/api/data/race_oikiri/r1": {"x": 1},
             "/api/prediction-results": {"highlights": [{"kind": "big_miss", "text": "◎A（勝率35%）は8着"}]}}
    monkeypatch.setattr(m, "_get", _fake(dirty, open_paths=("/api/data/race_oikiri/r1",)))
    got = m.check("http://x", ["20261123"], 10)
    assert "/api/data/race_oikiri/r1: HTTP 200（401 のはず）" in got
    assert "/api/prediction-results: .highlights[].text（文）" in got
