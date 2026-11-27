"""血統予想家（めぐ分析の欄）の規則・凍結・置き場・API の合わせ方の確かめ。データは使わない（手で作った小さな表だけ）。"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.pedigree_persona import columns as pc
from src.research.pedigree_persona import rules as R
from src.research.pedigree_persona import store

ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
USER_BANNED = ["回収率100%超", "もうかる", "勝てる", "期待値プラス", "妙味", "お宝", "狙い目", "穴", "推奨", "買い", "注目"]


# ── 凍結 ──────────────────────────────────────────────

def test_rules_file_matches_frozen_sha():
    """規則を直したら、版を分けて凍結し直す（会議の 1 号の条件 3）。直したまま push すると落ちる。"""
    fz = R.load_frozen()
    assert fz["version"] == R.RULES_VERSION
    assert set(fz["threshold"]) == {"ダート", "芝"} and set(fz["sd"]) == {"ダート", "芝"}
    assert fz["sd"]["ダート"] == 0.00539186                 # 前向きの登録の開発の定数
    assert fz["pop_min"] == R.POP_MIN and fz["top_frac"] == R.TOP_FRAC and fz["max_picks"] == R.MAX_PICKS


def test_research_tools_match_registered_sha():
    """研究の道具（前向きの登録で凍結）は読むだけ。1 行でも変わると止まる。"""
    pc.check_frozen()


def test_load_frozen_stops_when_rules_change(tmp_path):
    fz = json.loads(R.FROZEN_PATH.read_text(encoding="utf-8"))
    p = tmp_path / "frozen.json"
    p.write_text(json.dumps(fz | {"rules_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit):
        R.load_frozen(p)


# ── 規則 ──────────────────────────────────────────────

def test_score_centers_in_race_and_zeroes_no_pedigree():
    rid = np.array(["a", "a", "a", "b", "b"])
    raw = np.array([1.0, 2.0, 3.0, 10.0, 10.0])
    z = R.score(raw, rid, np.array([False, False, True, False, False]), 0.5)
    assert z.tolist() == [-2.0, 0.0, 0.0, 0.0, 0.0]


def test_pop_rank_by_odds_then_horse_number():
    rid = np.array(["a"] * 4 + ["b"] * 2)
    r = R.pop_rank(rid, np.array([5.0, 2.0, 5.0, 30.0, 3.0, 1.5]), np.array([1, 2, 3, 4, 1, 2]))
    assert r.tolist() == [2, 1, 3, 4, 2, 1]


def test_pick_only_from_6th_favourite_and_at_most_two():
    z = np.array([5.0, 0.1, 2.0, 3.0, 1.3, 0.5, 4.0])
    rank = np.array([1, 2, 6, 7, 8, 9, 5])
    assert R.pick(z, rank, 1.2).tolist() == [3, 2]          # 1・5 番人気は点が高くても選ばない。3 頭当たれば上位 2 頭
    assert R.pick(z, rank, 9.0).tolist() == []


@pytest.mark.parametrize("name, want", [("ジェイドロバリーJade Robbery(米)", "ジェイドロバリー"), ("Wild Again\xa0(米)", "Wild Again"),
                                        ("Ashkalani(愛)", "Ashkalani"), ("ソヴィエトスター", "ソヴィエトスター"), ("", "")])
def test_clean_name(name, want):
    assert R.clean_name(name) == want


def test_reason_is_fact_form_with_up_to_two_examples():
    s = R.reason("ダート", ["A.P. Indy\xa0(米)", "ミスターシービー", "Gulch"], True, 50)
    assert s == "母の側の 5 代に、ダートのレースで子孫の成績が良い祖先の血が多い（例: A.P. Indy・ミスターシービー）。今回が初ダート・父の産駒の成績がまだ少ない。"
    assert R.reason("芝", ["", ""], True, 500) == "母の側の 5 代に、芝のレースで子孫の成績が良い祖先の血が多い。"    # 芝では初ダートを書かない


def test_facts_are_counts_only():
    assert R.facts("ダート", [True, False, True], [False, True, False], [1, 2, 7], False) == "初ダート 2 頭・休み明け 1 頭・出走 1〜2 走 2 頭"
    assert R.facts("芝", [True], [False], [0], False) == "休み明け 0 頭・出走 1〜2 走 0 頭"
    assert R.facts("ダート", [], [], [], True) == "新馬戦"


def _module(surface="ダート", picks=None, state="ready"):
    return R.module(race_id="202606050101", surface=surface, facts_text="初ダート 1 頭・休み明け 0 頭・出走 1〜2 走 0 頭", state=state,
                    picks=picks or [], asof="2026-12-06", made_at="2026-12-06T09:00:00+09:00", rule_sha="x", inputs={})


def test_banned_words_cover_user_list_and_every_fixed_text():
    """禁止の語（ユーザーの決定）を、規則が作る文の全部の型・紹介文・注記に対して見張る。"""
    assert all(R.banned_in(w) for w in USER_BANNED)                 # ユーザーの決定の語は、どれも見張りに当たる（期待値プラスは「期待値」で当たる）
    texts = [R.INTRO, R.NOTE_UNVERIFIED, R.NOTE_TURF, R.TITLE, R.LABEL, *R.NOTES]
    for surf in ("ダート", "芝"):
        for fd in (True, False):
            for runs in (0, 1000):
                texts.append(R.reason(surf, ["○", "○"], fd, runs))
                texts.append(R.reason(surf, [], fd, runs))
        texts.append(R.facts(surf, [True], [True], [1], False))
    texts.append(R.facts("ダート", [], [], [], True))
    texts.append(json.dumps(_module("芝"), ensure_ascii=False))
    for t in texts:
        assert R.banned_in(t) == [], t
    assert "回収率は確かめていない。成績は 2028 年ごろまで出さない" in _module()["figure"]["note"]


def test_frontend_banned_list_matches_python():
    import re

    src = (ROOT / "frontend/lib/pedigreePersona.test.mjs").read_text(encoding="utf-8")
    body = src[src.index("const banned = [") : src.index("];", src.index("const banned = ["))]
    assert set(re.findall(r'"([^"]+)"', body)) == set(R.BANNED)


def test_module_refuses_banned_text(monkeypatch):
    monkeypatch.setattr(R, "INTRO", "この馬は狙い目")
    with pytest.raises(ValueError):
        _module()


def test_turf_is_marked_weak_and_unverified():
    m = _module("芝")
    assert m["figure"]["confidence"] == "弱め" and R.NOTE_TURF in m["notes"]
    assert _module("ダート")["figure"]["confidence"] is None


def test_module_has_no_numbers_of_score_odds_or_results():
    """点・閾値・人気・成績は欄に入れない（数字の誇張を避け、着順と結ばない）。"""
    fig = _module(picks=[{"horse_number": 3, "horse_name": "テスト", "reason": "r"}])["figure"]
    assert set(fig["picks"][0]) == {"horse_number", "horse_name", "reason"}
    for k in ("score", "z", "threshold", "odds", "popularity", "win_rate", "roi", "hit"):
        assert k not in json.dumps(fig)


# ── T-60 の段 ─────────────────────────────────────────

def _raw(n=10):
    return pd.DataFrame({
        "race_id": ["202606050101"] * n, "date": ["2026-12-06"] * n, "venue": ["中山"] * n, "surface": ["ダート"] * n, "distance": [1200] * n,
        "horse_number": list(range(1, n + 1)), "horse_id": [f"20220{i:05d}" for i in range(n)], "horse_name": [f"馬{i}" for i in range(1, n + 1)],
        "start_time": ["10:00"] * n, "FD": [i == 8 for i in range(n)], "layoff_y": [False] * n, "career": [3] * n, "new_race": [False] * n,
        "sire_runs": [500] * n, "no_pedigree": [False] * n,
        "dirt_sire": [0.0] * n, "dirt_bms": [0.0] * n, "dirt_mat5": [0.0] * (n - 2) + [0.05, 0.04], "dirt_lv5": [0.0] * n,
        "turf_mat5": [0.0] * n, "anc_top": ["[]"] * n})


def test_t60_finalize_uses_t60_starters_and_rank(monkeypatch):
    from src.research.pedigree_persona import t60

    monkeypatch.setattr(pc, "ancestor_names", lambda h, ids: {})
    fz = R.load_frozen()
    old = _module(state="pending")
    # 10 番は T-60 で取消（オッズ無し）。9 番は 9 番人気、10 番が居なければ 9 番だけが残る
    copy = {"snapshot_at": "2026-12-06T09:00:00+09:00",
            "entries": [{"horse_number": i, "win_odds": float(i), "popularity": i} for i in range(1, 10)] + [{"horse_number": 10, "win_odds": None}]}
    doc = t60.finalize(_raw(), copy, R, fz, old, datetime(2026, 12, 6, 9, 0, tzinfo=JST))
    fig = doc["figure"]
    assert fig["picks_state"] == "ready" and [p["horse_number"] for p in fig["picks"]] == [9]
    assert fig["picks"][0]["reason"].endswith("今回が初ダート。")
    assert fig["facts"].startswith("初ダート 1 頭")


def test_t60_ranks_over_all_t60_horses_and_needs_six_odds(monkeypatch):
    from src.research.pedigree_persona import t60

    monkeypatch.setattr(pc, "ancestor_names", lambda h, ids: {})
    fz = R.load_frozen()
    old = _module(state="pending")
    raw = _raw().iloc[4:].copy()          # 1〜4 番は生の値が無い（出馬表が変わった等）。それでも人気の順は写しの全頭で付ける
    copy = {"entries": [{"horse_number": i, "win_odds": float(i)} for i in range(1, 11)]}
    fig = t60.finalize(raw, copy, R, fz, old, datetime(2026, 12, 6, 9, 0, tzinfo=JST))["figure"]
    assert [p["horse_number"] for p in fig["picks"]] == [9, 10]        # 生の値のある 5〜10 番で中心化。順は写しの全頭で 9・10 番人気
    thin = {"entries": [{"horse_number": i, "win_odds": None} for i in range(1, 11)]}
    assert t60.finalize(_raw(), thin, R, fz, old, datetime(2026, 12, 6, 9, 0, tzinfo=JST))["figure"]["picks_state"] == "unavailable"


def test_t60_exits_fast_without_eve_output(tmp_path, monkeypatch):
    from src.research.pedigree_persona import t60

    monkeypatch.setenv("KEIBA_PEDIGREE_PERSONA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path / "ra"))
    monkeypatch.setattr("sys.argv", ["t60", "--date", "2026-12-06"])
    t0 = time.time()
    t60.main()
    assert time.time() - t0 < 1.0 and not (tmp_path / "ra").exists()


def test_t60_marks_unavailable_after_start_without_copy(tmp_path, monkeypatch):
    from src.research.pedigree_persona import t60

    monkeypatch.setenv("KEIBA_PEDIGREE_PERSONA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path / "ra"))
    monkeypatch.setenv("KEIBA_ODDS_T60_DIR", str(tmp_path / "odds"))
    monkeypatch.setenv("KEIBA_PEDIGREE_PERSONA_LOCK", str(tmp_path / "lock"))
    d = store.raw_dir("2026-12-06")
    d.mkdir(parents=True)
    _raw().to_parquet(d / "raw.parquet")
    (d / "races.json").write_text(json.dumps(["202606050101"]))
    store.write("202606050101", _module(state="pending") | {"start_time": "10:00"})
    monkeypatch.setattr("sys.argv", ["t60", "--date", "2026-12-06"])

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 12, 6, 10, 1, tzinfo=JST)
    monkeypatch.setattr(t60, "datetime", _DT)
    t60.main()
    assert store.read("202606050101")["figure"]["picks_state"] == "unavailable"


# ── 置き場・ミラー ────────────────────────────────────

def test_module_file_is_in_mirror_bundle_and_not_in_analysis_listing(tmp_path, monkeypatch):
    """欄のファイルはミラーの木の中で、束に入る（CLAUDE.md の再発防止 2）。めぐ分析の一覧（*.json）には混ざらない。"""
    import io
    import tarfile

    from src.ops import gcs_mirror

    f = tmp_path / "data/local/race_analysis/pedigree_persona/202606050101.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    (tmp_path / "data/local/race_analysis/202606050101.json").write_text("{}")
    assert "data/local/race_analysis" in gcs_mirror.BUNDLE_TREES
    blob = gcs_mirror.build_bundle(tmp_path, ("data/local/race_analysis",), window=14.0, now=time.time())
    names = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz").getnames()
    assert "data/local/race_analysis/pedigree_persona/202606050101.json" in names
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path / "data/local/race_analysis"))
    assert sorted(p.name for p in (tmp_path / "data/local/race_analysis").glob("*.json")) == ["202606050101.json"]
    assert store.module_dir() == tmp_path / "data/local/race_analysis/pedigree_persona"


# ── API の合わせ方 ────────────────────────────────────

def _payload(start="10:00"):
    return {"status": "ok", "race": {"race_id": "202606050101", "date": "2026-12-06", "start_time": start}, "entries": [], "modules": [{"id": "x"}]}


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path))
    doc = _module(picks=[{"horse_number": 9, "horse_name": "テスト", "reason": "r"}]) | {"visible": True, "start_time": "10:00"}
    store.write("202606050101", doc)
    return doc


def test_api_adds_module_before_start_and_strips_internal_keys(saved):
    from src.api.v1.routes.race_analysis import with_pedigree_persona

    out = with_pedigree_persona("202606050101", _payload(), now=datetime(2026, 12, 6, 9, 59, tzinfo=JST), developer=False)
    mod = out["modules"][-1]
    assert mod["id"] == "pedigree_persona" and len(out["modules"]) == 2
    assert set(mod) <= {"id", "view", "status", "label", "description", "figure", "notes"}


def test_api_drops_module_after_start(saved):
    """結果の画面・過去のレースで、一覧と着順を並べない（会議の 1 号の条件 1）。"""
    from src.api.v1.routes.race_analysis import with_pedigree_persona

    out = with_pedigree_persona("202606050101", _payload(), now=datetime(2026, 12, 6, 10, 0, tzinfo=JST), developer=True)
    assert [m["id"] for m in out["modules"]] == ["x"]


def test_api_shadow_only_for_developers(saved):
    """影の開催（ホストが visible を付けない）では、開発者にだけ出す。"""
    from src.api.v1.routes.race_analysis import with_pedigree_persona

    store.write("202606050101", saved | {"visible": False})
    now = datetime(2026, 12, 6, 9, 0, tzinfo=JST)
    assert len(with_pedigree_persona("202606050101", _payload(), now=now, developer=False)["modules"]) == 1
    assert len(with_pedigree_persona("202606050101", _payload(), now=now, developer=True)["modules"]) == 2


def test_api_hides_when_start_time_unknown(saved):
    from src.api.v1.routes.race_analysis import with_pedigree_persona

    store.write("202606050101", saved | {"start_time": ""})
    out = with_pedigree_persona("202606050101", _payload(start=""), now=datetime(2026, 12, 6, 8, 0, tzinfo=JST), developer=True)
    assert len(out["modules"]) == 1
    store.write("202606050101", saved | {"start_time": "10:00"})        # めぐ分析の race に時刻が無くても、欄のファイルの時刻で補う
    assert len(with_pedigree_persona("202606050101", _payload(start=""), now=datetime(2026, 12, 6, 9, 0, tzinfo=JST), developer=True)["modules"]) == 2


def test_api_route_is_member_only():
    from src.api.tier import member_only_path

    assert member_only_path("/api/v1/races/202606050101/analysis")
