"""smartrc の系統の台帳（smartrc_sire_lines）: コード→名前、手元の表からの台帳づくり、父系の祖先での補完。"""

import json

import pandas as pd
import pytest

from src.scraper import smartrc_sire_lines as ssl

MASTER = {
    "large": {"1000011": {"name": "サンデー系"}, "1000010": {"name": "ミスプロ系"}},
    "small": {"2000046": {"name": "ディープ系", "large": "1000011"}, "2000039": {"name": "キングマンボ系", "large": "1000010"}},
    "type": {"3000001": {"name": "日本型サンデー系"}},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "master.json").write_text(json.dumps(MASTER, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ssl, "MASTER", tmp_path / "master.json")
    sm = pd.DataFrame({"horse_id": ["h1", "h2", "h3", "h4"], "sire_id": ["S_DEEP", "S_DEEP", "S_KK", "S_NEW"],
                       "sire_name": ["ディープインパクト", "ディープインパクト", "キングカメハメハ", "新種牡馬"]})
    sm.to_parquet(tmp_path / "map.parquet")
    monkeypatch.setattr(ssl, "SIRE_MAP", tmp_path / "map.parquet")
    tp = tmp_path / "tp32"
    tp.mkdir()
    pd.DataFrame({"hcode": ["h1", "h2", "h3"], "f_code": ["c1", "c1", "c2"], "f_name": ["", "", ""],
                  "f_llcode": ["1000011", "1000011", "1000010"], "f_slcode": ["2000046", "2000046", "2000039"],
                  "f_clcode": ["3000001", "3000001", None]}).to_parquet(tp / "career_eq.parquet")
    monkeypatch.setattr(ssl, "TP32", tp)
    ped = tmp_path / "ped"
    (ped / "h4"[:4]).mkdir(parents=True)
    # 新種牡馬（h4 の父）の父はディープインパクト
    (ped / "h4" / "h4.json").write_text(json.dumps({"ancestors": [
        {"generation": 1, "position": 0, "horse_id": "S_NEW", "name": "新種牡馬"},
        {"generation": 2, "position": 0, "horse_id": "S_DEEP", "name": "ディープインパクト"}]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ssl, "PED_DIR", ped)
    return tmp_path / "out.json"


def test_names_fill_large_from_small():
    n = ssl.names({"small": "2000046", "type": "3000001"}, MASTER)
    assert n["small_name"] == "ディープ系" and n["large"] == "1000011" and n["large_name"] == "サンデー系" and n["type_name"] == "日本型サンデー系"


def test_build_from_tables_then_fill_by_paternal_ancestor(env):
    res = ssl.build(env)
    assert res == {"sires": 3, "known": 2, "missing": 1, "conflicts": 0}
    doc = ssl.load(env)
    assert doc["sires"]["S_DEEP"]["small_name"] == "ディープ系" and doc["sires"]["S_DEEP"]["n"] == 2
    assert not ssl.is_complete(env)
    assert ssl.fill_paternal(env) == {"filled": 1, "missing": 0, "known": 3}
    new = ssl.load(env)["sires"]["S_NEW"]
    assert new["small_name"] == "ディープ系" and new["from"] == "paternal" and new["via"]["sire_id"] == "S_DEEP"
    assert ssl.is_complete(env)
    assert ssl.labels(env) == {"S_DEEP": "ディープ系", "S_KK": "キングマンボ系", "S_NEW": "ディープ系"}
    # 作り直しても、父系で埋めた行は残る
    ssl.build(env)
    assert ssl.load(env)["sires"]["S_NEW"]["from"] == "paternal"
